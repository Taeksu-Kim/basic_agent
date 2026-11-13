"""Policy: the ReAct reasoning step -- decide the next action from the trajectory.

Given the question, available tools, and the scratchpad so far, the LLM emits a
single next action: call a ``tool`` (with args) or ``finish`` (with a final
answer). Output is coerced through a JSON schema (vLLM guided decoding).
"""

from __future__ import annotations

import json
from typing import Any

from agent.core.context import (ContextPolicy, obs_text, render_extras, render_steps, step_line,
                                 visible_tool_names)
from agent.core.contract import render_inputs
from agent.core.llm import (ContextOverflowError, LLMClient, OutputError, ask_parsed, extract_json,
                            with_guidance)
from agent.core.tools import ToolRegistry
from agent.prompts import ACTION_HINT, COMPACT_SYSTEM, REACT_SYSTEM

# Two shapes, so guided decoding can't emit a finish without its answer: with a
# flat schema requiring only "action", a real model (Qwen3.8-27B, 2026-11-01) put
# the answer in "result" and left "final" empty -- the run ended "ok" with no answer.
ACTION_SCHEMA: dict[str, Any] = {
    "anyOf": [
        {
            "type": "object",
            "properties": {
                "thought": {"type": "string"},
                "action": {"type": "string", "const": "tool"},
                "tool": {"type": "string"},
                "args": {"type": "object"},
            },
            "required": ["action", "tool", "args"],
        },
        {
            "type": "object",
            "properties": {
                "thought": {"type": "string"},
                "action": {"type": "string", "const": "finish"},
                "final": {"type": "string"},
                "result": {"type": "object"},  # optional structured payload on finish
            },
            "required": ["action", "final"],
        },
    ]
}


def answer_text(final: Any, result: Any) -> str:
    """The answer to record on finish. If the model left ``final`` empty but put
    the answer in ``result`` (seen with real models), use that instead of dropping it."""
    if isinstance(final, str) and final.strip():
        return final
    if result is None:
        return final or ""
    if isinstance(result, str):
        return result
    if isinstance(result, dict) and len(result) == 1:
        (only,) = result.values()
        if isinstance(only, (str, int, float)):
            return str(only)
    return json.dumps(result, default=str, ensure_ascii=False)


class ActionError(ValueError):
    """Raised when the model's action is malformed."""


def parse_action(text: str) -> dict[str, Any]:
    data = extract_json(text, what="policy output")
    if not isinstance(data, dict) or data.get("action") not in ("tool", "finish"):
        raise ActionError(f"invalid action: {data!r}")
    if data["action"] == "tool" and not data.get("tool"):
        raise ActionError("tool action requires a 'tool' name")
    return data


def build_user_prompt(state: dict[str, Any], registry: ToolRegistry,
                      policy: ContextPolicy | None = None) -> str:
    """``policy=None`` renders as ``ContextPolicy()`` (no feature on): every observation
    capped at ``inline_chars`` with ``…[truncated]``, no harness tools -- the original prompt."""
    policy = policy or ContextPolicy()
    parts = [f"Question: {state['query']}", *render_inputs(state), *render_extras(state, policy)]
    tools = registry.format_for_prompt(visible_tool_names(registry, policy,
                                                          state.get("loaded_tools") or []))
    harness = policy.harness_tools(registry)
    if harness:
        tools = "\n".join(filter(None, [tools, *(f"- {u}" for u in harness.values())]))
    parts.append(f"Tools:\n{tools}")
    steps = render_steps(state.get("scratchpad") or [], policy,
                         summary=state.get("summary", ""),
                         summarized_upto=int(state.get("summarized_upto", 0) or 0))
    parts.append(f"Steps so far:\n{steps}")
    parts.append("Decide the next action.")
    return "\n\n".join(parts)




def compact(state: dict[str, Any], llm: LLMClient, policy: ContextPolicy) -> dict[str, Any]:
    """Compaction: fold older steps into the running summary when the rendered
    scratchpad passes ``policy.compact_at_chars``. Returns the state update ({} if
    nothing to do). The newest ``compact_keep_recent`` steps stay verbatim."""
    if policy.compact_at_chars is None:
        return {}
    steps = state.get("scratchpad") or []
    upto = int(state.get("summarized_upto", 0) or 0)
    summary = state.get("summary", "")
    rendered = render_steps(steps, policy, summary=summary, summarized_upto=upto)
    new_upto = len(steps) - policy.compact_keep_recent
    if len(rendered) <= policy.compact_at_chars or new_upto <= upto:
        return {}
    chunk = "\n".join(step_line(i, s, obs_text(s.get("observation"))[: policy.inline_chars * 4])
                      for i, s in enumerate(steps[upto:new_upto], upto + 1))
    user = (f"Task: {state['query']}\n\n"
            + (f"Summary so far (steps 1-{upto}): {summary}\n\n" if summary else "")
            + f"Steps {upto + 1}-{new_upto}:\n{chunk}")
    new_summary = llm.complete(COMPACT_SYSTEM, user).strip()
    return {"summary": new_summary, "summarized_upto": new_upto}


def reason(state: dict[str, Any], llm: LLMClient, registry: ToolRegistry,
           *, system: str | None = None, retries: int = 1,
           policy: ContextPolicy | None = None, compact_llm: LLMClient | None = None) -> dict[str, Any]:
    """Reasoning node: decide the next action and count the step.

    ``system`` appends domain guidance to the base ReAct contract (suffix, so the
    action protocol always stays in force). A malformed action (real models emit
    them occasionally — e.g. ``action=tool`` without a tool name) is retried
    ``retries`` times, then degraded to a graceful ``finish`` carrying the parse
    error instead of crashing the whole run.

    With a ``policy`` the prompt is rendered through it (see agent.core.context),
    and compaction -- if enabled and due -- runs first.
    """
    sys_prompt = with_guidance(REACT_SYSTEM, system)
    compaction: dict[str, Any] = {}
    overflow = ""
    if policy is not None:
        try:
            compaction = compact(state, compact_llm or llm, policy)
        except ContextOverflowError as exc:  # the summary prompt itself overflowed: end, don't crash
            overflow = str(exc)
    action: dict[str, Any] | None = None
    if not overflow:
        user = build_user_prompt({**state, **compaction}, registry, policy)
        try:
            action = ask_parsed(llm, sys_prompt, user, parse_action, schema=ACTION_SCHEMA,
                                retries=retries, what="action", hint=ACTION_HINT)
        except ContextOverflowError as exc:  # end the run gracefully, don't crash it
            overflow = str(exc)
        except OutputError as exc:  # still malformed after retries: degrade to a graceful finish
            action = {"action": "finish", "final": "", "parse_error": str(exc.last)}
    if action is None:
        action = {"action": "finish", "final": "", "overflow": overflow}
    update: dict[str, Any] = {"action": action, "iteration": int(state.get("iteration", 0)) + 1,
                              **compaction}
    if action["action"] == "finish":
        update["final"] = answer_text(action.get("final", ""), action.get("result"))
        if action.get("result") is not None:  # structured channel (AgentTool result_key)
            update["result"] = action["result"]
        if "overflow" in action:
            update["status"] = "error"
            update["error"] = f"prompt exceeds the context window: {action['overflow'][:200]}"
        elif "parse_error" in action:  # contract: unusable output after retries
            update["status"] = "error"
            update["error"] = f"policy produced no valid action: {action['parse_error']}"
        else:
            update["status"] = "ok"
    else:
        # not finished: clear any status left on this thread by a previous run, so
        # hitting the step cap now reads as "capped" -- whoever started the run
        update["status"] = None
    return update
