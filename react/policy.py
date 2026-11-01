"""Policy: the ReAct reasoning step -- decide the next action from the trajectory.

Given the question, available tools, and the scratchpad so far, the LLM emits a
single next action: call a ``tool`` (with args) or ``finish`` (with a final
answer). Output is coerced through a JSON schema (vLLM guided decoding).
"""

from __future__ import annotations

import json
import re
from typing import Any

from agent.core.context import (ContextPolicy, obs_text, render_extras, render_steps,
                                 visible_tool_names)
from agent.core.llm import OUTPUT_ERRORS, ContextOverflowError, LLMClient
from agent.core.tools import ToolRegistry

REACT_SYSTEM = (
    "You are a ReAct agent. Solve the task step by step. At each step, look at "
    "the question and the steps taken so far, then decide ONE next action: either "
    "call a tool (action=tool, with tool name + args) to gather information, or "
    "finish (action=finish, with the final answer) when you have enough. Base tool "
    "args on prior observations in the scratchpad."
)

# Two shapes, so guided decoding can't emit a finish without its answer: with a
# flat schema requiring only "action", a real model (Qwen3.8-27B, 2026-11-01) put
# the answer in "result" and left "final" empty -- the run ended "ok" with no answer.
ACTION_SCHEMA: dict[str, Any] = {
    "anyOf": [
        {
            "type": "object",
            "properties": {
                "thought": {"type": "string"},
                "action": {"const": "tool"},
                "tool": {"type": "string"},
                "args": {"type": "object"},
            },
            "required": ["action", "tool", "args"],
        },
        {
            "type": "object",
            "properties": {
                "thought": {"type": "string"},
                "action": {"const": "finish"},
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


def _extract_json(text: str) -> Any:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        raise ActionError(f"no JSON found in policy output: {text[:200]!r}")
    return json.loads(m.group(0))


def parse_action(text: str) -> dict[str, Any]:
    data = _extract_json(text)
    if not isinstance(data, dict) or data.get("action") not in ("tool", "finish"):
        raise ActionError(f"invalid action: {data!r}")
    if data["action"] == "tool" and not data.get("tool"):
        raise ActionError("tool action requires a 'tool' name")
    return data


OBS_CHAR_LIMIT = 1500  # 스텝당 관측 상한 — 없으면 몇 스텝 만에 컨텍스트 초과 (실측)


def _format_scratchpad(steps: list[dict[str, Any]]) -> str:
    if not steps:
        return "(no steps yet)"
    lines = []
    for i, s in enumerate(steps, 1):
        obs = json.dumps(s.get("observation"), default=str, ensure_ascii=False)
        if len(obs) > OBS_CHAR_LIMIT:
            obs = obs[:OBS_CHAR_LIMIT] + "…[truncated]"
        lines.append(f"[{i}] tool={s.get('tool')} args={json.dumps(s.get('args', {}), ensure_ascii=False)} -> {obs}")
    return "\n".join(lines)


def _format_tools(registry: ToolRegistry, only: list[str] | None = None) -> str:
    return registry.format_for_prompt(only)


def build_user_prompt(state: dict[str, Any], registry: ToolRegistry,
                      policy: ContextPolicy | None = None) -> str:
    parts = [f"Question: {state['query']}"]
    # contract inputs: shown only when present, so the default prompt is unchanged
    if state.get("history"):
        parts.append("Conversation so far:\n" + "\n".join(str(h) for h in state["history"]))
    if state.get("context") is not None:
        ctx = state["context"]
        parts.append("Context from the caller:\n"
                     + (ctx if isinstance(ctx, str) else json.dumps(ctx, default=str, ensure_ascii=False)))
    if policy is None:  # the original prompt, unchanged
        parts.append(f"Tools:\n{_format_tools(registry)}")
        parts.append(f"Steps so far:\n{_format_scratchpad(state.get('scratchpad', []))}")
    else:
        parts += render_extras(state, policy)
        tools = _format_tools(registry, visible_tool_names(registry, policy,
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


COMPACT_SYSTEM = (
    "You compress an agent's working history. Write a concise summary of the steps "
    "below that keeps every fact, number, identifier, decision and open question the "
    "agent may still need; drop redundant tool output. Plain text, no preamble."
)


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
    chunk = "\n".join(
        f"[{i}] tool={s.get('tool')} args={json.dumps(s.get('args', {}), ensure_ascii=False)} -> "
        f"{obs_text(s.get('observation'))[: policy.inline_chars * 4]}"
        for i, s in enumerate(steps[upto:new_upto], upto + 1)
    )
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
    sys_prompt = f"{REACT_SYSTEM}\n\n{system}" if system else REACT_SYSTEM
    compaction: dict[str, Any] = {}
    if policy is not None:
        compaction = compact(state, compact_llm or llm, policy)
    user = build_user_prompt({**state, **compaction}, registry, policy)
    action: dict[str, Any] | None = None
    overflow = ""
    for _ in range(retries + 1):
        try:
            text = llm.complete(sys_prompt, user, schema=ACTION_SCHEMA)
        except ContextOverflowError as exc:  # end the run gracefully, don't crash it
            overflow = str(exc)
            break
        try:
            action = parse_action(text)
            break
        except OUTPUT_ERRORS as exc:  # incl. JSONDecodeError from a brace-y non-JSON reply
            last_err = exc
            # feed the error back so the retry can self-correct
            user = (
                f"{user}\n\n[Your previous action was invalid: {exc}. "
                'Emit a valid action: {"action": "tool", "tool": "<name>", "args": {...}} '
                'or {"action": "finish", "final": "<answer>"}.]'
            )
    if action is None and overflow:
        action = {"action": "finish", "final": "", "overflow": overflow}
    elif action is None:
        action = {"action": "finish", "final": "", "parse_error": str(last_err)}
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
