"""Policy: the ReAct reasoning step -- decide the next action from the trajectory.

Given the question, available tools, and the scratchpad so far, the LLM emits a
single next action: call a ``tool`` (with args) or ``finish`` (with a final
answer). Output is coerced through a JSON schema (vLLM guided decoding).
"""

from __future__ import annotations

import json
import re
from typing import Any

from agent.core.llm import LLMClient
from agent.core.tools import ToolRegistry

REACT_SYSTEM = (
    "You are a ReAct agent. Solve the task step by step. At each step, look at "
    "the question and the steps taken so far, then decide ONE next action: either "
    "call a tool (action=tool, with tool name + args) to gather information, or "
    "finish (action=finish, with the final answer) when you have enough. Base tool "
    "args on prior observations in the scratchpad."
)

ACTION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "thought": {"type": "string"},
        "action": {"type": "string", "enum": ["tool", "finish"]},
        "tool": {"type": "string"},
        "args": {"type": "object"},
        "final": {"type": "string"},
        "result": {"type": "object"},  # optional structured payload on finish
    },
    "required": ["action"],
}


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


def _format_tools(registry: ToolRegistry) -> str:
    lines = []
    for spec in registry.describe():
        line = f"- {spec['name']}"
        if spec.get("description"):
            line += f": {spec['description']}"
        lines.append(line)
    return "\n".join(lines)


def build_user_prompt(state: dict[str, Any], registry: ToolRegistry) -> str:
    return (
        f"Question: {state['query']}\n\n"
        f"Tools:\n{_format_tools(registry)}\n\n"
        f"Steps so far:\n{_format_scratchpad(state.get('scratchpad', []))}\n\n"
        "Decide the next action."
    )


def reason(state: dict[str, Any], llm: LLMClient, registry: ToolRegistry,
           *, system: str | None = None, retries: int = 1) -> dict[str, Any]:
    """Reasoning node: decide the next action and count the step.

    ``system`` appends domain guidance to the base ReAct contract (suffix, so the
    action protocol always stays in force). A malformed action (real models emit
    them occasionally — e.g. ``action=tool`` without a tool name) is retried
    ``retries`` times, then degraded to a graceful ``finish`` carrying the parse
    error instead of crashing the whole run.
    """
    sys_prompt = f"{REACT_SYSTEM}\n\n{system}" if system else REACT_SYSTEM
    user = build_user_prompt(state, registry)
    action: dict[str, Any] | None = None
    for _ in range(retries + 1):
        text = llm.complete(sys_prompt, user, schema=ACTION_SCHEMA)
        try:
            action = parse_action(text)
            break
        except ActionError as exc:
            last_err = exc
            # feed the error back so the retry can self-correct
            user = (
                f"{user}\n\n[Your previous action was invalid: {exc}. "
                'Emit a valid action: {"action": "tool", "tool": "<name>", "args": {...}} '
                'or {"action": "finish", "final": "<answer>"}.]'
            )
    if action is None:
        action = {"action": "finish", "final": "", "parse_error": str(last_err)}
    update: dict[str, Any] = {"action": action, "iteration": int(state.get("iteration", 0)) + 1}
    if action["action"] == "finish":
        update["final"] = action.get("final", "")
        if action.get("result") is not None:  # structured channel (AgentTool result_key)
            update["result"] = action["result"]
    return update
