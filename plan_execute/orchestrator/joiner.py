"""Joiner: after the whole plan runs, decide finish vs replan.

This is the LLMCompiler-style joiner: it looks at the accumulated results and
either produces a final answer (``finish``) or asks to go back to the planner
(``replan``) with a reason. It is a light classifier -- the heavy planning only
happens if it says ``replan``. The replan cap lives in the graph, not here.
"""

from __future__ import annotations

import json
import re
from typing import Any

from agent.core.llm import LLMClient
from agent.plan_execute.dag import PlanError

JOINER_SYSTEM = (
    "You are the joiner in a plan-and-execute agent. Given the user's task and "
    "the results of the executed plan, decide whether the task is complete. If "
    "so, return decision=finish with a final answer. If the results are "
    "insufficient or an error blocked progress, return decision=replan with a "
    "short reason describing what to try next."
)

DECISION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "decision": {"type": "string", "enum": ["finish", "replan"]},
        "final": {"type": "string"},
        "reason": {"type": "string"},
    },
    "required": ["decision"],
}


def _extract_json(text: str) -> Any:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        raise PlanError(f"no JSON found in joiner output: {text[:200]!r}")
    return json.loads(m.group(0))


def parse_decision(text: str) -> dict[str, Any]:
    data = _extract_json(text)
    if not isinstance(data, dict) or data.get("decision") not in ("finish", "replan"):
        raise PlanError(f"invalid joiner decision: {data!r}")
    return data


def build_user_prompt(state: dict[str, Any]) -> str:
    return (
        f"Task: {state['query']}\n\n"
        f"Results:\n{json.dumps(state.get('results', {}), default=str, indent=2)}"
    )


def join(state: dict[str, Any], llm: LLMClient) -> dict[str, Any]:
    """Joiner node: classify finish/replan and stash a final answer / reason."""
    text = llm.complete(JOINER_SYSTEM, build_user_prompt(state), schema=DECISION_SCHEMA)
    decision = parse_decision(text)
    return {
        "decision": decision["decision"],
        "final": decision.get("final", ""),
        "decision_reason": decision.get("reason", ""),
    }
