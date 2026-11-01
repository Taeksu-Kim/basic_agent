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
    "short reason describing what to try next. If the results show the requested "
    "information is not there, finish and say so instead of replanning."
)

FINAL_ROUND_NOTE = (
    "\n\nThis is the last round: you must finish now. Give the best answer the "
    "results support; if something could not be found, say so plainly."
)

# Two shapes so guided decoding can't emit "finish" without the answer or
# "replan" without a reason (same fix as react's ACTION_SCHEMA, see there).
DECISION_SCHEMA: dict[str, Any] = {
    "anyOf": [
        {
            "type": "object",
            "properties": {"decision": {"type": "string", "const": "finish"}, "final": {"type": "string"}},
            "required": ["decision", "final"],
        },
        {
            "type": "object",
            "properties": {"decision": {"type": "string", "const": "replan"}, "reason": {"type": "string"}},
            "required": ["decision", "reason"],
        },
    ]
}


FINISH_ONLY_SCHEMA: dict[str, Any] = DECISION_SCHEMA["anyOf"][0]


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


def build_user_prompt(state: dict[str, Any], policy: Any = None) -> str:
    if policy is None:  # the original prompt, unchanged
        results = json.dumps(state.get('results', {}), default=str, indent=2, ensure_ascii=False)
    else:  # ContextPolicy: cap each task's result (agent.core.context)
        from agent.core.context import render_results

        results = render_results(state.get("results", {}), policy)
    return f"Task: {state['query']}\n\nResults:\n{results}"


def join(state: dict[str, Any], llm: LLMClient, *, retries: int = 1,
         policy: Any = None, final_round: bool = False) -> dict[str, Any]:
    """Joiner node: classify finish/replan and stash a final answer / reason.

    An unusable decision is retried ``retries`` times with the error fed back;
    after that :class:`PlanError` propagates and the graph ends the run with
    ``error`` set instead of crashing.

    ``final_round`` (the replan cap is reached) allows only ``finish`` -- by
    schema, not just by instruction -- so a capped run still returns an answer.
    A prompt that overflows the context window is re-rendered with capped results.
    """
    from agent.core.context import complete_fitting
    from agent.core.llm import OUTPUT_ERRORS, ContextOverflowError
    from agent.plan_execute.orchestrator.planner import retry_feedback

    system = JOINER_SYSTEM + (FINAL_ROUND_NOTE if final_round else "")
    schema = FINISH_ONLY_SCHEMA if final_round else DECISION_SCHEMA
    decision: dict[str, Any] | None = None
    last: Exception | None = None
    user: str | None = None
    for _ in range(retries + 1):
        try:
            if user is None:
                text, user = complete_fitting(llm, system, lambda p: build_user_prompt(state, p),
                                              schema=schema, policy=policy)
            else:
                text = llm.complete(system, user, schema=schema)
        except ContextOverflowError as exc:
            raise PlanError(f"joiner prompt exceeds the context window: {exc}") from exc
        try:
            decision = parse_decision(text)
            break
        except OUTPUT_ERRORS as exc:
            last = exc
            user = retry_feedback(user, "decision", exc,
                                  '{"decision": "finish", "final": "<answer>"}')
    if decision is None:
        raise PlanError(f"joiner produced no valid decision after {retries + 1} attempts: {last}")
    return {
        "decision": decision["decision"],
        "final": decision.get("final", ""),
        "decision_reason": decision.get("reason", ""),
    }
