"""Joiner: after the whole plan runs, decide finish vs replan.

This is the LLMCompiler-style joiner: it looks at the accumulated results and
either produces a final answer (``finish``) or asks to go back to the planner
(``replan``) with a reason. It is a light classifier -- the heavy planning only
happens if it says ``replan``. The replan cap lives in the graph, not here.
"""

from __future__ import annotations

import json
from typing import Any

from agent.core.context import complete_fitting, render_results
from agent.core.llm import (ContextOverflowError, LLMClient, OutputError, ask_parsed, extract_json,
                            with_guidance)
from agent.plan_execute.dag import PlanError
from agent.prompts import DECISION_HINT, FINAL_ROUND_NOTE, JOINER_SYSTEM

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


def parse_decision(text: str) -> dict[str, Any]:
    data = extract_json(text, what="joiner output")
    if not isinstance(data, dict) or data.get("decision") not in ("finish", "replan"):
        raise PlanError(f"invalid joiner decision: {data!r}")
    return data


def build_user_prompt(state: dict[str, Any], policy: Any = None) -> str:
    if policy is None:  # the original prompt, unchanged
        results = json.dumps(state.get('results', {}), default=str, indent=2, ensure_ascii=False)
    else:  # ContextPolicy: cap each task's result (agent.core.context)
        results = render_results(state.get("results", {}), policy)
    return f"Task: {state['query']}\n\nResults:\n{results}"


def join(state: dict[str, Any], llm: LLMClient, *, retries: int = 1,
         policy: Any = None, final_round: bool = False,
         system: str | None = None) -> dict[str, Any]:
    """Joiner node: classify finish/replan and stash a final answer / reason.

    An unusable decision is retried ``retries`` times with the error fed back;
    after that :class:`PlanError` propagates and the graph ends the run with
    ``error`` set instead of crashing.

    ``final_round`` (the replan cap is reached) allows only ``finish`` -- by
    schema, not just by instruction -- so a capped run still returns an answer.
    A prompt that overflows the context window is re-rendered with capped results.
    ``system`` appends domain guidance (before the final-round note, which stays last).
    """
    system = with_guidance(JOINER_SYSTEM, system) + (FINAL_ROUND_NOTE if final_round else "")
    schema = FINISH_ONLY_SCHEMA if final_round else DECISION_SCHEMA
    try:
        decision = ask_parsed(
            llm, system, "", parse_decision, schema=schema, retries=retries, what="decision",
            hint=DECISION_HINT,
            first=lambda: complete_fitting(llm, system, lambda p: build_user_prompt(state, p),
                                           schema=schema, policy=policy),
        )
    except ContextOverflowError as exc:
        raise PlanError(f"joiner prompt exceeds the context window: {exc}") from exc
    except OutputError as exc:
        raise PlanError(f"joiner produced no valid decision after {exc.attempts} attempts: "
                        f"{exc.last}") from exc
    return {
        "decision": decision["decision"],
        "final": decision.get("final", ""),
        "decision_reason": decision.get("reason", ""),
    }
