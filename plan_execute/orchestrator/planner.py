"""Planner: turn a query (+ history / replan feedback) into a task DAG.

The planner emits a DAG (tasks with ``deps`` and ``$N`` arg references); the
*levels* (execution schedule) are computed here from that DAG -- the LLM never
produces level groupings itself. Routing is static: each task's ``tool`` field is
the binding. Output is coerced through a JSON schema (vLLM guided decoding).
"""

from __future__ import annotations

import json
from typing import Any

from agent.core.context import ContextPolicy, complete_fitting, render_results
from agent.core.contract import render_inputs
from agent.core.llm import (ContextOverflowError, LLMClient, OutputError, ask_parsed, extract_json,
                            with_guidance)
from agent.core.tools import ToolRegistry
from agent.plan_execute.dag import Plan, PlanError, topological_levels, validate_plan
from agent.prompts import PLANNER_SYSTEM, PLAN_HINT

# JSON schema handed to the model (vLLM json_schema / response_format).
PLAN_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "tasks": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "integer"},
                    "tool": {"type": "string"},
                    "args": {"type": "object"},
                    "deps": {"type": "array", "items": {"type": "integer"}},
                },
                "required": ["id", "tool"],
            },
        }
    },
    "required": ["tasks"],
}


def parse_plan(text: str, registry: ToolRegistry | None = None) -> Plan:
    """Parse and validate a plan from raw model output.

    With ``registry``, tasks naming unregistered tools are rejected too.
    """
    data = extract_json(text, what="planner output", allow_list=True)
    if isinstance(data, dict):
        data = data.get("tasks", [])
    if not isinstance(data, list):
        raise PlanError(f"expected a task list, got {type(data).__name__}")
    try:
        plan = Plan.from_list(data)
    except (TypeError, KeyError, AttributeError) as exc:  # e.g. tasks that are not objects
        raise PlanError(f"malformed task list: {exc}") from exc
    validate_plan(plan, registry)
    return plan


def build_user_prompt(state: dict[str, Any], registry: ToolRegistry,
                      policy: ContextPolicy | None = None) -> str:
    parts = [f"Task: {state['query']}", *render_inputs(state)]
    # On a replan, show what was tried and why we came back.
    if state.get("results"):
        rendered = (json.dumps(state["results"], default=str, ensure_ascii=False) if policy is None
                    else render_results(state["results"], policy))  # policy: cap each result
        parts.append("Results from the previous plan:\n" + rendered)
    if state.get("decision_reason"):
        parts.append(f"Why replanning: {state['decision_reason']}")
    parts.append("Available tools:\n" + _format_tools(registry))
    return "\n\n".join(parts)



def _format_tools(registry: ToolRegistry) -> str:
    return registry.format_for_prompt()


def plan(
    state: dict[str, Any], llm: LLMClient, registry: ToolRegistry, *, retries: int = 1,
    policy: ContextPolicy | None = None, system: str | None = None,
) -> dict[str, Any]:
    """Planner node: produce plan + computed levels, reset the level cursor.

    An unusable plan (not JSON, not a DAG, unknown tool) is retried ``retries``
    times with the error fed back; after that :class:`PlanError` propagates and
    the graph ends the run with ``error`` set instead of crashing. ``system``
    appends domain guidance to the planner prompt.
    """
    sys_prompt = with_guidance(PLANNER_SYSTEM, system)
    try:
        parsed = ask_parsed(
            llm, sys_prompt, "", lambda t: parse_plan(t, registry), schema=PLAN_SCHEMA,
            retries=retries, what="plan", hint=PLAN_HINT,
            # overflow (big previous results) -> re-render capped, once
            first=lambda: complete_fitting(llm, sys_prompt,
                                           lambda p: build_user_prompt(state, registry, p),
                                           schema=PLAN_SCHEMA, policy=policy),
        )
    except ContextOverflowError as exc:
        raise PlanError(f"planner prompt exceeds the context window: {exc}") from exc
    except OutputError as exc:
        raise PlanError(f"planner produced no valid plan after {exc.attempts} attempts: "
                        f"{exc.last}") from exc
    levels = topological_levels(parsed)
    return {
        "plan": parsed.to_list(),
        "levels": levels,
        "current_level": 0,
        "iteration": int(state.get("iteration", 0)) + 1,
    }
