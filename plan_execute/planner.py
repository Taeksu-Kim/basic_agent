"""Planner: turn a query (+ history / replan feedback) into a task DAG.

The planner emits a DAG (tasks with ``deps`` and ``$N`` arg references); the
*levels* (execution schedule) are computed here from that DAG -- the LLM never
produces level groupings itself. Routing is static: each task's ``tool`` field is
the binding. Output is coerced through a JSON schema (vLLM guided decoding).
"""

from __future__ import annotations

import json
import re
from typing import Any, Optional

from agent.llm import LLMClient
from agent.plan_execute.dag import Plan, PlanError, topological_levels, validate_plan
from agent.tools import ToolRegistry

PLANNER_SYSTEM = (
    "You are the planner in a plan-and-execute agent. Decompose the user's task "
    "into a DAG of tool calls. Each task has an integer id, a tool name (chosen "
    "from the available tools), an args object, and deps (ids it depends on). "
    "To feed one task's output into another, put \"$ID\" in the consumer's args; "
    "that also implies a dependency. Emit only tasks that are necessary. Do NOT "
    "group tasks into stages -- just give the DAG; dependencies determine order."
)

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


def _extract_json(text: str) -> Any:
    """Parse JSON from a model response, tolerating minor wrapping."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    m = re.search(r"(\{.*\}|\[.*\])", text, re.DOTALL)
    if not m:
        raise PlanError(f"no JSON found in planner output: {text[:200]!r}")
    return json.loads(m.group(1))


def parse_plan(text: str) -> Plan:
    """Parse and validate a plan from raw model output."""
    data = _extract_json(text)
    if isinstance(data, dict):
        data = data.get("tasks", [])
    if not isinstance(data, list):
        raise PlanError(f"expected a task list, got {type(data).__name__}")
    plan = Plan.from_list(data)
    validate_plan(plan)
    return plan


def build_user_prompt(state: dict[str, Any], registry: ToolRegistry) -> str:
    parts = [f"Task: {state['query']}"]
    history = state.get("history")
    if history:
        parts.append("Conversation so far:\n" + "\n".join(str(h) for h in history))
    # On a replan, show what was tried and why we came back.
    if state.get("results"):
        parts.append("Results from the previous plan:\n" + json.dumps(state["results"], default=str))
    if state.get("decision_reason"):
        parts.append(f"Why replanning: {state['decision_reason']}")
    parts.append("Available tools:\n" + _format_tools(registry))
    return "\n\n".join(parts)


def _format_tools(registry: ToolRegistry) -> str:
    lines = []
    for spec in registry.describe():
        line = f"- {spec['name']}"
        if spec.get("description"):
            line += f": {spec['description']}"
        if spec.get("args_schema"):
            line += f"\n    args: {json.dumps(spec['args_schema'])}"
        lines.append(line)
    return "\n".join(lines)


def plan(state: dict[str, Any], llm: LLMClient, registry: ToolRegistry) -> dict[str, Any]:
    """Planner node: produce plan + computed levels, reset the level cursor."""
    user = build_user_prompt(state, registry)
    text = llm.complete(PLANNER_SYSTEM, user, schema=PLAN_SCHEMA)
    parsed = parse_plan(text)
    levels = topological_levels(parsed)
    return {
        "plan": parsed.to_list(),
        "levels": levels,
        "current_level": 0,
        "iteration": int(state.get("iteration", 0)) + 1,
    }
