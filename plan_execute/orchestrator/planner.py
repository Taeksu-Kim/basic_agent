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

from agent.core.context import ContextPolicy, render_results
from agent.core.llm import OUTPUT_ERRORS, LLMClient
from agent.plan_execute.dag import Plan, PlanError, topological_levels, validate_plan
from agent.core.tools import ToolRegistry

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


def parse_plan(text: str, registry: ToolRegistry | None = None) -> Plan:
    """Parse and validate a plan from raw model output.

    With ``registry``, tasks naming unregistered tools are rejected too.
    """
    data = _extract_json(text)
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




def retry_feedback(user: str, what: str, exc: Exception, example: str) -> str:
    """Append the parse/validation error so the next attempt can self-correct."""
    return f"{user}\n\n[Your previous {what} was invalid: {exc}. Emit a valid one, e.g. {example}]"


def build_user_prompt(state: dict[str, Any], registry: ToolRegistry,
                      policy: ContextPolicy | None = None) -> str:
    parts = [f"Task: {state['query']}"]
    history = state.get("history")
    if history:
        parts.append("Conversation so far:\n" + "\n".join(str(h) for h in history))
    if state.get("context") is not None:  # contract input: shown only when present
        ctx = state["context"]
        parts.append("Context from the caller:\n"
                     + (ctx if isinstance(ctx, str) else json.dumps(ctx, default=str, ensure_ascii=False)))
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
    policy: ContextPolicy | None = None,
) -> dict[str, Any]:
    """Planner node: produce plan + computed levels, reset the level cursor.

    An unusable plan (not JSON, not a DAG, unknown tool) is retried ``retries``
    times with the error fed back; after that :class:`PlanError` propagates and
    the graph ends the run with ``error`` set instead of crashing.
    """
    from agent.core.context import complete_fitting
    from agent.core.llm import ContextOverflowError

    user: str | None = None
    parsed: Plan | None = None
    last: Exception | None = None
    for _ in range(retries + 1):
        try:
            if user is None:  # overflow (big previous results) -> re-render capped, once
                text, user = complete_fitting(llm, PLANNER_SYSTEM,
                                              lambda p: build_user_prompt(state, registry, p),
                                              schema=PLAN_SCHEMA, policy=policy)
            else:
                text = llm.complete(PLANNER_SYSTEM, user, schema=PLAN_SCHEMA)
        except ContextOverflowError as exc:
            raise PlanError(f"planner prompt exceeds the context window: {exc}") from exc
        try:
            parsed = parse_plan(text, registry)
            break
        except OUTPUT_ERRORS as exc:
            last = exc
            user = retry_feedback(
                user, "plan", exc,
                '{"tasks": [{"id": 1, "tool": "<registered tool>", "args": {...}, "deps": []}]}',
            )
    if parsed is None:
        raise PlanError(f"planner produced no valid plan after {retries + 1} attempts: {last}")
    levels = topological_levels(parsed)
    return {
        "plan": parsed.to_list(),
        "levels": levels,
        "current_level": 0,
        "iteration": int(state.get("iteration", 0)) + 1,
    }
