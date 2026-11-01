"""Executor: run a single (already-resolved) task against the tool registry.

One task per invocation -- the graph fans a whole level out to this node via
``Send``, so LangGraph runs the level's tasks concurrently and barriers before
advancing. Execution goes through :func:`agent.core.execution.execute_tool`, the
path every pattern shares: tool errors become a failure result the joiner /
early-abort logic can see, and interrupts raised inside the tool (``ask_human``,
a nested sub-agent's approval gate) pause the graph instead of being swallowed.
The id key lets ``$N`` references resolve later.
"""

from __future__ import annotations

from typing import Any

from agent.core.execution import execute_tool, failure
from agent.core.tools import ToolRegistry


async def run_task(payload: dict[str, Any], registry: ToolRegistry) -> dict[str, Any]:
    """Execute one resolved task, returning a ``{"results": {id: value}}`` update.

    ``payload`` carries ``id``, ``tool``, already-``$N``-substituted ``args`` and
    ``preapproved`` (set when the level gate already collected the human's
    approval, so no interrupt fires mid-fan-out). A ``requires_approval`` tool that
    somehow reaches here un-gated still pauses (``inline``) rather than running.
    """
    tid = payload["id"]
    if payload.get("arg_error"):  # its $N reference didn't resolve: fail the task, don't run it
        return {"results": {tid: failure(payload["arg_error"])}}
    value = await execute_tool(
        registry, payload["tool"], payload.get("args", {}),
        approval="preapproved" if payload.get("preapproved") else "inline",
    )
    return {"results": {tid: value}}
