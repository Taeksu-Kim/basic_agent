"""Executor: run a single (already-resolved) task against the tool registry.

One task per invocation -- the graph fans a whole level out to this node via
``Send``, so LangGraph runs the level's tasks concurrently and barriers before
advancing. Tool errors are captured into the result (not raised) so the joiner /
early-abort logic can see them; the id key lets ``$N`` references resolve later.
"""

from __future__ import annotations

from typing import Any

from agent.tools import ToolRegistry


async def run_task(payload: dict[str, Any], registry: ToolRegistry) -> dict[str, Any]:
    """Execute one resolved task, returning a ``{"results": {id: value}}`` update.

    ``payload`` carries ``id``, ``tool`` and already-``$N``-substituted ``args``.
    """
    tid = payload["id"]
    try:
        value = await registry.call(payload["tool"], payload.get("args", {}))
    except Exception as exc:  # noqa: BLE001 -- surface as data, not a crash
        value = {"error": f"{type(exc).__name__}: {exc}"}
    return {"results": {tid: value}}
