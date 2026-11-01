"""State for the ReAct agent.

A single reactive loop (no fan-out), so ``scratchpad`` just appends one step per
``act``. Each step records the thought, the tool + args, and the observation, so
the next ``reason`` sees the full trajectory.
"""

from __future__ import annotations

from typing import Annotated, Any, TypedDict


def append_steps(a: list | None, b: list | None) -> list:
    """Reducer: append newly-produced steps to the scratchpad."""
    return (a or []) + (b or [])


def union_names(a: list | None, b: list | None) -> list:
    """Reducer: order-preserving set union (tools loaded via tool_search)."""
    out = list(a or [])
    out += [x for x in (b or []) if x not in out]
    return out


class ReactState(TypedDict, total=False):
    """Keys follow :mod:`agent.core.contract`.

    Note: ``scratchpad`` appends across runs on the same ``thread_id`` (a thread
    is one request; reusing it for a new query carries the old steps along).
    """

    query: str
    history: list[Any]      # prior turns; shown to the model when non-empty
    context: Any            # caller-provided context; shown to the model when present
    scratchpad: Annotated[list[dict[str, Any]], append_steps]
    iteration: int          # number of reasoning steps taken
    action: dict[str, Any]  # the last decided action
    final: str
    result: Any             # optional structured payload set on finish
    status: str | None      # "ok" | "error" on finish; unset at END -> capped (contract)
    error: str
    # context management (only used with a ContextPolicy; see agent.core.context)
    notes: Annotated[list[str], append_steps]        # notes_write
    loaded_tools: Annotated[list[str], union_names]  # tools found via tool_search
    summary: str                                      # compaction: running summary
    summarized_upto: int                              # compaction: steps covered by summary
