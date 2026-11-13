"""State for the ReAct agent.

A single reactive loop (no fan-out), so ``scratchpad`` just appends one step per
``act``. Each step records the thought, the tool + args, and the observation, so
the next ``reason`` sees the full trajectory.
"""

from __future__ import annotations

from typing import Annotated, Any

from agent.core.contract import ContractState


def append_steps(a: list | None, b: list | None) -> list:
    """Reducer: append newly-produced steps to the scratchpad."""
    return (a or []) + (b or [])


def union_names(a: list | None, b: list | None) -> list:
    """Reducer: order-preserving set union (tools loaded via tool_search)."""
    out = list(a or [])
    out += [x for x in (b or []) if x not in out]
    return out


class ReactState(ContractState, total=False):
    """Contract keys (``iteration`` = reasoning steps taken) plus the ReAct loop state.

    Note: ``scratchpad`` appends across runs on the same ``thread_id`` (a thread
    is one request; reusing it for a new query carries the old steps along).
    """

    scratchpad: Annotated[list[dict[str, Any]], append_steps]
    action: dict[str, Any]  # the last decided action
    # context management (only used with a ContextPolicy; see agent.core.context)
    notes: Annotated[list[str], append_steps]        # notes_write
    loaded_tools: Annotated[list[str], union_names]  # tools found via tool_search
    summary: str                                      # compaction: running summary
    summarized_upto: int                              # compaction: steps covered by summary
