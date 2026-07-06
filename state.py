"""Shared graph state.

``results`` is written concurrently by the parallel task branches in a level, so
it carries a merge reducer. Everything else is written by a single node per step.
"""

from __future__ import annotations

from typing import Annotated, Any, TypedDict


RESET = "__RESET__"  # sentinel: replace (clear) the channel instead of merging


def merge_results(a: dict[int, Any] | None, b: dict[int, Any] | None) -> dict[int, Any]:
    """Reducer: shallow-merge partial result dicts from parallel task branches.

    A ``RESET`` sentinel clears the channel -- used by the planner node to give a
    new plan a fresh result namespace on replan.
    """
    if b == RESET:
        return {}
    out = dict(a or {})
    out.update(b or {})
    return out


class AgentState(TypedDict, total=False):
    # inputs
    query: str
    history: list[Any]
    # planning
    plan: list[dict[str, Any]]        # serialised Task dicts
    levels: list[list[int]]           # topological execution schedule
    current_level: int                # cursor into `levels`
    iteration: int                    # number of planning rounds so far
    # execution
    results: Annotated[dict[int, Any], merge_results]
    # joining
    decision: str                     # "finish" | "replan"
    decision_reason: str
    final: str
