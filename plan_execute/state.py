"""Shared graph state.

``results`` is written concurrently by the parallel task branches in a level, so
it carries a merge reducer. Everything else is written by a single node per step.
"""

from __future__ import annotations

from typing import Annotated, Any

from agent.core.contract import ContractState

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


class AgentState(ContractState, total=False):
    """Contract keys (``iteration`` = planning rounds so far) plus the plan/execute state."""

    # planning
    plan: list[dict[str, Any]]        # serialised Task dicts
    levels: list[list[int]]           # topological execution schedule
    current_level: int                # cursor into `levels`
    # execution
    results: Annotated[dict[int, Any], merge_results]
    # HITL
    plan_rejected: bool               # human rejected the plan -> replan
    preapproved: list[int]            # task ids the level gate already got approval for
    # joining
    decision: str                     # "finish" | "replan"
    decision_reason: str
