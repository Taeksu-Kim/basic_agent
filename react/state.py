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


class ReactState(TypedDict, total=False):
    query: str
    history: list[Any]  # accepted for a uniform invoke contract (unused by the loop)
    scratchpad: Annotated[list[dict[str, Any]], append_steps]
    iteration: int          # number of reasoning steps taken
    action: dict[str, Any]  # the last decided action
    final: str
