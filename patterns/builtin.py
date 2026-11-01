"""Built-in pattern registrations. Imported lazily by ``registry._ensure_builtins``.

The builders here point at the existing graph builders unchanged -- registering a
pattern adds a name and metadata, not a new code path.
"""

from __future__ import annotations

from agent.patterns.registry import PatternSpec, register
from agent.plan_execute.graph import build_graph
from agent.react.graph import build_react_agent

register(PatternSpec(
    name="react",
    std_name="ReAct",
    builder=build_react_agent,
    axis="2. planning style",
    summary="one reactive loop: reason -> one tool call -> observe -> ... -> finish",
    weak_model_ok="medium: drifts on long trajectories",
    sources=("P1",),
))

register(PatternSpec(
    name="plan_execute",
    std_name="Plan-and-Execute + LLMCompiler",
    builder=build_graph,
    axis="2. planning style",
    summary="planner writes a tool-call DAG -> levels run in parallel -> joiner finishes or replans",
    aliases=("plan",),
    weak_model_ok="high: structure compensates for a weaker model",
    sources=("P2", "P4"),
))
