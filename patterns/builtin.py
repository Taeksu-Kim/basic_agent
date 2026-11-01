"""Built-in pattern registrations. Imported lazily by ``registry._ensure_builtins``.

The builders here point at the existing graph builders unchanged -- registering a
pattern adds a name and metadata, not a new code path.
"""

from __future__ import annotations

from agent.patterns.registry import PatternSpec, register
from agent.plan_execute.graph import build_graph
from agent.patterns.workflows import (build_evaluator_optimizer, build_orchestrator_workers,
                                     build_parallel, build_prompt_chain, build_router)
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

# --- Anthropic workflow patterns (docs/PATTERNS.md §3). Delegation targets are
# registry tools; sub-agents of any pattern go in as AgentTool.

register(PatternSpec(
    name="prompt_chain",
    std_name="Prompt chaining",
    builder=build_prompt_chain,
    axis="workflow",
    summary="fixed steps in order (LLM prompt or tool), optional programmatic gate between steps",
    aliases=("chain",),
    weak_model_ok="high: each step is a small, fixed job",
    sources=("A1",),
))

register(PatternSpec(
    name="router",
    std_name="Routing",
    builder=build_router,
    axis="workflow",
    summary="classify the request (schema-constrained) and hand it to one route",
    weak_model_ok="high: classification",
    sources=("A1", "L2"),
))

register(PatternSpec(
    name="parallel",
    std_name="Parallelization (sectioning / voting)",
    builder=build_parallel,
    axis="workflow",
    summary="run branches concurrently, then concat / LLM-synthesise / majority-vote",
    aliases=("parallelization",),
    weak_model_ok="high",
    sources=("A1",),
))

register(PatternSpec(
    name="evaluator_optimizer",
    std_name="Evaluator-optimizer",
    builder=build_evaluator_optimizer,
    axis="workflow",
    summary="generate -> fresh-context evaluator grades against criteria -> revise, up to a cap",
    aliases=("evaluator",),
    weak_model_ok="medium: needs clear, checkable criteria",
    sources=("A1", "A6"),
))

register(PatternSpec(
    name="orchestrator_workers",
    std_name="Orchestrator-workers (dynamic)",
    builder=build_orchestrator_workers,
    axis="3. topology",
    summary="decide subtasks at run time, brief workers (read-only by default), synthesise or re-plan",
    aliases=("orchestrator",),
    weak_model_ok="low-medium: delegation quality depends on the model",
    sources=("A1", "A4", "A9"),
))
