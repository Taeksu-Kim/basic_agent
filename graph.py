"""Assemble the plan-centric agent graph.

Flow::

    START -> plan ─(dispatch)─▶ run_task* ─▶ advance ─(dispatch)─▶ ... ─▶ join
                                                                          │
                                    finish / cap ◀──(route_join)──────────┤
                                                                          │
                                    replan ───────────────────────────────┘  (back to plan)

* ``dispatch`` fans the current level out to ``run_task`` via ``Send`` (parallel
  within a level), or routes to ``join`` when the levels are exhausted -- or
  early, if any task errored. ``advance`` bumps the level cursor after the
  level's barrier. ``route_join`` ends on ``finish`` or when the replan cap is
  hit, otherwise loops back to ``plan`` (which resets results for the new plan).
"""

from __future__ import annotations

from typing import Any, Callable

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from agent.executor.executor import run_task
from agent.llm import LLMClient
from agent.orchestrator import joiner, planner
from agent.plan import substitute_args
from agent.state import RESET, AgentState
from agent.tools import ToolRegistry


def _has_error(results: dict[int, Any]) -> bool:
    return any(isinstance(v, dict) and "error" in v for v in results.values())


def build_graph(
    *, llm: LLMClient, registry: ToolRegistry, max_iterations: int = 3
) -> Any:
    """Compile the agent graph for a given LLM + tool registry."""

    def plan_node(state: AgentState) -> dict[str, Any]:
        update = planner.plan(state, llm, registry)
        update["results"] = RESET  # fresh result namespace for the new plan
        return update

    def dispatch(state: AgentState) -> list[Send] | str:
        results = state.get("results", {}) or {}
        if _has_error(results):  # early-abort: don't run further levels on failure
            return "join"
        levels = state.get("levels", [])
        cur = state.get("current_level", 0)
        if cur >= len(levels):
            return "join"
        by_id = {t["id"]: t for t in state["plan"]}
        sends: list[Send] = []
        for tid in levels[cur]:
            task = by_id[tid]
            args = substitute_args(task.get("args", {}), results)
            sends.append(Send("run_task", {"id": tid, "tool": task["tool"], "args": args}))
        return sends

    async def run_task_node(payload: dict[str, Any]) -> dict[str, Any]:
        return await run_task(payload, registry)

    def advance_node(state: AgentState) -> dict[str, Any]:
        return {"current_level": state.get("current_level", 0) + 1}

    def join_node(state: AgentState) -> dict[str, Any]:
        return joiner.join(state, llm)

    def route_join(state: AgentState) -> str:
        if state.get("decision") == "finish":
            return END
        if state.get("iteration", 0) >= max_iterations:
            return END  # replan cap reached -> give up gracefully
        return "plan"

    builder = StateGraph(AgentState)
    builder.add_node("plan", plan_node)
    builder.add_node("run_task", run_task_node)
    builder.add_node("advance", advance_node)
    builder.add_node("join", join_node)

    builder.add_edge(START, "plan")
    builder.add_conditional_edges("plan", dispatch, ["run_task", "join"])
    builder.add_edge("run_task", "advance")
    builder.add_conditional_edges("advance", dispatch, ["run_task", "join"])
    builder.add_conditional_edges("join", route_join, ["plan", END])

    return builder.compile()


async def arun(
    graph: Any, query: str, *, history: list[Any] | None = None
) -> AgentState:
    """Run the compiled graph to completion and return the final state."""
    return await graph.ainvoke({"query": query, "history": history or [], "iteration": 0})
