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
from langgraph.types import Command, Send, interrupt

from agent.plan_execute.executor import run_task
from agent.core.llm import LLMClient
from agent.plan_execute.orchestrator import joiner, planner
from agent.plan_execute.dag import Plan, substitute_args, topological_levels, validate_plan
from agent.plan_execute.state import RESET, AgentState
from agent.core.tools import ToolRegistry
from agent.core.hitl import approved, feedback_of


def _has_error(results: dict[int, Any]) -> bool:
    return any(isinstance(v, dict) and "error" in v for v in results.values())


def build_graph(
    *,
    llm: LLMClient,
    registry: ToolRegistry,
    max_iterations: int = 3,
    checkpointer: Any = None,
    approve_plans: bool = False,
    approve_levels: bool = False,
) -> Any:
    """Compile the agent graph for a given LLM + tool registry.

    ``approve_plans=True`` inserts a HITL gate after ``plan``: the graph pauses
    (``interrupt``) with the proposed plan and resumes on a human verdict —
    approve, ``{"plan": [...]}`` to substitute an edited plan, or reject (with
    optional feedback) to send the planner back around.

    ``approve_levels=True`` pauses at every level barrier (after a parallel
    level's tasks have all joined, before the next level dispatches) showing that
    level's results — approve continues, reject (with feedback) replans. The
    barrier is single-threaded, so no interrupt ever fires mid-fan-out.

    Both gates require ``checkpointer``.
    """

    def plan_node(state: AgentState) -> dict[str, Any]:
        update = planner.plan(state, llm, registry)
        update["results"] = RESET  # fresh result namespace for the new plan
        update["plan_rejected"] = False
        return update

    def approve_node(state: AgentState) -> dict[str, Any]:
        verdict = interrupt(
            {"type": "plan_approval", "plan": state["plan"], "levels": state["levels"]}
        )
        if approved(verdict):
            return {"plan_rejected": False}
        if isinstance(verdict, dict) and verdict.get("plan"):
            edited = Plan.from_list(verdict["plan"])  # human-substituted plan
            validate_plan(edited)
            return {
                "plan_rejected": False,
                "plan": edited.to_list(),
                "levels": topological_levels(edited),
                "current_level": 0,
                "results": RESET,
            }
        # rejected: feed the reason to the planner through the replan channel
        return {
            "plan_rejected": True,
            "decision_reason": f"human rejected the plan: {feedback_of(verdict)}",
        }

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
        cur = state.get("current_level", 0)
        if approve_levels:
            levels = state.get("levels", [])
            results = state.get("results", {}) or {}
            verdict = interrupt(
                {
                    "type": "level_review",
                    "completed_level": cur,
                    "level_results": {tid: results.get(tid) for tid in levels[cur]},
                    "next_level": levels[cur + 1] if cur + 1 < len(levels) else None,
                }
            )
            if not approved(verdict):
                return {
                    "current_level": cur + 1,
                    "plan_rejected": True,
                    "decision_reason": (
                        f"human stopped after level {cur}: {feedback_of(verdict)}"
                    ),
                }
        return {"current_level": cur + 1}

    def join_node(state: AgentState) -> dict[str, Any]:
        return joiner.join(state, llm)

    def route_join(state: AgentState) -> str:
        if state.get("decision") == "finish":
            return END
        if state.get("iteration", 0) >= max_iterations:
            return END  # replan cap reached -> give up gracefully
        return "plan"

    def route_approve(state: AgentState) -> list[Send] | str:
        if state.get("plan_rejected"):
            # respect the same replan cap as the joiner loop
            return "plan" if state.get("iteration", 0) < max_iterations else "join"
        return dispatch(state)

    builder = StateGraph(AgentState)
    builder.add_node("plan", plan_node)
    builder.add_node("run_task", run_task_node)
    builder.add_node("advance", advance_node)
    builder.add_node("join", join_node)

    builder.add_edge(START, "plan")
    if approve_plans:
        builder.add_node("approve", approve_node)
        builder.add_edge("plan", "approve")
        builder.add_conditional_edges("approve", route_approve, ["run_task", "join", "plan"])
    else:
        builder.add_conditional_edges("plan", dispatch, ["run_task", "join"])
    builder.add_edge("run_task", "advance")
    if approve_levels:  # a rejected level review routes back to plan
        builder.add_conditional_edges("advance", route_approve, ["run_task", "join", "plan"])
    else:
        builder.add_conditional_edges("advance", dispatch, ["run_task", "join"])
    builder.add_conditional_edges("join", route_join, ["plan", END])

    return builder.compile(checkpointer=checkpointer)


def _config(thread_id: str | None) -> dict[str, Any] | None:
    return {"configurable": {"thread_id": thread_id}} if thread_id else None


async def arun(
    graph: Any, query: str, *, history: list[Any] | None = None, thread_id: str | None = None
) -> AgentState:
    """Run the compiled graph and return the final (or interrupted) state.

    With a checkpointer, pass ``thread_id`` so the run can pause (HITL) and be
    resumed later via :func:`aresume`.
    """
    return await graph.ainvoke(
        {"query": query, "history": history or [], "iteration": 0}, _config(thread_id)
    )


async def aresume(graph: Any, verdict: Any, *, thread_id: str) -> AgentState:
    """Resume a paused (interrupted) run with the human's verdict."""
    return await graph.ainvoke(Command(resume=verdict), _config(thread_id))
