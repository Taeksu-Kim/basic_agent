"""Assemble the plan-centric agent graph.

Flow::

    START -> plan ─▶ [approve] ─▶ level_gate ─(dispatch)─▶ run_task* ─▶ advance ─▶ level_gate ─▶ … ─▶ join
                                                                                                    │
                                    finish / cap / error ◀──(route_join)────────────────────────────┤
                                    replan ─────────────────────────────────────────────────────────┘  (back to plan)

* ``level_gate`` runs before every level fans out. If any task in the level uses
  a ``requires_approval`` tool it pauses **once** (``interrupt``) for the whole
  level -- never mid-fan-out -- and records the verdicts: approved tasks run with
  ``preapproved``, denied ones get a denial result (which triggers early-abort,
  so the joiner sees why). No approval tools in the level -> it is a no-op.
* ``dispatch`` fans the current level out to ``run_task`` via ``Send`` (parallel
  within a level), or routes to ``join`` when the levels are exhausted -- or
  early, if any task failed. ``advance`` bumps the level cursor after the
  level's barrier. ``route_join`` ends on ``finish``, on the replan cap, or on a
  planner/joiner ``error``; otherwise loops back to ``plan`` (which resets
  results for the new plan).
* Interrupts raised *inside* a task (``ask_human``, a nested sub-agent's approval
  gate) pause the graph too; parallel tasks may pause at once -- resume those
  with an ``{interrupt_id: answer}`` dict (:func:`agent.core.hitl.aresume_graph`).
"""

from __future__ import annotations

from typing import Any

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send, interrupt

from agent.core import trace
from agent.core.context import ContextPolicy
from agent.core.execution import approval_payload, is_failure, needs_approval, record_verdict
from agent.core.hitl import approved, arun_graph, aresume_graph, feedback_of
from agent.core.llm import LLMClient
from agent.core.tools import ToolRegistry
from agent.plan_execute.dag import Plan, PlanError, substitute_args, topological_levels, validate_plan
from agent.plan_execute.executor import run_task
from agent.plan_execute.orchestrator import joiner, planner
from agent.plan_execute.state import RESET, AgentState


def _has_error(results: dict[int, Any]) -> bool:
    """True if any task *failed* (explicit failure marker, incl. denials).

    Data that merely contains an ``"error"`` field is not a failure.
    """
    return any(is_failure(v) for v in results.values())


def _verdict_for(verdict: Any, tid: int) -> Any:
    """Answer for one task of a ``tool_approvals`` batch.

    Either one verdict for every task (``True``, ``{"approved": False, ...}``), or
    per task: ``{"decisions": {task_id: verdict}}``. The ``decisions`` wrapper is
    deliberate -- a bare ``{task_id: ...}`` dict would collide with LangGraph's own
    ``{interrupt_id: ...}`` resume map. A task missing from ``decisions`` is denied.
    """
    if isinstance(verdict, dict) and isinstance(verdict.get("decisions"), dict):
        decisions = verdict["decisions"]
        for key in (tid, str(tid)):
            if key in decisions:
                return decisions[key]
        return {"approved": False, "feedback": "no decision given for this task"}
    return verdict


def build_graph(
    *,
    llm: LLMClient,
    registry: ToolRegistry,
    max_iterations: int = 3,
    checkpointer: Any = None,
    approve_plans: bool = False,
    approve_levels: bool = False,
    context_policy: ContextPolicy | None = None,
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

    Tools registered with ``requires_approval=True`` are always gated (see
    ``level_gate`` in the module docstring) -- no flag needed.

    Any pause requires ``checkpointer``.

    ``context_policy`` (opt-in) caps each task result shown to the planner (on
    replan) and the joiner at ``inline_chars``; ``None`` keeps the original,
    uncapped prompts. See :mod:`agent.core.context`.
    """
    planner_llm = trace.traced(llm, "plan_execute.planner")
    joiner_llm = trace.traced(llm, "plan_execute.joiner")

    def plan_node(state: AgentState) -> dict[str, Any]:
        try:
            update = planner.plan(state, planner_llm, registry, policy=context_policy)
        except PlanError as exc:  # output stayed unusable after retries -> end the run
            return {"error": str(exc), "status": "error", "final": "", "plan": [], "levels": [],
                    "iteration": int(state.get("iteration", 0)) + 1}
        update["results"] = RESET  # fresh result namespace for the new plan
        update["plan_rejected"] = False
        update["preapproved"] = []
        update["status"] = None  # clear a previous run's status on this thread (contract)
        update["error"] = ""
        return update

    def route_plan(state: AgentState) -> str:
        if state.get("error"):
            return END
        return "approve" if approve_plans else "level_gate"

    def approve_node(state: AgentState) -> dict[str, Any]:
        verdict = interrupt(
            {"type": "plan_approval", "plan": state["plan"], "levels": state["levels"]}
        )
        if approved(verdict):
            return {"plan_rejected": False}
        if isinstance(verdict, dict) and verdict.get("plan"):
            edited = Plan.from_list(verdict["plan"])  # human-substituted plan
            validate_plan(edited, registry)
            return {
                "plan_rejected": False,
                "plan": edited.to_list(),
                "levels": topological_levels(edited),
                "current_level": 0,
                "results": RESET,
                "preapproved": [],
            }
        # rejected: feed the reason to the planner through the replan channel
        return {
            "plan_rejected": True,
            "decision_reason": f"human rejected the plan: {feedback_of(verdict)}",
        }

    def level_gate_node(state: AgentState) -> dict[str, Any]:
        results = state.get("results", {}) or {}
        levels = state.get("levels", [])
        cur = state.get("current_level", 0)
        if _has_error(results) or cur >= len(levels):
            return {}
        by_id = {t["id"]: t for t in state["plan"]}
        done = set(state.get("preapproved") or [])
        gated = [tid for tid in levels[cur]
                 if tid not in done and tid not in results
                 and needs_approval(registry, by_id[tid]["tool"])]
        if not gated:
            return {}
        items = []
        for tid in gated:
            try:
                args = substitute_args(by_id[tid].get("args", {}), results)
            except PlanError:
                continue  # unresolvable reference: dispatch fails this task; nothing to approve
            items.append({"task_id": tid, "tool": by_id[tid]["tool"], "args": args})
        if not items:
            return {}
        if len(items) == 1:  # same payload shape as react's per-call gate
            it = items[0]
            payload = approval_payload(it["tool"], it["args"], thought="", task_id=it["task_id"])
        else:
            payload = {"type": "tool_approvals", "items": items}
        verdict = interrupt(payload)
        denied: dict[int, Any] = {}
        ok_ids: list[int] = []
        for it in items:
            refusal = record_verdict(it["tool"], _verdict_for(verdict, it["task_id"]))
            if refusal is None:
                ok_ids.append(it["task_id"])
            else:
                denied[it["task_id"]] = refusal
        update: dict[str, Any] = {"preapproved": sorted(done | set(ok_ids))}
        if denied:
            update["results"] = denied  # denial = failure -> early-abort -> joiner sees why
        return update

    def dispatch(state: AgentState) -> list[Send] | str:
        results = state.get("results", {}) or {}
        levels = state.get("levels", [])
        cur = state.get("current_level", 0)
        if cur >= len(levels):
            return "join"
        level = set(levels[cur])
        # early-abort: a failure in an *earlier* level stops further levels. A
        # denial inside this level (from level_gate) does not stop its siblings --
        # they are independent, and the human approved them individually.
        if _has_error({tid: v for tid, v in results.items() if tid not in level}):
            return "join"
        by_id = {t["id"]: t for t in state["plan"]}
        preapproved = set(state.get("preapproved") or [])
        sends: list[Send] = []
        for tid in levels[cur]:
            if tid in results:  # already settled (denied at the gate)
                continue
            task = by_id[tid]
            try:
                args = substitute_args(task.get("args", {}), results)
            except PlanError as exc:  # e.g. $1.total but task 1 returned text -> task fails
                sends.append(Send("run_task", {"id": tid, "tool": task["tool"],
                                               "arg_error": str(exc)}))
                continue
            sends.append(Send("run_task", {"id": tid, "tool": task["tool"], "args": args,
                                           "preapproved": tid in preapproved}))
        return sends or "join"  # whole level denied -> let the joiner decide

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
        try:
            update = joiner.join(state, joiner_llm, policy=context_policy,
                                 final_round=state.get("iteration", 0) >= max_iterations)
        except PlanError as exc:  # output stayed unusable after retries -> end the run
            return {"error": str(exc), "status": "error", "final": "", "decision": "error"}
        if update.get("decision") == "finish":
            update["status"] = "ok"  # contract; a replan that hits the cap leaves it unset -> capped
        return update

    def route_join(state: AgentState) -> str:
        if state.get("error") or state.get("decision") == "finish":
            return END
        if state.get("iteration", 0) >= max_iterations:
            return END  # replan cap reached -> give up gracefully
        return "plan"

    def route_approve(state: AgentState) -> str:
        if state.get("plan_rejected"):
            # respect the same replan cap as the joiner loop
            return "plan" if state.get("iteration", 0) < max_iterations else "join"
        return "level_gate"

    builder = StateGraph(AgentState)
    builder.add_node("plan", plan_node)
    builder.add_node("level_gate", level_gate_node)
    builder.add_node("run_task", run_task_node)
    builder.add_node("advance", advance_node)
    builder.add_node("join", join_node)

    builder.add_edge(START, "plan")
    if approve_plans:
        builder.add_node("approve", approve_node)
        builder.add_conditional_edges("plan", route_plan, ["approve", END])
        builder.add_conditional_edges("approve", route_approve, ["level_gate", "join", "plan"])
    else:
        builder.add_conditional_edges("plan", route_plan, ["level_gate", END])
    builder.add_conditional_edges("level_gate", dispatch, ["run_task", "join"])
    builder.add_edge("run_task", "advance")
    if approve_levels:  # a rejected level review routes back to plan
        builder.add_conditional_edges("advance", route_approve, ["level_gate", "join", "plan"])
    else:
        builder.add_edge("advance", "level_gate")
    builder.add_conditional_edges("join", route_join, ["plan", END])

    return builder.compile(checkpointer=checkpointer)


async def arun(
    graph: Any, query: str, *, history: list[Any] | None = None, thread_id: str | None = None
) -> AgentState:
    """Run the compiled graph and return the final (or interrupted) state.

    With a checkpointer, pass ``thread_id`` so the run can pause (HITL) and be
    resumed later via :func:`aresume`.
    """
    return await arun_graph(graph, query, history=history, thread_id=thread_id)


async def aresume(graph: Any, verdict: Any, *, thread_id: str) -> AgentState:
    """Resume a paused run. See :func:`agent.core.hitl.aresume_graph` for verdict forms."""
    return await aresume_graph(graph, verdict, thread_id=thread_id)
