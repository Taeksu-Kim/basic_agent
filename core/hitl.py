"""Shared HITL verdict contract.

A paused graph resumes with a human *verdict*. Both gates (plan approval,
per-tool approval) accept the same shapes:

* approve -- ``True`` | ``"approve"`` | ``{"approved": True}``
* reject  -- anything else; ``{"feedback": ...}`` or a plain string carries why
"""

from __future__ import annotations

from typing import Any, Literal, Optional

from langgraph.types import interrupt

# ``type`` of every interrupt payload the framework raises -- what a driver (run.py's
# console loop, a UI) dispatches on to know which verdict shape to send back.
InterruptType = Literal["tool_approval", "tool_approvals", "plan_approval", "level_review",
                        "question"]
TOOL_APPROVAL: InterruptType = "tool_approval"    # one tool call       -> verdict
TOOL_APPROVALS: InterruptType = "tool_approvals"  # plan level, N calls -> {"decisions": {task_id: verdict}}
PLAN_APPROVAL: InterruptType = "plan_approval"    # plan_execute plan   -> verdict | {"plan": [...]}
LEVEL_REVIEW: InterruptType = "level_review"      # plan_execute level  -> verdict
QUESTION: InterruptType = "question"              # ask_human           -> the answer itself


def approved(verdict: Any) -> bool:
    if verdict is True or verdict == "approve":
        return True
    return isinstance(verdict, dict) and bool(verdict.get("approved"))


def feedback_of(verdict: Any) -> str:
    """Human-supplied reason attached to a rejection ('' if none)."""
    if isinstance(verdict, dict):
        return str(verdict.get("feedback", ""))
    if isinstance(verdict, (bool, type(None))):
        return ""
    return str(verdict)


def pending_interrupts(state: Any) -> list[Any]:
    """The interrupts a run is paused on (``[]`` if it finished)."""
    return list((state or {}).get("__interrupt__", []) or [])


def _config(thread_id: Optional[str]) -> Optional[dict]:
    return {"configurable": {"thread_id": thread_id}} if thread_id else None


def _trace_outcome(state: Any) -> None:
    from agent.core import trace

    intrs = pending_interrupts(state)
    if intrs:
        for i in intrs:
            trace.emit("interrupt", interrupt_id=i.id, payload=trace.clip(i.value))
    else:
        trace.emit("run_end", final=trace.clip((state or {}).get("final", "")))


async def arun_graph(
    graph: Any, query: str, *, history: Optional[list] = None, thread_id: Optional[str] = None
) -> Any:
    """Start a run of any compiled agent graph (shared by every pattern)."""
    from agent.core import trace

    from agent.core.contract import initial_state

    trace.emit("run_start", query=trace.clip(query), thread_id=thread_id)
    state = await graph.ainvoke(initial_state(query, history=history), _config(thread_id))
    _trace_outcome(state)
    return state


async def aresume_graph(graph: Any, verdict: Any, *, thread_id: str) -> Any:
    """Resume a paused run.

    * one pending interrupt -- ``verdict`` is its answer (any verdict shape).
    * several pending (parallel branches paused at once) -- pass a dict mapping
      each interrupt id (``state["__interrupt__"][k].id``) to its answer. LangGraph
      pairs answers to interrupts by id, so order does not matter.
    """
    from langgraph.types import Command

    from agent.core import trace

    config = _config(thread_id)
    snapshot = await graph.aget_state(config)
    ids = [i.id for i in getattr(snapshot, "interrupts", ()) or ()]
    if len(ids) > 1:
        if not (isinstance(verdict, dict) and set(ids) <= set(verdict)):
            raise ValueError(
                f"{len(ids)} interrupts are pending; resume with a dict mapping every "
                f"interrupt id to its answer (ids: {ids})"
            )
        command = Command(resume={i: verdict[i] for i in ids})
    else:
        command = Command(resume=verdict)
    trace.emit("resume", interrupt_ids=ids)
    state = await graph.ainvoke(command, config)
    _trace_outcome(state)
    return state


def ask_human(question: str, options: Optional[list] = None) -> Any:
    """Tool: pause the graph and ask the human; the reply is the return value.

    Register it like any tool (``registry.register("ask_human", ask_human, ...)``)
    to let the *model* decide when to stop and ask — the complement of the
    structural gates, which pause regardless of what the model wants. Works in
    any pattern, including several parallel plan_execute tasks asking at once:
    resume those with an ``{interrupt_id: answer}`` dict (see :func:`aresume_graph`).
    """
    return interrupt({"type": QUESTION, "question": question, "options": options or []})
