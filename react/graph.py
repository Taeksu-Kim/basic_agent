"""Assemble the ReAct agent graph.

Flow::

    START -> reason ─(route)─▶ act ─▶ reason ─▶ ... ─▶ END

``reason`` decides the next action; ``route`` ends on ``finish`` or when the step
cap (``max_steps``) is reached, otherwise goes to ``act``. ``act`` runs the tool
and appends the observation to the scratchpad, then loops back. Tool errors are
captured as the observation (not aborted) so the agent can react to them -- the
point of a reactive loop.

Same contract as ``plan_execute``: ``ainvoke({"query": ...}) -> {"final": ...}``,
so a ReAct agent is `AgentTool`-wrappable and composes with plan_execute.
"""

from __future__ import annotations

from typing import Any

from langgraph.errors import GraphBubbleUp
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from agent.core.hitl import approved, feedback_of
from agent.core.llm import LLMClient
from agent.core.tools import ToolRegistry
from agent.react import policy
from agent.react.state import ReactState


async def _run_tool(registry: ToolRegistry, action: dict[str, Any]) -> Any:
    try:
        return await registry.call(action["tool"], action.get("args", {}))
    except GraphBubbleUp:
        raise  # an interrupt inside a tool (e.g. ask_human) must pause, not error
    except Exception as exc:  # noqa: BLE001 -- feed the error back as an observation
        return {"error": f"{type(exc).__name__}: {exc}"}


def _needs_approval(registry: ToolRegistry, name: str) -> bool:
    return registry.has(name) and getattr(registry.get(name), "requires_approval", False)


def build_react_agent(
    *, llm: LLMClient, registry: ToolRegistry, max_steps: int = 6, checkpointer: Any = None
) -> Any:
    """Compile a ReAct agent for a given LLM + tool registry.

    Tools registered with ``requires_approval=True`` trigger a HITL gate: the
    graph pauses (``interrupt``) before running them and resumes on a human
    verdict — approve runs the tool, reject feeds the denial back to the policy
    as the observation. Requires ``checkpointer``.
    """

    def reason_node(state: ReactState) -> dict[str, Any]:
        return policy.reason(state, llm, registry)

    async def act_node(state: ReactState) -> dict[str, Any]:
        action = state["action"]
        if _needs_approval(registry, action["tool"]):
            verdict = interrupt(
                {
                    "type": "tool_approval",
                    "tool": action["tool"],
                    "args": action.get("args", {}),
                    "thought": action.get("thought", ""),
                }
            )
            if not approved(verdict):
                # denial becomes the observation so the policy can react to it
                observation: Any = {
                    "denied": f"tool call denied by user: {feedback_of(verdict)}"
                }
                return {
                    "scratchpad": [
                        {
                            "thought": action.get("thought", ""),
                            "tool": action["tool"],
                            "args": action.get("args", {}),
                            "observation": observation,
                        }
                    ]
                }
        observation = await _run_tool(registry, action)
        step = {
            "thought": action.get("thought", ""),
            "tool": action["tool"],
            "args": action.get("args", {}),
            "observation": observation,
        }
        return {"scratchpad": [step]}

    def route(state: ReactState) -> str:
        if state.get("action", {}).get("action") == "finish":
            return END
        if state.get("iteration", 0) >= max_steps:
            return END  # step cap reached -> stop gracefully
        return "act"

    builder = StateGraph(ReactState)
    builder.add_node("reason", reason_node)
    builder.add_node("act", act_node)
    builder.add_edge(START, "reason")
    builder.add_conditional_edges("reason", route, ["act", END])
    builder.add_edge("act", "reason")
    return builder.compile(checkpointer=checkpointer)


def _config(thread_id: str | None) -> dict[str, Any] | None:
    return {"configurable": {"thread_id": thread_id}} if thread_id else None


async def arun(
    graph: Any, query: str, *, history: list[Any] | None = None, thread_id: str | None = None
) -> ReactState:
    """Run the compiled ReAct agent and return the final (or interrupted) state.

    With a checkpointer, pass ``thread_id`` so the run can pause (HITL) and be
    resumed later via :func:`aresume`.
    """
    return await graph.ainvoke(
        {"query": query, "history": history or [], "iteration": 0}, _config(thread_id)
    )


async def aresume(graph: Any, verdict: Any, *, thread_id: str) -> ReactState:
    """Resume a paused (interrupted) run with the human's verdict."""
    return await graph.ainvoke(Command(resume=verdict), _config(thread_id))
