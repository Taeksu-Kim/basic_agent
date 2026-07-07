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

from langgraph.graph import END, START, StateGraph

from agent.core.llm import LLMClient
from agent.core.tools import ToolRegistry
from agent.react import policy
from agent.react.state import ReactState


async def _run_tool(registry: ToolRegistry, action: dict[str, Any]) -> Any:
    try:
        return await registry.call(action["tool"], action.get("args", {}))
    except Exception as exc:  # noqa: BLE001 -- feed the error back as an observation
        return {"error": f"{type(exc).__name__}: {exc}"}


def build_react_agent(
    *, llm: LLMClient, registry: ToolRegistry, max_steps: int = 6
) -> Any:
    """Compile a ReAct agent for a given LLM + tool registry."""

    def reason_node(state: ReactState) -> dict[str, Any]:
        return policy.reason(state, llm, registry)

    async def act_node(state: ReactState) -> dict[str, Any]:
        action = state["action"]
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
    return builder.compile()


async def arun(graph: Any, query: str, *, history: list[Any] | None = None) -> ReactState:
    """Run the compiled ReAct agent to completion and return the final state."""
    return await graph.ainvoke({"query": query, "history": history or [], "iteration": 0})
