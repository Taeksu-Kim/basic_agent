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

from agent.core import trace
from agent.core.context import DEFAULT_REACT_POLICY, ContextPolicy, run_harness_tool
from agent.core.execution import execute_tool
from agent.core.hitl import arun_graph, aresume_graph
from agent.core.llm import LLMClient
from agent.core.tools import ToolRegistry
from agent.react import policy
from agent.react.state import ReactState


def build_react_agent(
    *, llm: LLMClient, registry: ToolRegistry, max_steps: int = 6,
    checkpointer: Any = None, system: str | None = None,
    context_policy: ContextPolicy | None = DEFAULT_REACT_POLICY,
) -> Any:
    """Compile a ReAct agent for a given LLM + tool registry.

    ``system`` appends domain guidance to the base ReAct system prompt (e.g. a
    retriever agent's query-rewriting rules).

    Tools registered with ``requires_approval=True`` trigger a HITL gate: the
    graph pauses (``interrupt``) before running them and resumes on a human
    verdict — approve runs the tool, reject feeds the denial back to the policy
    as the observation. Requires ``checkpointer``.

    ``context_policy`` sets context-management features (see
    :mod:`agent.core.context`). The default, ``DEFAULT_REACT_POLICY``, turns on
    just-in-time reading of long observations (``read_observation``); ``None``
    gives the original prompt (1,500-char truncation, no read tool).
    """
    if context_policy is not None:
        context_policy.check(registry)
    policy_llm = trace.traced(llm, "react.policy")
    compact_llm = trace.traced(llm, "react.compaction")
    harness = context_policy.harness_tools(registry) if context_policy else {}

    def reason_node(state: ReactState) -> dict[str, Any]:
        return policy.reason(state, policy_llm, registry, system=system,
                             policy=context_policy, compact_llm=compact_llm)

    async def act_node(state: ReactState) -> dict[str, Any]:
        action = state["action"]
        extra: dict[str, Any] = {}
        if action["tool"] in harness:
            # harness tools read/write run state; no registry, no approval gate
            observation, extra = run_harness_tool(action["tool"], action.get("args", {}),
                                                  state, registry, context_policy)
            trace.emit("tool_call", tool=action["tool"], args=trace.clip(action.get("args", {})),
                       latency_ms=0.0, outcome="harness", detail=None)
        else:
            # approval gate, error capture, interrupt pass-through: all in execute_tool.
            # A denial comes back as the observation so the policy can react to it.
            observation = await execute_tool(
                registry, action["tool"], action.get("args", {}),
                approval="inline", thought=action.get("thought", ""),
            )
        step = {
            "thought": action.get("thought", ""),
            "tool": action["tool"],
            "args": action.get("args", {}),
            "observation": observation,
        }
        return {"scratchpad": [step], **extra}

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


async def arun(
    graph: Any, query: str, *, history: list[Any] | None = None, thread_id: str | None = None
) -> ReactState:
    """Run the compiled ReAct agent and return the final (or interrupted) state.

    With a checkpointer, pass ``thread_id`` so the run can pause (HITL) and be
    resumed later via :func:`aresume`.
    """
    return await arun_graph(graph, query, history=history, thread_id=thread_id)


async def aresume(graph: Any, verdict: Any, *, thread_id: str) -> ReactState:
    """Resume a paused run. See :func:`agent.core.hitl.aresume_graph` for verdict forms."""
    return await aresume_graph(graph, verdict, thread_id=thread_id)
