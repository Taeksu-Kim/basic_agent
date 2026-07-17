"""Entrypoint: wire an LLM + tool registry into an agent and run a query.

Pick the agent type with ``--type plan|react``. Real runs use a vLLM server via
:class:`OpenAICompatLLM` (OpenAI-compatible endpoint). ``--demo`` runs the chosen
type fully offline with a ``FakeLLM`` and a couple of toy tools, so the wiring
can be smoke-tested without a model server.

    python -m agent.run "your task here" --type react
    python -m agent.run --demo --type plan

HITL: ``--approve-plans`` / ``--approve-levels`` turn on the plan_execute gates;
the driver loop below then pauses at each ``interrupt``, prompts on the console,
and resumes. ``--state <file>`` persists checkpoints to SQLite (pruned to the 30
most recent threads); without it an in-memory saver is used.

    python -m agent.run --demo --type plan --approve-levels   # interactive demo
"""

from __future__ import annotations

import argparse
import asyncio
import json
import uuid
from typing import Any

from langgraph.checkpoint.memory import InMemorySaver

from agent.core.checkpoint import open_async_saver, open_saver, prune_threads
from agent.core.hitl import ask_human
from agent.core.llm import FakeLLM, OpenAICompatLLM
from agent.core.tools import ToolRegistry
from agent.plan_execute.graph import aresume as plan_aresume
from agent.plan_execute.graph import arun as plan_arun
from agent.plan_execute.graph import build_graph as build_plan_agent
from agent.react.graph import aresume as react_aresume
from agent.react.graph import arun as react_arun
from agent.react.graph import build_react_agent


def demo_registry() -> ToolRegistry:
    """A couple of trivial tools for the offline demo."""
    reg = ToolRegistry()
    reg.register("uppercase", lambda text: str(text).upper(), description="uppercase text")
    reg.register("exclaim", lambda text: f"{text}!", description="add an exclamation mark")
    reg.register("ask_human", ask_human, description="ask the user a question and get the answer")
    return reg


# ---- HITL console driver ----------------------------------------------------


def _prompt_human(payload: dict[str, Any]) -> Any:
    """Render an interrupt payload on the console and collect the verdict."""
    kind = payload.get("type")
    print(f"\n=== paused: {kind} " + "=" * 40)
    if kind == "plan_approval":
        print(json.dumps(payload["plan"], indent=2))
        raw = input("approve plan? [y] / feedback to replan: ").strip()
    elif kind == "level_review":
        print(f"level {payload['completed_level']} results:")
        print(json.dumps(payload["level_results"], default=str, indent=2))
        print(f"next level: {payload['next_level']}")
        raw = input("continue? [y] / feedback to replan: ").strip()
    elif kind == "tool_approval":
        print(f"tool: {payload['tool']}  args: {json.dumps(payload['args'], default=str)}")
        raw = input("run this tool? [y] / feedback to deny: ").strip()
    else:  # "question" (ask_human) -- the answer is passed through verbatim
        print(payload.get("question", ""))
        if payload.get("options"):
            print("options:", " / ".join(map(str, payload["options"])))
        return input("answer: ").strip()
    if raw in ("", "y", "yes"):
        return True
    return {"approved": False, "feedback": raw}


async def drive(graph: Any, arun_fn: Any, aresume_fn: Any, query: str, thread_id: str) -> Any:
    """Run to completion, pausing on every interrupt for console input."""
    state = await arun_fn(graph, query, thread_id=thread_id)
    while "__interrupt__" in state:
        verdict = _prompt_human(state["__interrupt__"][0].value)
        state = await aresume_fn(graph, verdict, thread_id=thread_id)
    return state


def _demo_plan_llm() -> FakeLLM:
    # plan: uppercase, then exclaim its output; then finish.
    return FakeLLM.json(
        {
            "tasks": [
                {"id": 1, "tool": "uppercase", "args": {"text": "hello agent"}},
                {"id": 2, "tool": "exclaim", "args": {"text": "$1"}, "deps": [1]},
            ]
        },
        {"decision": "finish", "final": "pipeline ran: see results[2]"},
    )


def _demo_react_llm() -> FakeLLM:
    # reactive: uppercase -> exclaim (args based on the observation) -> finish.
    return FakeLLM.json(
        {"action": "tool", "tool": "uppercase", "args": {"text": "hello agent"}},
        {"action": "tool", "tool": "exclaim", "args": {"text": "HELLO AGENT"}},
        {"action": "finish", "final": "loop ran: see scratchpad"},
    )


async def _amain(
    query: str,
    *,
    agent_type: str,
    demo: bool,
    base_url: str,
    model: str,
    approve_plans: bool,
    approve_levels: bool,
    state_path: str | None,
) -> None:
    registry = demo_registry()  # replace with real tools per application
    llm = _demo_llm(agent_type) if demo else OpenAICompatLLM(base_url=base_url, model=model)

    # HITL / persistence: any pause needs a checkpointer + thread_id.
    # NOTE: the graphs run via ainvoke -> the SQLite saver must be the async one.
    saver = await open_async_saver(state_path) if state_path else InMemorySaver()
    thread_id = uuid.uuid4().hex[:8]

    if agent_type == "react":
        graph = build_react_agent(llm=llm, registry=registry, checkpointer=saver)
        state = await drive(graph, react_arun, react_aresume, query, thread_id)
        print("final:     ", state.get("final"))
        print("scratchpad:", json.dumps(state.get("scratchpad", []), default=str, indent=2))
    else:
        graph = build_plan_agent(
            llm=llm,
            registry=registry,
            checkpointer=saver,
            approve_plans=approve_plans,
            approve_levels=approve_levels,
        )
        state = await drive(graph, plan_arun, plan_aresume, query, thread_id)
        print("final:  ", state.get("final"))
        print("results:", json.dumps(state.get("results", {}), default=str, indent=2))

    if state_path:
        # prune via a sync handle on the same file: keep the 30 most recent threads
        prune_threads(open_saver(state_path))


def _demo_llm(agent_type: str) -> FakeLLM:
    return _demo_react_llm() if agent_type == "react" else _demo_plan_llm()


def main() -> None:
    ap = argparse.ArgumentParser(description="Run an agent (plan-and-execute or ReAct).")
    ap.add_argument("query", nargs="?", default="Say hello", help="task for the agent")
    ap.add_argument("--type", choices=["plan", "react"], default="plan", help="agent type")
    ap.add_argument("--demo", action="store_true", help="run offline with a FakeLLM")
    ap.add_argument("--base-url", default="http://localhost:8000/v1", help="vLLM endpoint")
    ap.add_argument("--model", default="local", help="model name served by vLLM")
    ap.add_argument("--approve-plans", action="store_true", help="HITL: approve each plan (plan type)")
    ap.add_argument("--approve-levels", action="store_true", help="HITL: review each level's results (plan type)")
    ap.add_argument("--state", default=None, help="SQLite checkpoint file (persistent HITL state)")
    args = ap.parse_args()
    asyncio.run(
        _amain(
            args.query,
            agent_type=args.type,
            demo=args.demo,
            base_url=args.base_url,
            model=args.model,
            approve_plans=args.approve_plans,
            approve_levels=args.approve_levels,
            state_path=args.state,
        )
    )


if __name__ == "__main__":
    main()
