"""Entrypoint: wire an LLM + tool registry into an agent and run a query.

Pick the agent type with ``--type plan|react``. Real runs use a vLLM server via
:class:`OpenAICompatLLM` (OpenAI-compatible endpoint). ``--demo`` runs the chosen
type fully offline with a ``FakeLLM`` and a couple of toy tools, so the wiring
can be smoke-tested without a model server.

    python -m agent.run "your task here" --type react
    python -m agent.run --demo --type plan
"""

from __future__ import annotations

import argparse
import asyncio
import json

from agent.core.llm import FakeLLM, OpenAICompatLLM
from agent.core.tools import ToolRegistry
from agent.plan_execute.graph import arun as plan_arun
from agent.plan_execute.graph import build_graph as build_plan_agent
from agent.react.graph import arun as react_arun
from agent.react.graph import build_react_agent


def demo_registry() -> ToolRegistry:
    """A couple of trivial tools for the offline demo."""
    reg = ToolRegistry()
    reg.register("uppercase", lambda text: str(text).upper(), description="uppercase text")
    reg.register("exclaim", lambda text: f"{text}!", description="add an exclamation mark")
    return reg


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


async def _amain(query: str, *, agent_type: str, demo: bool, base_url: str, model: str) -> None:
    registry = demo_registry()  # replace with real tools per application
    llm = _demo_llm(agent_type) if demo else OpenAICompatLLM(base_url=base_url, model=model)

    if agent_type == "react":
        graph = build_react_agent(llm=llm, registry=registry)
        state = await react_arun(graph, query)
        print("final:     ", state.get("final"))
        print("scratchpad:", json.dumps(state.get("scratchpad", []), default=str, indent=2))
    else:
        graph = build_plan_agent(llm=llm, registry=registry)
        state = await plan_arun(graph, query)
        print("final:  ", state.get("final"))
        print("results:", json.dumps(state.get("results", {}), default=str, indent=2))


def _demo_llm(agent_type: str) -> FakeLLM:
    return _demo_react_llm() if agent_type == "react" else _demo_plan_llm()


def main() -> None:
    ap = argparse.ArgumentParser(description="Run an agent (plan-and-execute or ReAct).")
    ap.add_argument("query", nargs="?", default="Say hello", help="task for the agent")
    ap.add_argument("--type", choices=["plan", "react"], default="plan", help="agent type")
    ap.add_argument("--demo", action="store_true", help="run offline with a FakeLLM")
    ap.add_argument("--base-url", default="http://localhost:8000/v1", help="vLLM endpoint")
    ap.add_argument("--model", default="local", help="model name served by vLLM")
    args = ap.parse_args()
    asyncio.run(
        _amain(args.query, agent_type=args.type, demo=args.demo, base_url=args.base_url, model=args.model)
    )


if __name__ == "__main__":
    main()
