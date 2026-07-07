"""Entrypoint: wire an LLM + tool registry into the graph and run a query.

Real runs use a vLLM server via :class:`OpenAICompatLLM` (OpenAI-compatible
endpoint). ``--demo`` runs the same graph fully offline with a ``FakeLLM`` and a
couple of toy tools, so the wiring can be smoke-tested without a model server.

    python -m agent.run "your task here"
    python -m agent.run --demo
"""

from __future__ import annotations

import argparse
import asyncio
import json

from agent.plan_execute.graph import arun, build_graph
from agent.core.llm import FakeLLM, OpenAICompatLLM
from agent.core.tools import ToolRegistry


def demo_registry() -> ToolRegistry:
    """A couple of trivial tools for the offline demo."""
    reg = ToolRegistry()
    reg.register("uppercase", lambda text: str(text).upper())
    reg.register("exclaim", lambda text: f"{text}!")
    return reg


def _demo_llm() -> FakeLLM:
    # plan: uppercase the query, then exclaim its output; then finish.
    return FakeLLM.json(
        {
            "tasks": [
                {"id": 1, "tool": "uppercase", "args": {"text": "hello agent"}},
                {"id": 2, "tool": "exclaim", "args": {"text": "$1"}, "deps": [1]},
            ]
        },
        {"decision": "finish", "final": "pipeline ran: see results[2]"},
    )


async def _amain(query: str, *, demo: bool, base_url: str, model: str) -> None:
    if demo:
        llm = _demo_llm()
        registry = demo_registry()
    else:
        llm = OpenAICompatLLM(base_url=base_url, model=model)
        registry = demo_registry()  # replace with real tools per application

    graph = build_graph(llm=llm, registry=registry)
    state = await arun(graph, query)
    print("final:  ", state.get("final"))
    print("results:", json.dumps(state.get("results", {}), default=str, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser(description="Run the plan-and-execute agent.")
    ap.add_argument("query", nargs="?", default="Say hello", help="task for the agent")
    ap.add_argument("--demo", action="store_true", help="run offline with a FakeLLM")
    ap.add_argument("--base-url", default="http://localhost:8000/v1", help="vLLM endpoint")
    ap.add_argument("--model", default="local", help="model name served by vLLM")
    args = ap.parse_args()
    asyncio.run(_amain(args.query, demo=args.demo, base_url=args.base_url, model=args.model))


if __name__ == "__main__":
    main()
