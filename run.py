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
from pathlib import Path
from typing import Any

from langgraph.checkpoint.memory import InMemorySaver

from agent.core.checkpoint import open_async_saver, open_saver, prune_threads
from agent.core.contract import envelope
from agent.core.hitl import aresume_graph, arun_graph, ask_human, pending_interrupts
from agent.core.llm import FakeLLM, OpenAICompatLLM
from agent.core.tools import ToolRegistry
from agent.core.trace import Tracer, tracing
from agent.patterns import registry as patterns


def demo_registry() -> ToolRegistry:
    """A couple of trivial tools for the offline demo."""
    reg = ToolRegistry()
    reg.register("uppercase", lambda text: str(text).upper(), description="uppercase text")
    reg.register("exclaim", lambda text: f"{text}!", description="add an exclamation mark")
    reg.register("ask_human", ask_human, description="ask the user a question and get the answer")
    reg.register("publish", lambda text: f"published: {text}",
                 description="publish text (side effect: needs human approval)",
                 requires_approval=True)
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
    elif kind == "tool_approvals":  # plan_execute level gate: several tasks, one pause
        decisions = {}
        for item in payload["items"]:
            print(f"task {item['task_id']}: {item['tool']}  "
                  f"args: {json.dumps(item['args'], default=str)}")
            raw = input("  run this task? [y] / feedback to deny: ").strip()
            decisions[item["task_id"]] = (
                True if raw in ("", "y", "yes") else {"approved": False, "feedback": raw}
            )
        return {"decisions": decisions}
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
    while pending_interrupts(state):
        intrs = pending_interrupts(state)
        if len(intrs) == 1:
            verdict = _prompt_human(intrs[0].value)
        else:  # parallel branches paused at once: answer each, keyed by interrupt id
            verdict = {i.id: _prompt_human(i.value) for i in intrs}
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


def _demo_hitl_llm(agent_type: str) -> FakeLLM:
    # uppercase, then `publish` (requires approval) with the uppercased text.
    if agent_type == "react":
        return FakeLLM.json(
            {"action": "tool", "tool": "uppercase", "args": {"text": "hello agent"}},
            {"action": "tool", "tool": "publish", "args": {"text": "HELLO AGENT"}},
            {"action": "finish", "final": "published (if approved)"},
        )
    return FakeLLM.json(
        {"tasks": [{"id": 1, "tool": "uppercase", "args": {"text": "hello agent"}},
                   {"id": 2, "tool": "publish", "args": {"text": "$1"}, "deps": [1]}]},
        {"decision": "finish", "final": "published (if approved)"},
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
    demo_hitl: bool = False,
    trace_dir: str | None = None,
    sampling: dict | None = None,
) -> None:
    tracer = Tracer(Path(trace_dir) / f"{uuid.uuid4().hex[:8]}.jsonl") if trace_dir else None
    with tracing(tracer):
        await _run(query, agent_type=agent_type, demo=demo, base_url=base_url, model=model,
                   approve_plans=approve_plans, approve_levels=approve_levels,
                   state_path=state_path, demo_hitl=demo_hitl, sampling=sampling)
    if tracer is not None:
        print(f"trace: {tracer.path}")


async def _run(
    query: str,
    *,
    agent_type: str,
    demo: bool,
    base_url: str,
    model: str,
    approve_plans: bool,
    approve_levels: bool,
    state_path: str | None,
    demo_hitl: bool,
    sampling: dict | None = None,
) -> None:
    spec = patterns.get(agent_type)  # aliases (e.g. "plan") resolve here
    registry = demo_registry()  # replace with real tools per application
    wf_config: dict[str, Any] = {}
    if spec.name in _WF_DEMOS:
        if not demo:
            raise SystemExit(f"{spec.name} needs pattern config; from the CLI only --demo is wired")
        responses, wf_config = _WF_DEMOS[spec.name]
        llm: Any = FakeLLM([r if isinstance(r, str) else json.dumps(r) for r in responses])
        if spec.name == "orchestrator_workers":
            registry.add(_demo_helper())
    elif demo_hitl:
        llm = _demo_hitl_llm(spec.name)
    elif demo:
        llm = _demo_llm(spec.name)
    else:
        llm = OpenAICompatLLM(base_url=base_url, model=model, sampling=sampling)

    # HITL / persistence: any pause needs a checkpointer + thread_id.
    # NOTE: the graphs run via ainvoke -> the SQLite saver must be the async one.
    saver = await open_async_saver(state_path) if state_path else InMemorySaver()
    thread_id = uuid.uuid4().hex[:8]

    config: dict[str, Any] = {"checkpointer": saver, **wf_config}
    if approve_plans or approve_levels:
        if spec.name != "plan_execute":
            raise SystemExit("--approve-plans / --approve-levels apply to plan_execute only")
        config.update(approve_plans=approve_plans, approve_levels=approve_levels)
    graph = spec.build(llm=llm, registry=registry, **config)
    state = await drive(graph, arun_graph, aresume_graph, query, thread_id)

    env = envelope(state)
    print("pattern:   ", f"{spec.name} ({spec.std_name})")
    print("status:    ", env["status"] + (f"  ({env['error']})" if env["error"] else ""))
    print("final:     ", env["final"])
    if "scratchpad" in state:
        print("scratchpad:", json.dumps(state.get("scratchpad", []), default=str, indent=2))
    if "results" in state:
        print("results:", json.dumps(state.get("results", {}), default=str, indent=2))

    if state_path:
        # prune via a sync handle on the same file: keep the 30 most recent threads
        prune_threads(open_saver(state_path))


_DEMO_SCRIPTS = {"react": _demo_react_llm, "plan_execute": _demo_plan_llm}

# Offline demos for the workflow patterns: (FakeLLM responses, pattern config). Their
# delegation targets are the demo registry's tools (+ a scripted ReAct "helper").
_WF_DEMOS: dict[str, tuple[list, dict]] = {
    "prompt_chain": ([], {"steps": [
        {"name": "upper", "tool": "uppercase", "args": {"text": "{query}"}},
        {"name": "excl", "tool": "exclaim", "args": {"text": "{previous}"}}]}),
    "router": ([{"route": "uppercase", "args": {"text": "hello agent"}}],
               {"routes": {"uppercase": "make text uppercase", "exclaim": "add an exclamation mark"}}),
    "parallel": ([], {"sections": [
        {"name": "upper", "tool": "uppercase", "args": {"text": "{query}"}},
        {"name": "excl", "tool": "exclaim", "args": {"text": "{query}"}}]}),
    "evaluator_optimizer": (["hello agent", {"pass": False, "feedback": "make it uppercase"},
                             "HELLO AGENT", {"pass": True, "feedback": "ok"}],
                            {"criteria": "the text must be uppercase"}),
    "orchestrator_workers": ([{"subtasks": [{"worker": "helper", "objective": "shout hello agent"}]},
                              {"decision": "finish", "final": "helper said: HELLO AGENT!"}],
                             {"workers": ["helper"]}),
}


def _demo_helper() -> Any:
    """A scripted ReAct sub-agent for the orchestrator demo."""
    from agent.core.tools import AgentTool

    child = patterns.get("react").build(llm=_demo_react_llm(), registry=demo_registry())
    return AgentTool("helper", child, description="a helper agent with text tools")


def _demo_llm(pattern: str) -> FakeLLM:
    if pattern not in _DEMO_SCRIPTS:
        raise SystemExit(f"no offline demo script for pattern {pattern!r} yet")
    return _DEMO_SCRIPTS[pattern]()


def _list_patterns() -> None:
    for s in patterns.specs():
        alias = f"  (aliases: {', '.join(s.aliases)})" if s.aliases else ""
        print(f"{s.name:<14} {s.std_name}{alias}")
        print(f"{'':<14} {s.summary}")
        print(f"{'':<14} config: {', '.join(s.config_keys())} | weak model: {s.weak_model_ok}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Run an agent pattern from the registry.")
    ap.add_argument("query", nargs="?", default="Say hello", help="task for the agent")
    ap.add_argument("--type", choices=patterns.choices(), default="plan",
                    help="pattern name or alias (see --list-patterns)")
    ap.add_argument("--list-patterns", action="store_true", help="list registered patterns and exit")
    ap.add_argument("--demo", action="store_true", help="run offline with a FakeLLM")
    ap.add_argument("--base-url", default="http://localhost:8000/v1", help="vLLM endpoint")
    ap.add_argument("--model", default="local", help="model name served by vLLM")
    ap.add_argument("--sampling", default=None, metavar="JSON",
                    help='extra request fields merged into every LLM call, e.g. '
                         '\'{"chat_template_kwargs": {"enable_thinking": false}}\' (Qwen3 thinking off)')
    ap.add_argument("--approve-plans", action="store_true", help="HITL: approve each plan (plan type)")
    ap.add_argument("--approve-levels", action="store_true", help="HITL: review each level's results (plan type)")
    ap.add_argument("--state", default=None, help="SQLite checkpoint file (persistent HITL state)")
    ap.add_argument("--demo-hitl", action="store_true",
                    help="offline demo that hits an approval-gated tool (`publish`)")
    ap.add_argument("--trace", default=None, metavar="DIR",
                    help="write a JSONL run trace (llm/tool calls, interrupts) into DIR")
    args = ap.parse_args()
    if args.list_patterns:
        _list_patterns()
        return
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
            demo_hitl=args.demo_hitl,
            trace_dir=args.trace,
            sampling=json.loads(args.sampling) if args.sampling else None,
        )
    )


if __name__ == "__main__":
    main()
