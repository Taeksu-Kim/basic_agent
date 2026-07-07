# Agent — reusable agent skeleton (LangGraph)

A general-purpose agent skeleton built on LangGraph, developed TDD-first. It is
organised around **two agent types that share one substrate**, so they compose
(either can be wrapped as a tool inside the other):

| type | package | paradigm | when |
|---|---|---|---|
| **Plan-and-Execute** | `plan_execute/` | orchestrated: plan a task DAG, execute, replan | structured / decomposable / parallelisable |
| **ReAct** | `react/` *(planned)* | reactive: LLM directs itself step by step | open-ended / unpredictable |

Shared substrate: `llm.py` (LLM abstraction) and `tools.py` (tools + registry).
Both types compile to a graph with the same `ainvoke` contract and are
`AgentTool`-wrappable. The stock-specific *news → price-reaction prediction*
agent is built on top of this by swapping in real tools and prompts.

## `plan_execute` — pattern

This is a merge of two standard patterns:

- **Plan-and-Execute** (planner → execute → replan loop), and
- **LLMCompiler** (planner emits a task **DAG**; `$N` args feed one task's output
  into another; a **joiner** decides finish-vs-replan after the whole plan runs).

The orchestrator = **planner + joiner + loop control** (no separate query
rewriter / intent classifier — the planner absorbs those). Routing is **static**:
each task's `tool` field is the binding.

```
START ─▶ plan ─(dispatch)─▶ run_task* ─▶ advance ─(dispatch)─▶ … ─▶ join
                                                                     │
                            finish / cap ◀──(route_join)─────────────┤
                            replan ───────────────────────────────────┘  (→ plan)
```

### Plan is a DAG; execution is level-synchronous

The planner outputs a DAG (`{id, tool, args, deps}`, with `$N` refs implying
deps). We compute **topological levels** from it and execute each level with a
LangGraph `Send` fan-out (parallel within a level, barrier between levels) — the
idiomatic LangGraph way. The LLM never emits level groupings; levels are derived.
This is a deliberate simplification of full eager DAG scheduling; the data model
stays a DAG so the executor can be upgraded to eager dispatch later without
touching the planner.

### Replan

LLMCompiler-style: the **joiner** runs once after the whole plan and returns
`finish` (with a final answer) or `replan` (with a reason). The graph caps
planning rounds (`max_iterations`) and **early-aborts** a plan if any task errors
(remaining levels are skipped and the joiner decides). On replan the result
namespace is reset for the new plan.

## Layout

| file | role |
|---|---|
| `llm.py` *(shared)* | `LLMClient` protocol; `FakeLLM` (tests); `OpenAICompatLLM` (vLLM), `AnthropicLLM` |
| `tools.py` *(shared)* | `BaseTool`/`FunctionTool`/`AgentTool` + `ToolRegistry` — the executable units tasks route to |
| `plan_execute/dag.py` | `Task`/`Plan`, validation, `topological_levels`, `$N` substitution (pure) |
| `plan_execute/planner.py` | query → DAG (JSON-schema constrained) + computed levels |
| `plan_execute/joiner.py` | finish / replan classifier |
| `plan_execute/executor.py` | run one resolved task against the registry |
| `plan_execute/state.py` | `AgentState` + `results` merge reducer |
| `plan_execute/graph.py` | assembles the StateGraph; `build_graph`, `arun` |
| `react/` | *(planned)* reactive tool-calling loop |
| `run.py` | CLI entrypoint (`--demo` runs offline) |

## LLM / inference

Designed for **vLLM behind an OpenAI-compatible endpoint**. `planner`/`joiner`
pass a JSON schema to `complete(...)`, which `OpenAICompatLLM` forwards as
`response_format: {type: json_schema}` (vLLM guided decoding) for valid JSON.

> The whole graph is exercised in tests with `FakeLLM` — **no real model call and
> no network in the test suite.**

## Run

```bash
conda activate stock-dataset
python -m agent.run --demo                    # offline smoke test (FakeLLM)
python -m agent.run "your task" \
    --base-url http://localhost:8000/v1 --model <served-model>   # real vLLM
python -m pytest agent/tests -q               # 54 tests
```

## Tools & sub-agents

An executable unit is a `BaseTool` (`name`, `description`, `args_schema`,
`run(**args)`). Register a plain callable via `registry.register(...)` /
`@registry.tool(...)`, or add a `BaseTool` subclass with `registry.add(...)`.
Tool `description`/`args_schema` are advertised to the planner so it routes well.

A **sub-agent is just a tool**: `AgentTool` wraps a compiled graph so the plan
can route to it by name; it runs its own plan/execute loop internally and returns
a result (the *agent-as-tool* pattern). This is how planned multi-agent is added
without touching the executor or the graph.

## Extending (→ news→reaction agent)

1. Register real tools on a `ToolRegistry` (e.g. `fetch_article`, `get_price_window`,
   `extract_catalyst`, `predict_reaction`).
2. Tailor `PLANNER_SYSTEM` / `JOINER_SYSTEM` prompts to the domain.
3. Point `OpenAICompatLLM` at the vLLM server and run.

Structural core (`plan_execute/dag.py`, `plan_execute/graph.py`,
`plan_execute/state.py`) stays unchanged.
