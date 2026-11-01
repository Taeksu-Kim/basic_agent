"""Anthropic's five workflow patterns ("Building effective agents", 2024-12) as
contract-compliant graphs: prompt chaining, routing, parallelization (sectioning /
voting), evaluator-optimizer, orchestrator-workers.

Design rule shared by all of them: **whatever a pattern delegates to is a tool in
its registry.** A sub-agent of any pattern is registered as an ``AgentTool``. So
everything the core already guarantees applies unchanged -- approval gates,
nested / parallel HITL, read-only scopes, failure propagation, tracing -- because
every delegation goes through :func:`agent.core.execution.execute_tool`.

Every graph honours :mod:`agent.core.contract` (input ``query``/``history``/
``context``; output ``final``/``result``/``status``/``error``), so each pattern can
itself be wrapped in an ``AgentTool`` and nested anywhere.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from typing import Annotated, Any, Callable, Optional, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from agent.core import trace
from agent.core.execution import execute_tool, is_failure, read_only_scope
from agent.core.llm import OUTPUT_ERRORS, LLMClient
from agent.core.tools import AgentTool, ToolRegistry


class PatternError(ValueError):
    """The model's output stayed unusable after retries (ends the run with status=error)."""


# ------------------------------------------------------------------ shared helpers


def _extract_json(text: str) -> Any:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if not m:
            raise PatternError(f"no JSON found in output: {text[:200]!r}")
        return json.loads(m.group(0))


def ask_json(llm: LLMClient, system: str, user: str, schema: dict, *,
             check: Callable[[Any], None] = lambda _: None, retries: int = 1, what: str = "output") -> dict:
    """One structured LLM call; malformed output is retried with the error fed back."""
    last: Exception | None = None
    for _ in range(retries + 1):
        text = llm.complete(system, user, schema=schema)
        try:
            data = _extract_json(text)
            if not isinstance(data, dict):
                raise PatternError(f"expected a JSON object, got {type(data).__name__}")
            check(data)
            return data
        except OUTPUT_ERRORS as exc:
            last = exc
            user = f"{user}\n\n[Your previous {what} was invalid: {exc}. Emit valid JSON.]"
    raise PatternError(f"no valid {what} after {retries + 1} attempts: {last}")


def text_of(value: Any) -> str:
    if is_failure(value):
        return f"[failed: {value.get('error') or value.get('denied')}]"
    if isinstance(value, str):
        return value
    return json.dumps(value, default=str, ensure_ascii=False)


def _accepts(registry: ToolRegistry, name: str, arg: str) -> bool:
    tool = registry.get(name)
    if isinstance(tool, AgentTool):
        return True  # sub-agents take query (+ context, extra state keys)
    props = ((tool.args_schema or {}).get("properties") or {})
    return arg in props


def delegate_args(registry: ToolRegistry, name: str, query: str, *,
                  context: Any = None, args: Optional[dict] = None) -> dict:
    """Arguments for handing ``query`` to tool ``name``.

    A sub-agent always gets the request as ``query`` (+ ``context``) -- a real model
    (Qwen3.8-27B) filled router ``args`` with structured fields for an agent route,
    which used to replace the query and crash the call. For plain tools, explicit
    ``args`` win, else the query if the tool takes one."""
    if isinstance(registry.get(name), AgentTool):
        out: dict[str, Any] = {"query": query}
        if context is not None:
            out["context"] = context
        return out
    if args:
        return dict(args)
    out = {}
    if _accepts(registry, name, "query"):
        out["query"] = query
        if context is not None and _accepts(registry, name, "context"):
            out["context"] = context
    return out


def _fmt(template: str, values: dict[str, Any]) -> str:
    """``{name}`` substitution for known keys only (other braces are left alone)."""
    def repl(m: re.Match[str]) -> str:
        key = m.group(1)
        return text_of(values[key]) if key in values else m.group(0)
    return re.sub(r"\{(\w+)\}", repl, template)


def _merge(a: dict | None, b: dict | None) -> dict:
    out = dict(a or {})
    out.update(b or {})
    return out


class _Base(TypedDict, total=False):
    query: str
    history: list[Any]
    context: Any
    iteration: int
    final: str
    result: Any
    status: Optional[str]
    error: str


def _ended_with(error: str) -> dict[str, Any]:
    return {"status": "error", "error": error, "final": ""}


# ------------------------------------------------------------------ prompt chaining


CHAIN_SYSTEM = "You are one step of a fixed multi-step pipeline. Do exactly what this step asks."


class ChainState(_Base, total=False):
    step: int
    outputs: Annotated[dict[str, Any], _merge]


def build_prompt_chain(*, llm: LLMClient, registry: ToolRegistry, steps: list[dict],
                       checkpointer: Any = None) -> Any:
    """Prompt chaining: run ``steps`` in order, each seeing the previous outputs.

    A step is ``{"name": ..., "prompt": "...{query}...{previous}...{<step name>}"}``
    (one LLM call) or ``{"name": ..., "tool": <registry tool>, "args": {...}}``
    (string args are templated the same way). Optional ``"gate": fn(output) -> str | None``
    is a programmatic check between steps; a returned string stops the chain
    (``status=error``, the string is the reason). ``final`` = last step's output;
    ``result`` = all step outputs by name.
    """
    if not steps:
        raise ValueError("prompt_chain needs at least one step")
    step_llm = trace.traced(llm, "prompt_chain.step")

    async def step_node(state: ChainState) -> dict[str, Any]:
        i = state.get("step", 0)
        spec = steps[i]
        name = spec.get("name") or f"step{i + 1}"
        outputs = state.get("outputs") or {}
        values = {"query": state["query"], "context": state.get("context") or "",
                  "previous": list(outputs.values())[-1] if outputs else "", **outputs}
        if "tool" in spec:
            args = {k: (_fmt(v, values) if isinstance(v, str) else v)
                    for k, v in (spec.get("args") or {}).items()}
            out = await execute_tool(registry, spec["tool"],
                                     args or delegate_args(registry, spec["tool"], state["query"]))
            if is_failure(out):
                return {**_ended_with(f"step {name!r} failed: {text_of(out)}"), "outputs": {name: out}}
        else:
            out = step_llm.complete(CHAIN_SYSTEM, _fmt(spec["prompt"], values)).strip()
        gate = spec.get("gate")
        problem = gate(out) if gate else None
        update: dict[str, Any] = {"outputs": {name: out}, "step": i + 1, "iteration": i + 1,
                                  "status": None}
        if problem:
            update.update(_ended_with(f"gate after step {name!r}: {problem}"))
        elif i + 1 == len(steps):
            update.update(final=text_of(out), status="ok",
                          result={**outputs, name: out})
        return update

    def route(state: ChainState) -> str:
        return END if state.get("status") in ("ok", "error") else "step"

    b = StateGraph(ChainState)
    b.add_node("step", step_node)
    b.add_edge(START, "step")
    b.add_conditional_edges("step", route, ["step", END])
    return b.compile(checkpointer=checkpointer)


# ------------------------------------------------------------------ routing


ROUTER_SYSTEM = (
    "You are a router. Pick the ONE route best suited to handle the user's request. "
    "If the chosen route needs arguments other than the request itself, provide them in args."
)


class RouterState(_Base, total=False):
    route: str
    route_args: dict[str, Any]
    route_reason: str


def build_router(*, llm: LLMClient, registry: ToolRegistry, routes: Optional[dict[str, str]] = None,
                 fallback: Optional[str] = None, checkpointer: Any = None) -> Any:
    """Routing: classify the request, hand it to exactly one route.

    ``routes`` maps a registry tool name to when to use it (default: every tool,
    described by its own description). The choice is constrained to the route names
    by schema; ``fallback`` is used if the model's choice is still unusable.
    ``final``/``result`` come from the chosen route; a failed route -> ``status=error``.
    """
    names = list(routes) if routes else registry.names()
    if not names:
        raise ValueError("router needs at least one route")
    desc = routes or {n: registry.get(n).description for n in names}
    route_llm = trace.traced(llm, "router.classify")
    schema = {"type": "object",
              "properties": {"route": {"type": "string", "enum": names}, "args": {"type": "object"},
                             "reason": {"type": "string"}},
              "required": ["route"]}

    def classify(state: RouterState) -> dict[str, Any]:
        listing = "\n".join(f"- {n}: {desc.get(n, '')}" for n in names)
        user = f"Request: {state['query']}\n\nRoutes:\n{listing}"
        if state.get("context") is not None:
            user += f"\n\nContext: {text_of(state['context'])}"

        def check(d: dict) -> None:
            if d.get("route") not in names:
                raise PatternError(f"route must be one of {names}")
        try:
            d = ask_json(route_llm, ROUTER_SYSTEM, user, schema, check=check, what="route")
        except PatternError as exc:
            if fallback is None:
                return _ended_with(str(exc))
            d = {"route": fallback, "reason": f"fallback ({exc})"}
        return {"route": d["route"], "route_args": d.get("args") or {},
                "route_reason": d.get("reason", ""), "status": None, "iteration": 1}

    async def call_route(state: RouterState) -> dict[str, Any]:
        name = state["route"]
        out = await execute_tool(registry, name, delegate_args(
            registry, name, state["query"], context=state.get("context"), args=state.get("route_args")))
        if is_failure(out):
            return _ended_with(f"route {name!r} failed: {text_of(out)}")
        return {"final": text_of(out), "result": None if isinstance(out, str) else out, "status": "ok"}

    def after_classify(state: RouterState) -> str:
        return END if state.get("status") == "error" else "call_route"

    b = StateGraph(RouterState)
    b.add_node("classify", classify)
    b.add_node("call_route", call_route)
    b.add_edge(START, "classify")
    b.add_conditional_edges("classify", after_classify, ["call_route", END])
    b.add_edge("call_route", END)
    return b.compile(checkpointer=checkpointer)


# ------------------------------------------------------------------ parallelization


SYNTH_SYSTEM = ("Combine the partial results below into one answer to the user's request. "
                "Use only what the results say; note any part that failed.")


class ParallelState(_Base, total=False):
    branches: Annotated[dict[int, Any], _merge]


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower().rstrip("."))


def build_parallel(*, llm: LLMClient, registry: ToolRegistry, mode: str = "sectioning",
                   sections: Optional[list[dict]] = None, voter: Optional[str] = None,
                   n: int = 3, aggregate: Optional[str] = None, checkpointer: Any = None) -> Any:
    """Parallelization, two variants.

    * ``mode="sectioning"``: ``sections = [{"name", "tool", "query": "...{query}..."}]``
      (or ``"args": {...}`` with templated strings, for plain tools without a query arg)
      run concurrently; ``aggregate`` = ``"concat"`` (default) or ``"llm"`` (synthesise).
    * ``mode="voting"``: tool ``voter`` runs ``n`` times on the same request;
      ``aggregate`` = ``"majority"`` (default; most common normalised answer) or ``"llm"``.

    Branches fan out with ``Send`` (several may pause for approval at once -- resume
    with an ``{interrupt_id: verdict}`` dict). Failed branches are reported, not
    hidden; all branches failing -> ``status=error``.
    """
    if mode == "sectioning":
        if not sections:
            raise ValueError("sectioning needs sections")
        plan = [{"name": s.get("name") or s["tool"], "tool": s["tool"],
                 "query": s.get("query", "{query}"), "args": s.get("args")} for s in sections]
        aggregate = aggregate or "concat"
    elif mode == "voting":
        if not voter:
            raise ValueError("voting needs a voter tool")
        plan = [{"name": f"vote{i + 1}", "tool": voter, "query": "{query}"} for i in range(n)]
        aggregate = aggregate or "majority"
    else:
        raise ValueError(f"unknown mode {mode!r}")
    synth_llm = trace.traced(llm, "parallel.aggregate")

    def fan_out(state: ParallelState) -> list[Send]:
        values = {"query": state["query"], "context": state.get("context") or ""}
        return [Send("branch", {"i": i, "tool": p["tool"],
                                "query": _fmt(p["query"], values), "context": state.get("context"),
                                "args": {k: (_fmt(v, values) if isinstance(v, str) else v)
                                         for k, v in (p.get("args") or {}).items()} or None})
                for i, p in enumerate(plan)]

    async def branch(payload: dict[str, Any]) -> dict[str, Any]:
        out = await execute_tool(registry, payload["tool"], delegate_args(
            registry, payload["tool"], payload["query"], context=payload.get("context"),
            args=payload.get("args")))
        return {"branches": {payload["i"]: out}}

    def combine(state: ParallelState) -> dict[str, Any]:
        outs = state.get("branches") or {}
        named = {plan[i]["name"]: outs.get(i) for i in range(len(plan))}
        ok = {k: v for k, v in named.items() if not is_failure(v)}
        if not ok:
            return {**_ended_with("all branches failed: " + "; ".join(
                f"{k}: {text_of(v)}" for k, v in named.items())), "result": named}
        if aggregate == "majority":
            votes = Counter(_norm(text_of(v)) for v in ok.values())
            top, count = votes.most_common(1)[0]
            winner = next(text_of(v) for v in ok.values() if _norm(text_of(v)) == top)
            return {"final": winner, "status": "ok", "iteration": 1,
                    "result": {"votes": dict(votes), "agreement": f"{count}/{len(plan)}",
                               "branches": named}}
        if aggregate == "llm":
            parts = "\n\n".join(f"[{k}] {text_of(v)}" for k, v in named.items())
            final = synth_llm.complete(SYNTH_SYSTEM, f"Request: {state['query']}\n\nResults:\n{parts}")
            return {"final": final.strip(), "status": "ok", "iteration": 1, "result": named}
        return {"final": "\n\n".join(f"[{k}] {text_of(v)}" for k, v in named.items()),
                "status": "ok", "iteration": 1, "result": named}

    b = StateGraph(ParallelState)
    b.add_node("branch", branch)
    b.add_node("combine", combine)
    b.add_conditional_edges(START, fan_out, ["branch"])
    b.add_edge("branch", "combine")
    b.add_edge("combine", END)
    return b.compile(checkpointer=checkpointer)


# ------------------------------------------------------------------ evaluator-optimizer


GEN_SYSTEM = ("Produce the requested output. If evaluator feedback on a previous draft is given, "
              "revise the draft to address every point. Output only the result.")
EVAL_SYSTEM = ("You are a strict evaluator. Judge the draft ONLY against the criteria. Pass it only "
               "if every criterion is met; otherwise give concrete, actionable feedback.")
EVAL_SCHEMA = {"type": "object",
               "properties": {"pass": {"type": "boolean"}, "feedback": {"type": "string"}},
               "required": ["pass", "feedback"]}


class EvalOptState(_Base, total=False):
    draft: Any
    feedback: str
    rounds: Annotated[list[dict[str, Any]], lambda a, b: (a or []) + (b or [])]


def build_evaluator_optimizer(*, llm: LLMClient, registry: ToolRegistry, criteria: str,
                              generator: Optional[str] = None, max_rounds: int = 3,
                              checkpointer: Any = None) -> Any:
    """Evaluator-optimizer: generate -> evaluate against ``criteria`` -> revise, until
    the evaluator passes it or ``max_rounds`` is reached.

    The generator is the LLM itself, or a registry tool / sub-agent (``generator``),
    which gets the previous draft + feedback as ``context``. The evaluator always runs
    in a **fresh context** (task, criteria, current draft only) -- the evaluator doesn't
    see how the draft was made, which is the point. Passing -> ``status=ok``; never
    passing within the cap -> ``status=capped`` with the last draft as ``final``.
    """
    gen_llm = trace.traced(llm, "evaluator_optimizer.generate")
    eval_llm = trace.traced(llm, "evaluator_optimizer.evaluate")

    async def generate(state: EvalOptState) -> dict[str, Any]:
        draft, feedback = state.get("draft"), state.get("feedback", "")
        revise = (f"Previous draft:\n{text_of(draft)}\n\nEvaluator feedback:\n{feedback}"
                  if draft is not None else None)
        if generator:
            ctx = "\n\n".join(filter(None, [text_of(state["context"]) if state.get("context") else "",
                                            f"Criteria: {criteria}", revise]))
            out = await execute_tool(registry, generator,
                                     delegate_args(registry, generator, state["query"], context=ctx))
            if is_failure(out):
                return _ended_with(f"generator {generator!r} failed: {text_of(out)}")
        else:
            user = f"Task: {state['query']}\n\nCriteria: {criteria}" + (f"\n\n{revise}" if revise else "")
            out = gen_llm.complete(GEN_SYSTEM, user).strip()
        return {"draft": out, "iteration": int(state.get("iteration", 0)) + 1, "status": None}

    def evaluate(state: EvalOptState) -> dict[str, Any]:
        user = (f"Task: {state['query']}\n\nCriteria: {criteria}\n\n"
                f"Draft:\n{text_of(state['draft'])}")
        try:
            d = ask_json(eval_llm, EVAL_SYSTEM, user, EVAL_SCHEMA, what="evaluation")
        except PatternError as exc:
            return _ended_with(str(exc))
        rnd = {"round": state.get("iteration", 0), "pass": bool(d["pass"]),
               "feedback": d.get("feedback", "")}
        update: dict[str, Any] = {"feedback": rnd["feedback"], "rounds": [rnd]}
        done = rnd["pass"] or state.get("iteration", 0) >= max_rounds
        if done:
            update.update(final=text_of(state["draft"]),
                          result={"passed": rnd["pass"], "rounds": len((state.get("rounds") or [])) + 1,
                                  "last_feedback": rnd["feedback"]})
            if rnd["pass"]:
                update["status"] = "ok"
        return update

    def after_generate(state: EvalOptState) -> str:
        return END if state.get("status") == "error" else "evaluate"

    def after_evaluate(state: EvalOptState) -> str:
        if state.get("status") in ("ok", "error") or state.get("iteration", 0) >= max_rounds:
            return END  # cap reached without a pass -> status unset -> "capped"
        return "generate"

    b = StateGraph(EvalOptState)
    b.add_node("generate", generate)
    b.add_node("evaluate", evaluate)
    b.add_edge(START, "generate")
    b.add_conditional_edges("generate", after_generate, ["evaluate", END])
    b.add_conditional_edges("evaluate", after_evaluate, ["generate", END])
    return b.compile(checkpointer=checkpointer)


# ------------------------------------------------------------------ orchestrator-workers


ORCH_SYSTEM = (
    "You are an orchestrator. Break the request into the subtasks needed and assign each to a "
    "worker. For each subtask write a clear brief: objective, the output format you need back, "
    "and boundaries (what NOT to do, to avoid overlap with other subtasks). Scale effort to the "
    "request: a simple lookup needs 1 subtask; comparisons need one per item compared. Never "
    "create two subtasks that do the same work."
)
SYNTH_ORCH_SYSTEM = (
    "You are the orchestrator. Using the workers' results, answer the user's request. If a "
    "needed piece is missing and another round of subtasks would get it, ask for more; "
    "otherwise finish, stating plainly anything that could not be found."
)
SYNTH_FINAL_NOTE = "\n\nThis is the last round: you must finish now."


class OrchState(_Base, total=False):
    subtasks: list[dict[str, Any]]
    work: Annotated[dict[int, Any], _merge]
    history_rounds: Annotated[list[dict[str, Any]], lambda a, b: (a or []) + (b or [])]
    more_reason: str


def build_orchestrator_workers(*, llm: LLMClient, registry: ToolRegistry,
                               workers: Optional[list[str]] = None, max_workers: int = 4,
                               max_rounds: int = 2, read_only_workers: bool = True,
                               checkpointer: Any = None) -> Any:
    """Orchestrator-workers (dynamic): the orchestrator decides the subtasks at run
    time (unlike plan_execute's fixed tool-call DAG), hands each to a worker with a
    brief (objective / output format / boundaries), workers run in parallel, and the
    orchestrator synthesises -- or asks for another round (up to ``max_rounds``; the
    last round may only finish, enforced by schema).

    ``workers``: registry tools / sub-agents it may delegate to (default: all).
    ``read_only_workers`` (default True): tools marked ``writes=True`` are refused
    inside workers -- single writer: delegation gathers intelligence, it doesn't act.
    """
    names = workers or registry.names()
    if not names:
        raise ValueError("orchestrator_workers needs at least one worker")
    orch_llm = trace.traced(llm, "orchestrator_workers.plan")
    synth_llm = trace.traced(llm, "orchestrator_workers.synthesize")
    plan_schema = {"type": "object", "properties": {"subtasks": {
        "type": "array", "minItems": 1, "maxItems": max_workers,
        "items": {"type": "object",
                  "properties": {"worker": {"type": "string", "enum": names}, "objective": {"type": "string"},
                                 "output_format": {"type": "string"},
                                 "boundaries": {"type": "string"}},
                  "required": ["worker", "objective"]}}},
        "required": ["subtasks"]}
    finish = {"type": "object", "properties": {"decision": {"type": "string", "const": "finish"},
                                               "final": {"type": "string"}},
              "required": ["decision", "final"]}
    more = {"type": "object", "properties": {"decision": {"type": "string", "const": "more"},
                                             "reason": {"type": "string"}},
            "required": ["decision", "reason"]}

    def orchestrate(state: OrchState) -> dict[str, Any]:
        listing = "\n".join(f"- {n}: {registry.get(n).description}" for n in names)
        user = f"Request: {state['query']}\n\nWorkers:\n{listing}"
        if state.get("context") is not None:
            user += f"\n\nContext: {text_of(state['context'])}"
        for r in state.get("history_rounds") or []:
            user += f"\n\nPrevious round results:\n{r['summary']}"
        if state.get("more_reason"):
            user += f"\n\nWhat is still missing: {state['more_reason']}"

        def check(d: dict) -> None:
            subs = d.get("subtasks")
            if not isinstance(subs, list) or not subs:
                raise PatternError("subtasks must be a non-empty list")
            bad = [s.get("worker") for s in subs if s.get("worker") not in names]
            if bad:
                raise PatternError(f"unknown workers {bad}; workers: {names}")
        try:
            d = ask_json(orch_llm, ORCH_SYSTEM, user, plan_schema, check=check, what="subtask plan")
        except PatternError as exc:
            return {**_ended_with(str(exc)), "subtasks": []}
        return {"subtasks": d["subtasks"][:max_workers], "work": {},
                "iteration": int(state.get("iteration", 0)) + 1, "status": None}

    def fan_out(state: OrchState) -> list[Send] | str:
        if state.get("status") == "error":
            return END
        return [Send("work", {"i": i, "sub": sub}) for i, sub in enumerate(state["subtasks"])]

    async def work(payload: dict[str, Any]) -> dict[str, Any]:
        sub = payload["sub"]
        brief = "\n".join(filter(None, [
            f"Objective: {sub['objective']}",
            f"Return: {sub['output_format']}" if sub.get("output_format") else "",
            f"Boundaries: {sub['boundaries']}" if sub.get("boundaries") else ""]))
        name = sub["worker"]
        with read_only_scope(f"worker:{name}" if read_only_workers else None):
            out = await execute_tool(registry, name, delegate_args(registry, name, brief))
        return {"work": {payload["i"]: out}}

    def synthesize(state: OrchState) -> dict[str, Any]:
        outs = state.get("work") or {}
        summary = "\n".join(f"[{i + 1}] {s['worker']} - {s['objective']}: {text_of(outs.get(i))}"
                            for i, s in enumerate(state["subtasks"]))
        last = state.get("iteration", 0) >= max_rounds
        schema = finish if last else {"anyOf": [finish, more]}
        system = SYNTH_ORCH_SYSTEM + (SYNTH_FINAL_NOTE if last else "")
        user = f"Request: {state['query']}\n\nWorker results:\n{summary}"
        def check(d: dict) -> None:
            allowed = ("finish",) if last else ("finish", "more")
            if d.get("decision") not in allowed:
                raise PatternError(f"decision must be one of {allowed}")
        try:
            d = ask_json(synth_llm, system, user, schema, what="synthesis", check=check)
        except PatternError as exc:
            return _ended_with(str(exc))
        rnd = {"round": state.get("iteration", 0), "summary": summary}
        if d["decision"] == "finish" or last:
            return {"final": d.get("final", ""), "status": "ok", "history_rounds": [rnd],
                    "result": {"rounds": state.get("iteration", 0),
                               "subtasks": [{**s, "result": outs.get(i)}
                                            for i, s in enumerate(state["subtasks"])]}}
        return {"more_reason": d.get("reason", ""), "history_rounds": [rnd]}

    def after_synth(state: OrchState) -> str:
        return END if state.get("status") in ("ok", "error") else "orchestrate"

    b = StateGraph(OrchState)
    b.add_node("orchestrate", orchestrate)
    b.add_node("work", work)
    b.add_node("synthesize", synthesize)
    b.add_edge(START, "orchestrate")
    b.add_conditional_edges("orchestrate", fan_out, ["work", END])
    b.add_edge("work", "synthesize")
    b.add_conditional_edges("synthesize", after_synth, ["orchestrate", END])
    return b.compile(checkpointer=checkpointer)
