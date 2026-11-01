"""Real-model evaluation set (WORKPLAN 0.3): fixed tasks over deterministic fake tools.

Runs every task under several conditions (pattern x context policy) against a real
OpenAI-compatible endpoint, traces each run, and reports per-run metrics:
success, status, steps, LLM calls, tokens, tool failures, wall time.

    python -m agent.examples.evals.run_evals \\
        --base-url http://<WSL IP>:8002/v1 --model qwen3.8-27b-redhat-int4 \\
        --sampling '{"chat_template_kwargs": {"enable_thinking": false}}' \\
        --out runs/evals            # writes results.json + one trace per run

Tools are fake and deterministic (fixed prices, rates, documents), so success is
checked against a known answer. Approval-gated tools are auto-approved by the
driver (the eval measures the agent, not the human).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

from langgraph.checkpoint.memory import InMemorySaver

from agent.core.context import ContextPolicy
from agent.core.contract import envelope
from agent.core.hitl import aresume_graph, arun_graph, pending_interrupts
from agent.core.llm import OpenAICompatLLM
from agent.core.tools import AgentTool, ToolRegistry
from agent.core.trace import Tracer, tracing
from agent.patterns import registry as patterns

# ------------------------------------------------------------------ fake world

PRICES = {"AAPL": 231.5, "MSFT": 418.2, "NVDA": 135.6}
CEOS = {"acme corp": "Jane Park"}
AGES = {"jane park": 52}
RATES = {("USD", "KRW"): 1380.0, ("EUR", "USD"): 1.08, ("USD", "EUR"): 0.926}
HISTORY = {("AAPL", "2026-10-30"): 229.1}


def _obj(props: dict, required: list[str]) -> dict:
    return {"type": "object", "properties": props, "required": required}


def _filing(ticker: str) -> str:
    filler = (f"{ticker} annual report. Section text about operations, risk factors, "
              "supply chain, governance and other routine disclosures. ")
    body = (filler * 60)[:4500]
    return body + f" Capital return: Dividend per share: 0.77 USD for fiscal 2026. " + (filler * 20)[:1500]


_EN = ("Section text about operations, risk factors, supply chain, governance and other "
       "routine disclosures. ")
_KO = "회사 개요와 사업 위험 요인, 공급망, 지배구조 등에 관한 일반적인 공시 내용입니다. "


def _doc(filler: str, length: int, facts: dict[int, str]) -> str:
    """``length`` chars of filler with each fact inserted at its character position."""
    text = (filler * (length // len(filler) + 2))[:length]
    for pos in sorted(facts, reverse=True):
        text = text[:pos] + f" {facts[pos]} " + text[pos:]
    return text


DOCS = {
    "acme-early": _doc(_EN, 6000, {400: "Dividend per share: 0.42 USD."}),
    "acme-absent": _doc(_EN, 6000, {}),
    "acme-split": _doc(_EN, 6000, {2000: "Total revenue: 120.0 billion USD.",
                                   4200: "Net income: 30.0 billion USD."}),
    "korea-report": _doc(_KO, 6000, {4500: "주당 배당금: 1,450원."}),
    "acme-deep": _doc(_EN, 12000, {10500: "Dividend per share: 1.15 USD."}),
    "huge-dump": _doc(_EN, 60000, {800: "Dividend per share: 0.31 USD."}),
}


def build_registry(log: list, *, distractors: int = 0, docs: bool = False) -> ToolRegistry:
    reg = ToolRegistry()

    def get_price(ticker: str) -> dict:
        t = str(ticker).upper()
        if t not in PRICES:
            raise KeyError(f"unknown ticker {ticker!r}; known: {sorted(PRICES)}")
        return {"ticker": t, "price_usd": PRICES[t]}

    def get_ceo(company: str) -> str:
        return CEOS.get(str(company).lower(), "unknown company")

    def get_age(person: str) -> Any:
        return AGES.get(str(person).lower(), "unknown person")

    def convert_currency(amount: float, from_ccy: str, to_ccy: str) -> dict:
        rate = RATES[(str(from_ccy).upper(), str(to_ccy).upper())]
        return {"amount": round(float(amount) * rate, 4), "currency": str(to_ccy).upper()}

    def get_filing(ticker: str) -> str:
        return _filing(str(ticker).upper())

    def place_order(ticker: str, qty: int, side: str) -> str:
        log.append(("order", str(ticker).upper(), int(qty), str(side).lower()))
        return f"order placed: {side} {qty} {ticker}"

    def get_history(ticker: str, date: str) -> dict:
        if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", str(date)):
            raise ValueError("date must be YYYY-MM-DD, e.g. 2026-10-30")
        key = (str(ticker).upper(), str(date))
        if key not in HISTORY:
            raise KeyError(f"no close for {key}")
        return {"ticker": key[0], "date": key[1], "close_usd": HISTORY[key]}

    def fx_rate(base: str, quote: str) -> dict:
        return {"base": base, "quote": quote, "rate": RATES[(str(base).upper(), str(quote).upper())]}

    s = {"type": "string"}
    reg.register("get_price", get_price, description="latest stock price in USD for a ticker",
                 args_schema=_obj({"ticker": s}, ["ticker"]))
    reg.register("get_ceo", get_ceo, description="name of a company's CEO",
                 args_schema=_obj({"company": s}, ["company"]))
    reg.register("get_age", get_age, description="age in years of a person",
                 args_schema=_obj({"person": s}, ["person"]))
    reg.register("convert_currency", convert_currency,
                 description="convert an amount between currencies (USD, KRW, EUR)",
                 args_schema=_obj({"amount": {"type": "number"}, "from_ccy": s, "to_ccy": s},
                                  ["amount", "from_ccy", "to_ccy"]))
    reg.register("get_filing", get_filing, description="full text of a company's latest annual filing",
                 args_schema=_obj({"ticker": s}, ["ticker"]))
    reg.register("place_order", place_order, description="place a stock order (needs human approval)",
                 args_schema=_obj({"ticker": s, "qty": {"type": "integer"}, "side": {"enum": ["buy", "sell"]}},
                                  ["ticker", "qty", "side"]),
                 requires_approval=True)
    reg.register("get_history", get_history, description="closing price of a ticker on a past date",
                 args_schema=_obj({"ticker": s, "date": s}, ["ticker", "date"]))
    if docs:
        def get_document(doc_id: str) -> str:
            if doc_id not in DOCS:
                raise KeyError(f"unknown document {doc_id!r}; available: {sorted(DOCS)}")
            return DOCS[doc_id]

        reg.register("get_document", get_document, description="full text of a company document by id",
                     args_schema=_obj({"doc_id": {"type": "string"}}, ["doc_id"]))
    if distractors:
        reg.register("fx_rate", fx_rate, description="foreign exchange rate between two currencies",
                     args_schema=_obj({"base": s, "quote": s}, ["base", "quote"]))
        topics = ["revenue", "headcount", "patents", "esg score", "credit rating", "beta",
                  "short interest", "insider trades", "analyst rating", "options volume"]
        for i in range(distractors):
            topic = topics[i % len(topics)]
            reg.register(f"metric_{i:02d}", (lambda topic=topic: f"{topic}: n/a"),
                         description=f"company {topic} metric (variant {i})")
    return reg


# ------------------------------------------------------------------ tasks


def _numbers(text: str) -> list[float]:
    return [float(x.replace(",", "")) for x in re.findall(r"-?\d[\d,]*\.?\d*", text or "")]


def expect_number(value: float, tol: float = 0.011) -> Callable[[dict, list], bool]:
    return lambda env, log: any(abs(n - value) <= tol * max(1.0, abs(value) * 0 + 1) for n in
                                _numbers(env["final"]))


@dataclass
class Task:
    name: str
    query: str
    check: Callable[[dict, list], bool]
    measures: str
    distractors: int = 0
    nested: bool = False
    docs: bool = False
    build: Optional[Callable[[Any, list], Any]] = None  # custom graph (workflow suite)


TASKS = [
    Task("single_lookup", "What is the current price of NVDA in USD?", expect_number(135.6),
         "단일 조회"),
    Task("dependent_chain", "How old is the CEO of Acme Corp?", expect_number(52),
         "의존 체인 (결과를 다음 호출 인자로)"),
    Task("parallel_sum", "What is the sum of the current prices of AAPL, MSFT and NVDA in USD?",
         expect_number(785.3), "독립 조회 3개 + 합산"),
    Task("arg_precision", "Convert 250 USD to KRW.", expect_number(345000), "인자 이름 정확도"),
    Task("large_observation",
         "According to NVDA's latest annual filing, what is the dividend per share?",
         expect_number(0.77, tol=0.001), "큰 관찰(4,500자 뒤의 정보)"),
    Task("approval", "Buy 10 shares of AAPL.",
         lambda env, log: ("order", "AAPL", 10, "buy") in log, "승인 필요 도구 + 정확한 인자"),
    Task("error_recovery", "What was AAPL's closing price on October 30, 2026?",
         expect_number(229.1), "도구 에러 메시지를 보고 인자 수정"),
    Task("many_tools", "What is the EUR to USD exchange rate?", expect_number(1.08, tol=0.001),
         "도구 31개 중 1개 찾기", distractors=30),
    Task("nested_researcher",
         "Ask the researcher for AAPL's current price, then convert that USD amount to KRW.",
         expect_number(319470), "서브에이전트 결과로 후속 계산", nested=True),
]

def says_not_found(env: dict, log: list) -> bool:
    """For a question whose answer is not in the document: must say so, not invent a figure."""
    final = env["final"]
    negates = re.search(r"(not (found|mention|disclos|provid|includ|contain|state|specif)|"
                        r"no (dividend|information|mention)|does not|doesn't|unable|cannot|isn't)",
                        final, re.I)
    invents = re.search(r"(\$\s*\d|\d+(\.\d+)?\s*(USD|dollars))", final)
    return bool(negates) and not invents


LO_TASKS = [
    Task("lo_original", "According to NVDA's latest annual filing, what is the dividend per share?",
         expect_number(0.77, tol=0.001), "답이 4,500자 지점 (기존 과제)"),
    Task("lo_early", "What is the dividend per share stated in document acme-early?",
         expect_number(0.42, tol=0.001), "답이 미리보기 안 (400자) — 더 읽지 않아야 함", docs=True),
    Task("lo_absent", "What is the dividend per share stated in document acme-absent?",
         says_not_found, "답이 문서에 없음 — '없다'고 해야 함", docs=True),
    Task("lo_split",
         "Using document acme-split, what is the net profit margin (net income / revenue) in percent?",
         expect_number(25, tol=0.01), "답이 두 곳(2,000·4,200자)에 나뉨", docs=True),
    Task("lo_korean", "문서 korea-report에 나온 주당 배당금은 얼마인가요?",
         expect_number(1450), "한글 문서, 답이 4,500자 지점", docs=True),
    Task("lo_deep", "What is the dividend per share stated in document acme-deep?",
         expect_number(1.15, tol=0.001), "12,000자 문서의 10,500자 지점 — 스텝 상한 확인", docs=True),
    Task("lo_huge", "What is the dividend per share stated in document huge-dump?",
         expect_number(0.31, tol=0.001), "60,000자 결과 — plan_execute 컨텍스트 초과 재현", docs=True),
]

# ------------------------------------------------------------------ workflow suite


def _price_agent(llm: Any, log: list, **kw: Any) -> AgentTool:
    child = patterns.get("react").build(llm=llm, registry=build_registry(log), max_steps=4)
    return AgentTool("price_agent", child, description="looks up stock prices, FX conversions, "
                                                       "company facts; give it a question in `query`",
                     **kw)


def _wf_router(llm: Any, log: list) -> Any:
    reg = build_registry(log)
    reg.add(_price_agent(llm, log))
    fx = patterns.get("react").build(llm=llm, registry=build_registry(log), max_steps=4)
    reg.add(AgentTool("fx_agent", fx, description="currency conversion questions"))
    return patterns.get("router").build(llm=llm, registry=reg, checkpointer=InMemorySaver(),
                                        routes={"price_agent": "stock price questions",
                                                "fx_agent": "currency conversion questions"})


def _wf_sections(llm: Any, log: list) -> Any:
    reg = ToolRegistry()
    reg.add(_price_agent(llm, log))
    return patterns.get("parallel").build(
        llm=llm, registry=reg, checkpointer=InMemorySaver(), aggregate="llm",
        sections=[{"name": t, "tool": "price_agent", "query": f"What is the current price of {t} in USD?"}
                  for t in ("AAPL", "MSFT", "NVDA")])


def _wf_voting(llm: Any, log: list) -> Any:
    reg = ToolRegistry()
    reg.add(_price_agent(llm, log))
    return patterns.get("parallel").build(llm=llm, registry=reg, checkpointer=InMemorySaver(),
                                          mode="voting", voter="price_agent", n=3)


EVAL_CRITERIA = ("1) mentions the ticker NVDA; 2) states the price 135.6 USD; "
                 "3) at most 20 words; 4) no investment advice")


def _wf_evaluator(llm: Any, log: list) -> Any:
    return patterns.get("evaluator_optimizer").build(
        llm=llm, registry=ToolRegistry(), checkpointer=InMemorySaver(), criteria=EVAL_CRITERIA,
        max_rounds=3)


def _wf_orchestrator(llm: Any, log: list) -> Any:
    reg = ToolRegistry()
    reg.add(_price_agent(llm, log))
    return patterns.get("orchestrator_workers").build(llm=llm, registry=reg,
                                                      checkpointer=InMemorySaver(), max_workers=3)


def _numeric_gate(out: Any) -> Optional[str]:
    try:
        float(str(out).replace(",", "").strip())
        return None
    except ValueError:
        return f"expected a bare number, got {str(out)[:40]!r}"


def _wf_chain(llm: Any, log: list) -> Any:
    reg = build_registry(log)
    reg.add(_price_agent(llm, log))
    return patterns.get("prompt_chain").build(llm=llm, registry=reg, checkpointer=InMemorySaver(), steps=[
        {"name": "fetch", "tool": "price_agent",
         "args": {"query": "What is NVDA's current price in USD?"}},
        {"name": "extract", "prompt": "Extract only the number from this text, nothing else: {fetch}",
         "gate": _numeric_gate},
        {"name": "krw", "tool": "convert_currency",
         "args": {"amount": "{extract}", "from_ccy": "USD", "to_ccy": "KRW"}},
    ])


def _wf_nested_structured(llm: Any, log: list) -> Any:
    reg = build_registry(log)
    reg.add(_price_agent(llm, log, output_schema={
        "type": "object", "properties": {"price_usd": {"type": "number"}}, "required": ["price_usd"]}))
    return patterns.get("plan_execute").build(llm=llm, registry=reg, checkpointer=InMemorySaver())


def _eval_check(env: dict, log: list) -> bool:
    f = env["final"]
    return (env["status"] == "ok" and "NVDA" in f and "135.6" in f and len(f.split()) <= 20)


WF_TASKS = [
    Task("router_fx", "Convert 250 USD to KRW.", expect_number(345000), "router: 환율 경로 선택",
         build=_wf_router),
    Task("router_price", "What is the current price of NVDA in USD?", expect_number(135.6),
         "router: 주가 경로 선택", build=_wf_router),
    Task("parallel_sections_sum", "What is the sum of the current prices of AAPL, MSFT and NVDA in USD?",
         expect_number(785.3), "parallel sectioning + LLM 종합", build=_wf_sections),
    Task("parallel_voting", "What is the current price of MSFT in USD?", expect_number(418.2),
         "parallel voting 3표", build=_wf_voting),
    Task("evaluator_summary", "Write a one-sentence note: NVDA's current price is 135.6 USD.",
         _eval_check, "evaluator: 기준 4개를 통과할 때까지 수정", build=_wf_evaluator),
    Task("orchestrator_compare",
         "Which is higher, AAPL's or MSFT's current price, and by how many USD?",
         lambda env, log: "MSFT" in env["final"] and expect_number(186.7)(env, log),
         "orchestrator: 비교 → 워커 2개 + 종합", build=_wf_orchestrator),
    Task("chain_price_to_krw", "NVDA price in KRW", expect_number(187128),
         "prompt_chain: 에이전트 → 숫자 추출(gate) → 환전 도구", build=_wf_chain),
    Task("nested_structured",
         "Ask the price_agent for AAPL's current price, then convert that USD amount to KRW.",
         expect_number(319470), "plan_execute + output_schema 자식 ($1.price_usd)",
         build=_wf_nested_structured),
]

SUITES = {"main": TASKS, "lo": LO_TASKS, "wf": WF_TASKS}

# ------------------------------------------------------------------ conditions


@dataclass
class Condition:
    name: str
    pattern: str
    policy: Optional[ContextPolicy] = None


POLICY = ContextPolicy(offload=True, keep_recent=4, tool_search_threshold=10)

CONDITIONS = [
    Condition("react", "react"),
    Condition("plan_execute", "plan_execute"),
    Condition("react+policy", "react", POLICY),
    Condition("react+offload", "react", ContextPolicy(offload=True)),
]

# ------------------------------------------------------------------ run


@dataclass
class RunResult:
    task: str
    condition: str
    success: bool
    status: str
    final: str
    steps: int
    llm_calls: int
    prompt_tokens: int
    completion_tokens: int
    tool_calls: int
    tool_failures: int
    interrupts: int
    reads: int
    wall_s: float
    error: str = ""
    notes: list = field(default_factory=list)


def _build(cond: Condition, task: Task, llm: Any, log: list) -> Any:
    if task.build is not None:
        return task.build(llm, log)
    reg = build_registry(log, distractors=task.distractors, docs=task.docs)
    cfg: dict[str, Any] = {"checkpointer": InMemorySaver()}
    if cond.pattern == "react":
        cfg["max_steps"] = 8
    if cond.policy is not None:
        cfg["context_policy"] = cond.policy
    if task.nested:
        child_reg = build_registry(log)
        child = patterns.get("react").build(llm=llm, registry=child_reg, max_steps=4,
                                            **({"context_policy": cond.policy} if cond.policy else {}))
        reg.add(AgentTool("researcher", child,
                          description="a research sub-agent; give it a question in `query`, "
                                      "it looks up stock prices and returns the answer"))
    return patterns.get(cond.pattern).build(llm=llm, registry=reg, **cfg)


async def run_one(task: Task, cond: Condition, llm: Any, trace_path: Path) -> RunResult:
    log: list = []
    graph = _build(cond, task, llm, log)
    t0 = time.perf_counter()
    err = ""
    interrupts = 0
    with tracing(Tracer(trace_path)):
        try:
            state = await arun_graph(graph, task.query, thread_id="eval")
            while pending_interrupts(state) and interrupts < 10:
                intrs = pending_interrupts(state)
                interrupts += len(intrs)
                verdict = True if len(intrs) == 1 else {i.id: True for i in intrs}
                state = await aresume_graph(graph, verdict, thread_id="eval")
        except Exception as e:  # noqa: BLE001 -- a crash is a result here
            state, err = {}, f"{type(e).__name__}: {e}"
    wall = time.perf_counter() - t0
    env = envelope(state)
    events = [json.loads(line) for line in trace_path.open(encoding="utf-8")] if trace_path.exists() else []
    llm_ev = [e for e in events if e["kind"] == "llm_call"]
    tool_ev = [e for e in events if e["kind"] == "tool_call" and not e["span"]]
    usage = [e.get("usage") or {} for e in llm_ev]
    steps = state.get("iteration", 0) if state else 0
    return RunResult(
        task=task.name, condition=cond.name, success=bool(state) and task.check(env, log),
        status=env["status"] if state else "crash", final=env["final"][:200], steps=steps,
        llm_calls=len(llm_ev), prompt_tokens=sum(u.get("prompt_tokens", 0) for u in usage),
        completion_tokens=sum(u.get("completion_tokens", 0) for u in usage),
        tool_calls=len(tool_ev), tool_failures=sum(e["outcome"] == "failure" for e in tool_ev),
        interrupts=interrupts, reads=sum(e["tool"] == "read_observation" for e in tool_ev),
        wall_s=round(wall, 1), error=err or env["error"][:200],
        notes=[f"{e['tool']}:{e['outcome']}" for e in tool_ev],
    )


async def main_async(args: argparse.Namespace) -> list[RunResult]:
    llm = OpenAICompatLLM(base_url=args.base_url, model=args.model, timeout=180,
                          sampling=json.loads(args.sampling) if args.sampling else None)
    out = Path(args.out)
    (out / "traces").mkdir(parents=True, exist_ok=True)
    tasks = [t for t in SUITES[args.suite] if not args.tasks or t.name in args.tasks]
    conds = [c for c in CONDITIONS if not args.conditions or c.name in args.conditions]
    if args.suite == "wf":  # each workflow task builds its own pattern -> one run per task
        conds = [Condition("workflow", "-")]
    results = []
    for task in tasks:
        for cond in conds:
            trace_path = out / "traces" / f"{task.name}__{cond.name}.jsonl"
            trace_path.unlink(missing_ok=True)
            r = await run_one(task, cond, llm, trace_path)
            results.append(r)
            print(f"{'OK  ' if r.success else 'FAIL'} {task.name:<18} {cond.name:<13} "
                  f"status={r.status:<6} steps={r.steps} llm={r.llm_calls} "
                  f"tok={r.prompt_tokens}/{r.completion_tokens} tools={r.tool_calls}"
                  f"(fail {r.tool_failures}) reads={r.reads} {r.wall_s}s | {r.final[:70]!r}"
                  + (f" | ERR {r.error[:80]}" if r.error else ""), flush=True)
    (out / "results.json").write_text(
        json.dumps([r.__dict__ for r in results], ensure_ascii=False, indent=2), encoding="utf-8")
    return results


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--sampling", default=None, metavar="JSON")
    ap.add_argument("--out", default="runs/evals")
    ap.add_argument("--suite", choices=sorted(SUITES), default="main",
                    help="main = 9 general tasks, lo = large-observation variants")
    ap.add_argument("--tasks", nargs="*", help="subset of task names")
    ap.add_argument("--conditions", nargs="*", help="subset of condition names")
    args = ap.parse_args()
    results = asyncio.run(main_async(args))
    ok = sum(r.success for r in results)
    print(f"\n{ok}/{len(results)} succeeded")


if __name__ == "__main__":
    main()
