"""조합 행렬 — 패턴 간 중첩·HITL·실패 처리를 E2E 로 검증한다 (FakeLLM, 네트워크 없음).

각 시나리오는 "코드가 하는 일"이 아니라 **"해야 하는 일"을 기대값으로** 정의한다.
그래서 결함이 남아 있으면 FAIL 로 나타난다.

    python -m agent.examples.compose_matrix            # 표 출력
    python -m agent.examples.compose_matrix --json out.json

구성
- PAIR 시나리오: (부모 패턴, 자식 패턴) 모든 쌍에서 실행. 자식은 ``AgentTool`` 로 감싼다.
- TOP 시나리오: 패턴 하나를 최상위로 실행.
패턴 목록은 ``PATTERNS`` 에 있다. 새 패턴은 여기에 (build, run) 어댑터를 추가하면
행렬에 자동으로 들어간다.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import traceback
from dataclasses import dataclass, field
from typing import Any, Callable

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from agent.core.context import ContextPolicy, Skill
from agent.core.contract import envelope
from agent.core.execution import is_failure
from agent.core.hitl import ask_human
from agent.core.llm import ContextOverflowError, FakeLLM
from agent.core.tools import AgentTool, ToolRegistry
from agent.patterns import registry as patterns
from agent.plan_execute.orchestrator.joiner import FINISH_ONLY_SCHEMA, JOINER_SYSTEM
from agent.react.policy import COMPACT_SYSTEM

# --------------------------------------------------------------------------
# 패턴 어댑터: 시나리오를 패턴 중립적으로 쓰기 위한 최소 인터페이스
# --------------------------------------------------------------------------


def _react_script(steps: list[list[tuple[str, dict]]], final: str) -> list[dict]:
    """steps 의 각 호출을 한 스텝씩 순서대로 실행한 뒤 finish."""
    out: list[dict] = []
    for step in steps:
        for tool, args in step:
            out.append({"action": "tool", "tool": tool, "args": args})
    out.append({"action": "finish", "final": final})
    return out


def _plan_script(steps: list[list[tuple[str, dict]]], final: str) -> list[dict]:
    """steps 하나가 DAG 레벨 하나. 각 레벨은 직전 레벨 전체에 의존."""
    tasks, prev, tid = [], [], 0
    for step in steps:
        cur = []
        for tool, args in step:
            tid += 1
            tasks.append({"id": tid, "tool": tool, "args": args, "deps": list(prev)})
            cur.append(tid)
        prev = cur
    return [{"tasks": tasks}, {"decision": "finish", "final": final}]


@dataclass
class Pattern:
    """행렬용 어댑터: 등록부 패턴 + FakeLLM 스크립트 형식 + '상한에 걸리게 하는' 설정."""

    name: str            # 행렬 표의 짧은 이름
    registry_name: str   # agent.patterns.registry 이름
    script: Callable[[list[list[tuple[str, dict]]], str], list[dict]]
    cap_config: dict[str, Any] = field(default_factory=dict)  # 1회 만에 상한 도달

    def build(self, llm: Any, reg: ToolRegistry, saver: Any = None, **config: Any) -> Any:
        return patterns.get(self.registry_name).build(llm=llm, registry=reg, checkpointer=saver,
                                                      **config)


PATTERNS: dict[str, Pattern] = {
    "react": Pattern("react", "react", _react_script, cap_config={"max_steps": 1}),
    "plan": Pattern("plan", "plan_execute", _plan_script, cap_config={"max_iterations": 1}),
}


def fake(objs: list[Any]) -> FakeLLM:
    return FakeLLM([o if isinstance(o, str) else json.dumps(o) for o in objs])


def _cfg(thread: str | None) -> dict | None:
    return {"configurable": {"thread_id": thread}} if thread else None


async def start(graph: Any, query: str = "q", thread: str | None = None) -> dict:
    return await graph.ainvoke({"query": query, "history": [], "iteration": 0}, _cfg(thread))


def pending(state: dict) -> list[Any]:
    return list(state.get("__interrupt__", []) or [])


async def resume(graph: Any, state: dict, verdict_of: Callable[[Any], Any], thread: str) -> dict:
    """대기 중인 interrupt 를 전부 응답해 재개. 여러 개면 interrupt id 로 매핑."""
    intrs = pending(state)
    if len(intrs) == 1:
        cmd = Command(resume=verdict_of(intrs[0].value))
    else:
        cmd = Command(resume={i.id: verdict_of(i.value) for i in intrs})
    return await graph.ainvoke(cmd, _cfg(thread))


# --------------------------------------------------------------------------
# 결과 수집
# --------------------------------------------------------------------------


@dataclass
class Outcome:
    scenario: str
    cell: str  # "parent⊃child" 또는 "pattern"
    expect: str
    ok: bool
    observed: str
    fix: str = ""  # 이 기대를 충족시키는 작업 (WORKPLAN 항목)


@dataclass
class Report:
    rows: list[Outcome] = field(default_factory=list)

    def add(self, *a: Any, **k: Any) -> None:
        self.rows.append(Outcome(*a, **k))


def _short(v: Any, n: int = 90) -> str:
    s = v if isinstance(v, str) else json.dumps(v, default=str, ensure_ascii=False)
    return s if len(s) <= n else s[: n - 1] + "…"


def child_output(parent_kind: str, state: dict) -> Any:
    """부모가 자식(첫 호출)에게서 받은 값."""
    if parent_kind == "react":
        pad = state.get("scratchpad") or []
        return pad[0].get("observation") if pad else None
    return (state.get("results") or {}).get(1)


# --------------------------------------------------------------------------
# PAIR 시나리오 (부모 ⊃ 자식)
# --------------------------------------------------------------------------


def _child_registry(log: list) -> ToolRegistry:
    reg = ToolRegistry()
    reg.register("up", lambda text: str(text).upper(), description="uppercase")

    def boom(**_: Any) -> Any:
        raise RuntimeError("tool exploded")

    reg.register("boom", boom, description="always fails")
    reg.register("danger", lambda: log.append("danger") or "done",
                 description="needs approval", requires_approval=True)

    def ask(question: str) -> Any:
        answer = ask_human(question)
        log.append(("answer", answer))
        return answer

    reg.register("ask", ask, description="ask the human")
    return reg


def _make_child(kind: str, steps, final: str, log: list, runs: int = 1, saver=None,
                skip: int = 0) -> Any:
    p = PATTERNS[kind]
    script = p.script(steps, final) * runs
    return p.build(fake(script[skip:]), _child_registry(log), saver)


def _make_parent(kind: str, child_graphs: dict[str, Any], steps, saver=None, skip: int = 0):
    p = PATTERNS[kind]
    reg = ToolRegistry()
    for name, g in child_graphs.items():
        reg.add(AgentTool(name, g, description=f"sub-agent {name}"))
    script = p.script(steps, "parent-done")
    return p.build(fake(script[skip:]), reg, saver)


async def pair_normal(rep: Report, pk: str, ck: str) -> None:
    log: list = []
    child = _make_child(ck, [[("up", {"text": "abc"})]], "CHILD:ABC", log)
    parent = _make_parent(pk, {"kid": child}, [[("kid", {"query": "go"})]])
    st = await start(parent)
    got = child_output(pk, st)
    rep.add("normal", f"{pk}⊃{ck}", "자식 결과가 부모에 전달", got == "CHILD:ABC", _short(got))


async def pair_child_tool_error(rep: Report, pk: str, ck: str) -> None:
    log: list = []
    child = _make_child(ck, [[("boom", {})]], "CHILD:HANDLED", log)
    parent = _make_parent(pk, {"kid": child}, [[("kid", {"query": "go"})]])
    try:
        st = await start(parent)
        ok = st.get("final") == "parent-done"
        obs = f"parent final={st.get('final')!r}, child={_short(child_output(pk, st), 50)}"
    except Exception as e:  # noqa: BLE001
        ok, obs = False, f"CRASH {type(e).__name__}: {e}"
    rep.add("child_tool_error", f"{pk}⊃{ck}", "자식 도구 에러에도 부모 정상 종료", ok, _short(obs))


async def pair_child_garbage(rep: Report, pk: str, ck: str) -> None:
    """자식 LLM 이 형식이 깨진 출력을 낸다 → 부모는 죽지 않고, 자식 실패를 구분할 수 있어야 한다."""
    child = PATTERNS[ck].build(fake(["not json", "still not json", "nope", "nope"]),
                               _child_registry([]))
    parent = _make_parent(pk, {"kid": child}, [[("kid", {"query": "go"})]])
    try:
        st = await start(parent)
        got = child_output(pk, st)
        ok = (st.get("final") == "parent-done" and is_failure(got)
              and "status=error" in str(got.get("error")))
        obs = f"parent final={st.get('final')!r}, child={_short(got, 70)}"
    except Exception as e:  # noqa: BLE001
        ok, obs = False, f"CRASH {type(e).__name__}: {e}"
    rep.add("child_llm_garbage", f"{pk}⊃{ck}",
            "자식 출력이 깨져도 부모 정상 종료 + 부모가 자식 실패(status=error)를 받음",
            ok, _short(obs), fix="1.4/3.1")


async def pair_child_capped(rep: Report, pk: str, ck: str) -> None:
    """자식이 상한(max_steps / max_iterations)에 걸려 끝내지 못함 → 부모가 그걸 구분해야 한다."""
    p = PATTERNS[ck]
    if ck == "react":
        script = [{"action": "tool", "tool": "up", "args": {"text": "a"}}]  # 1스텝 상한
    else:
        script = [{"tasks": [{"id": 1, "tool": "up", "args": {"text": "a"}}]},
                  {"decision": "replan", "reason": "more"}]  # 재계획 상한 1
    child = p.build(fake(script), _child_registry([]), **p.cap_config)
    parent = _make_parent(pk, {"kid": child}, [[("kid", {"query": "go"})]])
    st = await start(parent)
    got = child_output(pk, st)
    ok = (st.get("final") == "parent-done" and is_failure(got)
          and "status=capped" in str(got.get("error")))
    rep.add("child_capped", f"{pk}⊃{ck}", "자식이 상한에 걸림 → 부모가 status=capped 실패를 받음",
            ok, _short(got, 80), fix="3.1")


async def pair_context_to_child(rep: Report, pk: str, ck: str) -> None:
    """부모가 AgentTool 호출에 context 를 넘기면 자식 모델 프롬프트에 보여야 한다."""
    child_llm = fake(PATTERNS[ck].script([[("up", {"text": "a"})]], "CHILD:OK"))
    child = PATTERNS[ck].build(child_llm, _child_registry([]))
    parent = _make_parent(pk, {"kid": child},
                          [[("kid", {"query": "go", "context": "PARENT-NOTE-42"})]])
    await start(parent)
    first_prompt = child_llm.calls[0][1] if child_llm.calls else ""
    ok = "PARENT-NOTE-42" in first_prompt
    rep.add("context_to_child", f"{pk}⊃{ck}", "AgentTool 의 context 가 자식 첫 프롬프트에 포함", ok,
            "포함" if ok else _short(first_prompt, 80), fix="3.2")


async def _approval_flow(pk: str, ck: str, verdict: Any, *, restart: bool = False):
    log: list = []
    saver = InMemorySaver()
    steps_c = [[("danger", {})]]
    steps_p = [[("kid", {"query": "go"})]]
    child = _make_child(ck, steps_c, "CHILD:OK", log)
    parent = _make_parent(pk, {"kid": child}, steps_p, saver)
    st = await start(parent, thread="t")
    intrs = pending(st)
    paused = len(intrs) == 1 and intrs[0].value.get("type") == "tool_approval"
    if not paused:
        return False, f"멈추지 않음 (danger 실행 {log.count('danger')}회)", log
    if restart:  # 프로세스 재시작: 같은 saver 로 그래프를 새로 만든 뒤 재개
        child = _make_child(ck, steps_c, "CHILD:OK", log, skip=1)
        parent = _make_parent(pk, {"kid": child}, steps_p, saver, skip=1)
    st = await resume(parent, st, lambda _: verdict, "t")
    return st, None, log


async def pair_approval(rep: Report, pk: str, ck: str) -> None:
    for label, verdict, want in (("approve", True, 1), ("reject", {"approved": False, "feedback": "no"}, 0)):
        try:
            st, err, log = await _approval_flow(pk, ck, verdict)
            if err:
                rep.add(f"child_approval_{label}", f"{pk}⊃{ck}",
                        "자식의 승인 필요 도구에서 부모가 멈춤", False, err, fix="1.1/1.2")
                continue
            n = log.count("danger")
            ok = n == want and pending(st) == [] and st.get("final") == "parent-done"
            rep.add(f"child_approval_{label}", f"{pk}⊃{ck}",
                    f"멈춤 → {label} → 도구 {want}회 실행 → 부모 완료", ok,
                    f"도구 {n}회, final={st.get('final')!r}", fix="1.1/1.2")
        except Exception as e:  # noqa: BLE001
            rep.add(f"child_approval_{label}", f"{pk}⊃{ck}", "멈춤 → 재개", False,
                    _short(f"CRASH {type(e).__name__}: {e}"), fix="1.1/1.2")


async def pair_restart(rep: Report, pk: str, ck: str) -> None:
    try:
        st, err, log = await _approval_flow(pk, ck, True, restart=True)
        if err:
            rep.add("child_approval_restart", f"{pk}⊃{ck}", "재시작 후 재개", False, err, fix="1.1/1.2")
            return
        n = log.count("danger")
        ok = n == 1 and st.get("final") == "parent-done"
        rep.add("child_approval_restart", f"{pk}⊃{ck}",
                "멈춤 → 그래프 재생성(같은 saver) → 승인 → 도구 1회", ok,
                f"도구 {n}회, final={st.get('final')!r}", fix="1.1/1.2")
    except Exception as e:  # noqa: BLE001
        rep.add("child_approval_restart", f"{pk}⊃{ck}", "재시작 후 재개", False,
                _short(f"CRASH {type(e).__name__}: {e}"), fix="1.1/1.2")


async def pair_ask_human(rep: Report, pk: str, ck: str) -> None:
    log: list = []
    saver = InMemorySaver()
    child = _make_child(ck, [[("ask", {"question": "color?"})]], "CHILD:ASKED", log)
    parent = _make_parent(pk, {"kid": child}, [[("kid", {"query": "go"})]], saver)
    try:
        st = await start(parent, thread="t")
        intrs = pending(st)
        if not (len(intrs) == 1 and intrs[0].value.get("type") == "question"):
            rep.add("child_ask_human", f"{pk}⊃{ck}", "자식의 ask_human 에서 부모가 멈춤", False,
                    _short(f"멈추지 않음, child={child_output(pk, st)}"), fix="1.1/1.3")
            return
        st = await resume(parent, st, lambda _: "blue", "t")
        ok = ("answer", "blue") in log and st.get("final") == "parent-done"
        rep.add("child_ask_human", f"{pk}⊃{ck}", "멈춤 → 답변이 자식 도구에 전달 → 부모 완료",
                ok, f"log={log}, final={st.get('final')!r}", fix="1.1/1.3")
    except Exception as e:  # noqa: BLE001
        rep.add("child_ask_human", f"{pk}⊃{ck}", "멈춤 → 재개", False,
                _short(f"CRASH {type(e).__name__}: {e}"), fix="1.1/1.3")


async def pair_same_child_twice(rep: Report, pk: str, ck: str) -> None:
    """같은 자식을 한 실행에서 두 번 호출하고, 매번 승인 게이트를 거친다."""
    log: list = []
    saver = InMemorySaver()
    child = _make_child(ck, [[("danger", {})]], "CHILD:OK", log, runs=2)
    parent = _make_parent(pk, {"kid": child}, [[("kid", {"query": "a"})], [("kid", {"query": "b"})]],
                          saver)
    try:
        st = await start(parent, thread="t")
        pauses = 0
        while pending(st) and pauses < 5:
            pauses += 1
            st = await resume(parent, st, lambda _: True, "t")
        n = log.count("danger")
        ok = pauses == 2 and n == 2 and st.get("final") == "parent-done"
        rep.add("same_child_twice", f"{pk}⊃{ck}", "멈춤 2회 → 도구 2회 실행 → 부모 완료", ok,
                f"멈춤 {pauses}회, 도구 {n}회, final={st.get('final')!r}", fix="1.1/1.2")
    except Exception as e:  # noqa: BLE001
        rep.add("same_child_twice", f"{pk}⊃{ck}", "2회 호출", False,
                _short(f"CRASH {type(e).__name__}: {e}"), fix="1.1/1.2")


async def pair_parallel_interrupts(rep: Report, pk: str, ck: str) -> None:
    """부모의 같은 레벨에서 두 자식이 동시에 승인을 요청한다 (병렬 레벨이 있는 부모만)."""
    if pk != "plan":
        return
    log: list = []
    saver = InMemorySaver()
    kids = {n: _make_child(ck, [[("danger", {})]], f"CHILD:{n}", log) for n in ("kid1", "kid2")}
    parent = _make_parent(pk, kids, [[("kid1", {"query": "a"}), ("kid2", {"query": "b"})]], saver)
    try:
        st = await start(parent, thread="t")
        n_int = len(pending(st))
        if n_int != 2:
            rep.add("parallel_interrupts", f"{pk}⊃{ck}", "병렬 자식 2개가 동시에 멈춤", False,
                    f"interrupt {n_int}개, 도구 {log.count('danger')}회", fix="1.3")
            return
        st = await resume(parent, st, lambda _: True, "t")
        n = log.count("danger")
        ok = n == 2 and st.get("final") == "parent-done"
        rep.add("parallel_interrupts", f"{pk}⊃{ck}", "interrupt 2개 → id별 승인 → 도구 2회", ok,
                f"도구 {n}회, final={st.get('final')!r}", fix="1.3")
    except Exception as e:  # noqa: BLE001
        rep.add("parallel_interrupts", f"{pk}⊃{ck}", "병렬 interrupt", False,
                _short(f"CRASH {type(e).__name__}: {e}"), fix="1.3")


PAIR_SCENARIOS = [pair_normal, pair_child_tool_error, pair_child_garbage, pair_child_capped,
                  pair_context_to_child, pair_approval, pair_restart, pair_ask_human,
                  pair_same_child_twice, pair_parallel_interrupts]

# --------------------------------------------------------------------------
# TOP 시나리오 (단일 패턴)
# --------------------------------------------------------------------------


async def top_approval(rep: Report, k: str) -> None:
    log: list = []
    g = PATTERNS[k].build(fake(PATTERNS[k].script([[("danger", {})]], "done")),
                          _child_registry(log), InMemorySaver())
    st = await start(g, thread="t")
    paused = len(pending(st)) == 1
    ran_before = log.count("danger")
    if paused:
        st = await resume(g, st, lambda _: True, "t")
    ok = paused and ran_before == 0 and log.count("danger") == 1
    rep.add("approval_gate", k, "승인 필요 도구 앞에서 멈춤 → 승인 후 1회 실행", ok,
            f"멈춤={paused}, 승인 전 실행 {ran_before}회, 최종 {log.count('danger')}회", fix="1.2")


async def top_ask_human(rep: Report, k: str) -> None:
    log: list = []
    g = PATTERNS[k].build(fake(PATTERNS[k].script([[("ask", {"question": "color?"})]], "done")),
                          _child_registry(log), InMemorySaver())
    st = await start(g, thread="t")
    intrs = pending(st)
    paused = len(intrs) == 1 and intrs[0].value.get("type") == "question"
    if paused:
        st = await resume(g, st, lambda _: "blue", "t")
    ok = paused and ("answer", "blue") in log
    obs = f"멈춤={paused}" + ("" if paused else f", results={_short(st.get('results'), 70)}")
    rep.add("ask_human", k, "ask_human 에서 멈춤 → 답변 전달", ok, obs, fix="1.1/1.3")


async def top_llm_garbage(rep: Report, k: str) -> None:
    """출력이 계속 깨진다. 평문 / 중괄호는 있지만 JSON 이 아닌 출력 두 가지."""
    for label, junk in (("llm_garbage", "not json"), ("llm_brace_garbage", "{not: json,,}")):
        g = PATTERNS[k].build(fake([junk] * 6), _child_registry([]))
        try:
            env = envelope(await start(g))
            ok = env["status"] == "error" and bool(env["error"])
            rep.add(label, k, "출력이 계속 깨져도 죽지 않고 status=error 로 종료", ok,
                    f"status={env['status']}, error={_short(env['error'], 60)}", fix="1.4/3.1")
        except Exception as e:  # noqa: BLE001
            rep.add(label, k, "출력이 계속 깨져도 죽지 않고 status=error 로 종료", False,
                    _short(f"CRASH {type(e).__name__}: {e}"), fix="1.4")


async def top_status(rep: Report, k: str) -> None:
    """정상 종료 → status=ok, 상한 도달 → status=capped (계약)."""
    p = PATTERNS[k]
    env_ok = envelope(await start(p.build(fake(p.script([[("up", {"text": "a"})]], "done")),
                                          _child_registry([]))))
    if k == "react":
        capped_script = [{"action": "tool", "tool": "up", "args": {"text": "a"}}]
    else:
        capped_script = [{"tasks": [{"id": 1, "tool": "up", "args": {"text": "a"}}]},
                         {"decision": "replan", "reason": "more"}]
    env_cap = envelope(await start(p.build(fake(capped_script), _child_registry([]), **p.cap_config)))
    ok = env_ok["status"] == "ok" and env_cap["status"] == "capped"
    rep.add("status_contract", k, "정상 종료 status=ok, 상한 도달 status=capped", ok,
            f"정상={env_ok['status']}, 상한={env_cap['status']}", fix="3.1")


async def top_status_not_leaked(rep: Report, k: str) -> None:
    """같은 thread 로 다시 실행할 때 이전 실행의 status=ok 가 남아 상한 도달을 가리면 안 된다."""
    p = PATTERNS[k]
    ok_script = p.script([[("up", {"text": "a"})]], "done")
    if k == "react":  # 상한 2스텝: 1회차(도구→종료)는 통과, 2회차(도구→도구)는 상한
        cap_script = [{"action": "tool", "tool": "up", "args": {"text": "a"}}] * 2
        cfg = {"max_steps": 2}
    else:  # 재계획 상한 1: 1회차(계획→종료)는 통과, 2회차(계획→재계획)는 상한
        cap_script = [{"tasks": [{"id": 1, "tool": "up", "args": {"text": "a"}}]},
                      {"decision": "replan", "reason": "more"}]
        cfg = {"max_iterations": 1}
    # start() 는 계약의 initial_state 를 쓰지 않는다 → 호출자가 status 를 초기화하지 않아도 되는지 확인
    g = p.build(fake(ok_script + cap_script), _child_registry([]), InMemorySaver(), **cfg)
    first = envelope(await start(g, thread="same"))
    second = envelope(await start(g, thread="same"))
    ok = first["status"] == "ok" and second["status"] == "capped"
    rep.add("status_not_leaked", k, "같은 thread 2회차가 상한에 걸리면 status=capped (1회차 ok 가 안 남음)",
            ok, f"1회차={first['status']}, 2회차={second['status']}", fix="3.1")


async def top_unknown_tool(rep: Report, k: str) -> None:
    """모델이 없는 도구를 고른다. 실행 전에 걸러서 모델에게 다시 고르게 해야 한다."""
    if k != "plan":
        return
    llm = fake([{"tasks": [{"id": 1, "tool": "nope", "args": {}}]},
                {"tasks": [{"id": 1, "tool": "up", "args": {"text": "x"}}]},
                {"decision": "finish", "final": "done"}])
    g = patterns.get("plan_execute").build(llm=llm, registry=_child_registry([]))
    try:
        st = await start(g)
        ok = (st.get("results") or {}).get(1) == "X"
        rep.add("unknown_tool", k, "없는 도구는 계획 단계에서 걸러 재계획", ok,
                f"results={_short(st.get('results'), 70)}", fix="1.5")
    except Exception as e:  # noqa: BLE001
        rep.add("unknown_tool", k, "없는 도구는 계획 단계에서 걸러 재계획", False,
                _short(f"CRASH {type(e).__name__}: {e}"), fix="1.5")


async def top_error_field_data(rep: Report, k: str) -> None:
    """정상 데이터에 'error' 필드가 있어도 실패로 취급하면 안 된다."""
    if k != "plan":
        return
    reg = ToolRegistry()
    reg.register("fetch", lambda: {"error": None, "value": 1})
    reg.register("use", lambda x: f"used {x}")
    llm = fake([{"tasks": [{"id": 1, "tool": "fetch", "args": {}},
                           {"id": 2, "tool": "use", "args": {"x": "$1"}, "deps": [1]}]},
                {"decision": "finish", "final": "done"}])
    st = await start(patterns.get("plan_execute").build(llm=llm, registry=reg))
    ok = 2 in (st.get("results") or {})
    rep.add("error_field_in_data", k, "'error' 필드가 있는 정상 데이터에서 다음 레벨 실행", ok,
            f"results={_short(st.get('results'), 70)}", fix="1.6")


async def top_garbage_then_valid(rep: Report, k: str) -> None:
    """첫 출력이 깨지고 두 번째가 정상이면, 재시도로 정상 실행되어야 한다."""
    log: list = []
    script = PATTERNS[k].script([[("up", {"text": "abc"})]], "done")
    g = PATTERNS[k].build(fake(["not json"] + script), _child_registry(log))
    try:
        st = await start(g)
        ran = (st.get("results") or {}).get(1) == "ABC" if k == "plan" else \
            any(s.get("observation") == "ABC" for s in st.get("scratchpad") or [])
        ok = ran and st.get("final") == "done"
        rep.add("garbage_then_valid", k, "깨진 출력 1회 → 재시도로 정상 실행", ok,
                f"도구 실행={ran}, final={st.get('final')!r}", fix="1.4")
    except Exception as e:  # noqa: BLE001
        rep.add("garbage_then_valid", k, "깨진 출력 1회 → 재시도", False,
                _short(f"CRASH {type(e).__name__}: {e}"), fix="1.4")


def _two_danger_registry(log: list) -> ToolRegistry:
    reg = ToolRegistry()
    reg.register("up", lambda text: str(text).upper())
    for n in ("d1", "d2"):
        reg.register(n, (lambda name: (lambda **a: log.append((name, a)) or f"{name} done"))(n),
                     requires_approval=True)
    return reg


async def top_batch_mixed_verdict(rep: Report, k: str) -> None:
    """같은 레벨의 승인 필요 task 2개: 하나 승인, 하나 거절 → 한 번만 멈추고 승인된 것만 실행."""
    if k != "plan":
        return
    log: list = []
    llm = fake([{"tasks": [{"id": 1, "tool": "d1", "args": {}}, {"id": 2, "tool": "d2", "args": {}},
                           {"id": 3, "tool": "up", "args": {"text": "next"}, "deps": [1, 2]}]},
                {"decision": "finish", "final": "done"}])
    g = patterns.get("plan_execute").build(llm=llm, registry=_two_danger_registry(log), checkpointer=InMemorySaver())
    st = await start(g, thread="t")
    intrs = pending(st)
    payload = intrs[0].value if len(intrs) == 1 else None
    if not (payload and payload.get("type") == "tool_approvals" and len(payload["items"]) == 2):
        rep.add("batch_mixed_verdict", k, "레벨 승인 1회 묶음 요청", False,
                f"interrupt {len(intrs)}개, payload={_short(payload, 60)}", fix="1.2")
        return
    st = await resume(g, st, lambda _: {"decisions": {1: True, 2: {"approved": False,
                                                                    "feedback": "no"}}}, "t")
    res = st.get("results") or {}
    ok = ([n for n, _ in log] == ["d1"] and "denied" in (res.get(2) or {}) and 3 not in res
          and not pending(st))
    rep.add("batch_mixed_verdict", k,
            "1회 멈춤 → d1 승인·d2 거절 → d1만 실행, d2 거절 기록, 다음 레벨(3) 미실행", ok,
            f"실행={[n for n, _ in log]}, results={_short(res, 60)}", fix="1.2")


async def top_approval_with_ref(rep: Report, k: str) -> None:
    """2레벨 task 가 1레벨 결과($1)를 인자로 받는다 → 승인 요청에 치환된 값이 보여야 한다."""
    if k != "plan":
        return
    log: list = []
    llm = fake([{"tasks": [{"id": 1, "tool": "up", "args": {"text": "abc"}},
                           {"id": 2, "tool": "d1", "args": {"v": "$1"}, "deps": [1]}]},
                {"decision": "finish", "final": "done"}])
    g = patterns.get("plan_execute").build(llm=llm, registry=_two_danger_registry(log), checkpointer=InMemorySaver())
    st = await start(g, thread="t")
    intrs = pending(st)
    shown = intrs[0].value.get("args") if len(intrs) == 1 else None
    if intrs:
        st = await resume(g, st, lambda _: True, "t")
    ok = shown == {"v": "ABC"} and log == [("d1", {"v": "ABC"})]
    rep.add("approval_shows_resolved_args", k, "승인 요청에 $1 이 치환된 인자 표시 → 그 인자로 실행", ok,
            f"표시={shown}, 실행={log}", fix="1.2")


async def top_replan_resets_approval(rep: Report, k: str) -> None:
    """승인 후 재계획되면, 새 계획의 같은 id task 는 다시 승인받아야 한다."""
    if k != "plan":
        return
    log: list = []
    llm = fake([{"tasks": [{"id": 1, "tool": "d1", "args": {"n": 1}}]},
                {"decision": "replan", "reason": "again"},
                {"tasks": [{"id": 1, "tool": "d1", "args": {"n": 2}}]},
                {"decision": "finish", "final": "done"}])
    g = patterns.get("plan_execute").build(llm=llm, registry=_two_danger_registry(log), checkpointer=InMemorySaver())
    st = await start(g, thread="t")
    pauses = 0
    while pending(st) and pauses < 5:
        pauses += 1
        st = await resume(g, st, lambda _: True, "t")
    ok = pauses == 2 and [a for _, a in log] == [{"n": 1}, {"n": 2}]
    rep.add("replan_resets_approval", k, "재계획 후 같은 id task 도 다시 승인 요청 (2회 멈춤)", ok,
            f"멈춤 {pauses}회, 실행={log}", fix="1.2")


async def top_final_in_result(rep: Report, k: str) -> None:
    """실모델(27B)이 답을 final 이 아니라 result 에 넣은 출력을 재현 → 답이 버려지면 안 된다."""
    if k != "react":
        return
    llm = fake([{"action": "finish", "result": {"answer": "HELLO AGENT!"}}])
    env = envelope(await start(PATTERNS[k].build(llm, _child_registry([]))))
    ok = env["final"] == "HELLO AGENT!" and env["status"] == "ok"
    rep.add("final_in_result", k, "final 없이 result 에만 답 → final 로 복구", ok,
            f"final={env['final']!r}", fix="D7")


TOP_SCENARIOS = [top_approval, top_ask_human, top_llm_garbage, top_garbage_then_valid,
                 top_status, top_status_not_leaked, top_final_in_result,
                 top_unknown_tool, top_error_field_data, top_batch_mixed_verdict,
                 top_approval_with_ref, top_replan_resets_approval]

# --------------------------------------------------------------------------


# --------------------------------------------------------------------------
# CONTEXT 시나리오 (ContextPolicy, opt-in 컨텍스트 관리)
# --------------------------------------------------------------------------


class RoutingFake(FakeLLM):
    """compaction 요약 요청에는 고정 요약을, 그 외에는 큐의 다음 응답을 돌려준다.

    요약이 언제 일어나는지에 테스트가 묶이지 않도록 하기 위함."""

    def __init__(self, responses: list[str], summary: str = "SUMMARY-XYZ") -> None:
        super().__init__(responses)
        self.summary = summary
        self.compactions = 0

    def complete(self, system: str, user: str, *, schema=None) -> str:
        if system == COMPACT_SYSTEM:
            self.compactions += 1
            self.calls.append((system, user, schema))
            return self.summary
        return super().complete(system, user, schema=schema)


def _acts(*actions: tuple[str, dict], final: str = "done") -> list[str]:
    out = [json.dumps({"action": "tool", "tool": t, "args": a}) for t, a in actions]
    return out + [json.dumps({"action": "finish", "final": final})]


def _prompts(llm: FakeLLM) -> list[str]:
    return [u for s, u, _ in llm.calls if s != COMPACT_SYSTEM]


def _ctx_registry(log: list) -> ToolRegistry:
    reg = ToolRegistry()
    reg.register("big", lambda: "a" * 3000 + "NEEDLE" + "b" * 500, description="large output")
    reg.register("small", lambda n: f"OBS-{n}-" + "x" * 140, description="medium output")
    reg.register("danger", lambda: log.append("danger") or "done", requires_approval=True)
    return reg


async def ctx_offload(rep: Report) -> None:
    pol = ContextPolicy(offload=True)
    llm = fake(_acts(("big", {}), ("read_observation", {"step": 1, "offset": 2990})))
    st = await start(patterns.get("react").build(llm=llm, registry=_ctx_registry([]),
                                                 context_policy=pol))
    pr = _prompts(llm)
    ok = ("NEEDLE" not in pr[1] and "read_observation(step=1" in pr[1] and "NEEDLE" in pr[2]
          and envelope(st)["status"] == "ok")
    # 대조: 정책 없이 같은 흐름이면 NEEDLE 을 볼 방법이 없다 (read_observation 은 없는 도구)
    llm0 = fake(_acts(("big", {}), ("read_observation", {"step": 1, "offset": 2990})))
    await start(patterns.get("react").build(llm=llm0, registry=_ctx_registry([]),
                                            context_policy=None))  # 원래 프롬프트
    never = all("NEEDLE" not in u for u in _prompts(llm0))
    rep.add("ctx_offload", "react+policy",
            "긴 관찰은 미리보기+포인터 → read_observation 으로 뒷부분(NEEDLE) 읽음 / 정책 없으면 못 봄",
            ok and never, f"2번째 프롬프트 NEEDLE={'NEEDLE' in pr[1]}, 3번째={'NEEDLE' in pr[2]}, "
                          f"정책없음 전체={'못 봄' if never else '봄'}", fix="5.1")


async def ctx_clearing(rep: Report) -> None:
    pol = ContextPolicy(keep_recent=1)
    llm = fake(_acts(("small", {"n": 1}), ("small", {"n": 2}), ("small", {"n": 3})))
    await start(patterns.get("react").build(llm=llm, registry=_ctx_registry([]),
                                            context_policy=pol))
    last = _prompts(llm)[3]
    ok = ("OBS-1-" not in last and "OBS-2-" not in last and "OBS-3-" in last
          and last.count("[cleared:") == 2)
    rep.add("ctx_clearing", "react+policy", "최근 1개만 전문, 이전 2개는 cleared 표시", ok,
            f"cleared={last.count('[cleared:')}, OBS-3 표시={'OBS-3-' in last}", fix="5.2")


async def ctx_compaction(rep: Report) -> None:
    pol = ContextPolicy(compact_at_chars=400, compact_keep_recent=1)
    llm = RoutingFake(_acts(("small", {"n": 1}), ("small", {"n": 2}), ("small", {"n": 3}),
                            ("small", {"n": 4})))
    st = await start(patterns.get("react").build(llm=llm, registry=_ctx_registry([]),
                                                 context_policy=pol))
    pr = _prompts(llm)
    last = pr[-1]
    ok = (llm.compactions >= 1 and "SUMMARY-XYZ" in last and "OBS-1-" not in last
          and "OBS-4-" in last and st.get("summarized_upto", 0) >= 1
          and envelope(st)["status"] == "ok")
    rep.add("ctx_compaction", "react+policy",
            "누적이 임계 초과 → 요약 생성, 마지막 프롬프트에 요약+최근 스텝, 오래된 원문 없음", ok,
            f"요약 {llm.compactions}회, summarized_upto={st.get('summarized_upto')}, "
            f"OBS-1 남음={'OBS-1-' in last}", fix="5.3")


async def ctx_notes_survive_pause(rep: Report) -> None:
    """메모 → 승인 대기로 멈춤 → 그래프 재생성 후 재개 → 메모가 프롬프트에 남아 있음."""
    log: list = []
    pol = ContextPolicy(notes=True)
    saver = InMemorySaver()
    script = _acts(("notes_write", {"text": "remember 42"}), ("danger", {}))
    llm = fake(script)
    g = patterns.get("react").build(llm=llm, registry=_ctx_registry(log), checkpointer=saver,
                                    context_policy=pol)
    st = await start(g, thread="t")
    paused = bool(pending(st))
    llm2 = fake(script[2:])  # 재시작: 남은 응답만 가진 새 LLM·새 그래프
    g2 = patterns.get("react").build(llm=llm2, registry=_ctx_registry(log), checkpointer=saver,
                                     context_policy=pol)
    st = await resume(g2, st, lambda _: True, "t")
    last = _prompts(llm2)[-1] if llm2.calls else ""
    ok = paused and log == ["danger"] and "- remember 42" in last
    rep.add("ctx_notes_survive_pause", "react+policy", "메모가 멈춤·재시작 후에도 프롬프트에 남음", ok,
            f"멈춤={paused}, 도구={log}, 메모 표시={'remember 42' in last}", fix="5.4")


async def ctx_skills(rep: Report) -> None:
    pol = ContextPolicy(skills=(Skill("refund", "refund policy steps", "BODY-REFUND-77"),))
    llm = fake(_acts(("load_skill", {"name": "refund"})))
    await start(patterns.get("react").build(llm=llm, registry=_ctx_registry([]),
                                            context_policy=pol))
    pr = _prompts(llm)
    ok = ("refund: refund policy steps" in pr[0] and "BODY-REFUND-77" not in pr[0]
          and "BODY-REFUND-77" in pr[1])
    rep.add("ctx_skills", "react+policy", "처음엔 스킬 이름·설명만, load_skill 후 본문", ok,
            f"처음 본문={'BODY-REFUND-77' in pr[0]}, 로드 후={'BODY-REFUND-77' in pr[1]}", fix="5.5")


async def ctx_tool_search(rep: Report) -> None:
    log: list = []
    reg = ToolRegistry()
    for i in range(25):
        reg.register(f"t{i:02d}", lambda: "x", description=f"generic tool number {i}")
    reg.register("weather_lookup", lambda city: log.append(city) or "sunny",
                 description="weather forecast for a city")
    pol = ContextPolicy(tool_search_threshold=10, core_tools=("t00",))
    llm = fake(_acts(("tool_search", {"query": "weather"}), ("weather_lookup", {"city": "Seoul"})))
    await start(patterns.get("react").build(llm=llm, registry=reg, context_policy=pol))
    pr = _prompts(llm)
    ok = ("weather_lookup" not in pr[0] and "- t00" in pr[0] and "- t05" not in pr[0]
          and "- weather_lookup" in pr[1] and log == ["Seoul"])
    rep.add("ctx_tool_search", "react+policy",
            "도구 26개 중 핵심만 노출 → tool_search 로 찾은 도구가 다음 프롬프트에 노출·실행", ok,
            f"처음 노출={'weather_lookup' in pr[0]}, 검색 후={'- weather_lookup' in pr[1]}, 실행={log}",
            fix="5.6")


async def ctx_name_clash(rep: Report) -> None:
    reg = ToolRegistry()
    reg.register("read_observation", lambda: "mine")
    try:
        patterns.get("react").build(llm=fake([]), registry=reg,
                                    context_policy=ContextPolicy(offload=True))
        ok, obs = False, "빌드 통과 (충돌 미검출)"
    except ValueError as e:
        ok, obs = True, _short(str(e), 70)
    rep.add("ctx_name_clash", "react+policy", "하네스 도구와 이름이 겹치는 등록 도구 → 빌드 시 거부", ok,
            obs, fix="5.7")


async def ctx_nested_child_policy(rep: Report) -> None:
    """정책을 켠 react 자식이 plan 부모 안에서 동작 (자식이 NEEDLE 을 읽음)."""
    child_llm = fake(_acts(("big", {}), ("read_observation", {"step": 1, "offset": 2990}),
                           final="CHILD:FOUND"))
    child = patterns.get("react").build(llm=child_llm, registry=_ctx_registry([]),
                                        context_policy=ContextPolicy(offload=True))
    parent = _make_parent("plan", {"kid": child}, [[("kid", {"query": "find"})]])
    st = await start(parent)
    ok = (st.get("results") or {}).get(1) == "CHILD:FOUND" and "NEEDLE" in _prompts(child_llm)[2]
    rep.add("ctx_nested_child", "plan⊃react+policy", "정책 켠 자식이 부모 안에서 뒷부분을 읽고 결과 반환",
            ok, f"results={_short(st.get('results'), 50)}", fix="5.7")


async def ctx_plan_results_cap(rep: Report) -> None:
    reg = ToolRegistry()
    reg.register("huge", lambda: "z" * 5000)
    script = [{"tasks": [{"id": 1, "tool": "huge", "args": {}}]}, {"decision": "finish", "final": "ok"}]
    llm = fake(script)
    await start(patterns.get("plan_execute").build(llm=llm, registry=reg,
                                                   context_policy=ContextPolicy(inline_chars=200)))
    capped = llm.calls[1][1]
    llm0 = fake(script)
    await start(patterns.get("plan_execute").build(llm=llm0, registry=reg))
    uncapped = llm0.calls[1][1]
    ok = len(capped) < 600 and "chars total, truncated" in capped and len(uncapped) > 5000
    rep.add("ctx_plan_results_cap", "plan+policy", "joiner 프롬프트의 task 결과를 inline_chars 로 제한",
            ok, f"정책 {len(capped)}자 / 정책없음 {len(uncapped)}자", fix="5.7")


async def ctx_compaction_chain(rep: Report) -> None:
    """요약이 여러 번 일어나면 이전 요약을 입력으로 이어받아야 한다."""
    pol = ContextPolicy(compact_at_chars=400, compact_keep_recent=1)
    llm = RoutingFake(_acts(*[("small", {"n": i}) for i in range(1, 9)]))
    st = await start(patterns.get("react").build(llm=llm, registry=_ctx_registry([]),
                                                 context_policy=pol, max_steps=12))
    comp_inputs = [u for sys_, u, _ in llm.calls if sys_ == COMPACT_SYSTEM]
    chained = any("Summary so far" in u for u in comp_inputs[1:])
    ok = len(comp_inputs) >= 2 and chained and envelope(st)["status"] == "ok"
    rep.add("ctx_compaction_chain", "react+policy", "요약 2회 이상, 2회차부터 이전 요약을 이어받음", ok,
            f"요약 {len(comp_inputs)}회, 이어받음={chained}, upto={st.get('summarized_upto')}", fix="5.3")


async def ctx_read_bad_args(rep: Report) -> None:
    pol = ContextPolicy(offload=True)
    llm = fake(_acts(("read_observation", {"step": 9}), ("read_observation", {"step": "x"})))
    st = await start(patterns.get("react").build(llm=llm, registry=_ctx_registry([]),
                                                 context_policy=pol))
    obs = [s_.get("observation") for s_ in st.get("scratchpad") or []]
    ok = (len(obs) == 2 and all(isinstance(o, dict) and "error" in o for o in obs)
          and envelope(st)["status"] == "ok")
    rep.add("ctx_read_bad_args", "react+policy", "잘못된 step 인자 → 에러 관찰로 돌려주고 계속 진행", ok,
            _short(obs, 80), fix="5.1")



class OverflowFake(FakeLLM):
    """프롬프트가 limit 자를 넘으면 실서버처럼 ContextOverflowError 를 낸다."""

    def __init__(self, responses: list[str], limit: int) -> None:
        super().__init__(responses)
        self.limit = limit
        self.overflows = 0

    def complete(self, system: str, user: str, *, schema=None) -> str:
        if len(system) + len(user) > self.limit:
            self.overflows += 1
            self.calls.append((system, user, schema))
            raise ContextOverflowError("maximum context length exceeded (fake)")
        return super().complete(system, user, schema=schema)


async def ctx_overflow_plan(rep: Report) -> None:
    """60,000자 결과 → joiner 프롬프트 초과 → 결과를 잘라 재시도 → 정상 종료."""
    reg = ToolRegistry()
    reg.register("dump", lambda: "Dividend 0.31. " + "z" * 60000)
    llm = OverflowFake([json.dumps({"tasks": [{"id": 1, "tool": "dump", "args": {}}]}),
                        json.dumps({"decision": "finish", "final": "0.31"})], limit=20000)
    try:
        env = envelope(await start(patterns.get("plan_execute").build(llm=llm, registry=reg)))
        ok = llm.overflows == 1 and env["status"] == "ok" and env["final"] == "0.31"
        obs = f"초과 {llm.overflows}회 → status={env['status']}, final={env['final']!r}"
    except Exception as e:  # noqa: BLE001
        ok, obs = False, _short(f"CRASH {type(e).__name__}: {e}")
    rep.add("ctx_overflow_plan", "plan", "joiner 프롬프트 초과 → 결과 잘라 1회 재시도 → 정상 종료", ok, obs,
            fix="D9")


async def ctx_overflow_react(rep: Report) -> None:
    """react 프롬프트가 초과 → 실행이 죽지 않고 status=error 로 종료."""
    llm = OverflowFake([], limit=10)
    try:
        env = envelope(await start(patterns.get("react").build(llm=llm, registry=_ctx_registry([]))))
        ok = env["status"] == "error" and "context window" in env["error"]
        obs = f"status={env['status']}, error={_short(env['error'], 50)}"
    except Exception as e:  # noqa: BLE001
        ok, obs = False, _short(f"CRASH {type(e).__name__}: {e}")
    rep.add("ctx_overflow_react", "react", "프롬프트 초과 → 죽지 않고 status=error", ok, obs, fix="D9")


async def ctx_joiner_final_round(rep: Report) -> None:
    """재계획 상한 라운드의 joiner 는 finish 만 허용하는 스키마를 받아야 한다."""
    reg = ToolRegistry()
    reg.register("up", lambda text: str(text).upper())
    script = [{"tasks": [{"id": 1, "tool": "up", "args": {"text": "a"}}]},
              {"decision": "replan", "reason": "more"},
              {"tasks": [{"id": 1, "tool": "up", "args": {"text": "b"}}]},
              {"decision": "finish", "final": "best effort"}]
    llm = fake(script)
    env = envelope(await start(patterns.get("plan_execute").build(llm=llm, registry=reg,
                                                                  max_iterations=2)))
    schemas = [sc for sys_, _, sc in llm.calls if sys_.startswith(JOINER_SYSTEM)]
    finish_only = schemas[-1] == FINISH_ONLY_SCHEMA and schemas[0] != FINISH_ONLY_SCHEMA
    ok = finish_only and env["status"] == "ok" and env["final"] == "best effort"
    rep.add("joiner_final_round", "plan", "마지막 라운드 joiner 는 finish-only 스키마 → 답을 내고 종료", ok,
            f"finish-only={finish_only}, status={env['status']}", fix="5.x")


async def ctx_read_find_and_next(rep: Report) -> None:
    pol = ContextPolicy(offload=True)
    llm = fake(_acts(("big", {}), ("read_observation", {"step": 1, "find": "needle"}),
                     ("read_observation", {"step": 1, "offset": 1500})))
    st = await start(patterns.get("react").build(llm=llm, registry=_ctx_registry([]),
                                                 context_policy=pol))
    obs = [x.get("observation") for x in st.get("scratchpad") or []]
    # 오프셋은 관찰의 JSON 텍스트 기준 (앞 따옴표 1자 포함) — 표시·읽기가 같은 기준
    found = isinstance(obs[1], str) and "NEEDLE" in obs[1] and "(at char 3001)" in obs[1]
    nxt = isinstance(obs[2], str) and "read_observation(step=1, offset=3000)" in obs[2]
    rep.add("ctx_read_find_next", "react+policy", "find 로 위치 점프 + 순차 읽기 결과에 다음 호출 명시", found and nxt,
            f"find={found}, 다음 호출 안내={nxt}", fix="5.1")


CONTEXT_SCENARIOS = [ctx_offload, ctx_clearing, ctx_compaction, ctx_notes_survive_pause,
                     ctx_skills, ctx_tool_search, ctx_name_clash, ctx_nested_child_policy,
                     ctx_plan_results_cap, ctx_compaction_chain, ctx_read_bad_args,
                     ctx_overflow_plan, ctx_overflow_react, ctx_joiner_final_round,
                     ctx_read_find_and_next]


async def run_all() -> Report:
    rep = Report()
    # 등록부의 모든 패턴이 행렬에 들어 있어야 한다 (새 패턴은 어댑터를 추가해야 통과)
    covered = {p.registry_name for p in PATTERNS.values()}
    missing = sorted(set(patterns.names()) - covered)
    rep.add("registry_coverage", "-", "등록된 모든 패턴에 행렬 어댑터가 있음", not missing,
            f"누락={missing}" if missing else f"{sorted(covered)}", fix="3.3")
    for k in PATTERNS:
        for sc in TOP_SCENARIOS:
            await _guard(rep, sc, k)
    for pk in PATTERNS:
        for ck in PATTERNS:
            for sc in PAIR_SCENARIOS:
                await _guard(rep, sc, pk, ck)
    for sc in CONTEXT_SCENARIOS:
        await _guard(rep, sc)
    return rep


async def _guard(rep: Report, sc: Callable, *args: str) -> None:
    try:
        await sc(rep, *args)
    except Exception as e:  # noqa: BLE001 -- 시나리오 자체 버그도 행으로 남긴다
        rep.add(sc.__name__, "⊃".join(args), "(시나리오 실행)", False,
                _short(f"HARNESS {type(e).__name__}: {e} @ {traceback.format_exc(limit=1)}"))


def print_report(rep: Report) -> None:
    w = max(len(r.scenario) for r in rep.rows)
    for r in rep.rows:
        mark = "PASS" if r.ok else "FAIL"
        fix = f"  [{r.fix}]" if (r.fix and not r.ok) else ""
        print(f"{mark}  {r.scenario:<{w}}  {r.cell:<11}  {r.expect}  →  {r.observed}{fix}")
    n_ok = sum(r.ok for r in rep.rows)
    print(f"\n{n_ok}/{len(rep.rows)} PASS, {len(rep.rows) - n_ok} FAIL")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--json", help="결과를 JSON 으로도 저장")
    args = ap.parse_args()
    rep = asyncio.run(run_all())
    print_report(rep)
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump([r.__dict__ for r in rep.rows], f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
