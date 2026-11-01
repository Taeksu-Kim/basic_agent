"""Held-out evaluation set (2026-11-01) -- written and committed BEFORE its first run.

Purpose: check that the framework generalises beyond the tasks used while fixing it
(``run_evals.py`` main / lo / wf: English, finance, keyword-findable facts). Rules:

* different domain (Korean HR), tables, paraphrased questions (the question's words
  are not in the document), date arithmetic, every registered pattern;
* tasks and pass criteria are frozen in the commit that adds this file; results are
  recorded as they come -- no prompt or code tuning against this set. A failure found
  here is fixed only with a NEW held-out check afterwards.

Run: ``python -m agent.examples.evals.run_evals --suite heldout ...``
"""

from __future__ import annotations

import re
from datetime import date
from typing import Any, Optional

from langgraph.checkpoint.memory import InMemorySaver

from agent.core.tools import AgentTool, ToolRegistry
from agent.patterns import registry as patterns

# ------------------------------------------------------------------ fake HR world

EMPLOYEES = {
    "E1001": {"name": "김민지", "dept": "재무팀", "hire_date": "2021-03-02", "grade": "G3"},
    "E1002": {"name": "Park Jun", "dept": "Engineering", "hire_date": "2024-08-19", "grade": "G2"},
}
LEAVE = {"E1001": {"annual_total": 15, "used": 6}, "E1002": {"annual_total": 15, "used": 2}}

_FILL = "본 조항은 회사의 인사 운영 원칙에 따라 적용되며 세부 절차는 인사팀 공지를 따른다. "


def _section(title: str, body: str, filler: int) -> str:
    return f"\n\n## {title}\n" + _FILL * filler + body + " " + _FILL * 2


HANDBOOK = (
    "# 사내 취업규칙 (2026년판)\n" + _FILL * 6
    + _section("제1장 근무시간", "근무시간은 09:00~18:00을 원칙으로 한다.", 8)
    + _section("제2장 재택근무", "재택근무는 주 2회까지 허용하며, 사전에 소속 팀장의 승인을 받아야 한다.", 10)
    + _section("제3장 연차휴가",
               "연차휴가 일수는 근속기간에 따라 아래 표와 같다.\n"
               "| 근속기간 | 연차 일수 |\n|---|---|\n| 1년 미만 | 매월 1일 |\n| 1년 이상 3년 미만 | 15일 |\n"
               "| 3년 이상 5년 미만 | 16일 |\n| 5년 이상 | 18일 |", 8)
    + _section("제4장 육아휴직",
               "만 8세 이하 자녀를 양육하는 직원은 자녀 1명당 최대 1년의 육아휴직을 사용할 수 있다.", 14)
    + _section("제5장 출장비",
               "출장 숙박비 한도는 직급별로 아래 표와 같다.\n"
               "| 직급 | 1박 숙박비 한도 |\n|---|---|\n| G1 | 10만원 |\n| G2 | 12만원 |\n| G3 | 15만원 |", 12)
    + _section("제6장 기타", "본 규칙에 정하지 않은 사항은 관계 법령에 따른다.", 4)
)


def build_hr_registry(log: list) -> ToolRegistry:
    reg = ToolRegistry()
    s = {"type": "string"}

    def obj(props: dict, req: list) -> dict:
        return {"type": "object", "properties": props, "required": req}

    def find_employee(name: str) -> Any:
        hits = [{"employee_id": k, **v} for k, v in EMPLOYEES.items() if name.strip() in v["name"]]
        return hits or {"error": f"no employee named {name!r}"}

    def get_employee(employee_id: str) -> Any:
        if employee_id not in EMPLOYEES:
            raise KeyError(f"unknown employee_id {employee_id!r}")
        return {"employee_id": employee_id, **EMPLOYEES[employee_id]}

    def get_leave_balance(employee_id: str) -> Any:
        if employee_id not in LEAVE:
            raise KeyError(f"unknown employee_id {employee_id!r}")
        return dict(LEAVE[employee_id])

    def get_handbook() -> str:
        return HANDBOOK

    def tenure_years(hire_date: str, as_of: str) -> float:
        d0, d1 = date.fromisoformat(hire_date), date.fromisoformat(as_of)
        return round((d1 - d0).days / 365.25, 1)

    def create_ticket(category: str, summary: str) -> str:
        log.append(("ticket", str(category), str(summary)))
        return f"ticket created: [{category}] {summary}"

    def draft_leave_request(employee_id: str, days: float) -> str:
        log.append(("leave_draft", str(employee_id), days))
        return f"leave request draft: {employee_id}, {days} days"

    reg.register("find_employee", find_employee, description="직원 이름으로 사번·부서·입사일·직급 조회",
                 args_schema=obj({"name": s}, ["name"]))
    reg.register("get_employee", get_employee, description="사번으로 직원 정보 조회",
                 args_schema=obj({"employee_id": s}, ["employee_id"]))
    reg.register("get_leave_balance", get_leave_balance, description="사번으로 연차 총일수·사용일수 조회",
                 args_schema=obj({"employee_id": s}, ["employee_id"]))
    reg.register("get_handbook", get_handbook, description="사내 취업규칙 전문")
    reg.register("tenure_years", tenure_years, description="입사일부터 기준일까지 근속 연수(소수 1자리)",
                 args_schema=obj({"hire_date": s, "as_of": s}, ["hire_date", "as_of"]))
    reg.register("create_ticket", create_ticket, description="IT/시설 요청 티켓 생성 (승인 필요)",
                 args_schema=obj({"category": {"enum": ["IT", "시설", "기타"]}, "summary": s},
                                 ["category", "summary"]),
                 requires_approval=True, writes=True)
    reg.register("draft_leave_request", draft_leave_request, description="휴가 신청서 초안 작성",
                 args_schema=obj({"employee_id": s, "days": {"type": "number"}}, ["employee_id", "days"]))
    return reg


# ------------------------------------------------------------------ builders


def _react(llm: Any, log: list) -> Any:
    return patterns.get("react").build(llm=llm, registry=build_hr_registry(log), max_steps=8,
                                       checkpointer=InMemorySaver())


def _plan(llm: Any, log: list) -> Any:
    return patterns.get("plan_execute").build(llm=llm, registry=build_hr_registry(log),
                                              checkpointer=InMemorySaver())


def _hr_agent(llm: Any, log: list, **kw: Any) -> AgentTool:
    child = patterns.get("react").build(llm=llm, registry=build_hr_registry(log), max_steps=6)
    return AgentTool("hr_agent", child, description="인사 정보·연차·사내 규정 질문을 처리하는 에이전트", **kw)


def _router(llm: Any, log: list) -> Any:
    reg = ToolRegistry()
    reg.add(_hr_agent(llm, log))
    it_reg = build_hr_registry(log)
    it = patterns.get("react").build(llm=llm, registry=it_reg, max_steps=4)
    reg.add(AgentTool("it_agent", it, description="IT·시설 장애 신고와 요청 티켓을 처리하는 에이전트"))
    return patterns.get("router").build(llm=llm, registry=reg, checkpointer=InMemorySaver(),
                                        routes={"hr_agent": "인사·연차·규정 질문",
                                                "it_agent": "장비 고장, 수리, IT·시설 요청"})


def _parallel(llm: Any, log: list) -> Any:
    reg = ToolRegistry()
    reg.add(_hr_agent(llm, log))
    return patterns.get("parallel").build(
        llm=llm, registry=reg, checkpointer=InMemorySaver(), aggregate="llm",
        sections=[{"name": "재택근무", "tool": "hr_agent", "query": "사내 규정의 재택근무 허용 횟수와 조건은?"},
                  {"name": "숙박비", "tool": "hr_agent", "query": "G2 직급의 출장 숙박비 한도는?"}])


def _orchestrator(llm: Any, log: list) -> Any:
    reg = ToolRegistry()
    reg.add(_hr_agent(llm, log))
    return patterns.get("orchestrator_workers").build(llm=llm, registry=reg,
                                                      checkpointer=InMemorySaver(), max_workers=3)


def _nested_structured(llm: Any, log: list) -> Any:
    reg = build_hr_registry(log)
    reg.add(_hr_agent(llm, log, output_schema={
        "type": "object", "properties": {"employee_id": {"type": "string"},
                                         "days_left": {"type": "number"}},
        "required": ["employee_id", "days_left"]}))
    return patterns.get("plan_execute").build(llm=llm, registry=reg, checkpointer=InMemorySaver())


WELCOME_CRITERIA = ("1) 한국어로 작성 2) 세 문장 이하 3) '재무팀'이라는 단어 포함 4) 이모지 사용 금지 "
                    "5) 신입사원 김민지 님을 환영하는 내용")


def _evaluator(llm: Any, log: list) -> Any:
    return patterns.get("evaluator_optimizer").build(llm=llm, registry=ToolRegistry(),
                                                     checkpointer=InMemorySaver(),
                                                     criteria=WELCOME_CRITERIA, max_rounds=3)


def _date_gate(out: Any) -> Optional[str]:
    return None if re.fullmatch(r"\d{4}-\d{2}-\d{2}", str(out).strip()) else \
        f"expected YYYY-MM-DD, got {str(out)[:40]!r}"


def _chain(llm: Any, log: list) -> Any:
    reg = build_hr_registry(log)
    return patterns.get("prompt_chain").build(llm=llm, registry=reg, checkpointer=InMemorySaver(), steps=[
        {"name": "emp", "tool": "get_employee", "args": {"employee_id": "E1002"}},
        {"name": "hire", "prompt": "다음 데이터에서 입사일만 YYYY-MM-DD 형식으로, 다른 글자 없이 출력: {emp}",
         "gate": _date_gate},
        {"name": "tenure", "tool": "tenure_years", "args": {"hire_date": "{hire}", "as_of": "2026-11-01"}},
    ])


# ------------------------------------------------------------------ checks (frozen)


def _nums(text: str) -> list[float]:
    return [float(x.replace(",", "")) for x in re.findall(r"-?\d[\d,]*\.?\d*", text or "")]


def has_num(value: float, tol: float = 0.011):
    return lambda env, log: any(abs(n - value) <= tol for n in _nums(env["final"]))


def has_all(*parts: str):
    return lambda env, log: all(p in (env["final"] or "") for p in parts)


def has_any(*parts: str):
    return lambda env, log: any(p in (env["final"] or "") for p in parts)


def _welcome_ok(env: dict, log: list) -> bool:
    f = env["final"] or ""
    sentences = [x for x in re.split(r"[.!?。]\s*", f.strip()) if x.strip()]
    emoji = re.search(r"[\U0001F300-\U0001FAFF☀-➿]", f)
    return (env["status"] == "ok" and re.search(r"[가-힣]", f) is not None and len(sentences) <= 3
            and "재무팀" in f and not emoji)


def _ticket_with(word: str):
    return lambda env, log: any(e[0] == "ticket" and word in e[2] for e in log)


def _leave_draft_days(days: float):
    return lambda env, log: any(e[0] == "leave_draft" and isinstance(e[2], (int, float))
                                and abs(float(e[2]) - days) < 0.01 for e in log)


def heldout_tasks(Task: Any) -> list:
    """Build the frozen task list with ``run_evals.Task`` (passed in to avoid an import cycle)."""
    t = []
    for pat, build in (("react", _react), ("plan", _plan)):
        t += [
            Task(f"h_leave_left_{pat}", "김민지 직원의 남은 연차는 며칠인가요?", has_num(9),
                 "직원 조회 → 연차 조회 → 뺄셈", build=build),
            Task(f"h_paraphrase_{pat}", "회사 규정상 아이를 돌보기 위해 쉴 수 있는 기간은 최대 얼마인가요?",
                 has_any("1년", "12개월", "1 year"), "질문 단어가 문서에 없음 (문서: 육아휴직)", build=build),
            Task(f"h_table_{pat}", "G3 직급의 출장 1박 숙박비 한도는 얼마인가요?",
                 has_any("15만", "150,000", "150000"), "문서 뒤쪽 표 (약 3,600자 지점, react 미리보기 1,500자 밖)", build=build),
            Task(f"h_tenure_table_{pat}",
                 "2026-11-01 기준으로 Park Jun 직원의 근속기간에 해당하는 규정상 연차 일수는?",
                 has_num(15), "입사일 조회 → 근속 계산 → 표 조회", build=build),
        ]
    t += [
        Task("h_ids_plan", "재무팀 김민지 직원의 사번과 입사일을 알려줘", has_all("E1001", "2021-03-02"),
             "plan_execute 단일 조회", build=_plan),
        Task("h_ticket_react", "김민지 직원 명의로 '노트북 교체' IT 티켓을 생성해줘", _ticket_with("노트북"),
             "승인 필요 쓰기 도구 (자동 승인)", build=_react),
        Task("h_router_it", "모니터가 고장났어요. 수리 요청 넣어주세요.", _ticket_with("모니터"),
             "router → it_agent → 티켓 (승인)", build=_router),
        Task("h_parallel", "재택근무 규정과 G2 숙박비 한도를 정리해줘",
             lambda env, log: has_any("2회", "두 번", "주 2")(env, log) and has_any("12만", "120,000", "120000")(env, log),
             "parallel 섹션 2개 + LLM 종합", build=_parallel),
        Task("h_orchestrator", "김민지와 Park Jun 중 누가 남은 연차가 더 많나요?",
             has_any("Park", "박준"), "orchestrator: 비교 → 워커 + 종합", build=_orchestrator),
        Task("h_nested_structured",
             "hr_agent에게 김민지 직원의 사번과 남은 연차 일수를 물어본 뒤, 그 일수로 휴가 신청서 초안을 작성해줘",
             _leave_draft_days(9), "output_schema(days_left) → $1.days_left (planner 예시와 다른 필드명)",
             build=_nested_structured),
        Task("h_evaluator", "신입사원 김민지 님 환영 메시지를 써줘", _welcome_ok,
             "evaluator: 형식 기준 5개", build=_evaluator),
        Task("h_chain", "Park Jun의 근속 연수", has_num(2.2, tol=0.051),
             "prompt_chain: 조회 → 날짜 추출(gate) → 계산 도구", build=_chain),
    ]
    return t
