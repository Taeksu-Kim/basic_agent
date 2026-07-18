import pytest

from agent.core.llm import FakeLLM
from agent.core.tools import ToolRegistry
from agent.react import policy
from agent.react.policy import ActionError


def _reg(*names, desc=""):
    r = ToolRegistry()
    for n in names:
        r.register(n, lambda *a, **k: None, description=desc)
    return r


def test_parse_action_tool():
    a = policy.parse_action('{"action": "tool", "tool": "search", "args": {"q": "x"}}')
    assert a["action"] == "tool" and a["tool"] == "search"


def test_parse_action_finish():
    a = policy.parse_action('{"action": "finish", "final": "done"}')
    assert a["action"] == "finish" and a["final"] == "done"


def test_parse_action_rejects_bad_action():
    with pytest.raises(ActionError, match="invalid"):
        policy.parse_action('{"action": "nope"}')


def test_parse_action_tool_requires_name():
    with pytest.raises(ActionError, match="requires"):
        policy.parse_action('{"action": "tool"}')


def test_reason_finish_sets_final_and_counts_step():
    llm = FakeLLM.json({"action": "finish", "final": "ans"})
    out = policy.reason({"query": "q", "iteration": 0}, llm, _reg("search"))
    assert out["action"]["action"] == "finish"
    assert out["final"] == "ans"
    assert out["iteration"] == 1


def test_reason_tool_has_no_final():
    llm = FakeLLM.json({"action": "tool", "tool": "search", "args": {}})
    out = policy.reason({"query": "q", "iteration": 0}, llm, _reg("search"))
    assert out["action"]["tool"] == "search"
    assert "final" not in out


def test_reason_advertises_tools_and_passes_schema():
    llm = FakeLLM.json({"action": "finish", "final": "x"})
    policy.reason({"query": "q", "iteration": 0}, llm, _reg("search", desc="find stuff"))
    system, user, schema = llm.calls[0]
    assert "search" in user and "find stuff" in user
    assert schema == policy.ACTION_SCHEMA


def test_reason_renders_scratchpad_observations():
    llm = FakeLLM.json({"action": "finish", "final": "x"})
    state = {
        "query": "q",
        "iteration": 1,
        "scratchpad": [{"tool": "search", "args": {"q": "a"}, "observation": "found-it"}],
    }
    policy.reason(state, llm, _reg("search"))
    _, user, _ = llm.calls[0]
    assert "found-it" in user


def test_finish_with_structured_result():
    from agent.core.llm import FakeLLM
    from agent.core.tools import ToolRegistry
    from agent.react import policy

    evidence = [{"law": "근로기준법", "clause": "60", "score": 0.9}]
    llm = FakeLLM.json({"action": "finish", "final": "답변", "result": {"evidence": evidence}})
    update = policy.reason({"query": "q", "iteration": 0}, llm, ToolRegistry())

    assert update["final"] == "답변"
    assert update["result"] == {"evidence": evidence}  # 구조화 결과가 상태로 전달


def test_finish_without_result_leaves_result_unset():
    from agent.core.llm import FakeLLM
    from agent.core.tools import ToolRegistry
    from agent.react import policy

    llm = FakeLLM.json({"action": "finish", "final": "그냥 답"})
    update = policy.reason({"query": "q", "iteration": 0}, llm, ToolRegistry())
    assert "result" not in update


def test_reason_accepts_system_prompt_override():
    from agent.core.llm import FakeLLM
    from agent.core.tools import ToolRegistry
    from agent.react import policy

    llm = FakeLLM.json({"action": "finish", "final": "ok"})
    policy.reason({"query": "q", "iteration": 0}, llm, ToolRegistry(),
                  system="너는 법령 검색 에이전트다.")
    system_sent = llm.calls[0][0]
    assert "법령 검색" in system_sent           # 커스텀 지침이 전달됨
    assert policy.REACT_SYSTEM in system_sent   # 기본 ReAct 계약은 유지(suffix 방식)


def test_reason_retries_once_on_malformed_action():
    from agent.core.llm import FakeLLM
    from agent.core.tools import ToolRegistry
    from agent.react import policy

    llm = FakeLLM(['{"action": "tool"}',  # tool 이름 누락 -> 재시도
                   '{"action": "tool", "tool": "lookup", "args": {}}'])
    update = policy.reason({"query": "q", "iteration": 0}, llm, ToolRegistry())
    assert update["action"]["tool"] == "lookup"
    assert len(llm.calls) == 2


def test_reason_gives_up_gracefully_after_retry():
    from agent.core.llm import FakeLLM
    from agent.core.tools import ToolRegistry
    from agent.react import policy

    llm = FakeLLM(["not json at all", "{}"])  # 두 번 다 실패 -> 우아한 finish
    update = policy.reason({"query": "q", "iteration": 0}, llm, ToolRegistry())
    assert update["action"]["action"] == "finish"
    assert "parse_error" in update["action"]


def test_scratchpad_observation_is_truncated_in_prompt():
    from agent.react import policy

    big = [{"snippet": "가" * 5000}]
    text = policy._format_scratchpad([{"tool": "t", "args": {}, "observation": big}])
    assert len(text) < 2000                 # 관측이 프롬프트를 폭파시키지 않음
    assert "truncated" in text


def test_retry_feeds_error_back_to_model():
    from agent.core.llm import FakeLLM
    from agent.core.tools import ToolRegistry
    from agent.react import policy

    llm = FakeLLM(['{"action": "tool"}',
                   '{"action": "finish", "final": "고침"}'])
    update = policy.reason({"query": "q", "iteration": 0}, llm, ToolRegistry())
    assert update["final"] == "고침"
    # 두 번째 호출의 user 프롬프트에 이전 오류가 피드백돼야 모델이 교정 가능
    assert "invalid" in llm.calls[1][1] and "tool" in llm.calls[1][1]


def test_parse_action_json_embedded_in_prose():
    # thinking 모델이 JSON 앞뒤로 텍스트를 붙여도 추출된다 — 캡처 그룹 없는 정규식에
    # group(1)을 호출해 IndexError가 나던 회귀 (ablation 라이브에서 실측)
    a = policy.parse_action('생각: 검색하자.\n{"action": "tool", "tool": "s", "args": {}}\n끝.')
    assert a["action"] == "tool" and a["tool"] == "s"
