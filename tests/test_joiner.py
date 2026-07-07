import pytest

from agent.llm import FakeLLM
from agent.plan_execute import joiner
from agent.plan_execute.dag import PlanError


def test_parse_decision_finish():
    d = joiner.parse_decision('{"decision": "finish", "final": "done"}')
    assert d["decision"] == "finish" and d["final"] == "done"


def test_parse_decision_replan_with_reason():
    d = joiner.parse_decision('{"decision": "replan", "reason": "need more data"}')
    assert d["decision"] == "replan" and d["reason"] == "need more data"


def test_parse_decision_rejects_bad_value():
    with pytest.raises(PlanError, match="invalid"):
        joiner.parse_decision('{"decision": "maybe"}')


def test_join_node_finish():
    llm = FakeLLM.json({"decision": "finish", "final": "the answer"})
    out = joiner.join({"query": "q", "results": {"1": "r"}}, llm)
    assert out["decision"] == "finish"
    assert out["final"] == "the answer"


def test_join_node_replan_stashes_reason():
    llm = FakeLLM.json({"decision": "replan", "reason": "insufficient"})
    out = joiner.join({"query": "q", "results": {}}, llm)
    assert out["decision"] == "replan"
    assert out["decision_reason"] == "insufficient"


def test_join_node_passes_schema():
    llm = FakeLLM.json({"decision": "finish", "final": "x"})
    joiner.join({"query": "q", "results": {}}, llm)
    _, _, schema = llm.calls[0]
    assert schema == joiner.DECISION_SCHEMA
