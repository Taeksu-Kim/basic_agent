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
