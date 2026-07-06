import pytest

from agent.llm import FakeLLM, LLMClient


def test_fakellm_is_llmclient():
    assert isinstance(FakeLLM([]), LLMClient)


def test_fakellm_returns_queued_in_order_and_records_calls():
    llm = FakeLLM(["a", "b"])
    assert llm.complete("sys", "u1") == "a"
    assert llm.complete("sys", "u2", schema={"type": "object"}) == "b"
    assert llm.calls == [("sys", "u1", None), ("sys", "u2", {"type": "object"})]


def test_fakellm_json_helper():
    llm = FakeLLM.json({"x": 1}, [1, 2])
    assert llm.complete("s", "u") == '{"x": 1}'
    assert llm.complete("s", "u") == "[1, 2]"


def test_fakellm_exhausted_raises():
    llm = FakeLLM([])
    with pytest.raises(AssertionError, match="ran out"):
        llm.complete("s", "u")
