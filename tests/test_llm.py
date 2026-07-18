import pytest

from agent.core.llm import FakeLLM, LLMClient


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


def test_openai_compat_sampling_overrides_in_payload(monkeypatch):
    from agent.core import llm as llm_mod

    captured = {}

    class _Resp:
        status_code = 200

        @staticmethod
        def json():
            return {"choices": [{"message": {"content": "ok"}}]}

    def fake_post(url, headers=None, json=None, timeout=None):
        captured.update(json)
        return _Resp()

    monkeypatch.setattr("requests.post", fake_post)
    llm = llm_mod.OpenAICompatLLM(base_url="http://x/v1", model="m",
                                  sampling={"temperature": 0.2, "frequency_penalty": 0.5})
    assert llm.complete("s", "u") == "ok"
    assert captured["temperature"] == 0.2          # 기본 0을 덮어씀
    assert captured["frequency_penalty"] == 0.5
    assert captured["max_tokens"] == 1024          # 폭주 방지 기본 상한
