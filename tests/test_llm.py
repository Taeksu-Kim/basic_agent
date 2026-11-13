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


# ---- OpenAICompatLLM: transient errors are retried with backoff (network faked)

class _R:
    def __init__(self, code, text="", headers=None):
        self.status_code, self.text, self.headers = code, text, headers or {}

    def json(self):
        return {"choices": [{"message": {"content": "ok"}}]}


def _scripted(monkeypatch, outcomes):
    """requests.post returns / raises ``outcomes`` in order; time.sleep is recorded."""
    import requests

    from agent.core import llm as llm_mod

    posts, sleeps = [], []
    seq = list(outcomes)

    def fake_post(*a, **k):
        posts.append(1)
        out = seq.pop(0)
        if isinstance(out, Exception):
            raise out
        return out

    monkeypatch.setattr(requests, "post", fake_post)
    monkeypatch.setattr(llm_mod.time, "sleep", sleeps.append)
    return posts, sleeps


def test_openai_compat_retries_timeout_then_succeeds(monkeypatch):
    import requests

    from agent.core.llm import OpenAICompatLLM

    posts, sleeps = _scripted(monkeypatch, [requests.Timeout("t"), requests.ConnectionError("c"),
                                            _R(200)])
    assert OpenAICompatLLM().complete("s", "u") == "ok"
    assert len(posts) == 3 and sleeps == [1.0, 2.0]   # exponential backoff


def test_openai_compat_gives_up_with_named_error(monkeypatch):
    from agent.core.llm import LLMUnavailableError, OpenAICompatLLM

    posts, _ = _scripted(monkeypatch, [_R(503, "busy")] * 3)
    with pytest.raises(LLMUnavailableError, match="3 attempts.*503"):
        OpenAICompatLLM().complete("s", "u")
    assert len(posts) == 3


def test_openai_compat_does_not_retry_client_errors(monkeypatch):
    from agent.core.llm import ContextOverflowError, OpenAICompatLLM

    posts, _ = _scripted(monkeypatch, [_R(400, "unknown field")])
    with pytest.raises(RuntimeError, match="400"):
        OpenAICompatLLM().complete("s", "u")
    posts2, _ = _scripted(monkeypatch, [_R(400, "This model's maximum context length is 8192")])
    with pytest.raises(ContextOverflowError):
        OpenAICompatLLM().complete("s", "u")
    assert len(posts) == 1 and len(posts2) == 1


def test_openai_compat_honours_retry_after_and_max_retries(monkeypatch):
    from agent.core.llm import LLMUnavailableError, OpenAICompatLLM

    posts, sleeps = _scripted(monkeypatch, [_R(429, "slow down", {"Retry-After": "7"}), _R(200)])
    assert OpenAICompatLLM().complete("s", "u") == "ok"
    assert sleeps == [7.0]
    posts, sleeps = _scripted(monkeypatch, [_R(502)])
    with pytest.raises(LLMUnavailableError):
        OpenAICompatLLM(max_retries=0).complete("s", "u")
    assert len(posts) == 1 and sleeps == []
