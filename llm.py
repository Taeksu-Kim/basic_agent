"""LLM client abstraction.

The orchestrator (planner, joiner) depends only on the :class:`LLMClient`
protocol -- a single ``complete(system, user, *, schema) -> str`` method. This is
what lets the whole graph be tested with :class:`FakeLLM` and **never** make a
real model call in the test suite.

The design assumes **vLLM inference behind an OpenAI-compatible endpoint**. When
a ``schema`` is passed, :class:`OpenAICompatLLM` forwards it as
``response_format: {"type": "json_schema", ...}`` -- vLLM's guided-decoding path,
which guarantees the planner/joiner get schema-valid JSON back. Providers are
lazily imported, so importing ``agent`` never needs those deps or a network.
"""

from __future__ import annotations

import json
from typing import Any, Optional, Protocol, runtime_checkable


@runtime_checkable
class LLMClient(Protocol):
    def complete(self, system: str, user: str, *, schema: Optional[dict] = None) -> str: ...


class FakeLLM:
    """Deterministic stand-in for tests. Returns queued responses in order.

    Records the (system, user, schema) it was called with for assertions. The
    ``schema`` is accepted but ignored -- tests assert on structure, not decoding.
    """

    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)
        self.calls: list[tuple[str, str, Optional[dict]]] = []

    def complete(self, system: str, user: str, *, schema: Optional[dict] = None) -> str:
        self.calls.append((system, user, schema))
        if not self._responses:
            raise AssertionError("FakeLLM ran out of queued responses")
        return self._responses.pop(0)

    @classmethod
    def json(cls, *objs: Any) -> "FakeLLM":
        """Convenience: queue JSON-serialised objects as responses."""
        return cls([json.dumps(o) for o in objs])


class OpenAICompatLLM:
    """OpenAI-compatible chat endpoint. This is how a local vLLM server is reached.

    Note (from project setup): under WSL the loopback address can be broken --
    point ``base_url`` at the WSL host IP (e.g. ``http://10.5.0.2:8000/v1``)
    rather than ``localhost`` if requests hang.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8000/v1",
        model: str = "local",
        api_key: str = "not-needed",
        timeout: float = 120.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout = timeout

    def complete(self, system: str, user: str, *, schema: Optional[dict] = None) -> str:
        import requests  # lazy

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": 0,
        }
        if schema is not None:
            # vLLM guided decoding via OpenAI structured-output form.
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "response", "schema": schema, "strict": True},
            }
        resp = requests.post(
            f"{self.base_url}/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json=payload,
            timeout=self.timeout,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]


class AnthropicLLM:
    """Anthropic Messages API adapter (secondary provider).

    Anthropic has no ``response_format``; when a ``schema`` is supplied it is
    appended to the system prompt as a soft instruction.
    """

    def __init__(self, model: str = "claude-opus-4-8", max_tokens: int = 4096) -> None:
        self.model = model
        self.max_tokens = max_tokens

    def complete(self, system: str, user: str, *, schema: Optional[dict] = None) -> str:
        import anthropic  # lazy

        if schema is not None:
            system = f"{system}\n\nRespond with JSON matching this schema:\n{json.dumps(schema)}"
        client = anthropic.Anthropic()
        msg = client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return "".join(block.text for block in msg.content if block.type == "text")
