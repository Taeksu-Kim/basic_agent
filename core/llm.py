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
import re
import time
from typing import Any, Callable, Optional, Protocol, TypeVar, runtime_checkable

T = TypeVar("T")

# Exceptions that mean "the model's output was unusable" (not JSON, wrong shape,
# failed validation) -- worth a retry with the error fed back. Every pattern's
# parser catches exactly this set (PlanError, ActionError and JSONDecodeError are
# all ValueErrors).
OUTPUT_ERRORS: tuple[type[Exception], ...] = (ValueError, TypeError, KeyError, AttributeError)


class OutputError(ValueError):
    """The model's output stayed unusable. From :func:`ask_parsed`, ``last`` is the
    final attempt's parse error and ``attempts`` how many calls were made."""

    def __init__(self, message: str, *, last: Optional[Exception] = None, attempts: int = 0) -> None:
        super().__init__(message)
        self.last = last
        self.attempts = attempts


_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)
_OBJECT_OR_LIST_RE = re.compile(r"(\{.*\}|\[.*\])", re.DOTALL)


def extract_json(text: str, *, what: str = "output", allow_list: bool = False) -> Any:
    """Parse JSON from a model response, tolerating prose around it (seen with real
    models: '생각: ... {json} 끝.'). Falls back to the outermost ``{...}`` span (or
    ``{...}``/``[...]``, whichever starts first, with ``allow_list``). One parser for
    every pattern -- four private copies used to drift."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    m = (_OBJECT_OR_LIST_RE if allow_list else _OBJECT_RE).search(text)
    if not m:
        raise OutputError(f"no JSON found in {what}: {text[:200]!r}")
    return json.loads(m.group(0))


def ask_parsed(
    llm: "LLMClient",
    system: str,
    user: str,
    parse: Callable[[str], T],
    *,
    schema: Optional[dict] = None,
    retries: int = 1,
    what: str = "output",
    hint: str = "Emit valid JSON.",
    first: Optional[Callable[[], tuple[str, str]]] = None,
) -> T:
    """Call the model and return ``parse(output)``; the one retry loop every pattern uses.

    Output that ``parse`` rejects (any of :data:`OUTPUT_ERRORS`) is retried ``retries``
    times with ``[Your previous <what> was invalid: <error>. <hint>]`` appended, so the
    model can self-correct. After that :class:`OutputError` is raised (``last`` = the
    final parse error). :class:`ContextOverflowError` is never retried here (the same
    prompt would overflow again) -- it propagates for the caller to end the run.

    ``first``: produce the first attempt's ``(output, user_prompt)`` some other way --
    planner/joiner pass :func:`agent.core.context.complete_fitting` so an overflowing
    prompt is re-rendered once; retries then build on the prompt that was used.
    """
    last: Optional[Exception] = None
    for attempt in range(retries + 1):
        if attempt == 0 and first is not None:
            text, user = first()
        else:
            text = llm.complete(system, user, schema=schema)
        try:
            return parse(text)
        except OUTPUT_ERRORS as exc:  # incl. JSONDecodeError from a brace-y non-JSON reply
            last = exc
            user = f"{user}\n\n[Your previous {what} was invalid: {exc}. {hint}]"
    raise OutputError(f"no valid {what} after {retries + 1} attempts: {last}",
                      last=last, attempts=retries + 1)


class ContextOverflowError(RuntimeError):
    """The prompt (+ requested output) exceeds the model's context window.

    Raised instead of a generic HTTP error so callers can shrink the prompt and
    retry, or end the run gracefully -- an unhandled overflow used to crash the run
    (plan_execute joiner with a 60k-char tool result, 2026-11-01)."""


class LLMUnavailableError(RuntimeError):
    """The model server stayed unreachable / overloaded after the client's retries.

    Deliberately *not* turned into ``status=error``: it is an infrastructure failure,
    not the agent's, and with a checkpointer the run can be resumed from its last
    completed step (``graph.ainvoke(None, config)``) once the server is back."""


TRANSIENT_HTTP = frozenset({429, 500, 502, 503, 504})  # worth retrying; other 4xx are not
MAX_RETRY_WAIT = 30.0  # cap on a server-sent Retry-After (seconds)


def _retry_after(resp: Any) -> Optional[float]:
    try:
        return min(float(resp.headers.get("Retry-After")), MAX_RETRY_WAIT)
    except (TypeError, ValueError):  # absent, or an HTTP-date we don't parse
        return None


# Server/SDK error texts that mean "context window exceeded" (one list for every client --
# OpenAICompatLLM used to check only the first one).
_OVERFLOW_HINTS = ("maximum context length", "context_length_exceeded", "prompt is too long",
                   "context window", "too many tokens")


def _overflow_text(text: str) -> bool:
    text = text.lower()
    return any(h in text for h in _OVERFLOW_HINTS)


# Request defaults for every client (override per key with ``sampling``). max_tokens 기본
# 상한: 미설정 시 소형 모델의 반복 폭주가 컨텍스트를 다 먹고 JSON이 중간에 잘리는 사고가 난다 (실측).
DEFAULT_SAMPLING: dict[str, Any] = {"temperature": 0, "max_tokens": 1024}


def with_guidance(base: str, guidance: Optional[str]) -> str:
    """A pattern's built-in system prompt plus the caller's domain guidance (each builder's
    ``system=``). Appended *after* the base so the pattern's output protocol stays in force;
    ``None``/empty leaves the base unchanged."""
    return f"{base}\n\n{guidance}" if guidance else base


def with_schema_instruction(system: str, schema: dict) -> str:
    """System prompt plus the schema as an instruction -- for providers/modes without
    native structured output; the patterns' parse-and-retry handles what slips through."""
    return (f"{system}\n\nRespond with ONLY a JSON value matching this JSON schema "
            f"(no prose, no code fences):\n{json.dumps(schema, ensure_ascii=False)}")


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
        sampling: Optional[dict] = None,
        max_retries: int = 2,
        backoff: float = 1.0,
    ) -> None:
        """``max_retries``: extra attempts on a transient failure (timeout, connection
        error, HTTP 429 / 5xx), waiting ``backoff * 2**i`` seconds (or the server's
        ``Retry-After``, capped at :data:`MAX_RETRY_WAIT`). The SDK-based clients get the
        same from their SDK; this one talks HTTP directly, so it has to do it itself."""
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout = timeout
        self.sampling = {**DEFAULT_SAMPLING, **(sampling or {})}
        self.max_retries = max_retries
        self.backoff = backoff
        import threading

        self._local = threading.local()  # per-thread: parallel nodes call from several threads

    @property
    def last_usage(self) -> Optional[dict]:
        """Token usage of this thread's most recent call (read by the tracer)."""
        return getattr(self._local, "usage", None)

    def complete(self, system: str, user: str, *, schema: Optional[dict] = None) -> str:
        import requests  # lazy

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            **self.sampling,
        }
        if schema is not None:
            # vLLM guided decoding via OpenAI structured-output form.
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "response", "schema": schema, "strict": True},
            }
        last = ""
        for attempt in range(self.max_retries + 1):
            if attempt:
                time.sleep(wait)
            try:
                resp = requests.post(
                    f"{self.base_url}/chat/completions",
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    json=payload,
                    timeout=self.timeout,
                )
            except (requests.Timeout, requests.ConnectionError) as exc:
                last, wait = f"{type(exc).__name__}: {exc}", self.backoff * 2 ** attempt
                continue
            if resp.status_code in TRANSIENT_HTTP:
                last = f"HTTP {resp.status_code}: {resp.text[:200]}"
                wait = _retry_after(resp) or self.backoff * 2 ** attempt
                continue
            break
        else:
            raise LLMUnavailableError(
                f"LLM unreachable after {self.max_retries + 1} attempts: {last}")
        if resp.status_code == 400 and _overflow_text(resp.text):
            raise ContextOverflowError(f"LLM context overflow: {resp.text[:300]}")
        if resp.status_code >= 400:
            # surface the server's reason -- a bare 400 is undebuggable
            raise RuntimeError(f"LLM HTTP {resp.status_code}: {resp.text[:500]}")
        data = resp.json()
        self._local.usage = data.get("usage")
        return data["choices"][0]["message"]["content"]


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
            system = with_schema_instruction(system, schema)
        client = anthropic.Anthropic()
        try:
            msg = client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                system=system,
                messages=[{"role": "user", "content": user}],
            )
        except Exception as exc:  # noqa: BLE001 -- map overflow, re-raise everything else
            # unmapped, an overflow crashed the run instead of ending it with status=error
            if _is_overflow(exc):
                raise ContextOverflowError(f"LLM context overflow (anthropic): {exc}") from exc
            raise
        return "".join(block.text for block in msg.content if block.type == "text")


# ---------------------------------------------------------------- any-llm adapter

# Providers whose native structured output cannot express this framework's schemas.
# Checked offline (any-llm 1.37.0 + anthropic SDK, examples/llm_adapter_check.py): the
# Anthropic schema transform turns a free-form ``{"type": "object"}`` (tool ``args``,
# ``result``) into ``additionalProperties: false`` with no properties -- an object that
# must stay EMPTY -- and drops ``const`` into the description. Native mode would let
# the model call tools only without arguments, so these default to "prompt" mode.
PROMPT_SCHEMA_PROVIDERS = frozenset({"anthropic", "azureanthropic", "vertexaianthropic"})

class AnyLLMClient:
    """Any provider through mozilla-ai **any-llm** (optional: ``pip install 'any-llm-sdk[<provider>]'``).

    ``provider`` is an any-llm provider id (``openai``, ``anthropic``, ``gemini``, ``mistral``,
    ``ollama``, ...). For an OpenAI-compatible server such as vLLM, pass ``api_base`` with
    ``provider="openai_compatible"`` -- any-llm's dedicated path (it never sends your
    ``OPENAI_API_KEY`` to a custom endpoint).

    ``structured`` -- how a ``schema`` is enforced:

    * ``"native"``: the provider's structured output (``response_format=json_schema``).
      vLLM turns it into guided decoding -- the schema is guaranteed.
    * ``"prompt"``: the schema is appended to the system prompt; the patterns' own
      parse-and-retry handles bad output. Needed where native mode can't express the
      schemas (see ``PROMPT_SCHEMA_PROVIDERS``).
    * ``"auto"`` (default): ``"prompt"`` for those providers, ``"native"`` otherwise.

    ``strict`` is sent with native schemas (default False: hosted APIs with a strict
    mode reject free-form objects; vLLM ignores the flag). ``sampling`` adds request
    fields (``temperature``, ``max_tokens``, ...); ``extra_body`` adds raw body fields the
    SDK doesn't know (e.g. ``{"chat_template_kwargs": {"enable_thinking": False}}`` for
    Qwen3 on vLLM). ``client_kwargs`` go to ``AnyLLM.create`` (e.g. ``http_client``).

    Context-window overflows raise :class:`ContextOverflowError` like the other clients,
    so the patterns' shrink-and-retry / graceful-error paths work unchanged.
    """

    def __init__(
        self,
        provider: str,
        model: str,
        *,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        structured: str = "auto",
        strict: bool = False,
        sampling: Optional[dict] = None,
        extra_body: Optional[dict] = None,
        client_kwargs: Optional[dict] = None,
    ) -> None:
        from any_llm import AnyLLM  # lazy: optional dependency
        import threading

        if structured not in ("auto", "native", "prompt"):
            raise ValueError("structured must be 'auto', 'native' or 'prompt'")
        self.provider = provider
        self.model = model
        self.structured = (("prompt" if provider in PROMPT_SCHEMA_PROVIDERS else "native")
                           if structured == "auto" else structured)
        self.strict = strict
        self.sampling = {**DEFAULT_SAMPLING, **(sampling or {})}
        self.extra_body = extra_body
        kw = dict(client_kwargs or {})
        if provider == "openai_compatible":
            if not api_base:
                raise ValueError("provider='openai_compatible' needs api_base")
            self._llm = AnyLLM.create_openai_compatible(name="openai_compatible", api_base=api_base,
                                                        api_key=api_key, unified_exceptions=True, **kw)
        else:
            self._llm = AnyLLM.create(provider, api_key=api_key, api_base=api_base,
                                      unified_exceptions=True, **kw)
        self._local = threading.local()

    @property
    def last_usage(self) -> Optional[dict]:
        return getattr(self._local, "usage", None)

    def complete(self, system: str, user: str, *, schema: Optional[dict] = None) -> str:
        kwargs: dict[str, Any] = dict(self.sampling)
        if self.extra_body:
            kwargs["extra_body"] = self.extra_body
        if schema is not None and self.structured == "native":
            kwargs["response_format"] = {"type": "json_schema", "json_schema": {
                "name": "response", "schema": schema, "strict": self.strict}}
        elif schema is not None:
            system = with_schema_instruction(system, schema)
        try:
            resp = self._llm.completion(
                model=self.model,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                **kwargs,
            )
        except Exception as exc:  # noqa: BLE001 -- map overflow, re-raise everything else
            if _is_overflow(exc):
                raise ContextOverflowError(f"LLM context overflow ({self.provider}): {exc}") from exc
            raise
        usage = getattr(resp, "usage", None)
        self._local.usage = ({"prompt_tokens": usage.prompt_tokens,
                              "completion_tokens": usage.completion_tokens} if usage else None)
        return resp.choices[0].message.content or ""


def _is_overflow(exc: Exception) -> bool:
    try:
        from any_llm.exceptions import ContextLengthExceededError

        if isinstance(exc, ContextLengthExceededError):
            return True
    except ImportError:  # pragma: no cover
        pass
    return _overflow_text(str(exc))
