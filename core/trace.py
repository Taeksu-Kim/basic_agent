"""Run trace: an append-only JSONL event log per run.

Off by default -- with no active tracer every hook is a cheap no-op. Turn it on
around a run::

    from agent.core.trace import Tracer, tracing
    with tracing(Tracer("runs/r1.jsonl")):
        state = await arun(graph, "task")

Events carry ``span``: the path of nested sub-agents that emitted them
(``[]`` = top level, ``["researcher"]`` = inside the ``researcher`` AgentTool), so
one file holds the whole nested run. The active tracer and span live in
``contextvars``; LangGraph copies the context into sync nodes and nested graphs,
so nothing has to be threaded through call signatures.

Event kinds (fields besides ``ts``/``run``/``span``/``kind``):

* ``llm_call``  -- ``caller``, ``prompt_chars``, ``output_chars``, ``latency_ms``,
  ``schema`` (bool), ``usage`` (if the client reports it), ``error``
* ``tool_call`` -- ``tool``, ``args``, ``latency_ms``, ``outcome``
  (``ok`` | ``failure`` | ``denied``), ``detail``
* ``approval``  -- ``tool``, ``approved``, ``feedback`` (recorded once a verdict
  arrives; interrupts themselves re-run nodes, so they are logged by the driver)
* ``run_start`` / ``run_end`` / ``interrupt`` -- emitted by ``arun``/``aresume``
"""

from __future__ import annotations

import contextlib
import contextvars
import json
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Iterator, Optional

_TRACER: contextvars.ContextVar[Optional["Tracer"]] = contextvars.ContextVar("agent_tracer", default=None)
_SPAN: contextvars.ContextVar[tuple[str, ...]] = contextvars.ContextVar("agent_span", default=())

ARG_CHARS = 300  # args/detail are truncated in the log (the log is for measurement, not replay)


class Tracer:
    """Appends events to a JSONL file. Thread-safe (sync nodes run in threads)."""

    def __init__(self, path: str | Path, run_id: Optional[str] = None) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.run_id = run_id or uuid.uuid4().hex[:8]
        self._lock = threading.Lock()

    def emit(self, kind: str, **fields: Any) -> None:
        event = {"ts": round(time.time(), 3), "run": self.run_id, "span": list(_SPAN.get()),
                 "kind": kind, **fields}
        line = json.dumps(event, default=str, ensure_ascii=False)
        with self._lock, self.path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")


@contextlib.contextmanager
def tracing(tracer: Optional[Tracer]) -> Iterator[Optional[Tracer]]:
    """Activate ``tracer`` for the enclosed code (``None`` = leave tracing off)."""
    token = _TRACER.set(tracer)
    try:
        yield tracer
    finally:
        _TRACER.reset(token)


@contextlib.contextmanager
def span(name: str) -> Iterator[None]:
    """Nest subsequent events under ``name`` (used by AgentTool around a sub-agent)."""
    token = _SPAN.set(_SPAN.get() + (name,))
    try:
        yield
    finally:
        _SPAN.reset(token)


def active() -> Optional[Tracer]:
    return _TRACER.get()


def emit(kind: str, **fields: Any) -> None:
    tracer = _TRACER.get()
    if tracer is not None:
        tracer.emit(kind, **fields)


def clip(value: Any, n: int = ARG_CHARS) -> str:
    s = value if isinstance(value, str) else json.dumps(value, default=str, ensure_ascii=False)
    return s if len(s) <= n else s[: n - 1] + "…"


class TracedLLM:
    """Wraps an ``LLMClient`` so each ``complete`` call emits an ``llm_call`` event.

    Transparent when tracing is off. ``caller`` labels the call site (planner,
    joiner, react policy, ...). Attribute access falls through to the wrapped
    client, so e.g. ``FakeLLM.calls`` still works on the original object.
    """

    def __init__(self, llm: Any, caller: str) -> None:
        self._llm = llm
        self._caller = caller

    def complete(self, system: str, user: str, *, schema: Optional[dict] = None) -> str:
        if _TRACER.get() is None:
            return self._llm.complete(system, user, schema=schema)
        t0 = time.perf_counter()
        out, err = None, None
        try:
            out = self._llm.complete(system, user, schema=schema)
            return out
        except Exception as exc:  # noqa: BLE001 -- record, then re-raise unchanged
            err = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            emit("llm_call", caller=self._caller, prompt_chars=len(system) + len(user),
                 output_chars=len(out) if out is not None else 0,
                 latency_ms=round((time.perf_counter() - t0) * 1000, 1),
                 schema=schema is not None, usage=getattr(self._llm, "last_usage", None),
                 error=err)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._llm, name)


def traced(llm: Any, caller: str) -> Any:
    """Idempotent wrap (don't double-wrap an already traced client)."""
    return llm if isinstance(llm, TracedLLM) else TracedLLM(llm, caller)
