"""The single tool-execution path every pattern goes through.

Why one path: approval, error capture, interrupt pass-through and tracing used to
be implemented separately in ``react`` and ``plan_execute``, and they drifted --
plan_execute ran ``requires_approval`` tools without asking and swallowed
``GraphInterrupt`` (so ``ask_human`` and nested approvals silently failed). Every
pattern now calls :func:`execute_tool`; a new pattern gets the same semantics for
free.

Result shapes (what the model / joiner sees):

* success  -- the tool's return value, unchanged
* failure  -- ``{"error": "<Type>: <msg>", "__failure__": True}``
* denied   -- ``{"denied": "tool call denied by user: <why>", "__failure__": True}``

``__failure__`` is the explicit marker :func:`is_failure` checks. Data that merely
*contains* an ``"error"`` field is not a failure.
"""

from __future__ import annotations

import contextlib
import contextvars
import time
from typing import Any, Iterator, Literal, Optional

from langgraph.errors import GraphBubbleUp
from langgraph.types import interrupt

from agent.core import trace
from agent.core.hitl import TOOL_APPROVAL, approved, feedback_of
from agent.core.tools import ToolRegistry

FAILURE_KEY = "__failure__"

# Names of the read-only sub-agents we are currently inside (innermost last).
_READ_ONLY: contextvars.ContextVar[tuple[str, ...]] = contextvars.ContextVar(
    "agent_read_only", default=())


@contextlib.contextmanager
def read_only_scope(agent_name: Optional[str]) -> Iterator[None]:
    """Inside, tools marked ``writes=True`` are refused (``None`` = no change).
    Nested scopes stack, so a read-only sub-agent's own sub-agents stay read-only."""
    if agent_name is None:
        yield
        return
    token = _READ_ONLY.set(_READ_ONLY.get() + (agent_name,))
    try:
        yield
    finally:
        _READ_ONLY.reset(token)

Approval = Literal["inline", "preapproved"]


def failure(message: str) -> dict[str, Any]:
    return {"error": message, FAILURE_KEY: True}


def denial(feedback: str) -> dict[str, Any]:
    return {"denied": f"tool call denied by user: {feedback}", FAILURE_KEY: True}


def is_failure(value: Any) -> bool:
    return isinstance(value, dict) and value.get(FAILURE_KEY) is True


def needs_approval(registry: ToolRegistry, name: str) -> bool:
    return registry.has(name) and bool(getattr(registry.get(name), "requires_approval", False))


def approval_payload(name: str, args: dict[str, Any], **extra: Any) -> dict[str, Any]:
    """The ``tool_approval`` interrupt payload (same shape for every pattern)."""
    return {"type": TOOL_APPROVAL, "tool": name, "args": args, **extra}


def record_verdict(name: str, verdict: Any) -> Optional[dict[str, Any]]:
    """Trace a verdict; return the denial result if it rejects, else ``None``."""
    ok = approved(verdict)
    trace.emit("approval", tool=name, approved=ok, feedback=feedback_of(verdict))
    return None if ok else denial(feedback_of(verdict))


def _arg_hint(registry: ToolRegistry, name: str, exc: Exception) -> str:
    """For a call with wrong argument names/count, tell the model the expected
    args -- a bare "unexpected keyword argument 'from'" made a real model keep
    guessing names (from_currency -> from -> source)."""
    if not (isinstance(exc, TypeError) and "argument" in str(exc)) or not registry.has(name):
        return ""
    import json

    schema = getattr(registry.get(name), "args_schema", None)
    if not schema:
        return ""
    return f". Expected args for {name}: {json.dumps(schema, ensure_ascii=False)}"


async def execute_tool(
    registry: ToolRegistry,
    name: str,
    args: Optional[dict[str, Any]] = None,
    *,
    approval: Approval = "inline",
    thought: str = "",
) -> Any:
    """Run one tool call and return what the caller should record as its result.

    ``approval="inline"`` pauses (``interrupt``) before a ``requires_approval``
    tool and resumes on the human verdict -- for loops that run one call at a
    time (react). ``"preapproved"`` skips the gate because the caller already
    collected the verdict before fanning out (plan_execute's level gate), so no
    interrupt ever fires mid-fan-out.

    Never raises for tool errors (they become a failure result the model can
    react to), but always re-raises ``GraphBubbleUp`` so interrupts raised inside
    the tool -- ``ask_human``, or a nested sub-agent's approval gate -- pause the
    whole graph instead of being recorded as an error.
    """
    args = args or {}
    scope = _READ_ONLY.get()
    if scope and registry.has(name) and getattr(registry.get(name), "writes", False):
        refused = failure(f"tool {name!r} has side effects and is not allowed inside "
                          f"read-only sub-agent {scope[-1]!r}; report what should be done instead")
        trace.emit("tool_call", tool=name, args=trace.clip(args), latency_ms=0.0,
                   outcome="failure", detail=refused["error"])
        return refused
    if approval == "inline" and needs_approval(registry, name):
        verdict = interrupt(approval_payload(name, args, thought=thought))
        denied = record_verdict(name, verdict)
        if denied is not None:
            trace.emit("tool_call", tool=name, args=trace.clip(args), latency_ms=0.0,
                       outcome="denied", detail=denied["denied"])
            return denied

    t0 = time.perf_counter()
    try:
        value = await registry.call(name, args)
    except GraphBubbleUp:
        raise  # a pause, not an error
    except Exception as exc:  # noqa: BLE001 -- surface as data the model can react to
        value = failure(f"{type(exc).__name__}: {exc}" + _arg_hint(registry, name, exc))
    trace.emit("tool_call", tool=name, args=trace.clip(args),
               latency_ms=round((time.perf_counter() - t0) * 1000, 1),
               outcome="failure" if is_failure(value) else "ok",
               detail=trace.clip(value["error"]) if is_failure(value) else None)
    return value
