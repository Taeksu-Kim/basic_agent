"""The contract every agent pattern implements -- what makes patterns nest.

Any compiled graph that honours this contract can be wrapped in an ``AgentTool``
and used as a tool inside any other pattern, and can be driven by the shared
``arun_graph`` / ``aresume_graph``.

**Input** (the state ``ainvoke`` starts from)

* ``query``     -- the task (str, required)
* ``history``   -- prior conversation turns (list, may be empty)
* ``context``   -- context the *caller* chose to pass down (optional; e.g. a
  parent agent's relevant findings). Patterns show it to the model when present.
* ``iteration`` -- 0 at start (step / planning-round counter)
* ``status``    -- reset to ``None`` at start (so a reused thread can't leak the
  previous run's status)

**Output** (keys in the final state)

* ``final``  -- the answer text (str)
* ``result`` -- optional structured payload (Any, ``None`` if the pattern has none)
* ``status`` -- how the run ended:
    - ``"ok"``      the pattern reached its own finish
    - ``"error"``   the model's output stayed unusable (after retries)
    - ``"capped"``  a step / replan cap stopped it before it finished
  A pattern sets ``"ok"`` / ``"error"`` itself; ending without setting it means
  a cap ended the run, so :func:`status_of` reports ``"capped"``.
* ``error``  -- human-readable reason when ``status == "error"``

:func:`envelope` normalises a final state into that shape regardless of pattern.
"""

from __future__ import annotations

import json
from typing import Any, Callable, Literal, Optional, Protocol, TypedDict, runtime_checkable

QUERY, HISTORY, CONTEXT, ITERATION = "query", "history", "context", "iteration"
FINAL, RESULT, STATUS, ERROR = "final", "result", "status", "error"

Status = Literal["ok", "error", "capped"]
STATUSES: tuple[str, ...] = ("ok", "error", "capped")


class ContractState(TypedDict, total=False):
    """The contract's keys as graph state; every pattern's state extends this."""

    query: str
    history: list[Any]      # prior turns; shown to the model when non-empty
    context: Any            # caller-provided context; shown to the model when present
    iteration: int          # step / planning-round counter
    final: str
    result: Any             # optional structured payload
    status: Optional[str]   # "ok" | "error" when the pattern ends itself; unset at END -> capped
    error: str


# A pattern's ``verify=``: checks the final answer in code before the run may end "ok".
# ``(final, result) -> problem`` -- ``None`` / "" passes; a string says what is wrong and
# is fed back to the model (react: as an observation; plan_execute: as the replan reason).
Verifier = Callable[[str, Any], Optional[str]]


def verification_failed(problem: str) -> str:
    """The one wording of a verifier rejection (feedback to the model and ``error``)."""
    return f"final answer failed verification: {problem}"


def ended_with(error: str) -> dict[str, Any]:
    """State update that ends a run as ``status="error"`` (the contract's failure shape)."""
    return {STATUS: "error", ERROR: error, FINAL: ""}


def as_text(value: Any) -> str:
    """A value as prompt text: strings verbatim, anything else as (unicode) JSON."""
    return value if isinstance(value, str) else json.dumps(value, default=str, ensure_ascii=False)


def render_inputs(state: dict[str, Any]) -> list[str]:
    """Prompt sections for the contract inputs ``history`` / ``context``, each shown only
    when present (so a prompt without them is unchanged)."""
    parts = []
    if state.get(HISTORY):
        parts.append("Conversation so far:\n" + "\n".join(str(h) for h in state[HISTORY]))
    if state.get(CONTEXT) is not None:
        parts.append("Context from the caller:\n" + as_text(state[CONTEXT]))
    return parts


@runtime_checkable
class AgentGraph(Protocol):
    """A compiled agent graph (LangGraph ``CompiledStateGraph`` satisfies this)."""

    async def ainvoke(self, input: Any, config: Optional[dict] = None, **kwargs: Any) -> Any: ...


class Envelope(TypedDict):
    final: str
    result: Any
    status: Status
    error: str


def initial_state(query: str, *, history: Optional[list] = None, context: Any = None,
                  **extra: Any) -> dict[str, Any]:
    """The input every pattern is started with (``arun_graph`` and ``AgentTool``)."""
    state: dict[str, Any] = {**extra, QUERY: query, HISTORY: history or [], ITERATION: 0,
                             STATUS: None}
    if context is not None:
        state[CONTEXT] = context
    return state


def status_of(state: dict[str, Any]) -> Status:
    s = (state or {}).get(STATUS)
    return s if s in STATUSES else "capped"


def envelope(state: dict[str, Any]) -> Envelope:
    """Normalise a pattern's final state to the contract's output shape."""
    state = state or {}
    return {
        "final": state.get(FINAL) or "",
        "result": state.get(RESULT),
        "status": status_of(state),
        "error": state.get(ERROR) or "",
    }
