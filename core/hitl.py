"""Shared HITL verdict contract.

A paused graph resumes with a human *verdict*. Both gates (plan approval,
per-tool approval) accept the same shapes:

* approve -- ``True`` | ``"approve"`` | ``{"approved": True}``
* reject  -- anything else; ``{"feedback": ...}`` or a plain string carries why
"""

from __future__ import annotations

from typing import Any, Optional

from langgraph.types import interrupt


def approved(verdict: Any) -> bool:
    if verdict is True or verdict == "approve":
        return True
    return isinstance(verdict, dict) and bool(verdict.get("approved"))


def feedback_of(verdict: Any) -> str:
    """Human-supplied reason attached to a rejection ('' if none)."""
    if isinstance(verdict, dict):
        return str(verdict.get("feedback", ""))
    if isinstance(verdict, (bool, type(None))):
        return ""
    return str(verdict)


def ask_human(question: str, options: Optional[list] = None) -> Any:
    """Tool: pause the graph and ask the human; the reply is the return value.

    Register it like any tool (``registry.register("ask_human", ask_human, ...)``)
    to let the *model* decide when to stop and ask — the complement of the
    structural gates, which pause regardless of what the model wants. React-only
    for now: inside a parallel plan_execute level, several tasks could interrupt
    at once and resuming would need per-interrupt id mapping.
    """
    return interrupt({"type": "question", "question": question, "options": options or []})
