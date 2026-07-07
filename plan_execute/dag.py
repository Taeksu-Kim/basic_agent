"""Plan data structures and pure DAG algorithms.

A plan is a DAG of :class:`Task` nodes. Each task names a ``tool`` to run, an
``args`` mapping, and its ``deps`` (task ids it depends on). Task args may embed
``$N`` references to the *output* of task ``N``; such references are treated as
implicit dependencies on top of the explicit ``deps``.

This module is deliberately free of any LLM or LangGraph coupling so it can be
unit-tested in isolation. It provides:

* validation (unique ids, resolvable deps/refs, acyclicity),
* topological *level* grouping (the execution schedule), and
* ``$N`` argument substitution.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

_REF_RE = re.compile(r"\$(\d+)")


class PlanError(ValueError):
    """Raised when a plan is structurally invalid (bad deps, cycle, ...)."""


@dataclass(frozen=True)
class Task:
    """A single node in the plan DAG."""

    id: int
    tool: str
    args: dict[str, Any] = field(default_factory=dict)
    deps: tuple[int, ...] = ()

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "Task":
        if "id" not in d or "tool" not in d:
            raise PlanError(f"task missing 'id'/'tool': {d!r}")
        return Task(
            id=int(d["id"]),
            tool=str(d["tool"]),
            args=dict(d.get("args") or {}),
            deps=tuple(int(x) for x in (d.get("deps") or ())),
        )

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "tool": self.tool, "args": self.args, "deps": list(self.deps)}

    def effective_deps(self) -> set[int]:
        """Explicit deps plus dependencies implied by ``$N`` refs in args."""
        return set(self.deps) | refs_in(self.args)


@dataclass
class Plan:
    tasks: list[Task]

    @staticmethod
    def from_list(raw: list[dict[str, Any]]) -> "Plan":
        return Plan([Task.from_dict(t) for t in raw])

    def to_list(self) -> list[dict[str, Any]]:
        return [t.to_dict() for t in self.tasks]

    def by_id(self) -> dict[int, Task]:
        return {t.id: t for t in self.tasks}

    def __len__(self) -> int:  # convenience
        return len(self.tasks)


def refs_in(value: Any) -> set[int]:
    """Collect all ``$N`` reference ids appearing anywhere in ``value``."""
    out: set[int] = set()
    if isinstance(value, str):
        out.update(int(n) for n in _REF_RE.findall(value))
    elif isinstance(value, dict):
        for v in value.values():
            out |= refs_in(v)
    elif isinstance(value, (list, tuple)):
        for v in value:
            out |= refs_in(v)
    return out


def validate_plan(plan: Plan) -> None:
    """Raise :class:`PlanError` if the plan is not a valid DAG."""
    ids = [t.id for t in plan.tasks]
    if len(ids) != len(set(ids)):
        raise PlanError(f"duplicate task ids: {ids}")
    id_set = set(ids)
    for t in plan.tasks:
        missing = t.effective_deps() - id_set
        if missing:
            raise PlanError(f"task {t.id} references unknown ids {sorted(missing)}")
        if t.id in t.effective_deps():
            raise PlanError(f"task {t.id} depends on itself")
    # acyclicity is checked by attempting a topological sort
    topological_levels(plan)


def topological_levels(plan: Plan) -> list[list[int]]:
    """Group task ids into topological *levels* (Kahn's algorithm).

    Level ``k`` contains every task all of whose (effective) deps live in levels
    ``< k`` — i.e. each task is placed at the earliest level its deps allow.
    Within a level, ids are sorted for determinism. Raises on cycles.
    """
    by_id = plan.by_id()
    remaining = {t.id: t.effective_deps() for t in plan.tasks}
    levels: list[list[int]] = []
    done: set[int] = set()
    while remaining:
        ready = sorted(tid for tid, deps in remaining.items() if deps <= done)
        if not ready:
            raise PlanError(f"cycle detected among tasks {sorted(remaining)}")
        levels.append(ready)
        done.update(ready)
        for tid in ready:
            del remaining[tid]
    # touch by_id so a stray unknown dep surfaces here too (defensive)
    assert all(tid in by_id for level in levels for tid in level)
    return levels


def substitute_args(args: Any, results: dict[int, Any]) -> Any:
    """Resolve ``$N`` references in ``args`` against ``results``.

    A value that is *exactly* ``"$N"`` is replaced by the raw result object
    (preserving type). A ``$N`` embedded in a larger string is string-formatted.
    Missing results raise :class:`PlanError`.
    """
    if isinstance(args, str):
        whole = _REF_RE.fullmatch(args)
        if whole:
            key = int(whole.group(1))
            if key not in results:
                raise PlanError(f"unresolved reference ${key}")
            return results[key]

        def _repl(m: re.Match[str]) -> str:
            key = int(m.group(1))
            if key not in results:
                raise PlanError(f"unresolved reference ${key}")
            return str(results[key])

        return _REF_RE.sub(_repl, args)
    if isinstance(args, dict):
        return {k: substitute_args(v, results) for k, v in args.items()}
    if isinstance(args, list):
        return [substitute_args(v, results) for v in args]
    return args
