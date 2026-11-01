"""Pattern registry: every agent pattern, under its industry-standard name.

A pattern is registered once with its builder and metadata. Builders share one
signature -- ``build(llm=..., registry=..., **config) -> compiled graph`` -- and
every built graph honours :mod:`agent.core.contract`, so any pattern can be run by
``arun_graph`` and nested in any other via ``AgentTool``.

    from agent.patterns import registry as patterns
    patterns.names()                         # ['plan_execute', 'react', ...]
    spec = patterns.get("plan")              # aliases resolve too
    graph = spec.build(llm=llm, registry=tools, checkpointer=saver)

Terminology and sources for each ``std_name`` live in ``docs/PATTERNS.md``.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from typing import Any, Callable

Builder = Callable[..., Any]


@dataclass(frozen=True)
class PatternSpec:
    name: str                       # registry key (snake_case)
    std_name: str                   # industry-standard name (docs/PATTERNS.md)
    builder: Builder                # build(llm=..., registry=..., **config) -> graph
    axis: str                       # which docs/PATTERNS.md axis it belongs to
    summary: str                    # one line: what it does
    aliases: tuple[str, ...] = ()   # other names it is known by / CLI shorthands
    weak_model_ok: str = ""         # judgment for small local models (not a measurement)
    sources: tuple[str, ...] = ()   # docs/PATTERNS.md source ids
    extra: dict[str, Any] = field(default_factory=dict)

    def build(self, *, llm: Any, registry: Any, **config: Any) -> Any:
        unknown = set(config) - set(self.config_keys())
        if unknown and not self._accepts_var_kwargs():
            raise TypeError(f"pattern {self.name!r} does not accept config {sorted(unknown)}; "
                            f"accepted: {self.config_keys()}")
        return self.builder(llm=llm, registry=registry, **config)

    def config_keys(self) -> list[str]:
        """Keyword config the builder accepts (besides ``llm`` / ``registry``)."""
        params = inspect.signature(self.builder).parameters
        return [n for n, p in params.items()
                if n not in ("llm", "registry") and p.kind is not p.VAR_KEYWORD]

    def _accepts_var_kwargs(self) -> bool:
        return any(p.kind is p.VAR_KEYWORD for p in inspect.signature(self.builder).parameters.values())


_PATTERNS: dict[str, PatternSpec] = {}
_ALIASES: dict[str, str] = {}


def register(spec: PatternSpec) -> PatternSpec:
    for key in (spec.name, *spec.aliases):
        if key in _PATTERNS or key in _ALIASES:
            raise ValueError(f"pattern name/alias {key!r} already registered")
    _PATTERNS[spec.name] = spec
    for a in spec.aliases:
        _ALIASES[a] = spec.name
    return spec


def get(name: str) -> PatternSpec:
    _ensure_builtins()
    key = _ALIASES.get(name, name)
    if key not in _PATTERNS:
        raise KeyError(f"unknown pattern {name!r}; registered: {names()}")
    return _PATTERNS[key]


def names() -> list[str]:
    _ensure_builtins()
    return sorted(_PATTERNS)


def specs() -> list[PatternSpec]:
    _ensure_builtins()
    return [_PATTERNS[n] for n in sorted(_PATTERNS)]


def choices() -> list[str]:
    """Names + aliases (for CLI ``--type``)."""
    _ensure_builtins()
    return sorted([*_PATTERNS, *_ALIASES])


_builtins_loaded = False


def _ensure_builtins() -> None:
    """Register the built-in patterns on first use (lazy: avoids import cycles)."""
    global _builtins_loaded
    if _builtins_loaded:
        return
    _builtins_loaded = True
    from agent.patterns import builtin  # noqa: F401 -- registers on import
