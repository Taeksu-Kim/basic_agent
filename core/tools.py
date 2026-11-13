"""Tool registry and the base structure for executable units.

A *tool* is the unit the executor dispatches a task to -- the ``tool`` field of a
plan task names one. Tools carry structure so the planner can route well:

* ``name``        -- the routing key,
* ``description`` -- advertised to the planner so it knows when to use the tool,
* ``args_schema`` -- optional JSON schema describing the tool's arguments,
* ``run(**args)`` -- the (sync or async) implementation.

Register a plain function with :meth:`ToolRegistry.register` (wrapped into a
:class:`FunctionTool`), a decorator with :meth:`ToolRegistry.tool`, or add a
:class:`BaseTool` subclass instance with :meth:`ToolRegistry.add` for richer
tools (state, setup, validation, ...).
"""

from __future__ import annotations

import inspect
from abc import ABC, abstractmethod
from typing import Any, Callable, Optional


class ToolError(KeyError):
    """Raised when a task names a tool that is not registered."""


class BaseTool(ABC):
    """Base class for an executable unit the executor can run."""

    name: str = ""
    description: str = ""
    args_schema: Optional[dict] = None
    requires_approval: bool = False  # HITL: pause for human approval before running
    writes: bool = False             # has side effects; refused inside read-only sub-agents
    output_schema: Optional[dict] = None  # shape of the result, advertised to planners

    @abstractmethod
    def run(self, **kwargs: Any) -> Any:
        """Execute the tool. May be sync or return an awaitable."""
        raise NotImplementedError

    def spec(self) -> dict[str, Any]:
        """Machine-readable description advertised to the planner."""
        spec = {"name": self.name, "description": self.description, "args_schema": self.args_schema}
        if self.output_schema:
            spec["output_schema"] = self.output_schema
        return spec


class FunctionTool(BaseTool):
    """Adapter that wraps a plain (sync or async) callable as a tool."""

    def __init__(
        self,
        name: str,
        func: Callable[..., Any],
        description: str = "",
        args_schema: Optional[dict] = None,
        requires_approval: bool = False,
        writes: bool = False,
        output_schema: Optional[dict] = None,
    ) -> None:
        self.name = name
        self.description = description
        self.args_schema = args_schema
        self.requires_approval = requires_approval
        self.writes = writes
        self.output_schema = output_schema
        self._func = func

    def run(self, **kwargs: Any) -> Any:
        return self._func(**kwargs)


class AgentTool(BaseTool):
    """Expose a sub-agent (a compiled graph) as a tool -- the *agent-as-tool*
    pattern. From the executor's view a sub-agent is just a tool: the plan routes
    to it by name, it runs its own plan/execute loop internally and returns a
    result. This is how planned multi-agent is added without changing the core.

    ``graph`` is any object with an async ``ainvoke(state) -> state`` (e.g. the
    output of :func:`agent.graph.build_graph`). The tool's ``query_arg`` becomes
    the sub-agent's query; any *other* kwargs are forwarded into the sub-agent's
    initial state (so e.g. a retriever sub-agent can take ``filters``/``top_k``),
    except the reserved run-bookkeeping keys. ``result_key`` is pulled from the
    final state -- point it at a structured channel (e.g. react's ``result``) to
    get non-string results out of a sub-agent.
    """

    _RESERVED = ("query", "history", "iteration", "status")  # contract-managed keys

    def __init__(
        self,
        name: str,
        graph: Any,
        *,
        description: str = "",
        args_schema: Optional[dict] = None,
        query_arg: str = "query",
        result_key: str = "final",
        read_only: bool = False,
        output_schema: Optional[dict] = None,
    ) -> None:
        """``read_only``: tools marked ``writes=True`` are refused anywhere inside this
        sub-agent (and its own sub-agents) -- single-writer: only the caller acts.

        ``output_schema``: ask the sub-agent to finish with a structured ``result`` of
        this shape and return that object (planners see it as ``returns`` and can
        reference fields: ``$1.total``). If the sub-agent doesn't produce a
        matching ``result``, the call comes back as a failure -- never as free text
        silently standing in for a number.
        """
        self.name = name
        self.description = description
        self.args_schema = args_schema or {
            "type": "object",
            "properties": {query_arg: {"type": "string"}},
            "required": [query_arg],
        }
        self._graph = graph
        self._query_arg = query_arg
        self._result_key = result_key
        self.read_only = read_only
        self.output_schema = output_schema

    def _context_with_schema(self, context: Any) -> Any:
        import json

        ask = ("Finish with `result`: a JSON object matching this schema "
               f"{json.dumps(self.output_schema, ensure_ascii=False)}, and a one-line `final`.")
        if context is None:
            return ask
        base = context if isinstance(context, str) else json.dumps(context, ensure_ascii=False,
                                                                   default=str)
        return f"{base}\n\n{ask}"

    async def run(self, **kwargs: Any) -> Any:
        # local imports: contract/execution import this module
        from agent.core import trace
        from agent.core.contract import initial_state, status_of
        from agent.core.execution import failure, read_only_scope

        query = kwargs[self._query_arg]
        extra = {k: v for k, v in kwargs.items()
                 if k != self._query_arg and k not in self._RESERVED}
        if self.output_schema is not None:
            extra["context"] = self._context_with_schema(extra.get("context"))
        with trace.span(self.name), read_only_scope(self.name if self.read_only else None):
            state = await self._graph.ainvoke(initial_state(query, **extra))
        # Contract-aware sub-agents report how they ended. Anything but "ok" comes
        # back as a failure, so the parent can tell "the child failed / was cut
        # off" from "the child answered ''". Graphs without a ``status`` key
        # (pre-contract) are passed through unchanged.
        if isinstance(state, dict) and "status" in state:
            status = status_of(state)
            if status != "ok":
                why = state.get("error") or "stopped before finishing"
                partial = state.get("final") or ""
                return failure(f"sub-agent {self.name!r} ended with status={status}: {why}"
                               + (f" (partial answer: {partial})" if partial else ""))
        if self.output_schema is not None:
            result = state.get("result") if isinstance(state, dict) else None
            problem = schema_mismatch(self.output_schema, result)
            if problem:
                return failure(f"sub-agent {self.name!r} returned no `result` matching its "
                               f"output_schema ({problem}); its answer was: "
                               f"{(state or {}).get('final', '')!r}")
            return result
        return state.get(self._result_key)


_JSON_TYPES = {"string": str, "number": (int, float), "integer": int, "boolean": bool,
               "object": dict, "array": list}


def schema_mismatch(schema: dict, value: Any) -> str:
    """Shallow check of ``value`` against a JSON schema ('' if it fits): top-level
    type, required keys, and the declared type of each present property."""
    want = schema.get("type")
    if want and want in _JSON_TYPES and not isinstance(value, _JSON_TYPES[want]):
        return f"expected {want}, got {type(value).__name__}"
    if isinstance(value, dict):
        missing = [k for k in schema.get("required", []) if k not in value]
        if missing:
            return f"missing {missing}"
        for key, sub in (schema.get("properties") or {}).items():
            t = sub.get("type") if isinstance(sub, dict) else None
            if key in value and t in _JSON_TYPES:
                ok = isinstance(value[key], _JSON_TYPES[t]) and not (
                    t in ("number", "integer") and isinstance(value[key], bool))
                if not ok:
                    return f"{key}: expected {t}, got {type(value[key]).__name__}"
    return ""


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, BaseTool] = {}

    def add(self, tool: BaseTool) -> BaseTool:
        """Register a :class:`BaseTool` instance."""
        if not tool.name:
            raise ValueError("tool must have a non-empty name")
        if tool.name in self._tools:
            raise ValueError(f"tool {tool.name!r} already registered")
        self._tools[tool.name] = tool
        return tool

    def register(
        self,
        name: str,
        func: Callable[..., Any],
        *,
        description: str = "",
        args_schema: Optional[dict] = None,
        requires_approval: bool = False,
        writes: bool = False,
        output_schema: Optional[dict] = None,
    ) -> FunctionTool:
        """Wrap and register a plain callable."""
        tool = FunctionTool(name, func, description, args_schema, requires_approval,
                            writes, output_schema)
        self.add(tool)
        return tool

    def tool(
        self,
        name: str,
        *,
        description: str = "",
        args_schema: Optional[dict] = None,
        requires_approval: bool = False,
        writes: bool = False,
        output_schema: Optional[dict] = None,
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        """Decorator form of :meth:`register`: ``@registry.tool("search", description=...)``.
        Takes the same options -- it used to lack ``requires_approval``/``writes``, so a
        decorated tool could not be approval-gated or refused in read-only sub-agents."""

        def deco(fn: Callable[..., Any]) -> Callable[..., Any]:
            self.register(name, fn, description=description, args_schema=args_schema,
                          requires_approval=requires_approval, writes=writes,
                          output_schema=output_schema)
            return fn

        return deco

    def has(self, name: str) -> bool:
        return name in self._tools

    def names(self) -> list[str]:
        return sorted(self._tools)

    def get(self, name: str) -> BaseTool:
        try:
            return self._tools[name]
        except KeyError:
            raise ToolError(f"unknown tool {name!r}; registered: {self.names()}") from None

    def describe(self) -> list[dict[str, Any]]:
        """Specs of all tools, sorted by name -- fed to the planner prompt."""
        return [self._tools[n].spec() for n in self.names()]

    def format_for_prompt(self, only: Optional[list[str]] = None) -> str:
        """The tool list every pattern shows the model: name, description and the
        args schema. One formatter for all patterns -- react used to omit the args
        schema, and a real model (Qwen3.8-27B) then guessed argument names
        (``from_currency`` for ``from_ccy``, ``quantity`` for ``qty``) and failed."""
        import json

        lines = []
        for spec in self.describe():
            if only is not None and spec["name"] not in only:
                continue
            line = f"- {spec['name']}"
            if spec.get("description"):
                line += f": {spec['description']}"
            if spec.get("args_schema"):
                line += f"\n    args: {json.dumps(spec['args_schema'], ensure_ascii=False)}"
            if spec.get("output_schema"):
                line += f"\n    returns: {json.dumps(spec['output_schema'], ensure_ascii=False)}"
            lines.append(line)
        return "\n".join(lines)

    async def call(self, name: str, args: dict[str, Any]) -> Any:
        """Invoke a tool, awaiting it if it returns a coroutine."""
        result = self.get(name).run(**(args or {}))
        if inspect.isawaitable(result):
            return await result  # type: ignore[no-any-return]
        return result
