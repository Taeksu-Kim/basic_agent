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

    @abstractmethod
    def run(self, **kwargs: Any) -> Any:
        """Execute the tool. May be sync or return an awaitable."""
        raise NotImplementedError

    def spec(self) -> dict[str, Any]:
        """Machine-readable description advertised to the planner."""
        return {"name": self.name, "description": self.description, "args_schema": self.args_schema}


class FunctionTool(BaseTool):
    """Adapter that wraps a plain (sync or async) callable as a tool."""

    def __init__(
        self,
        name: str,
        func: Callable[..., Any],
        description: str = "",
        args_schema: Optional[dict] = None,
        requires_approval: bool = False,
    ) -> None:
        self.name = name
        self.description = description
        self.args_schema = args_schema
        self.requires_approval = requires_approval
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

    _RESERVED = ("query", "history", "iteration")

    def __init__(
        self,
        name: str,
        graph: Any,
        *,
        description: str = "",
        args_schema: Optional[dict] = None,
        query_arg: str = "query",
        result_key: str = "final",
    ) -> None:
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

    async def run(self, **kwargs: Any) -> Any:
        query = kwargs[self._query_arg]
        extra = {k: v for k, v in kwargs.items()
                 if k != self._query_arg and k not in self._RESERVED}
        state = await self._graph.ainvoke(
            {**extra, "query": query, "history": [], "iteration": 0}
        )
        return state.get(self._result_key)


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
    ) -> FunctionTool:
        """Wrap and register a plain callable."""
        tool = FunctionTool(name, func, description, args_schema, requires_approval)
        self.add(tool)
        return tool

    def tool(
        self, name: str, *, description: str = "", args_schema: Optional[dict] = None
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        """Decorator form: ``@registry.tool("search", description=...)``."""

        def deco(fn: Callable[..., Any]) -> Callable[..., Any]:
            self.register(name, fn, description=description, args_schema=args_schema)
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

    async def call(self, name: str, args: dict[str, Any]) -> Any:
        """Invoke a tool, awaiting it if it returns a coroutine."""
        result = self.get(name).run(**(args or {}))
        if inspect.isawaitable(result):
            return await result  # type: ignore[no-any-return]
        return result
