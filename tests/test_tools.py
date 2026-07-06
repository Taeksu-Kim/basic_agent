import pytest

from agent.tools import BaseTool, FunctionTool, ToolError, ToolRegistry


async def test_register_and_call_sync_tool():
    reg = ToolRegistry()
    reg.register("add", lambda a, b: a + b)
    assert await reg.call("add", {"a": 2, "b": 3}) == 5


async def test_decorator_registration():
    reg = ToolRegistry()

    @reg.tool("greet", description="say hi")
    def greet(name):
        return f"hi {name}"

    assert reg.has("greet")
    assert await reg.call("greet", {"name": "x"}) == "hi x"


async def test_call_awaits_async_tool():
    reg = ToolRegistry()

    @reg.tool("aecho")
    async def aecho(v):
        return v

    assert await reg.call("aecho", {"v": 7}) == 7


async def test_unknown_tool_raises_toolerror():
    reg = ToolRegistry()
    with pytest.raises(ToolError, match="unknown tool"):
        await reg.call("nope", {})


def test_duplicate_registration_rejected():
    reg = ToolRegistry()
    reg.register("t", lambda: 1)
    with pytest.raises(ValueError, match="already registered"):
        reg.register("t", lambda: 2)


def test_names_sorted():
    reg = ToolRegistry()
    reg.register("b", lambda: 1)
    reg.register("a", lambda: 1)
    assert reg.names() == ["a", "b"]


# ---- base structure / describe ---------------------------------------------


def test_function_tool_carries_metadata_and_describe():
    reg = ToolRegistry()
    reg.register("search", lambda q: q, description="web search", args_schema={"type": "object"})
    (spec,) = reg.describe()
    assert spec == {"name": "search", "description": "web search", "args_schema": {"type": "object"}}


async def test_add_basetool_subclass_instance():
    reg = ToolRegistry()

    class Doubler(BaseTool):
        name = "double"
        description = "double a number"

        def run(self, x):
            return x * 2

    reg.add(Doubler())
    assert reg.has("double")
    assert await reg.call("double", {"x": 4}) == 8
    assert reg.describe()[0]["description"] == "double a number"


async def test_basetool_can_be_async():
    reg = ToolRegistry()

    class AsyncTool(BaseTool):
        name = "afetch"

        async def run(self, url):
            return f"body:{url}"

    reg.add(AsyncTool())
    assert await reg.call("afetch", {"url": "u"}) == "body:u"


def test_add_rejects_nameless_tool():
    reg = ToolRegistry()

    class Bad(BaseTool):
        def run(self):
            return 1

    with pytest.raises(ValueError, match="non-empty name"):
        reg.add(Bad())


def test_function_tool_is_basetool():
    assert isinstance(FunctionTool("t", lambda: 1), BaseTool)
