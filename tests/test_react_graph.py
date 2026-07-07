from agent.core.llm import FakeLLM
from agent.core.tools import AgentTool, ToolRegistry
from agent.react.graph import arun, build_react_agent
from agent.react.state import append_steps


def test_append_steps_reducer():
    assert append_steps([1], [2]) == [1, 2]
    assert append_steps(None, [1]) == [1]


def _reg():
    r = ToolRegistry()
    r.register("lookup", lambda q: f"result:{q}")
    return r


async def test_react_loop_tool_then_finish():
    llm = FakeLLM.json(
        {"action": "tool", "tool": "lookup", "args": {"q": "cats"}},
        {"action": "finish", "final": "the answer"},
    )
    graph = build_react_agent(llm=llm, registry=_reg())
    state = await arun(graph, "question")

    assert state["final"] == "the answer"
    assert state["scratchpad"][0]["observation"] == "result:cats"
    assert state["iteration"] == 2  # two reasoning steps


async def test_react_error_is_fed_back_not_aborted():
    reg = ToolRegistry()

    def boom():
        raise ValueError("kaboom")

    reg.register("boom", boom)
    llm = FakeLLM.json(
        {"action": "tool", "tool": "boom", "args": {}},
        {"action": "finish", "final": "recovered"},
    )
    graph = build_react_agent(llm=llm, registry=reg)
    state = await arun(graph, "q")

    assert "error" in state["scratchpad"][0]["observation"]  # captured as observation
    assert state["final"] == "recovered"  # loop continued past the error


async def test_react_max_steps_cap():
    # policy always asks for a tool; the cap must stop the loop.
    llm = FakeLLM.json(*([{"action": "tool", "tool": "lookup", "args": {"q": "x"}}] * 3))
    graph = build_react_agent(llm=llm, registry=_reg(), max_steps=3)
    state = await arun(graph, "q")

    assert state["iteration"] == 3
    assert state.get("final", "") == ""  # never finished, stopped at cap


async def test_react_agent_composes_as_tool_in_plan():
    # A ReAct sub-agent, wrapped as a tool, dispatched by a plan_execute plan.
    from agent.plan_execute.graph import arun as plan_arun
    from agent.plan_execute.graph import build_graph

    sub_llm = FakeLLM.json(
        {"action": "tool", "tool": "lookup", "args": {"q": "z"}},
        {"action": "finish", "final": "react-done"},
    )
    sub = build_react_agent(llm=sub_llm, registry=_reg())

    reg = ToolRegistry()
    reg.add(AgentTool("investigate", sub, description="a ReAct sub-agent"))

    plan_llm = FakeLLM.json(
        {"tasks": [{"id": 1, "tool": "investigate", "args": {"query": "dig in"}}]},
        {"decision": "finish", "final": "plan used $1"},
    )
    plan = build_graph(llm=plan_llm, registry=reg)
    state = await plan_arun(plan, "top task")

    assert state["results"][1] == "react-done"  # ReAct sub-agent ran to completion
    assert state["final"] == "plan used $1"
