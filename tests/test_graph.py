from agent.graph import arun, build_graph
from agent.llm import FakeLLM
from agent.state import merge_results
from agent.tools import AgentTool, ToolRegistry


# ---- state reducer ---------------------------------------------------------


def test_merge_results_merges():
    assert merge_results({1: "a"}, {2: "b"}) == {1: "a", 2: "b"}
    assert merge_results(None, {1: "x"}) == {1: "x"}


def test_merge_results_reset_sentinel_clears():
    assert merge_results({1: "a", 2: "b"}, "__RESET__") == {}


# ---- end-to-end with fakes (no real LLM, no network) -----------------------


def _gen_registry():
    reg = ToolRegistry()
    reg.register("gen", lambda n: f"g{n}")
    reg.register("combine", lambda x, y: {"x": x, "y": y})
    return reg


async def test_parallel_level_and_cross_level_substitution():
    llm = FakeLLM.json(
        {
            "tasks": [
                {"id": 1, "tool": "gen", "args": {"n": 1}},
                {"id": 2, "tool": "gen", "args": {"n": 2}},
                {"id": 3, "tool": "combine", "args": {"x": "$1", "y": "$2"}},
            ]
        },
        {"decision": "finish", "final": "ok"},
    )
    graph = build_graph(llm=llm, registry=_gen_registry())
    state = await arun(graph, "do it")

    assert state["final"] == "ok"
    # both parallel tasks ran
    assert state["results"][1] == "g1" and state["results"][2] == "g2"
    # $1/$2 were substituted with the upstream outputs
    assert state["results"][3] == {"x": "g1", "y": "g2"}


async def test_replan_loop_then_finish():
    llm = FakeLLM.json(
        {"tasks": [{"id": 1, "tool": "gen", "args": {"n": 1}}]},
        {"decision": "replan", "reason": "need more"},
        {"tasks": [{"id": 1, "tool": "gen", "args": {"n": 9}}]},
        {"decision": "finish", "final": "done"},
    )
    graph = build_graph(llm=llm, registry=_gen_registry())
    state = await arun(graph, "do it")

    assert state["final"] == "done"
    assert state["iteration"] == 2  # planned twice
    assert state["results"][1] == "g9"  # results were reset to the 2nd plan


async def test_replan_cap_stops_gracefully():
    # joiner always says replan; cap at 2 planning rounds must terminate.
    llm = FakeLLM.json(
        {"tasks": [{"id": 1, "tool": "gen", "args": {"n": 1}}]},
        {"decision": "replan", "reason": "again"},
        {"tasks": [{"id": 1, "tool": "gen", "args": {"n": 1}}]},
        {"decision": "replan", "reason": "again"},
    )
    graph = build_graph(llm=llm, registry=_gen_registry(), max_iterations=2)
    state = await arun(graph, "do it")

    assert state["iteration"] == 2
    assert state["decision"] == "replan"  # hit the cap, did not loop forever


async def test_sub_agent_as_tool():
    # A sub-agent, wrapped as a tool, is dispatched by a parent plan (agent-as-tool).
    sub_llm = FakeLLM.json(
        {"tasks": [{"id": 1, "tool": "gen", "args": {"n": 5}}]},
        {"decision": "finish", "final": "sub-answer"},
    )
    sub_graph = build_graph(llm=sub_llm, registry=_gen_registry())

    parent_reg = ToolRegistry()
    parent_reg.add(AgentTool("researcher", sub_graph, description="delegates research"))

    parent_llm = FakeLLM.json(
        {"tasks": [{"id": 1, "tool": "researcher", "args": {"query": "dig in"}}]},
        {"decision": "finish", "final": "parent used $1"},
    )
    parent = build_graph(llm=parent_llm, registry=parent_reg)
    state = await arun(parent, "top task")

    assert state["results"][1] == "sub-answer"  # sub-agent ran and returned
    assert state["final"] == "parent used $1"


async def test_error_triggers_early_abort():
    ran: list[str] = []
    reg = ToolRegistry()

    def boom():
        raise ValueError("kaboom")

    reg.register("boom", boom)
    reg.register("after", lambda x: ran.append("after") or "late")

    llm = FakeLLM.json(
        {
            "tasks": [
                {"id": 1, "tool": "boom"},
                {"id": 2, "tool": "after", "args": {"x": "$1"}, "deps": [1]},
            ]
        },
        {"decision": "finish", "final": "handled"},
    )
    graph = build_graph(llm=llm, registry=reg)
    state = await arun(graph, "do it")

    assert "error" in state["results"][1]  # error captured as data
    assert 2 not in state["results"]        # level-1 task never dispatched
    assert ran == []                        # early-abort skipped it
    assert state["final"] == "handled"
