"""HITL (human-in-the-loop) approval gates.

Both graphs pause via ``interrupt()`` (which requires a checkpointer) and resume
with ``aresume``. The verdict contract, shared by both gates:

* approve  -- ``True`` | ``"approve"`` | ``{"approved": True}``
* edit     -- plan gate only: ``{"plan": [...]}`` replaces the plan (validated)
* reject   -- anything else; ``{"feedback": ...}`` / str carries the reason
"""

from langgraph.checkpoint.memory import InMemorySaver

from agent.core.llm import FakeLLM
from agent.core.tools import ToolRegistry
from agent.plan_execute.graph import arun as pe_arun, aresume as pe_aresume, build_graph
from agent.react.graph import arun as react_arun, aresume as react_aresume, build_react_agent


def _interrupt_payload(state) -> dict:
    (intr,) = state["__interrupt__"]  # exactly one pending interrupt
    return intr.value


# ---- react: per-tool approval gate -----------------------------------------


def _react_registry(calls: list):
    reg = ToolRegistry()
    reg.register("lookup", lambda q: f"result:{q}")
    reg.register(
        "wipe",
        lambda target: calls.append(target) or "wiped",
        description="destructive",
        requires_approval=True,
    )
    return reg


async def test_react_flagged_tool_pauses_then_approve_runs_it():
    calls: list = []
    llm = FakeLLM.json(
        {"action": "tool", "tool": "wipe", "args": {"target": "db"}},
        {"action": "finish", "final": "done"},
    )
    graph = build_react_agent(llm=llm, registry=_react_registry(calls), checkpointer=InMemorySaver())

    state = await react_arun(graph, "clean up", thread_id="t1")
    payload = _interrupt_payload(state)
    assert payload["type"] == "tool_approval"
    assert payload["tool"] == "wipe" and payload["args"] == {"target": "db"}
    assert calls == []  # tool did NOT run while paused

    state = await react_aresume(graph, True, thread_id="t1")
    assert calls == ["db"]  # ran after approval
    assert state["scratchpad"][0]["observation"] == "wiped"
    assert state["final"] == "done"


async def test_react_deny_feeds_denial_as_observation():
    calls: list = []
    llm = FakeLLM.json(
        {"action": "tool", "tool": "wipe", "args": {"target": "db"}},
        {"action": "finish", "final": "aborted"},
    )
    graph = build_react_agent(llm=llm, registry=_react_registry(calls), checkpointer=InMemorySaver())

    await react_arun(graph, "clean up", thread_id="t1")
    state = await react_aresume(graph, {"approved": False, "feedback": "too risky"}, thread_id="t1")

    assert calls == []  # denied -> never executed
    obs = state["scratchpad"][0]["observation"]
    assert "denied" in obs and "too risky" in obs["denied"]
    assert state["final"] == "aborted"  # loop continued past the denial


async def test_react_unflagged_tool_never_pauses():
    llm = FakeLLM.json(
        {"action": "tool", "tool": "lookup", "args": {"q": "cats"}},
        {"action": "finish", "final": "ok"},
    )
    graph = build_react_agent(llm=llm, registry=_react_registry([]), checkpointer=InMemorySaver())
    state = await react_arun(graph, "q", thread_id="t1")
    assert "__interrupt__" not in state and state["final"] == "ok"


# ---- plan_execute: plan approval gate ---------------------------------------


def _pe_registry():
    reg = ToolRegistry()
    reg.register("gen", lambda n: f"g{n}")
    return reg


async def test_plan_gate_pauses_then_approve_executes():
    llm = FakeLLM.json(
        {"tasks": [{"id": 1, "tool": "gen", "args": {"n": 1}}]},
        {"decision": "finish", "final": "ok"},
    )
    graph = build_graph(
        llm=llm, registry=_pe_registry(), approve_plans=True, checkpointer=InMemorySaver()
    )

    state = await pe_arun(graph, "do it", thread_id="t1")
    payload = _interrupt_payload(state)
    assert payload["type"] == "plan_approval"
    assert payload["plan"][0]["tool"] == "gen"
    assert "results" not in state or not state["results"]  # nothing ran yet

    state = await pe_aresume(graph, "approve", thread_id="t1")
    assert state["results"][1] == "g1"
    assert state["final"] == "ok"


async def test_plan_gate_edit_replaces_plan():
    llm = FakeLLM.json(
        {"tasks": [{"id": 1, "tool": "gen", "args": {"n": 1}}]},
        {"decision": "finish", "final": "ok"},
    )
    graph = build_graph(
        llm=llm, registry=_pe_registry(), approve_plans=True, checkpointer=InMemorySaver()
    )

    await pe_arun(graph, "do it", thread_id="t1")
    edited = {"plan": [{"id": 1, "tool": "gen", "args": {"n": 42}}]}
    state = await pe_aresume(graph, edited, thread_id="t1")

    assert state["results"][1] == "g42"  # the human's plan ran, not the LLM's
    assert state["final"] == "ok"


async def test_plan_gate_reject_feeds_feedback_into_replan():
    llm = FakeLLM.json(
        {"tasks": [{"id": 1, "tool": "gen", "args": {"n": 1}}]},
        {"tasks": [{"id": 1, "tool": "gen", "args": {"n": 2}}]},  # replan after reject
        {"decision": "finish", "final": "ok"},
    )
    graph = build_graph(
        llm=llm, registry=_pe_registry(), approve_plans=True, checkpointer=InMemorySaver()
    )

    await pe_arun(graph, "do it", thread_id="t1")
    state = await pe_aresume(graph, {"approved": False, "feedback": "use n=2"}, thread_id="t1")

    # second interrupt: the replanned plan comes back for approval
    payload = _interrupt_payload(state)
    assert payload["plan"][0]["args"] == {"n": 2}
    # the replan prompt carried the human feedback
    assert "use n=2" in llm.calls[1][1]

    state = await pe_aresume(graph, True, thread_id="t1")
    assert state["results"][1] == "g2"
    assert state["final"] == "ok"


async def test_level_gate_pauses_at_each_barrier_with_level_results():
    llm = FakeLLM.json(
        {
            "tasks": [
                {"id": 1, "tool": "gen", "args": {"n": 1}},
                {"id": 2, "tool": "gen", "args": {"n": 2}},
                {"id": 3, "tool": "gen", "args": {"n": "$1"}},  # level 2
            ]
        },
        {"decision": "finish", "final": "ok"},
    )
    graph = build_graph(
        llm=llm, registry=_pe_registry(), approve_levels=True, checkpointer=InMemorySaver()
    )

    state = await pe_arun(graph, "do it", thread_id="t1")
    payload = _interrupt_payload(state)
    assert payload["type"] == "level_review"
    assert payload["completed_level"] == 0
    assert payload["level_results"] == {1: "g1", 2: "g2"}  # the whole parallel level
    assert payload["next_level"] == [3]

    state = await pe_aresume(graph, True, thread_id="t1")
    payload = _interrupt_payload(state)  # second barrier
    assert payload["completed_level"] == 1
    assert payload["level_results"] == {3: "gg1"}
    assert payload["next_level"] is None

    state = await pe_aresume(graph, True, thread_id="t1")
    assert state["final"] == "ok"


async def test_level_gate_reject_replans_with_feedback():
    llm = FakeLLM.json(
        {"tasks": [{"id": 1, "tool": "gen", "args": {"n": 1}}]},
        {"tasks": [{"id": 1, "tool": "gen", "args": {"n": 2}}]},  # replan after reject
        {"decision": "finish", "final": "ok"},
    )
    graph = build_graph(
        llm=llm, registry=_pe_registry(), approve_levels=True, checkpointer=InMemorySaver()
    )

    await pe_arun(graph, "do it", thread_id="t1")
    state = await pe_aresume(graph, {"approved": False, "feedback": "wrong, use n=2"}, thread_id="t1")

    assert "wrong, use n=2" in llm.calls[1][1]  # replan prompt carried the feedback
    payload = _interrupt_payload(state)  # new plan ran; its level comes up for review
    assert payload["level_results"] == {1: "g2"}

    state = await pe_aresume(graph, True, thread_id="t1")
    assert state["results"][1] == "g2"
    assert state["final"] == "ok"


# ---- ask_human: model-initiated question ------------------------------------


async def test_ask_human_pauses_and_answer_becomes_observation():
    from agent.core.hitl import ask_human

    reg = ToolRegistry()
    reg.register("ask_human", ask_human, description="ask the user a question")
    llm = FakeLLM.json(
        {"action": "tool", "tool": "ask_human", "args": {"question": "continue?", "options": ["yes", "no"]}},
        {"action": "finish", "final": "user said yes"},
    )
    graph = build_react_agent(llm=llm, registry=reg, checkpointer=InMemorySaver())

    state = await react_arun(graph, "q", thread_id="t1")
    payload = _interrupt_payload(state)
    assert payload["type"] == "question"
    assert payload["question"] == "continue?" and payload["options"] == ["yes", "no"]

    state = await react_aresume(graph, "yes", thread_id="t1")
    assert state["scratchpad"][0]["observation"] == "yes"  # answer = tool return value
    assert state["final"] == "user said yes"


async def test_plan_gate_off_by_default_no_pause():
    llm = FakeLLM.json(
        {"tasks": [{"id": 1, "tool": "gen", "args": {"n": 1}}]},
        {"decision": "finish", "final": "ok"},
    )
    graph = build_graph(llm=llm, registry=_pe_registry(), checkpointer=InMemorySaver())
    state = await pe_arun(graph, "do it", thread_id="t1")
    assert "__interrupt__" not in state and state["final"] == "ok"
