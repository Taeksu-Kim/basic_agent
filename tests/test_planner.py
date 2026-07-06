import pytest

from agent.llm import FakeLLM
from agent.orchestrator import planner
from agent.plan import PlanError
from agent.tools import ToolRegistry


def _registry(*names):
    reg = ToolRegistry()
    for n in names:
        reg.register(n, lambda *a, **k: None)
    return reg


def test_parse_plan_from_tasks_object():
    text = '{"tasks": [{"id": 1, "tool": "search", "args": {"q": "x"}}]}'
    p = parse = planner.parse_plan(text)
    assert len(p) == 1 and p.tasks[0].tool == "search"


def test_parse_plan_from_bare_list():
    p = planner.parse_plan('[{"id": 1, "tool": "a"}]')
    assert len(p) == 1


def test_parse_plan_tolerates_surrounding_text():
    p = planner.parse_plan('here you go: {"tasks": [{"id": 1, "tool": "a"}]} done')
    assert len(p) == 1


def test_parse_plan_validates_dag():
    with pytest.raises(PlanError, match="unknown"):
        planner.parse_plan('{"tasks": [{"id": 1, "tool": "a", "deps": [9]}]}')


def test_plan_node_returns_levels_and_resets_cursor():
    llm = FakeLLM.json(
        {
            "tasks": [
                {"id": 1, "tool": "search"},
                {"id": 2, "tool": "search"},
                {"id": 3, "tool": "summarize", "args": {"x": "$1", "y": "$2"}},
            ]
        }
    )
    out = planner.plan({"query": "do it", "iteration": 0}, llm, _registry("search", "summarize"))
    assert out["levels"] == [[1, 2], [3]]
    assert out["current_level"] == 0
    assert out["iteration"] == 1
    assert len(out["plan"]) == 3


def test_plan_node_passes_schema_and_tools_to_llm():
    llm = FakeLLM.json({"tasks": [{"id": 1, "tool": "search"}]})
    planner.plan({"query": "q", "iteration": 0}, llm, _registry("search", "fetch"))
    system, user, schema = llm.calls[0]
    assert schema == planner.PLAN_SCHEMA
    assert "search" in user and "fetch" in user  # tools advertised to the model


def test_plan_node_advertises_tool_descriptions():
    reg = ToolRegistry()
    reg.register("search", lambda q: q, description="find articles")
    llm = FakeLLM.json({"tasks": [{"id": 1, "tool": "search"}]})
    planner.plan({"query": "q", "iteration": 0}, llm, reg)
    _, user, _ = llm.calls[0]
    assert "search" in user and "find articles" in user


def test_plan_node_increments_iteration_on_replan():
    llm = FakeLLM.json({"tasks": [{"id": 1, "tool": "search"}]})
    out = planner.plan({"query": "q", "iteration": 2}, llm, _registry("search"))
    assert out["iteration"] == 3
