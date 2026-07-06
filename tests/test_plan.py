import pytest

from agent.plan import (
    Plan,
    PlanError,
    Task,
    refs_in,
    substitute_args,
    topological_levels,
    validate_plan,
)


def _plan(*tasks: dict) -> Plan:
    return Plan.from_list(list(tasks))


def test_task_from_dict_and_roundtrip():
    t = Task.from_dict({"id": 1, "tool": "search", "args": {"q": "x"}, "deps": [0]})
    assert t.id == 1 and t.tool == "search" and t.deps == (0,)
    assert Task.from_dict(t.to_dict()) == t


def test_task_from_dict_requires_id_and_tool():
    with pytest.raises(PlanError):
        Task.from_dict({"tool": "search"})


def test_refs_in_scans_nested_structures():
    assert refs_in({"a": "$1", "b": ["hi $2", {"c": "$3"}]}) == {1, 2, 3}
    assert refs_in("no refs here") == set()


def test_effective_deps_merges_explicit_and_refs():
    t = Task.from_dict({"id": 5, "tool": "t", "args": {"x": "$2"}, "deps": [1]})
    assert t.effective_deps() == {1, 2}


# ---- topological levels ----------------------------------------------------


def test_levels_linear_chain():
    p = _plan(
        {"id": 1, "tool": "a"},
        {"id": 2, "tool": "b", "deps": [1]},
        {"id": 3, "tool": "c", "deps": [2]},
    )
    assert topological_levels(p) == [[1], [2], [3]]


def test_levels_parallel_then_join():
    # 1,2 parallel; 3 depends on both -> [[1,2],[3]]
    p = _plan(
        {"id": 1, "tool": "a"},
        {"id": 2, "tool": "b"},
        {"id": 3, "tool": "c", "args": {"x": "$1", "y": "$2"}},
    )
    assert topological_levels(p) == [[1, 2], [3]]


def test_levels_uneven_deps_place_task_at_earliest_level():
    # 3 depends only on 2, so it lands in level 1 next to nothing, not waiting on 1
    p = _plan(
        {"id": 1, "tool": "slow"},
        {"id": 2, "tool": "quick"},
        {"id": 3, "tool": "refine", "deps": [2]},
    )
    assert topological_levels(p) == [[1, 2], [3]]


def test_levels_within_level_sorted_for_determinism():
    p = _plan({"id": 3, "tool": "a"}, {"id": 1, "tool": "b"}, {"id": 2, "tool": "c"})
    assert topological_levels(p) == [[1, 2, 3]]


def test_cycle_raises():
    p = _plan(
        {"id": 1, "tool": "a", "deps": [2]},
        {"id": 2, "tool": "b", "deps": [1]},
    )
    with pytest.raises(PlanError, match="cycle"):
        topological_levels(p)


# ---- validation ------------------------------------------------------------


def test_validate_rejects_duplicate_ids():
    p = _plan({"id": 1, "tool": "a"}, {"id": 1, "tool": "b"})
    with pytest.raises(PlanError, match="duplicate"):
        validate_plan(p)


def test_validate_rejects_unknown_dep():
    p = _plan({"id": 1, "tool": "a", "deps": [9]})
    with pytest.raises(PlanError, match="unknown"):
        validate_plan(p)


def test_validate_rejects_ref_to_missing_task():
    p = _plan({"id": 1, "tool": "a", "args": {"x": "$7"}})
    with pytest.raises(PlanError, match="unknown"):
        validate_plan(p)


def test_validate_rejects_self_dependency():
    p = _plan({"id": 1, "tool": "a", "deps": [1]})
    with pytest.raises(PlanError, match="itself"):
        validate_plan(p)


def test_validate_accepts_valid_dag():
    p = _plan(
        {"id": 1, "tool": "a"},
        {"id": 2, "tool": "b", "args": {"x": "$1"}},
    )
    validate_plan(p)  # should not raise


# ---- $N substitution -------------------------------------------------------


def test_substitute_whole_string_preserves_type():
    assert substitute_args("$1", {1: [1, 2, 3]}) == [1, 2, 3]


def test_substitute_embedded_ref_stringifies():
    assert substitute_args("value is $1!", {1: 42}) == "value is 42!"


def test_substitute_nested_and_multiple():
    out = substitute_args({"a": "$1", "b": ["$2", "n=$1"]}, {1: 5, 2: "hi"})
    assert out == {"a": 5, "b": ["hi", "n=5"]}


def test_substitute_unresolved_raises():
    with pytest.raises(PlanError, match="unresolved"):
        substitute_args("$9", {1: "x"})
