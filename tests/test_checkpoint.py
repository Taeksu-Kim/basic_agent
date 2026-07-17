"""Local SQLite checkpoint store + thread pruning (cap = N most recent threads).

A *thread* is one user request; each run writes several per-step checkpoints
under its thread_id. The cap counts threads, and pruning deletes a thread's
checkpoints wholesale (a partially-deleted thread could not be resumed).
"""

from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

from agent.core.checkpoint import list_threads, open_async_saver, open_saver, prune_threads


class _S(TypedDict, total=False):
    x: int


def _tiny_graph(saver):
    b = StateGraph(_S)
    b.add_node("bump", lambda s: {"x": s.get("x", 0) + 1})
    b.add_edge(START, "bump")
    b.add_edge("bump", END)
    return b.compile(checkpointer=saver)


def _run_threads(saver, ids):
    g = _tiny_graph(saver)
    for tid in ids:
        g.invoke({"x": 0}, {"configurable": {"thread_id": tid}})


def test_open_saver_persists_and_resumes_state(tmp_path):
    db = tmp_path / "ckpt.sqlite"
    saver = open_saver(db)
    _run_threads(saver, ["a"])
    # a fresh saver on the same file sees the stored state
    saver2 = open_saver(db)
    snap = _tiny_graph(saver2).get_state({"configurable": {"thread_id": "a"}})
    assert snap.values["x"] == 1


def test_list_threads_newest_first(tmp_path):
    saver = open_saver(tmp_path / "ckpt.sqlite")
    _run_threads(saver, ["a", "b", "c"])
    assert list_threads(saver) == ["c", "b", "a"]


def test_prune_keeps_most_recent_threads(tmp_path):
    saver = open_saver(tmp_path / "ckpt.sqlite")
    _run_threads(saver, [f"t{i}" for i in range(5)])

    deleted = prune_threads(saver, keep=3)

    assert deleted == ["t1", "t0"]  # the two oldest, oldest last
    assert list_threads(saver) == ["t4", "t3", "t2"]
    # surviving threads still fully resumable
    snap = _tiny_graph(saver).get_state({"configurable": {"thread_id": "t2"}})
    assert snap.values["x"] == 1
    # pruned thread leaves no rows behind in any table
    for table in ("checkpoints", "writes"):
        n = saver.conn.execute(
            f"SELECT COUNT(*) FROM {table} WHERE thread_id = ?", ("t0",)
        ).fetchone()[0]
        assert n == 0


async def test_async_saver_works_with_ainvoke(tmp_path):
    # the agent graphs run via ainvoke -- the saver must support the async path
    # (the sync SqliteSaver raises NotImplementedError there).
    db = tmp_path / "ckpt.sqlite"
    saver = await open_async_saver(db)
    g = _tiny_graph(saver)
    out = await g.ainvoke({"x": 0}, {"configurable": {"thread_id": "a"}})
    assert out["x"] == 1
    # same file is readable by the sync saver (shared pruning/inspection path)
    assert list_threads(open_saver(db)) == ["a"]


def test_prune_noop_under_cap(tmp_path):
    saver = open_saver(tmp_path / "ckpt.sqlite")
    _run_threads(saver, ["a", "b"])
    assert prune_threads(saver, keep=30) == []
    assert list_threads(saver) == ["b", "a"]
