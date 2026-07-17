"""Local checkpoint persistence + thread-capped pruning.

HITL pauses (``interrupt()``) need a checkpointer to park the graph state and
resume it later. This module provides a local SQLite saver plus a pruner that
caps retention at the N most recent *threads*.

Granularity: LangGraph writes one checkpoint per super-step, so a single request
accumulates many checkpoints under one ``thread_id``. The cap counts *threads*
(one thread = one user request); pruning deletes a thread's checkpoints
wholesale — a partially-deleted thread could not be resumed.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Union

from langgraph.checkpoint.sqlite import SqliteSaver

DEFAULT_KEEP = 30


def open_saver(path: Union[str, Path]) -> SqliteSaver:
    """Open (creating if needed) a *sync* SQLite checkpointer at ``path``.

    Works with ``graph.invoke``; the agent graphs run via ``ainvoke``, which the
    sync saver refuses — use :func:`open_async_saver` for those. Both write the
    same schema, so this one remains the handle for pruning/inspection.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # check_same_thread=False: LangGraph may touch the saver from worker threads.
    saver = SqliteSaver(sqlite3.connect(str(path), check_same_thread=False))
    saver.setup()
    return saver


async def open_async_saver(path: Union[str, Path]):
    """Open (creating if needed) an *async* SQLite checkpointer at ``path``.

    This is the one to pass to the agent graphs (they run via ``ainvoke``).
    Lazily imports ``aiosqlite``.
    """
    import aiosqlite
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    saver = AsyncSqliteSaver(await aiosqlite.connect(str(path)))
    await saver.setup()
    return saver


def list_threads(saver: SqliteSaver) -> list[str]:
    """Thread ids, most recently checkpointed first.

    Ordered by each thread's latest checkpoint_id — a UUIDv6-style string that
    sorts lexicographically by creation time.
    """
    rows = saver.conn.execute(
        "SELECT thread_id, MAX(checkpoint_id) AS latest"
        " FROM checkpoints GROUP BY thread_id ORDER BY latest DESC"
    ).fetchall()
    return [r[0] for r in rows]


def prune_threads(saver: SqliteSaver, keep: int = DEFAULT_KEEP) -> list[str]:
    """Delete all but the ``keep`` most recent threads; return the deleted ids."""
    doomed = list_threads(saver)[keep:]
    if doomed:
        marks = ",".join("?" * len(doomed))
        with saver.conn:  # one transaction: never leave a half-deleted thread
            saver.conn.execute(f"DELETE FROM checkpoints WHERE thread_id IN ({marks})", doomed)
            saver.conn.execute(f"DELETE FROM writes WHERE thread_id IN ({marks})", doomed)
    return doomed
