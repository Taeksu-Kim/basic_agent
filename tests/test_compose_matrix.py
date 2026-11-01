"""Entry-point check: the whole composition matrix must pass.

The matrix (``examples/compose_matrix.py``) runs every (parent, child) pattern pair
and the single-pattern scenarios end to end with a FakeLLM, judging each against
what *should* happen (pauses, approvals, retries, failure handling). This test
only asserts the matrix has no failing row, and prints the failing rows if any.
"""

import asyncio

from agent.examples.compose_matrix import run_all


def test_compose_matrix_all_pass():
    report = asyncio.run(run_all())
    failed = [f"{r.scenario} [{r.cell}] {r.expect} -> {r.observed}" for r in report.rows if not r.ok]
    assert len(report.rows) >= 47
    assert not failed, "\n".join(failed)
