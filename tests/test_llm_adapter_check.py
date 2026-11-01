"""Entry-point check: LLM adapters conform offline (mock HTTP, no model, no network).

Skipped when the optional any-llm dependency isn't installed.
"""

import pytest

pytest.importorskip("any_llm")


def test_llm_adapter_check_all_pass(capsys):
    from agent.examples import llm_adapter_check as chk

    chk.rows.clear()
    chk.main()
    failed = [r for r in chk.rows if not r[2]]
    assert len(chk.rows) >= 17
    assert not failed, "\n".join(f"{p} {n}: {d}" for p, n, _, d in failed)
