"""Tests for run persistence, trace export, and regression diffs."""

import pytest

from agent_evals.loop import run_suite
from agent_evals.models import CheckSpec
from agent_evals.runners import MockRunner
from tests.conftest import make_task


def _run_with_scores(mem_store, scores):
    """Save a run whose tasks score exactly ``scores`` (list of 0.0/1.0)."""
    tasks = []
    for i, s in enumerate(scores):
        final = "42" if s == 1.0 else "wrong"
        tasks.append(
            make_task(
                task_id=f"t{i}",
                mock_script=[{"final": final}],
                scoring=[CheckSpec(type="exact_match", expected="42")],
            )
        )
    run = run_suite(tasks, MockRunner(), suite_name="s")
    mem_store.save_run(run)
    return run.run_id


def test_save_and_list_runs(mem_store):
    run_id = _run_with_scores(mem_store, [1.0, 1.0, 0.0])
    rows = mem_store.list_runs()
    assert len(rows) == 1
    assert rows[0]["id"] == run_id
    assert rows[0]["total"] == 3
    assert rows[0]["passed"] == 2
    assert rows[0]["avg_score"] == 2 / 3


def test_trace_roundtrip(mem_store):
    task = make_task(
        task_id="t1",
        tools=["calculator"],
        mock_script=[
            {"tool": "calculator", "args": {"expression": "1+1"}},
            {"final": "42"},
        ],
    )
    run = run_suite([task], MockRunner())
    mem_store.save_run(run)
    trace = mem_store.get_trace(run.run_id, "t1")
    assert len(trace) == 2
    assert trace[0].tool_name == "calculator"
    assert trace[0].tool_args == {"expression": "1+1"}
    assert trace[0].tool_result == "2"
    assert trace[1].role == "assistant"
    assert trace[1].content == "42"


def test_export_jsonl(mem_store, tmp_path):
    run_id = _run_with_scores(mem_store, [1.0])
    out = tmp_path / "trace.jsonl"
    n = mem_store.export_jsonl(run_id, out)
    assert n == 1
    import json

    row = json.loads(out.read_text().strip())
    assert row["run_id"] == run_id
    assert row["task_id"] == "t0"


def test_compare_runs_regression_and_improvement(mem_store):
    run_a = _run_with_scores(mem_store, [1.0, 1.0, 1.0])
    run_b = _run_with_scores(mem_store, [1.0, 0.0, 1.0])
    d = mem_store.compare_runs(run_a, run_b)
    assert d["regressed"] == ["t1"]
    assert d["improved"] == []
    assert sorted(d["unchanged"]) == ["t0", "t2"]
    assert d["avg_a"] == 1.0
    assert d["avg_b"] == pytest.approx(2 / 3, abs=1e-4)
    assert d["avg_delta"] < 0
    per = {p["task_id"]: p for p in d["per_task"]}
    assert per["t1"]["delta"] == -1.0
    assert per["t1"]["passed_a"] and not per["t1"]["passed_b"]


def test_compare_runs_added_and_removed(mem_store):
    run_a = _run_with_scores(mem_store, [1.0])
    run_b = _run_with_scores(mem_store, [1.0, 1.0])
    d = mem_store.compare_runs(run_a, run_b)
    assert d["added"] == ["t1"]
    assert d["removed"] == []
    d2 = mem_store.compare_runs(run_b, run_a)
    assert d2["removed"] == ["t1"]


def test_get_run_summary_unknown(mem_store):
    assert mem_store.get_run_summary("nope") is None
