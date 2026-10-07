"""Tests for the HTML report renderer."""

from agent_evals.models import StepRecord
from agent_evals.report import render_report


def _summary():
    return {
        "id": "abc123def456",
        "started_at": 1791000000.0,
        "suite": "builtin",
        "runner": "mock",
    }


def _tasks():
    return [
        {
            "task_id": "t1",
            "task_name": "Task one",
            "category": "math",
            "score": 1.0,
            "passed": 1,
            "final_answer": "42",
            "error": "",
        },
        {
            "task_id": "t2",
            "task_name": "Task two",
            "category": "math",
            "score": 0.0,
            "passed": 0,
            "final_answer": "nope",
            "error": "",
        },
    ]


def test_report_contains_summary_and_tasks():
    html = render_report(_summary(), _tasks())
    assert "agent-evals run report" in html
    assert "abc123def456" in html
    assert "t1" in html and "t2" in html
    assert "1/2" in html
    assert "PASS" in html and "FAIL" in html


def test_report_escapes_html():
    tasks = _tasks()
    tasks[0]["final_answer"] = "<script>alert(1)</script>"
    traces = {"t1": [StepRecord(seq=0, role="assistant", content="<b>hi</b>")], "t2": []}
    html = render_report(_summary(), tasks, traces)
    assert "<script>" not in html
    assert "&lt;b&gt;hi&lt;/b&gt;" in html


def test_report_with_diff():
    diff = {
        "run_a": "aaaa",
        "run_b": "bbbb",
        "per_task": [
            {
                "task_id": "t1",
                "task_name": "Task one",
                "score_a": 1.0,
                "score_b": 0.0,
                "delta": -1.0,
                "passed_a": True,
                "passed_b": False,
            }
        ],
        "improved": [],
        "regressed": ["t1"],
        "unchanged": [],
        "added": [],
        "removed": [],
        "avg_a": 1.0,
        "avg_b": 0.0,
        "avg_delta": -1.0,
    }
    html = render_report(_summary(), _tasks(), diff=diff)
    assert "Regression diff" in html
    assert "-1.00" in html


def test_report_is_standalone():
    html = render_report(_summary(), _tasks())
    assert html.startswith("<!DOCTYPE html>")
    assert "http" not in html.split("<style>")[1].split("</style>")[0]
