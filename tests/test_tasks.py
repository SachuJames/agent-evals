"""Tests for task loading and the built-in suite."""

import pytest

from agent_evals.loop import run_task
from agent_evals.runners import MockRunner
from agent_evals.tasks import builtin_tasks, load_suite, load_task_file


def test_builtin_suite_loads_ten_tasks():
    tasks = builtin_tasks()
    assert len(tasks) == 10
    assert len({t.id for t in tasks}) == 10


def test_builtin_tasks_have_required_fields():
    for t in builtin_tasks():
        assert t.id and t.name and t.category and t.prompt
        assert t.scoring
        assert t.max_steps >= 1
        assert t.pass_threshold > 0


def test_builtin_mock_scripts_use_known_tools():
    from agent_evals.tools import REGISTRY

    for t in builtin_tasks():
        for step in t.mock_script:
            if "tool" in step:
                assert step["tool"] in REGISTRY, f"{t.id}: unknown tool {step['tool']}"
                assert step["tool"] in t.tools, f"{t.id}: tool not whitelisted"


def test_all_builtin_tasks_pass_with_mock():
    failures = []
    for t in builtin_tasks():
        result = run_task(t, MockRunner())
        if not result.passed:
            failures.append((t.id, result.error, result.final_answer))
    assert not failures, f"mock should pass every builtin task: {failures}"


def test_flaky_mode_fails_exactly_two_builtin_tasks():
    failed = [t.id for t in builtin_tasks() if not run_task(t, MockRunner(mode="flaky")).passed]
    assert sorted(failed) == ["format_three_bullets", "math_pythagoras"]


def test_load_task_file_missing_field(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text("id: x\nname: y\n")
    with pytest.raises(ValueError, match="missing required fields"):
        load_task_file(p)


def test_load_suite_rejects_duplicate_ids(tmp_path):
    for name in ("a.yaml", "b.yaml"):
        (tmp_path / name).write_text(
            "id: dup\nname: n\ncategory: c\nprompt: p\n"
            "scoring:\n  - type: contains\n    expected: x\n"
        )
    with pytest.raises(ValueError, match="duplicate task ids"):
        load_suite(tmp_path)
