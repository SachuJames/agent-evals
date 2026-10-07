"""Tests for scorer correctness."""

import pytest

from agent_evals.models import CheckSpec
from agent_evals.scorers import (
    StubJudge,
    contains,
    exact_match,
    json_schema_match,
    regex_match,
    score_task,
)
from tests.conftest import make_task


def test_exact_match_pass():
    assert exact_match("42", "42") == 1.0


def test_exact_match_normalizes_whitespace():
    assert exact_match("  42\n", "42") == 1.0


def test_exact_match_fail():
    assert exact_match("43", "42") == 0.0


def test_contains_single_and_list():
    assert contains("the cat sat", "cat") == 1.0
    assert contains("the cat sat", ["cat", "sat"]) == 1.0
    assert contains("the cat sat", ["cat", "dog"]) == 0.0


def test_regex_match():
    assert regex_match("- a\n- b", r"^(- .+\n)+- .+$") == 1.0
    assert regex_match("no bullets", r"^(- .+\n)+- .+$") == 0.0


def test_json_schema_match_valid():
    schema = {"type": "object", "required": ["a"], "properties": {"a": {"type": "integer"}}}
    assert json_schema_match('{"a": 1}', schema) == 1.0


def test_json_schema_match_invalid_json():
    assert json_schema_match("not json", {"type": "object"}) == 0.0


def test_json_schema_match_schema_violation():
    schema = {"type": "object", "required": ["a"], "properties": {"a": {"type": "integer"}}}
    assert json_schema_match('{"a": "x"}', schema) == 0.0


def test_weighted_average():
    task = make_task(
        scoring=[
            CheckSpec(type="exact_match", expected="42", weight=3.0),
            CheckSpec(type="contains", expected="nope", weight=1.0),
        ]
    )
    score, details = score_task(task, "42")
    assert score == pytest.approx(0.75)
    assert [d["type"] for d in details] == ["exact_match", "contains"]


def test_unknown_check_type_raises():
    task = make_task(scoring=[CheckSpec(type="bogus")])
    with pytest.raises(ValueError, match="unknown check type"):
        score_task(task, "42")


def test_stub_judge_keywords():
    task = make_task()
    judge = StubJudge()
    s, _ = judge.judge(task, "the answer mentions Paris", "must_contain: paris")
    assert s == 1.0
    s, rationale = judge.judge(task, "the answer mentions London", "must_contain: paris")
    assert s == 0.0
    assert "paris" in rationale


def test_stub_judge_no_keywords_always_passes():
    task = make_task()
    s, _ = StubJudge().judge(task, "anything", "be helpful")
    assert s == 1.0


def test_llm_judge_check_uses_judge():
    task = make_task(scoring=[CheckSpec(type="llm_judge", rubric="must_contain: 42")])
    score, _ = score_task(task, "the answer is 42", judge=StubJudge())
    assert score == 1.0


def test_llm_judge_without_judge_scores_zero():
    task = make_task(scoring=[CheckSpec(type="llm_judge", rubric="must_contain: 42")])
    score, details = score_task(task, "the answer is 42", judge=None)
    assert score == 0.0
    assert details[0]["detail"] == "no judge configured"


def test_python_scorer(tmp_path):
    mod = tmp_path / "myscorers.py"
    mod.write_text("def is_42(answer, task):\n    return 1.0 if answer.strip() == '42' else 0.0\n")
    import sys

    sys.path.insert(0, str(tmp_path))
    try:
        task = make_task(scoring=[CheckSpec(type="python", function="myscorers.is_42")])
        score, _ = score_task(task, "42")
        assert score == 1.0
        score, _ = score_task(task, "43")
        assert score == 0.0
    finally:
        sys.path.remove(str(tmp_path))
