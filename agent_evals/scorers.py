"""Scorers: turn a task's final answer into a 0.0-1.0 score per check."""

from __future__ import annotations

import abc
import importlib
import json
import re

import jsonschema

from agent_evals.models import CheckSpec, TaskSpec


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip())


def exact_match(answer: str, expected: str) -> float:
    return 1.0 if _normalize(answer) == _normalize(expected) else 0.0


def contains(answer: str, expected: str | list[str]) -> float:
    needles = [expected] if isinstance(expected, str) else list(expected)
    return 1.0 if all(n in answer for n in needles) else 0.0


def regex_match(answer: str, pattern: str) -> float:
    return 1.0 if re.search(pattern, answer.strip(), re.DOTALL) else 0.0


def json_schema_match(answer: str, schema: dict) -> float:
    try:
        payload = json.loads(answer.strip())
    except (json.JSONDecodeError, ValueError):
        return 0.0
    try:
        jsonschema.validate(payload, schema)
    except jsonschema.ValidationError:
        return 0.0
    return 1.0


def python_scorer(answer: str, dotted_path: str, task: TaskSpec) -> float:
    """Call a custom scorer at ``module:function`` returning 0.0-1.0."""
    module_name, _, func_name = dotted_path.rpartition(".")
    if not module_name or not func_name:
        raise ValueError(f"bad scorer path: {dotted_path!r}")
    func = getattr(importlib.import_module(module_name), func_name)
    return float(func(answer, task))


class Judge(abc.ABC):
    """LLM-as-judge interface: score an answer against a rubric."""

    @abc.abstractmethod
    def judge(self, task: TaskSpec, answer: str, rubric: str) -> tuple[float, str]:
        """Return (score 0.0-1.0, short rationale)."""


class StubJudge(Judge):
    """Deterministic offline judge.

    Scores 1.0 when every ``must_contain:`` keyword in the rubric appears in
    the answer (case-insensitive), else 0.0. Rubrics without keywords always
    pass. This keeps the demo and CI fully offline.
    """

    def judge(self, task: TaskSpec, answer: str, rubric: str) -> tuple[float, str]:
        keywords = [
            line.split(":", 1)[1].strip().lower()
            for line in rubric.splitlines()
            if line.strip().lower().startswith("must_contain:")
        ]
        lowered = answer.lower()
        missing = [k for k in keywords if k and k not in lowered]
        if missing:
            return 0.0, f"missing keywords: {', '.join(missing)}"
        return 1.0, "all rubric keywords present"


def score_check(
    check: CheckSpec, answer: str, task: TaskSpec, judge: Judge | None
) -> tuple[float, str]:
    """Score one check. Returns (score, detail string)."""
    t = check.type
    if t == "exact_match":
        s = exact_match(answer, str(check.expected or ""))
        return s, f"expected {check.expected!r}"
    if t == "contains":
        s = contains(answer, check.expected or "")
        return s, f"expected to contain {check.expected!r}"
    if t == "regex":
        s = regex_match(answer, check.pattern or "")
        return s, f"pattern {check.pattern!r}"
    if t == "json_schema":
        s = json_schema_match(answer, check.schema or {})
        return s, "validates against schema" if s else "schema validation failed"
    if t == "llm_judge":
        if judge is None:
            return 0.0, "no judge configured"
        s, rationale = judge.judge(task, answer, check.rubric or "")
        return s, rationale
    if t == "python":
        s = python_scorer(answer, check.function or "", task)
        return s, f"custom scorer {check.function}"
    raise ValueError(f"unknown check type: {t!r}")


def score_task(task: TaskSpec, answer: str, judge: Judge | None = None) -> tuple[float, list[dict]]:
    """Score every check in a task; return (weighted average, per-check details)."""
    details: list[dict] = []
    total_weight = 0.0
    weighted = 0.0
    for check in task.scoring:
        s, detail = score_check(check, answer, task, judge)
        details.append({"type": check.type, "score": s, "weight": check.weight, "detail": detail})
        weighted += s * check.weight
        total_weight += check.weight
    score = weighted / total_weight if total_weight else 0.0
    return score, details
