"""Loading and validating YAML task suites."""

from __future__ import annotations

from pathlib import Path

import yaml

from agent_evals.models import CheckSpec, TaskSpec

BUILTIN_DIR = Path(__file__).parent / "builtin_tasks"

_REQUIRED_TASK_FIELDS = ("id", "name", "category", "prompt", "scoring")


def _load_check(raw: dict) -> CheckSpec:
    if "type" not in raw:
        raise ValueError("scoring check is missing 'type'")
    return CheckSpec(
        type=raw["type"],
        weight=float(raw.get("weight", 1.0)),
        expected=raw.get("expected"),
        pattern=raw.get("pattern"),
        schema=raw.get("schema"),
        rubric=raw.get("rubric"),
        function=raw.get("function"),
    )


def load_task_file(path: str | Path) -> TaskSpec:
    """Load and validate a single task YAML file."""
    path = Path(path)
    with open(path) as fh:
        raw = yaml.safe_load(fh)
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: task file must contain a mapping")
    missing = [f for f in _REQUIRED_TASK_FIELDS if f not in raw]
    if missing:
        raise ValueError(f"{path}: missing required fields: {', '.join(missing)}")
    if not raw["scoring"]:
        raise ValueError(f"{path}: 'scoring' must list at least one check")
    return TaskSpec(
        id=str(raw["id"]),
        name=str(raw["name"]),
        category=str(raw["category"]),
        prompt=str(raw["prompt"]),
        tools=list(raw.get("tools", [])),
        max_steps=int(raw.get("max_steps", 8)),
        scoring=[_load_check(c) for c in raw["scoring"]],
        pass_threshold=float(raw.get("pass_threshold", 1.0)),
        fetch_stubs=dict(raw.get("fetch_stubs", {}) or {}),
        mock_script=list(raw.get("mock_script", []) or []),
    )


def load_suite(directory: str | Path) -> list[TaskSpec]:
    """Load every *.yaml task in a directory, sorted by task id."""
    directory = Path(directory)
    tasks = [load_task_file(p) for p in sorted(directory.glob("*.yaml"))]
    ids = [t.id for t in tasks]
    if len(set(ids)) != len(ids):
        dupes = sorted({i for i in ids if ids.count(i) > 1})
        raise ValueError(f"duplicate task ids: {', '.join(dupes)}")
    return tasks


def builtin_tasks() -> list[TaskSpec]:
    """Load the tasks shipped with the harness."""
    return load_suite(BUILTIN_DIR)
