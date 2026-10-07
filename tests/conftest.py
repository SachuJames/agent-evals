"""Shared fixtures for the agent-evals test suite."""

import pytest

from agent_evals.models import CheckSpec, TaskSpec
from agent_evals.store import RunStore


def make_task(task_id="t1", **overrides):
    fields = dict(
        id=task_id,
        name="Test task",
        category="test",
        prompt="Answer 42.",
        tools=[],
        max_steps=4,
        scoring=[CheckSpec(type="exact_match", expected="42")],
        pass_threshold=1.0,
        fetch_stubs={},
        mock_script=[{"final": "42"}],
    )
    fields.update(overrides)
    return TaskSpec(**fields)


@pytest.fixture()
def mem_store(tmp_path):
    store = RunStore(tmp_path / "runs.db")
    yield store
    store.close()
