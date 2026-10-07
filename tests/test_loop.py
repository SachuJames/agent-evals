"""Tests for the agent tool-use loop and runners."""

from agent_evals.loop import run_suite, run_task
from agent_evals.models import AgentAction, CheckSpec
from agent_evals.runners import AgentRunner, HttpRunner, MockRunner
from tests.conftest import make_task


def test_mock_runner_perfect_passes():
    task = make_task()
    result = run_task(task, MockRunner())
    assert result.passed
    assert result.score == 1.0
    assert result.final_answer == "42"


def test_multi_step_tool_loop():
    task = make_task(
        tools=["calculator", "kv_store"],
        mock_script=[
            {"tool": "kv_store", "args": {"op": "set", "key": "a", "value": "17"}},
            {"tool": "kv_store", "args": {"op": "set", "key": "b", "value": "25"}},
            {"tool": "calculator", "args": {"expression": "17 + 25"}},
            {"final": "42"},
        ],
    )
    result = run_task(task, MockRunner())
    assert result.passed
    tool_steps = [s for s in result.steps if s.tool_name]
    assert [s.tool_name for s in tool_steps] == ["kv_store", "kv_store", "calculator"]
    assert tool_steps[2].tool_result == "42"


def test_disallowed_tool_fails_task():
    task = make_task(
        tools=["calculator"],
        mock_script=[{"tool": "web_fetch", "args": {"url": "https://x"}}],
    )
    result = run_task(task, MockRunner())
    assert not result.passed
    assert "not allowed" in result.error


def test_max_steps_reached():
    task = make_task(
        tools=["calculator"],
        max_steps=2,
        mock_script=[
            {"tool": "calculator", "args": {"expression": "1+1"}},
            {"tool": "calculator", "args": {"expression": "2+2"}},
            {"tool": "calculator", "args": {"expression": "3+3"}},
        ],
    )
    result = run_task(task, MockRunner())
    assert not result.passed
    assert "max steps" in result.error


def test_flaky_mode_corrupts_fixed_tasks():
    task = make_task(
        task_id="math_pythagoras",
        mock_script=[{"final": "15"}],
        scoring=[CheckSpec(type="exact_match", expected="15")],
    )
    result = run_task(task, MockRunner(mode="flaky"))
    assert not result.passed
    assert result.final_answer == "14"


def test_flaky_mode_leaves_other_tasks_alone():
    task = make_task()
    result = run_task(task, MockRunner(mode="flaky"))
    assert result.passed


def test_runner_exception_ends_task():
    class Boom(AgentRunner):
        name = "boom"

        def next_action(self, task, messages, step):
            raise RuntimeError("kaput")

    result = run_task(make_task(), Boom())
    assert not result.passed
    assert "runner error" in result.error


def test_run_suite_aggregates():
    tasks = [make_task(f"t{i}") for i in range(3)]
    run = run_suite(tasks, MockRunner(), suite_name="s")
    assert run.total == 3
    assert run.passed == 3
    assert run.avg_score == 1.0
    assert run.suite == "s"
    assert run.runner == "mock"


def test_http_runner_needs_base_url(monkeypatch):
    monkeypatch.delenv("AGENT_EVALS_BASE_URL", raising=False)
    try:
        HttpRunner(base_url="")
    except ValueError as exc:
        assert "base URL" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_http_runner_parses_tool_call(monkeypatch):
    import agent_evals.runners as runners_mod

    payload = {
        "choices": [
            {
                "message": {
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "1",
                            "type": "function",
                            "function": {
                                "name": "calculator",
                                "arguments": '{"expression": "1+1"}',
                            },
                        }
                    ],
                }
            }
        ]
    }

    class FakeResp:
        def raise_for_status(self):
            pass

        def json(self):
            return payload

    monkeypatch.setattr(runners_mod.requests, "post", lambda *a, **k: FakeResp())
    runner = HttpRunner(base_url="https://llm.test/v1", api_key="k")
    action = runner.next_action(make_task(tools=["calculator"]), [], 0)
    assert isinstance(action, AgentAction)
    assert action.kind == "tool"
    assert action.tool_name == "calculator"
    assert action.tool_args == {"expression": "1+1"}
