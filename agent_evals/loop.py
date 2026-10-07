"""The agent tool-use loop: drive a runner, execute tools, record the trace."""

from __future__ import annotations

import time

from agent_evals.models import StepRecord, TaskResult, TaskSpec, now
from agent_evals.runners import AgentRunner
from agent_evals.scorers import score_task
from agent_evals.tools import REGISTRY, ToolContext


def _tool_prompt(task: TaskSpec) -> str:
    if not task.tools:
        return ""
    names = ", ".join(task.tools)
    return (
        f"You have these tools available: {names}. "
        "To use one, reply with a tool call. When you know the answer, "
        "reply with the final answer text and no tool call."
    )


def run_task(
    task: TaskSpec,
    runner: AgentRunner,
    judge=None,
    trace: list[StepRecord] | None = None,
) -> TaskResult:
    """Run one task through the agent loop and score the final answer."""
    ctx = ToolContext(fetch_stubs=task.fetch_stubs)
    steps: list[StepRecord] = trace if trace is not None else []
    messages: list[dict] = [
        {"role": "system", "content": "You are an agent being evaluated. " + _tool_prompt(task)},
        {"role": "user", "content": task.prompt},
    ]

    final_answer = ""
    error = ""
    seq = 0

    for step in range(task.max_steps):
        started = time.perf_counter()
        try:
            action = runner.next_action(task, messages, step)
        except Exception as exc:  # noqa: BLE001 - runner failures end the task
            error = f"runner error: {exc}"
            break
        latency_ms = (time.perf_counter() - started) * 1000

        if action.kind == "final":
            final_answer = action.text
            steps.append(
                StepRecord(seq=seq, role="assistant", content=final_answer, latency_ms=latency_ms)
            )
            messages.append({"role": "assistant", "content": final_answer})
            break

        # Tool call path.
        tool = REGISTRY.get(action.tool_name)
        if tool is None or action.tool_name not in task.tools:
            error = f"tool not allowed: {action.tool_name!r}"
            steps.append(
                StepRecord(
                    seq=seq,
                    role="assistant",
                    content=f"[tried to call disallowed tool {action.tool_name}]",
                    latency_ms=latency_ms,
                )
            )
            break
        try:
            result = tool.execute(action.tool_args, ctx)
        except Exception as exc:  # noqa: BLE001 - tool failures are fed back, not fatal
            result = f"error: {exc}"
        steps.append(
            StepRecord(
                seq=seq,
                role="assistant",
                content=f"[tool call {action.tool_name}]",
                tool_name=action.tool_name,
                tool_args=dict(action.tool_args),
                tool_result=result,
                latency_ms=latency_ms,
            )
        )
        seq += 1
        messages.append(
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": f"call_{seq}",
                        "type": "function",
                        "function": {
                            "name": action.tool_name,
                            "arguments": action.tool_args,
                        },
                    }
                ],
            }
        )
        messages.append({"role": "tool", "tool_call_id": f"call_{seq}", "content": result})
    else:
        error = error or "max steps reached without a final answer"

    score, checks = score_task(task, final_answer, judge=judge)
    return TaskResult(
        task_id=task.id,
        task_name=task.name,
        category=task.category,
        score=score,
        passed=bool(score >= task.pass_threshold and not error),
        final_answer=final_answer,
        checks=checks,
        steps=steps,
        error=error,
    )


def run_suite(
    tasks: list[TaskSpec],
    runner: AgentRunner,
    suite_name: str = "builtin",
    judge=None,
) -> "RunResult":
    """Run every task in a suite with the same runner."""
    from agent_evals.models import RunResult, new_run_id

    run = RunResult(run_id=new_run_id(), suite=suite_name, runner=runner.name, started_at=now())
    for task in tasks:
        run.task_results.append(run_task(task, runner, judge=judge))
    return run
