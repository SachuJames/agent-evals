"""agent-evals: eval harness for AI agents.

Task suites in YAML, pluggable agent runners, a tool-use loop,
scorers, trace capture in SQLite, and run-over-run regression tracking.
"""

from agent_evals.loop import run_suite, run_task
from agent_evals.models import AgentAction, CheckSpec, RunResult, StepRecord, TaskResult, TaskSpec
from agent_evals.runners import AgentRunner, HttpRunner, MockRunner
from agent_evals.scorers import StubJudge, score_check, score_task
from agent_evals.store import RunStore
from agent_evals.tasks import builtin_tasks, load_suite

__all__ = [
    "AgentAction",
    "AgentRunner",
    "CheckSpec",
    "HttpRunner",
    "MockRunner",
    "RunResult",
    "RunStore",
    "StepRecord",
    "StubJudge",
    "TaskResult",
    "TaskSpec",
    "builtin_tasks",
    "load_suite",
    "run_suite",
    "run_task",
    "score_check",
    "score_task",
]

__version__ = "0.1.0"
