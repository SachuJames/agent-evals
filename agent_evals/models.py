"""Core data models for the eval harness."""

from __future__ import annotations

import dataclasses
import time
import uuid


@dataclasses.dataclass
class CheckSpec:
    """One scoring check inside a task."""

    type: str  # exact_match | contains | regex | json_schema | llm_judge | python
    weight: float = 1.0
    expected: str | list[str] | None = None
    pattern: str | None = None
    schema: dict | None = None
    rubric: str | None = None
    function: str | None = None  # dotted path for type=python


@dataclasses.dataclass
class TaskSpec:
    """A single eval task."""

    id: str
    name: str
    category: str
    prompt: str
    tools: list[str]
    max_steps: int = 8
    scoring: list[CheckSpec] = dataclasses.field(default_factory=list)
    pass_threshold: float = 1.0
    fetch_stubs: dict[str, str] = dataclasses.field(default_factory=dict)
    mock_script: list[dict] = dataclasses.field(default_factory=list)


@dataclasses.dataclass
class AgentAction:
    """What the agent decided to do on one loop step."""

    kind: str  # "final" | "tool"
    text: str = ""
    tool_name: str = ""
    tool_args: dict = dataclasses.field(default_factory=dict)


@dataclasses.dataclass
class StepRecord:
    """One recorded step of the agent loop."""

    seq: int
    role: str  # assistant | tool
    content: str
    tool_name: str = ""
    tool_args: dict = dataclasses.field(default_factory=dict)
    tool_result: str = ""
    latency_ms: float = 0.0


@dataclasses.dataclass
class TaskResult:
    """Outcome of running one task."""

    task_id: str
    task_name: str
    category: str
    score: float
    passed: bool
    final_answer: str
    checks: list[dict] = dataclasses.field(default_factory=list)
    steps: list[StepRecord] = dataclasses.field(default_factory=list)
    error: str = ""


@dataclasses.dataclass
class RunResult:
    """Outcome of running a whole suite."""

    run_id: str
    suite: str
    runner: str
    started_at: float
    task_results: list[TaskResult] = dataclasses.field(default_factory=list)

    @property
    def total(self) -> int:
        return len(self.task_results)

    @property
    def passed(self) -> int:
        return sum(1 for r in self.task_results if r.passed)

    @property
    def avg_score(self) -> float:
        if not self.task_results:
            return 0.0
        return sum(r.score for r in self.task_results) / len(self.task_results)


def new_run_id() -> str:
    return uuid.uuid4().hex[:12]


def now() -> float:
    return time.time()
