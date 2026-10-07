"""Pluggable agent runners.

A runner decides the agent's next action given the conversation so far.
MockRunner is deterministic and scripted (used by CI, tests, and the demo).
HttpRunner speaks the OpenAI chat-completions format against any compatible
endpoint, so real models plug in through environment configuration.
"""

from __future__ import annotations

import abc
import copy
import json
import os

import requests

from agent_evals.models import AgentAction, TaskSpec


class AgentRunner(abc.ABC):
    """Decides the agent's next action."""

    name: str = "base"

    @abc.abstractmethod
    def next_action(self, task: TaskSpec, messages: list[dict], step: int) -> AgentAction:
        """Return the agent's next action (a tool call or a final answer)."""


class MockRunner(AgentRunner):
    """Deterministic scripted runner.

    Each task's YAML may define ``mock_script``: a list of steps where each
    step is either ``{"tool": name, "args": {...}}`` or ``{"final": text}``.

    ``mode="flaky"`` deterministically corrupts a fixed set of tasks so that
    run-vs-run diffs have something to show. It never touches randomness.
    """

    name = "mock"

    # task_id -> replacement final answer used in flaky mode
    FLAKY_ANSWERS = {
        "math_pythagoras": "14",
        "format_three_bullets": "- Mars\n- Venus\n- Jupiter\n- Saturn",
    }

    def __init__(self, mode: str = "perfect"):
        if mode not in ("perfect", "flaky"):
            raise ValueError("mode must be 'perfect' or 'flaky'")
        self.mode = mode
        self.name = f"mock-{mode}" if mode != "perfect" else "mock"

    def next_action(self, task: TaskSpec, messages: list[dict], step: int) -> AgentAction:
        script = copy.deepcopy(task.mock_script)
        if not script:
            return AgentAction(kind="final", text="")
        if step >= len(script):
            # Script exhausted: close out with the last final we saw, if any.
            finals = [s.get("final", "") for s in script if "final" in s]
            return AgentAction(kind="final", text=finals[-1] if finals else "")
        entry = script[step]
        if "tool" in entry:
            return AgentAction(
                kind="tool",
                tool_name=str(entry["tool"]),
                tool_args=dict(entry.get("args", {})),
            )
        text = str(entry.get("final", ""))
        if self.mode == "flaky" and task.id in self.FLAKY_ANSWERS:
            text = self.FLAKY_ANSWERS[task.id]
        return AgentAction(kind="final", text=text)


class HttpRunner(AgentRunner):
    """Runner for any OpenAI chat-completions compatible endpoint.

    Configure with env vars (or constructor args):
      AGENT_EVALS_BASE_URL  e.g. https://api.openai.com/v1  (required)
      AGENT_EVALS_API_KEY   bearer token (optional)
      AGENT_EVALS_MODEL     model name (default: gpt-4o-mini)
    """

    name = "http"

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        timeout: float = 120.0,
    ):
        self.base_url = (base_url or os.environ.get("AGENT_EVALS_BASE_URL", "")).rstrip("/")
        if not self.base_url:
            raise ValueError(
                "HttpRunner needs a base URL: pass base_url or set AGENT_EVALS_BASE_URL"
            )
        self.api_key = api_key if api_key is not None else os.environ.get("AGENT_EVALS_API_KEY", "")
        self.model = model or os.environ.get("AGENT_EVALS_MODEL", "gpt-4o-mini")
        self.timeout = timeout

    def next_action(self, task: TaskSpec, messages: list[dict], step: int) -> AgentAction:
        from agent_evals import tools as tool_module

        payload = {
            "model": self.model,
            "messages": messages,
            "tools": [
                tool_module.openai_tool_schema(t) for t in tool_module.registry_for(task.tools)
            ],
            "tool_choice": "auto",
        }
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        resp = requests.post(
            f"{self.base_url}/chat/completions",
            headers=headers,
            data=json.dumps(payload),
            timeout=self.timeout,
        )
        resp.raise_for_status()
        choice = resp.json()["choices"][0]["message"]
        tool_calls = choice.get("tool_calls") or []
        if tool_calls:
            call = tool_calls[0]
            args = call["function"].get("arguments", "{}")
            return AgentAction(
                kind="tool",
                tool_name=call["function"]["name"],
                tool_args=json.loads(args) if isinstance(args, str) else dict(args),
            )
        return AgentAction(kind="final", text=choice.get("content") or "")
