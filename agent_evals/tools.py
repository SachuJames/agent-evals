"""Whitelisted tools the harness can execute inside the agent loop."""

from __future__ import annotations

import abc
import ast
import math
import operator
from collections.abc import Callable


class Tool(abc.ABC):
    """A tool the agent may call. Execution is synchronous and sandboxed."""

    name: str = ""
    description: str = ""
    parameters: dict = {}

    @abc.abstractmethod
    def execute(self, args: dict, ctx: ToolContext) -> str:
        """Run the tool. Return a string that is fed back to the agent."""


class ToolContext:
    """Per-task-run mutable state shared by tools."""

    def __init__(self, fetch_stubs: dict[str, str] | None = None):
        self.kv: dict[str, str] = {}
        self.fetch_stubs: dict[str, str] = dict(fetch_stubs or {})


_SAFE_BINOPS: dict[type, Callable[[float, float], float]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_SAFE_UNARYOPS: dict[type, Callable[[float], float]] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}
_SAFE_FUNCS: dict[str, Callable[..., float]] = {
    "sqrt": math.sqrt,
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
    "pow": pow,
    "floor": math.floor,
    "ceil": math.ceil,
}


def _eval_node(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _eval_node(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.BinOp) and type(node.op) in _SAFE_BINOPS:
        return _SAFE_BINOPS[type(node.op)](_eval_node(node.left), _eval_node(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _SAFE_UNARYOPS:
        return _SAFE_UNARYOPS[type(node.op)](_eval_node(node.operand))
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        func = _SAFE_FUNCS.get(node.func.id)
        if func is None or node.keywords:
            raise ValueError(f"function not allowed: {getattr(node.func, 'id', '?')}")
        return float(func(*[_eval_node(a) for a in node.args]))
    raise ValueError(f"expression not allowed: {ast.dump(node)[:60]}")


class Calculator(Tool):
    name = "calculator"
    description = "Evaluate a numeric arithmetic expression."
    parameters = {
        "type": "object",
        "properties": {"expression": {"type": "string"}},
        "required": ["expression"],
    }

    def execute(self, args: dict, ctx: ToolContext) -> str:
        expr = str(args.get("expression", ""))
        try:
            tree = ast.parse(expr, mode="eval")
            value = _eval_node(tree)
        except Exception as exc:
            return f"error: {exc}"
        if value == int(value):
            return str(int(value))
        return f"{value:.6f}".rstrip("0").rstrip(".")


class KVStore(Tool):
    name = "kv_store"
    description = "A small key-value store: set, get, delete, or list keys."
    parameters = {
        "type": "object",
        "properties": {
            "op": {"type": "string", "enum": ["set", "get", "delete", "list"]},
            "key": {"type": "string"},
            "value": {"type": "string"},
        },
        "required": ["op"],
    }

    def execute(self, args: dict, ctx: ToolContext) -> str:
        op = str(args.get("op", ""))
        key = str(args.get("key", ""))
        if op == "set":
            ctx.kv[key] = str(args.get("value", ""))
            return "ok"
        if op == "get":
            return ctx.kv.get(key, "error: key not found")
        if op == "delete":
            return "ok" if ctx.kv.pop(key, None) is not None else "error: key not found"
        if op == "list":
            return ",".join(sorted(ctx.kv))
        return f"error: unknown op {op!r}"


class WebFetch(Tool):
    name = "web_fetch"
    description = "Fetch a URL. In evals this is stubbed per task."
    parameters = {
        "type": "object",
        "properties": {"url": {"type": "string"}},
        "required": ["url"],
    }

    def execute(self, args: dict, ctx: ToolContext) -> str:
        url = str(args.get("url", ""))
        if url in ctx.fetch_stubs:
            return ctx.fetch_stubs[url]
        return f"error: no stub for url {url!r}"


REGISTRY: dict[str, Tool] = {
    "calculator": Calculator(),
    "kv_store": KVStore(),
    "web_fetch": WebFetch(),
}


def registry_for(names: list[str]) -> list[Tool]:
    """Return tool objects for whitelisted names, raising on unknown names."""
    tools = []
    for name in names:
        if name not in REGISTRY:
            raise ValueError(f"unknown tool: {name!r}")
        tools.append(REGISTRY[name])
    return tools


def openai_tool_schema(tool: Tool) -> dict:
    """Render a tool as an OpenAI chat-completions tool definition."""
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.parameters,
        },
    }
