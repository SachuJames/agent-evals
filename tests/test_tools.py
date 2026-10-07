"""Tests for the whitelisted tools."""

import pytest

from agent_evals.tools import REGISTRY, ToolContext, openai_tool_schema, registry_for


def ctx():
    return ToolContext(fetch_stubs={"https://x.test/a": "stub content"})


def test_calculator_basic():
    assert REGISTRY["calculator"].execute({"expression": "2 + 3 * 4"}, ctx()) == "14"


def test_calculator_float_formatting():
    assert REGISTRY["calculator"].execute({"expression": "1000 * 1.05 ** 2"}, ctx()) == "1102.5"


def test_calculator_functions():
    assert REGISTRY["calculator"].execute({"expression": "sqrt(16)"}, ctx()) == "4"


def test_calculator_rejects_unsafe():
    out = REGISTRY["calculator"].execute({"expression": "__import__('os').system('x')"}, ctx())
    assert out.startswith("error:")


def test_calculator_rejects_names():
    out = REGISTRY["calculator"].execute({"expression": "foo + 1"}, ctx())
    assert out.startswith("error:")


def test_kv_store_roundtrip():
    kv = REGISTRY["kv_store"]
    c = ctx()
    assert kv.execute({"op": "set", "key": "a", "value": "17"}, c) == "ok"
    assert kv.execute({"op": "get", "key": "a"}, c) == "17"
    assert kv.execute({"op": "get", "key": "missing"}, c).startswith("error:")
    assert kv.execute({"op": "list"}, c) == "a"
    assert kv.execute({"op": "delete", "key": "a"}, c) == "ok"


def test_web_fetch_stub_hit_and_miss():
    wf = REGISTRY["web_fetch"]
    c = ctx()
    assert wf.execute({"url": "https://x.test/a"}, c) == "stub content"
    assert wf.execute({"url": "https://x.test/nope"}, c).startswith("error:")


def test_registry_for_unknown_raises():
    with pytest.raises(ValueError, match="unknown tool"):
        registry_for(["nope"])


def test_openai_tool_schema_shape():
    schema = openai_tool_schema(REGISTRY["calculator"])
    assert schema["type"] == "function"
    assert schema["function"]["name"] == "calculator"
    assert "expression" in schema["function"]["parameters"]["properties"]
