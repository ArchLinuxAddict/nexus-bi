import pytest

from nexus.core.errors import BudgetExceeded, ValidationError
from nexus.reasoning.investigation import investigate
from nexus.reasoning.tools import ToolRegistry, ToolSpec


def test_investigation_report(simple_binary):
    report = investigate(str(simple_binary), "understand the binary", max_steps=6)
    assert report["status"] == "completed"
    assert report["binary_sha256"]
    assert report["facts"]
    assert report["hypotheses"]
    assert report["actions"]
    assert report["conclusion"]["verification"].startswith("static only")


def test_investigation_budget(simple_binary):
    with pytest.raises(BudgetExceeded):
        investigate(str(simple_binary), "x", max_steps=0)


def test_tool_registry_validation():
    registry = ToolRegistry()
    registry.register(ToolSpec(
        name="echo",
        description="echo",
        input_schema={"type": "object", "required": ["value"]},
        handler=lambda value: {"value": value},
    ))
    with pytest.raises(ValidationError):
        registry.invoke("echo", {})
    result = registry.invoke("echo", {"value": 3})
    assert result["output"]["value"] == 3
    assert result["output_sha256"]
