from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from nexus.core.errors import ValidationError
from nexus.core.hashing import stable_json_hash


@dataclass
class ToolSpec:
    name: str
    description: str
    input_schema: dict[str, Any]
    handler: Callable[..., Any]
    output_schema: dict[str, Any] = field(default_factory=dict)
    timeout_seconds: int = 30
    permission: str = "read"

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
            "output_schema": self.output_schema,
            "timeout_seconds": self.timeout_seconds,
            "permission": self.permission,
        }


def builtin_registry() -> "ToolRegistry":
    from nexus.program.bundle import analyze_binary

    registry = ToolRegistry()
    registry.register(ToolSpec(
        name="analyze_binary",
        description="Run deterministic ELF, disassembly, function, CFG, JNI and security analysis.",
        input_schema={"type": "object", "required": ["path"], "properties": {"path": {"type": "string"}}},
        handler=lambda path: analyze_binary(path).summary(),
    ))
    registry.register(ToolSpec(
        name="build_program_record",
        description="Build the canonical evidence-linked ProgramRecord.",
        input_schema={"type": "object", "required": ["path"], "properties": {"path": {"type": "string"}}},
        handler=lambda path: _record_summary(path),
    ))
    return registry


def _record_summary(path: str) -> dict:
    from nexus.program.bundle import analyze_binary
    from nexus.program.record import build_program_record
    record = build_program_record(analyze_binary(path))
    return {"record_keys": sorted(record.keys()), "functions": len(record.get("functions", []))}


class ToolRegistry:
    def __init__(self):
        self._tools: dict[str, ToolSpec] = {}

    def register(self, spec: ToolSpec) -> None:
        self._tools[spec.name] = spec

    def get(self, name: str) -> ToolSpec:
        if name not in self._tools:
            raise ValidationError(f"tool is not registered: {name}")
        return self._tools[name]

    def list(self) -> list[dict[str, Any]]:
        return [spec.to_dict() for spec in self._tools.values()]

    def invoke(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        spec = self.get(name)
        required = spec.input_schema.get("required", [])
        missing = [key for key in required if key not in arguments]
        if missing:
            raise ValidationError(f"missing tool arguments for {name}: {missing}")
        output = spec.handler(**arguments)
        return {
            "tool": name,
            "arguments_sha256": stable_json_hash(arguments),
            "output_sha256": stable_json_hash(output),
            "output": output,
            "permission": spec.permission,
            "schema": spec.output_schema,
        }
