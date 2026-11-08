"""Typed evidence records for external tool runs."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

SCHEMA_VERSION = "nexus.tool-result/v1"


def _canonical(payload: dict[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)


@dataclass(frozen=True)
class ToolIdentity:
    name: str
    version: str
    path: str | None
    status: str
    capabilities: tuple[str, ...] = ()
    license: str | None = None
    notes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "path": self.path,
            "status": self.status,
            "capabilities": list(self.capabilities),
            "license": self.license,
            "notes": list(self.notes),
        }


@dataclass
class ToolResult:
    tool: str
    version: str
    input_sha256: str
    arguments: dict[str, Any] = field(default_factory=dict)
    command: list[str] = field(default_factory=list)
    status: str = "unknown"
    started_at: str | None = None
    finished_at: str | None = None
    artifacts: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    limitations: list[str] = field(default_factory=list)
    cached: bool = False
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        identity = {
            "schema_version": SCHEMA_VERSION,
            "tool": self.tool,
            "version": self.version,
            "input_sha256": self.input_sha256,
            "arguments": self.arguments,
            "command": self.command,
            "status": self.status,
            "artifacts": self.artifacts,
            "warnings": self.warnings,
            "limitations": self.limitations,
            "error": self.error,
        }
        payload = {
            **identity,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "cached": self.cached,
        }
        payload["evidence_id"] = "tool_" + hashlib.sha256(_canonical(identity).encode("utf-8")).hexdigest()[:24]
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ToolResult":
        return cls(
            tool=str(payload.get("tool") or ""),
            version=str(payload.get("version") or ""),
            input_sha256=str(payload.get("input_sha256") or ""),
            arguments=dict(payload.get("arguments") or {}),
            command=list(payload.get("command") or []),
            status=str(payload.get("status") or "unknown"),
            started_at=payload.get("started_at"),
            finished_at=payload.get("finished_at"),
            artifacts=dict(payload.get("artifacts") or {}),
            warnings=list(payload.get("warnings") or []),
            limitations=list(payload.get("limitations") or []),
            cached=bool(payload.get("cached")),
            error=payload.get("error"),
        )


def cache_key(input_sha256: str, tool: str, version: str, arguments: dict[str, Any] | None = None) -> str:
    payload = {
        "input_sha256": input_sha256,
        "tool": tool,
        "version": version,
        "arguments": arguments or {},
    }
    return hashlib.sha256(_canonical(payload).encode("utf-8")).hexdigest()
