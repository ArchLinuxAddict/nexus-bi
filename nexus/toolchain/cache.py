"""Content-addressed cache for external tool results."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from nexus.toolchain.models import ToolResult

ROOT = Path(__file__).resolve().parents[2]


class ToolCache:
    def __init__(self, root: str | Path | None = None) -> None:
        self.root = Path(root) if root else ROOT / "artifacts" / "toolchain" / "cache"

    def _path(self, key: str) -> Path:
        return self.root / key[:2] / f"{key}.json"

    def get(self, key: str) -> dict[str, Any] | None:
        path = self._path(key)
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return None

    def get_result(self, key: str) -> ToolResult | None:
        payload = self.get(key)
        if payload is None:
            return None
        result = ToolResult.from_dict(payload)
        result.cached = True
        return result

    def put(self, key: str, payload: dict[str, Any]) -> Path:
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temporary, path)
        return path

    def put_result(self, key: str, result: ToolResult) -> Path:
        return self.put(key, result.to_dict())
