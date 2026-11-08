"""Structured JSONL telemetry and terminal status helpers."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any


class Telemetry:
    def __init__(self, run_dir: str | Path):
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_path = self.run_dir / "telemetry.jsonl"
        self.status_path = self.run_dir / "status.json"
        self.started = time.monotonic()
        self.last: dict[str, Any] = {}

    def emit(self, event: str, payload: dict[str, Any] | None = None) -> None:
        record = {"timestamp": time.time(), "elapsed_seconds": round(time.monotonic() - self.started, 2), "event": event, **(payload or {})}
        self.last = record
        with self.metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, default=str) + "\n")
        temporary = self.status_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(record, indent=2, default=str) + "\n", encoding="utf-8")
        temporary.replace(self.status_path)

    def progress_line(self, record: dict[str, Any]) -> str:
        objectives = record.get("objectives") or []
        losses = ", ".join(f"{item['objective_id']}:{item.get('raw', 0):.3f}" for item in objectives if item.get("active"))
        return (
            f"step {record.get('step', '?')} | loss {record.get('loss', 0):.4f} | "
            f"lr {record.get('learning_rate', 0):.2e} | rss {record.get('rss_mb', 0):.1f} MB | {losses}"
        )

    def status(self) -> dict[str, Any]:
        if self.status_path.exists():
            try:
                return json.loads(self.status_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                pass
        return {}


def latest_metric(run_dir: str | Path) -> dict[str, Any]:
    path = Path(run_dir) / "metrics.jsonl"
    if not path.exists():
        return {}
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not lines:
        return {}
    try:
        return json.loads(lines[-1])
    except json.JSONDecodeError:
        return {}
