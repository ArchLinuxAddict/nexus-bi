"""Checkpoint discovery and resume validation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from nexus_train.checkpoint import inspect


def latest_checkpoint(roots: list[str | Path] | None = None, *, require_training_state: bool = True) -> Path | None:
    candidates: list[Path] = []
    for root in roots or [Path("checkpoints"), Path("runs")]:
        base = Path(root)
        if not base.exists():
            continue
        candidates.extend(base.rglob("*.pt"))
    candidates = [path for path in candidates if path.is_file() and not path.name.endswith(".metadata.json")]
    if not candidates:
        return None
    candidates.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    for path in candidates:
        try:
            info = inspect(path)
        except Exception:
            continue
        if require_training_state and not (info["has_optimizer_state"] and info["has_scheduler_state"]):
            continue
        return path
    return None


def checkpoint_summary(path: str | Path) -> dict[str, Any]:
    return inspect(path)
