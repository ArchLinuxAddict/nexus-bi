"""Standalone checkpoint inspection helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from nexus.foundation.checkpoint import inspect_checkpoint, load_foundation_checkpoint


def inspect(path: str | Path) -> dict[str, Any]:
    info = inspect_checkpoint(path, include_state=False)
    metadata = info.get("metadata") or {}
    return {
        "path": str(Path(path).resolve()),
        "sha256": info.get("sha256"),
        "size": info.get("size"),
        "format": info.get("format"),
        "architecture_version": info.get("architecture_version"),
        "config_hash": info.get("config_hash"),
        "vocab_sha256": info.get("vocab_sha256"),
        "parameter_count": metadata.get("parameter_count"),
        "initialization": metadata.get("initialization"),
        "pretrained_weights_loaded": metadata.get("pretrained_weights_loaded"),
        "step": metadata.get("step"),
        "frozen_dataset": metadata.get("frozen_dataset"),
        "has_optimizer_state": info.get("has_optimizer_state"),
        "has_scheduler_state": info.get("has_scheduler_state"),
        "safe_loaded": True,
    }


def load_for_inference(path: str | Path):
    return load_foundation_checkpoint(path, load_training_state=False)


def write_metadata(path: str | Path, payload: dict[str, Any]) -> None:
    target = Path(path).with_suffix(Path(path).suffix + ".metadata.json")
    target.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
