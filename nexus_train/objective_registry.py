"""Objective readiness policy for the standalone runtime."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
READY_STATES = {"READY", "PARTIAL"}
FINAL_READY_STATES = {"READY"}


def load_objective_readiness(path: str | Path | None = None) -> dict[str, Any]:
    target = Path(path) if path else ROOT / "configs" / "objective_readiness.json"
    return json.loads(target.read_text(encoding="utf-8"))


def validate_objective_selection(enabled: list[str], run_mode: str, path: str | Path | None = None) -> dict[str, Any]:
    registry = load_objective_readiness(path)
    entries = registry.get("objectives") or {}
    allowed = FINAL_READY_STATES if run_mode == "final" else READY_STATES
    problems = []
    selected = {}
    for objective_id in enabled:
        entry = entries.get(objective_id)
        if entry is None:
            problems.append(f"objective has no readiness entry: {objective_id}")
            continue
        status = str(entry.get("status"))
        selected[objective_id] = entry
        if status not in allowed:
            problems.append(f"{objective_id} is {status} and cannot enter {run_mode} training")
        if run_mode == "final" and entry.get("final_ready") is False:
            problems.append(f"{objective_id} is not verified for final training")
        validation = entry.get("validation_coverage") or {}
        if run_mode in {"main", "final"} and validation.get("ready") is False:
            problems.append(f"{objective_id} lacks validation coverage for {run_mode} training")
    return {
        "schema_version": "nexus.objective-selection/v1",
        "run_mode": run_mode,
        "enabled": list(enabled),
        "selected": selected,
        "problems": problems,
        "status": "PASS" if not problems else "FAIL",
    }
