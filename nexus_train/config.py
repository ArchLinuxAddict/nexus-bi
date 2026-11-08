"""Configuration loading and validation (JSON-compatible YAML)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from nexus.foundation.config import FoundationConfig

ROOT = Path(__file__).resolve().parents[1]


def load_document(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        try:
            import yaml  # type: ignore
        except ImportError as exc:
            raise ValueError(f"{path} is not JSON and PyYAML is unavailable; use JSON-compatible YAML") from exc
        payload = yaml.safe_load(text)
    if not isinstance(payload, dict):
        raise ValueError(f"configuration must be a mapping: {path}")
    return payload


def load_local_config(path: str | Path) -> dict[str, Any]:
    payload = load_document(path)
    required = {"model", "training", "dataset", "safety"}
    missing = sorted(required - set(payload))
    if missing:
        raise ValueError(f"configuration missing sections: {missing}")
    model = payload["model"]
    try:
        config = FoundationConfig.from_dict(model)
    except Exception as exc:
        raise ValueError(f"invalid model configuration: {exc}") from exc
    training = payload["training"]
    if training.get("precision", "fp32") != "fp32":
        raise ValueError("only CPU fp32 is supported by this runtime")
    if training.get("device", "cpu") != "cpu":
        raise ValueError("only device=cpu is supported; no verified GPU backend exists")
    if int(training.get("batch_size", 0)) <= 0 or int(training.get("gradient_accumulation_steps", 0)) <= 0:
        raise ValueError("batch_size and gradient_accumulation_steps must be positive")
    if int(training.get("max_steps", 0)) <= 0:
        raise ValueError("max_steps must be positive")
    if int(training.get("torch_threads", 0)) <= 0:
        raise ValueError("torch_threads must be positive")
    from nexus_train.objective_registry import validate_objective_selection

    readiness = validate_objective_selection(
        list(training.get("objectives", [])),
        str(training.get("run_mode", "pilot")),
    )
    if readiness["status"] != "PASS":
        raise ValueError(f"objective readiness gate failed: {readiness['problems']}")
    payload["_objective_readiness"] = readiness
    from nexus.foundation.objectives import OBJECTIVES

    known_objectives = {objective.objective_id for objective in OBJECTIVES}
    unknown = set(training.get("objectives", [])) - known_objectives
    if unknown:
        raise ValueError(f"unsupported training objectives: {sorted(unknown)}")
    safety = payload["safety"]
    if float(safety.get("max_rss_gib", 0)) > 12.0:
        raise ValueError("max_rss_gib cannot silently exceed the audited 12 GiB cap")
    if float(safety.get("min_free_disk_gib", 0)) < 1:
        raise ValueError("min_free_disk_gib must be at least 1 GiB")
    payload["_model_config"] = config
    return payload


def model_sweep(path: str | Path | None = None) -> dict[str, Any]:
    path = Path(path) if path else ROOT / "configs" / "model_sweep.yaml"
    payload = load_document(path)
    candidates = payload.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise ValueError("model sweep needs a non-empty candidates list")
    for candidate in candidates:
        FoundationConfig.from_dict(candidate["model"])
    return payload


def objectives_config(path: str | Path | None = None) -> dict[str, Any]:
    path = Path(path) if path else ROOT / "configs" / "objectives.yaml"
    return load_document(path)
