"""Frozen-test evaluation for the standalone runtime."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from nexus.foundation.checkpoint import load_foundation_checkpoint
from nexus.foundation.evaluation import evaluate_checkpoint, markdown_report
from nexus_train.config import load_local_config
from nexus_train.data_validation import load_gate

ROOT = Path(__file__).resolve().parents[1]


def verify_checkpoint_gate_identity(loaded, gate) -> dict[str, Any]:
    metadata = getattr(loaded, "metadata", {}) or {}
    frozen = metadata.get("frozen_dataset") or {}
    checkpoint_frozen_id = frozen.get("frozen_id")
    if checkpoint_frozen_id != gate.frozen_id:
        raise ValueError(
            "checkpoint was not trained on the configured frozen dataset: "
            f"checkpoint={checkpoint_frozen_id!r} gate={gate.frozen_id!r}"
        )
    if frozen.get("vocab_sha256") not in {None, gate.vocab.digest()}:
        raise ValueError("checkpoint frozen-dataset vocabulary digest mismatch")
    return {"frozen_id": checkpoint_frozen_id, "manifest_sha256": frozen.get("manifest_sha256"), "vocab_sha256": frozen.get("vocab_sha256")}


def evaluate_frozen(checkpoint: str | Path, config_path: str | Path | None = None, *, reconstruction_samples: int = 2, split: str = "test") -> dict[str, Any]:
    if split not in {"test", "extended_test"}:
        raise ValueError(f"evaluation split must be test or extended_test: {split}")
    local = load_local_config(config_path or ROOT / "configs" / "local_cpu.yaml")
    gate = load_gate(local["dataset"]["frozen_dir"])
    loaded = load_foundation_checkpoint(
        checkpoint,
        load_training_state=False,
        expected_vocab_sha256=gate.vocab.digest(),
    )
    identity = verify_checkpoint_gate_identity(loaded, gate)
    max_functions = int(local["dataset"].get("max_functions", 8))
    train = gate.build_split("train", loaded.config, max_functions=max_functions)
    test = gate.build_split(split, loaded.config, max_functions=max_functions)
    objectives = list(local["training"].get("objectives", []))
    report = evaluate_checkpoint(checkpoint, test, train, objectives=objectives, reconstruction_samples=reconstruction_samples, permutations=50, seed_repeats=2, workspace=ROOT / "artifacts" / "training_runtime" / "evaluation")
    report["frozen_dataset"] = {**gate.identity(), "checkpoint_identity": identity}
    report["evaluation_split"] = split
    report["checkpoint"]["sha256"] = loaded.sha256
    return report


def write_evaluation(report: dict[str, Any], output: str | Path) -> dict[str, str]:
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")
    markdown = target.with_suffix(".md")
    markdown.write_text(markdown_report(report), encoding="utf-8")
    return {"json": str(target), "markdown": str(markdown)}
