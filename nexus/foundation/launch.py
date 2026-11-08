"""Canonical frozen-corpus training launch and safe preflight.

``nexus foundation train`` delegates here.  The optimization engine remains
``nexus_train.StandaloneRun``/``FoundationTrainer``; this module only resolves
the frozen corpus, checkpoint mode, objective selection, overrides and safety
checks.  It never trains by itself.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from nexus.foundation.checkpoint import inspect_checkpoint, validate_initialization_checkpoint
from nexus.foundation.config import config_for_vocabulary
from nexus.foundation.dataset import resolve_dataset_identity
from nexus_train.config import load_local_config
from nexus_train.trainer import StandaloneRun

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs" / "nexus_bi_realworld_v6.yaml"
DRY_RUN_ROOT = ROOT / "artifacts" / "training_runtime" / "dry_runs"

FROZEN_DATASET_LABELS = {
    "realworld-v6": "realworld_v6",
}

FROZEN_CORPUS_ROOT = ROOT / "artifacts" / "training_runtime" / "frozen_dataset" / "v6"


def _load_config(path: str | Path | None) -> dict[str, Any]:
    local = load_local_config(path or DEFAULT_CONFIG)
    if local.get("blocked") is True:
        blockers = (local.get("blockers") or {})
        raise ValueError(
            f"training config {path or DEFAULT_CONFIG} is documentation-only and intentionally blocked: "
            f"{blockers.get('status', 'BLOCKED')} {blockers.get('reasons', [])}"
        )
    return local


def _resolve_frozen_dataset(id_or_alias: str, frozen_datasets: dict[str, Any]) -> tuple[str, str]:
    component_label = FROZEN_DATASET_LABELS.get(id_or_alias)
    if component_label in frozen_datasets and component_label in {
        "java_controlled_v1", "dex_controlled_v1", "jni_declaration_controlled", "jni_native_controlled"
    }:
        return id_or_alias, component_label
    entry, canonical = resolve_dataset_identity(id_or_alias)
    del entry
    label = FROZEN_DATASET_LABELS.get(canonical) or FROZEN_DATASET_LABELS.get(id_or_alias)
    if label is None:
        raise ValueError(
            f"dataset {id_or_alias!r} resolves to {canonical!r} but is not a member of frozen corpus v3"
        )
    if label not in frozen_datasets:
        raise ValueError(
            f"dataset {id_or_alias!r} maps to frozen label {label!r}, which is absent from the frozen corpus"
        )
    return canonical, label


def _validate_frozen_request(requested: list[str], local: dict[str, Any]) -> dict[str, Any]:
    frozen = local.get("frozen_corpus") or {}
    frozen_datasets = frozen.get("datasets") or {}
    if not frozen_datasets:
        raise ValueError("training config has no frozen dataset composition")
    ids = requested or list((local.get("training") or {}).get("dataset_ids") or [])
    if not ids:
        raise ValueError("no dataset IDs were requested and the config declares none")
    resolved = {}
    labels = set()
    for dataset_id in ids:
        canonical, label = _resolve_frozen_dataset(dataset_id, frozen_datasets)
        resolved[dataset_id] = {"canonical_id": canonical, "frozen_label": label, "component": False}
        labels.add(label)
    if not requested:
        for dataset_id in local.get("component_dataset_ids") or []:
            label = FROZEN_DATASET_LABELS.get(dataset_id, dataset_id)
            if label not in frozen_datasets:
                raise ValueError(f"component dataset {dataset_id!r} is absent from the frozen corpus")
            resolved[dataset_id] = {"canonical_id": dataset_id, "frozen_label": label, "component": True}
            labels.add(label)
    if not labels:
        raise ValueError("requested datasets do not resolve to any frozen corpus member")
    return {"requested": ids, "resolved": resolved, "frozen_labels": sorted(labels)}


def _check_profile(profile: str | None, local: dict[str, Any]) -> str:
    configured = str((local.get("model") or {}).get("profile") or "")
    if profile and configured and profile != configured:
        raise ValueError(
            f"profile {profile!r} does not match the frozen effective model profile {configured!r}; "
            "a different profile requires a new tokenizer, corpus encoding and initialization checkpoint"
        )
    return configured or profile or "compact"


def _resolve_init_path(init_from: str | Path | None, local: dict[str, Any]) -> Path | None:
    configured = (local.get("initialization") or {}).get("checkpoint")
    if init_from is None and configured:
        init_from = configured
    if init_from is None:
        return None
    path = Path(init_from)
    if not path.is_absolute():
        path = ROOT / path
    if not path.exists():
        raise FileNotFoundError(f"initialization checkpoint does not exist: {path}")
    expected_sha = (local.get("initialization") or {}).get("sha256")
    if expected_sha and configured and str(path.resolve()) == str((ROOT / configured).resolve()):
        inspected = inspect_checkpoint(path)
        if inspected.get("sha256") and inspected["sha256"] != expected_sha:
            raise ValueError("initialization checkpoint hash does not match the frozen training configuration")
    return path


def _preflight(run: StandaloneRun, trainer, init_validation: dict[str, Any] | None) -> dict[str, Any]:
    optimizer_state = trainer.optimizer.state_dict()
    populated = {key: value for key, value in optimizer_state.get("state", {}).items() if value}
    scheduler_state = trainer.scheduler.state_dict()
    checkpoints = sorted(str(path.name) for path in run.telemetry.run_dir.glob("*.pt"))
    if checkpoints:
        raise RuntimeError(f"preflight unexpectedly produced checkpoints: {checkpoints}")
    return {
        "status": "PREFLIGHT_OK",
        "created_at": datetime.now(UTC).isoformat(),
        "run_dir": str(trainer.run_dir),
        "frozen_id": run.gate.frozen_id,
        "frozen_identity": run.identity,
        "train_samples": len(run.train_dataset),
        "validation_samples": len(run.validation_dataset),
        "model_profile": run.model_config.profile,
        "model_config_hash": run.model_config.config_hash(),
        "parameter_count": trainer.model.count_parameters(),
        "vocab_sha256": run.vocab.digest(),
        "objectives": list(run.training_config.objectives),
        "objective_readiness": run.local.get("_objective_readiness"),
        "initialization": init_validation,
        "optimizer": {
            "step": trainer.step,
            "state_entries": len(populated),
            "fresh": not populated and trainer.step == 0,
        },
        "scheduler": {
            "last_epoch": int(scheduler_state.get("last_epoch", -1)),
            "fresh": int(scheduler_state.get("last_epoch", -1)) in {-1, 0},
        },
        "checkpoint_written": False,
        "training_performed": False,
    }


def preflight_training(
    *,
    config_path: str | Path | None = None,
    init_from: str | Path | None = None,
    resume_from: str | Path | None = None,
    dataset_ids: list[str] | None = None,
    objectives: list[str] | None = None,
    profile: str | None = None,
    run_name: str | None = None,
    steps: int | None = None,
    batch_size: int | None = None,
    seed: int | None = None,
    output_dir: str | Path | None = None,
    frozen_dir: str | Path | None = None,
) -> dict[str, Any]:
    if init_from and resume_from:
        raise ValueError("--init-from and --resume-from are mutually exclusive")
    local = _load_config(config_path)
    if frozen_dir:
        local["dataset"]["frozen_dir"] = str(frozen_dir)
    profile_name = _check_profile(profile, local)
    requested = _validate_frozen_request(dataset_ids or [], local)
    overrides: dict[str, Any] = {}
    if objectives:
        overrides["objectives"] = list(objectives)
    if batch_size is not None:
        overrides["batch_size"] = int(batch_size)
    if seed is not None:
        overrides["seed"] = int(seed)
    overrides["dataset_ids"] = [
        item["canonical_id"] for item in requested["resolved"].values() if not item.get("component")
    ]
    if frozen_dir:
        local["dataset"]["frozen_dir"] = str(frozen_dir)
    if not init_from and not resume_from:
        init_from = _resolve_init_path(None, local)
    if profile_name != str((local.get("model") or {}).get("profile")):
        raise ValueError("profile override changes the effective architecture and is not supported for this frozen corpus")
    run_dir_root = Path(output_dir) if output_dir else DRY_RUN_ROOT
    dry_name = run_name or f"dry-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}"
    run = StandaloneRun(
        config_path or DEFAULT_CONFIG,
        run_name=dry_name,
        max_steps=steps,
        init_checkpoint=init_from if init_from and not resume_from else None,
        training_overrides=overrides,
        output_dir=run_dir_root,
        run_root=run_dir_root,
        dataset_filter_labels=set(requested["frozen_labels"]) | {item["canonical_id"] for item in requested["resolved"].values()},
    )
    trainer = run.prepare()
    init_validation = None
    if init_from and not resume_from:
        init_validation = validate_initialization_checkpoint(
            init_from,
            expected_config=config_for_vocabulary(run.model_config, run.vocab),
            expected_vocab=run.vocab,
        )
        init_validation = {
            key: init_validation.get(key)
            for key in (
                "status", "sha256", "checkpoint_type", "parameter_count", "config_hash",
                "vocab_sha256", "step", "training_started", "optimizer_steps", "pretrained_weights",
                "initialized_from_scratch", "from_scratch_state_verified", "metadata_schema",
            )
        }
    if resume_from:
        trainer.resume(resume_from)
    proof = _preflight(run, trainer, init_validation)
    proof["dataset_resolution"] = requested
    proof["profile"] = profile_name
    proof["config_path"] = str(Path(config_path or DEFAULT_CONFIG).resolve())
    return proof


def launch_training(
    *,
    config_path: str | Path | None = None,
    init_from: str | Path | None = None,
    resume_from: str | Path | None = None,
    dataset_ids: list[str] | None = None,
    objectives: list[str] | None = None,
    profile: str | None = None,
    run_name: str | None = None,
    steps: int | None = None,
    batch_size: int | None = None,
    seed: int | None = None,
    output_dir: str | Path | None = None,
    frozen_dir: str | Path | None = None,
    dry_run: bool = False,
    evaluate: bool = True,
) -> dict[str, Any]:
    if dry_run:
        return preflight_training(
            config_path=config_path,
            init_from=init_from,
            resume_from=resume_from,
            dataset_ids=dataset_ids,
            objectives=objectives,
            profile=profile,
            run_name=run_name,
            steps=steps,
            batch_size=batch_size,
            seed=seed,
            output_dir=output_dir,
            frozen_dir=frozen_dir,
        )
    if init_from and resume_from:
        raise ValueError("--init-from and --resume-from are mutually exclusive")
    if not init_from and not resume_from:
        raise ValueError("a new frozen-corpus run requires --init-from; use --resume-from only for a complete training checkpoint")
    local = _load_config(config_path)
    if frozen_dir:
        local["dataset"]["frozen_dir"] = str(frozen_dir)
    _check_profile(profile, local)
    requested = _validate_frozen_request(dataset_ids or [], local)
    overrides: dict[str, Any] = {}
    if objectives:
        overrides["objectives"] = list(objectives)
    if batch_size is not None:
        overrides["batch_size"] = int(batch_size)
    if seed is not None:
        overrides["seed"] = int(seed)
    overrides["dataset_ids"] = [
        item["canonical_id"] for item in requested["resolved"].values() if not item.get("component")
    ]
    if init_from is None and not resume_from:
        init_from = _resolve_init_path(None, local)
    run_root = Path(output_dir) if output_dir else None
    run = StandaloneRun(
        config_path or DEFAULT_CONFIG,
        run_name=run_name,
        max_steps=steps,
        init_checkpoint=init_from if not resume_from else None,
        training_overrides=overrides,
        output_dir=run_root,
        run_root=run_root,
        dataset_filter_labels=set(requested["frozen_labels"]) | {item["canonical_id"] for item in requested["resolved"].values()},
    )
    if resume_from:
        result = run.run(resume_path=resume_from, evaluate=evaluate)
    else:
        result = run.run(evaluate=evaluate)
    result["dataset_resolution"] = requested
    result["config_path"] = str(Path(config_path or DEFAULT_CONFIG).resolve())
    return result
