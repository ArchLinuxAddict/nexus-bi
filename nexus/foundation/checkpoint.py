"""Versioned foundation-model checkpoints.

A checkpoint always records the architecture, parameter count, initialization
method, vocabulary digest, dataset fingerprints, objective configuration,
training configuration, seeds, optimizer/scheduler state, progress, evaluation
results and integrity digest.  Loading refuses pretrained-weight claims and
mismatched architecture/vocabulary by default.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch

from nexus.core.hashing import sha256_file
from nexus.foundation.config import ARCHITECTURE_VERSION, CHECKPOINT_FORMAT, CONFIG_HASH_SCHEME, FoundationConfig
from nexus.foundation.model import NexusFoundationModel, build_foundation_model
from nexus.foundation.vocab import Vocabulary

CHECKPOINT_TYPES = ("INITIALIZATION", "TRAINING", "RESUMABLE_TRAINING", "EVALUATION")
INITIALIZATION_STATUSES = {"random_initialization", "untrained_initialization", "initialization"}


@dataclass
class LoadedFoundationCheckpoint:
    model: NexusFoundationModel
    config: FoundationConfig
    vocab: Vocabulary | None
    metadata: dict[str, Any]
    sha256: str
    optimizer_state: dict[str, Any] | None
    scheduler_state: dict[str, Any] | None
    aggregator_state: dict[str, Any] | None
    step: int
    torch_rng_state: torch.Tensor | None = None
    python_rng_state: Any | None = None
    checkpoint_type: str = "TRAINING"


def classify_checkpoint(payload: dict[str, Any]) -> str:
    metadata = payload.get("metadata") or {}
    declared = str(metadata.get("checkpoint_type") or "").upper()
    if declared in CHECKPOINT_TYPES:
        return declared
    status = str(metadata.get("status") or "").lower()
    if (
        metadata.get("training_started") is False
        or status in INITIALIZATION_STATUSES
        or (metadata.get("trained") is False and int(payload.get("training_step", metadata.get("step", 0)) or 0) == 0)
    ):
        return "INITIALIZATION"
    if payload.get("optimizer_state") is not None and payload.get("scheduler_state") is not None:
        return "RESUMABLE_TRAINING"
    if metadata.get("evaluation") is not None or status.lower() == "evaluation":
        return "EVALUATION"
    return "TRAINING"


def checkpoint_type_errors(payload: dict[str, Any], checkpoint_type: str | None = None) -> list[str]:
    metadata = payload.get("metadata") or {}
    kind = (checkpoint_type or classify_checkpoint(payload)).upper()
    errors = []
    if kind == "INITIALIZATION":
        if metadata.get("training_started") is not False and metadata.get("trained") is not False:
            errors.append("initialization checkpoint must declare training_started=false")
        if int(metadata.get("optimizer_steps", payload.get("training_step", metadata.get("step", 0))) or 0) != 0:
            errors.append("initialization checkpoint must have optimizer_steps=0")
        if metadata.get("pretrained_weights") is not False and metadata.get("pretrained_weights_loaded") is not False:
            errors.append("initialization checkpoint must declare pretrained_weights=false")
        if metadata.get("initialized_from_scratch") is not True:
            errors.append("initialization checkpoint must declare initialized_from_scratch=true")
        if payload.get("optimizer_state") is not None or payload.get("scheduler_state") is not None:
            errors.append("initialization checkpoint must not contain optimizer or scheduler state")
    elif kind in {"RESUMABLE_TRAINING", "TRAINING"}:
        if payload.get("optimizer_state") is None:
            errors.append("training checkpoint is missing optimizer state")
        if payload.get("scheduler_state") is None:
            errors.append("training checkpoint is missing scheduler state")
    return errors


def _atomic_torch_save(payload: dict[str, Any], target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=target.parent, prefix=f".{target.name}.")
    os.close(descriptor)
    try:
        torch.save(payload, temporary)
        os.replace(temporary, target)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def save_foundation_checkpoint(
    path: str | Path,
    model: NexusFoundationModel,
    config: FoundationConfig,
    vocab: Vocabulary,
    metadata: dict[str, Any] | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    aggregator: torch.nn.Module | None = None,
    step: int = 0,
    include_training_state: bool = False,
    checkpoint_type: str | None = None,
) -> dict[str, Any]:
    target = Path(path)
    parameter_count = model.count_parameters()
    metadata = dict(metadata or {})
    kind = (checkpoint_type or metadata.get("checkpoint_type") or ("RESUMABLE_TRAINING" if include_training_state else "INITIALIZATION")).upper()
    if kind not in CHECKPOINT_TYPES:
        raise ValueError(f"invalid checkpoint type: {kind}")
    defaults = {
        "training_started": bool(include_training_state),
        "optimizer_steps": int(step),
        "pretrained_weights": False,
        "initialized_from_scratch": True,
    }
    for key, value in defaults.items():
        metadata.setdefault(key, value)
    if kind == "INITIALIZATION":
        metadata["training_started"] = False
        metadata["optimizer_steps"] = 0
        metadata["pretrained_weights"] = False
        metadata["initialized_from_scratch"] = True
    reserved = {
        "format", "checkpoint_type", "architecture_version", "created_at", "parameter_count",
        "initialization", "pretrained_weights_loaded", "config_hash", "config_hash_scheme",
        "legacy_config_hash", "vocab_sha256", "step",
    }
    state_metadata = {
        "format": CHECKPOINT_FORMAT,
        "checkpoint_type": kind,
        "architecture_version": ARCHITECTURE_VERSION,
        "created_at": datetime.now(UTC).isoformat(),
        "parameter_count": parameter_count,
        "initialization": "random",
        "pretrained_weights_loaded": False,
        "config_hash": config.config_hash(),
        "config_hash_scheme": CONFIG_HASH_SCHEME,
        "legacy_config_hash": config.legacy_config_hash(),
        "vocab_sha256": vocab.digest(),
        "step": int(step),
        **{key: value for key, value in metadata.items() if key not in reserved},
    }
    payload: dict[str, Any] = {
        "format": CHECKPOINT_FORMAT,
        "checkpoint_type": kind,
        "architecture_version": ARCHITECTURE_VERSION,
        "model_config": config.to_dict(),
        "config_hash": config.config_hash(),
        "config_hash_scheme": CONFIG_HASH_SCHEME,
        "legacy_config_hash": config.legacy_config_hash(),
        "vocab": vocab.to_dict(),
        "vocab_sha256": vocab.digest(),
        "model_state": model.state_dict(),
        "metadata": state_metadata,
    }
    if include_training_state:
        payload["optimizer_state"] = optimizer.state_dict() if optimizer is not None else None
        payload["scheduler_state"] = scheduler.state_dict() if scheduler is not None else None
        payload["aggregator_state"] = aggregator.state_dict() if aggregator is not None else None
        payload["training_step"] = int(step)
        payload["torch_rng_state"] = torch.get_rng_state()
        try:
            import random

            payload["python_rng_state"] = random.getstate()
        except Exception:
            payload["python_rng_state"] = None
    _atomic_torch_save(payload, target)
    digest = sha256_file(target)
    sidecar = target.with_suffix(target.suffix + ".sha256")
    sidecar.write_text(json.dumps({"sha256": digest, "size": target.stat().st_size}, indent=2) + "\n", encoding="utf-8")
    return {
        "path": str(target),
        "sha256": digest,
        "size": target.stat().st_size,
        "parameter_count": parameter_count,
        "metadata": state_metadata,
    }


def load_foundation_checkpoint(
    path: str | Path,
    map_location: str = "cpu",
    expected_vocab_sha256: str | None = None,
    allow_pretrained: bool = False,
    load_training_state: bool = True,
    safe_only: bool = True,
) -> LoadedFoundationCheckpoint:
    """Load a checkpoint.

    Phase 13 defaults to ``weights_only=True`` so a checkpoint cannot execute
    arbitrary Python objects during deserialization.  ``safe_only=False`` exists
    only for explicitly trusted legacy tooling and must not be used on untrusted
    files.
    """
    source = Path(path)
    try:
        payload = torch.load(source, map_location=map_location, weights_only=True)
    except Exception as exc:
        if safe_only:
            raise ValueError(
                "checkpoint contains objects outside the safe tensor/schema allowlist; "
                "refusing to deserialize with unrestricted pickle loading"
            ) from exc
        payload = torch.load(source, map_location=map_location, weights_only=False)
    if payload.get("format") != CHECKPOINT_FORMAT:
        raise ValueError(f"unsupported checkpoint format: {payload.get('format')}")
    if payload.get("architecture_version") != ARCHITECTURE_VERSION:
        raise ValueError("checkpoint architecture version mismatch")
    metadata = payload.get("metadata", {})
    if metadata.get("pretrained_weights_loaded") and not allow_pretrained:
        raise ValueError("checkpoint claims pretrained weights; loading is prohibited by mission policy")
    config = FoundationConfig.from_dict(payload["model_config"])
    stored_config_hash = payload.get("config_hash")
    if stored_config_hash not in {config.config_hash(), config.legacy_config_hash(), metadata.get("model_config_hash")}:
        raise ValueError("checkpoint configuration hash mismatch")
    vocab_payload = payload.get("vocab")
    vocab = None
    if vocab_payload:
        digest = vocab_payload.pop("vocab_sha256", None)
        vocab = Vocabulary(**{key: value for key, value in vocab_payload.items() if key in Vocabulary.__dataclass_fields__})
        if digest is not None and vocab.digest() != digest:
            raise ValueError("checkpoint vocabulary digest mismatch")
        if expected_vocab_sha256 and vocab.digest() != expected_vocab_sha256:
            raise ValueError("checkpoint vocabulary does not match the requested vocabulary")
    model = build_foundation_model(config)
    model.load_state_dict(payload["model_state"], strict=True)
    model.eval()
    actual_parameters = model.count_parameters()
    if actual_parameters != metadata.get("parameter_count"):
        raise ValueError(
            f"checkpoint parameter count mismatch: metadata={metadata.get('parameter_count')} actual={actual_parameters}"
        )
    return LoadedFoundationCheckpoint(
        model=model,
        config=config,
        vocab=vocab,
        metadata=metadata,
        sha256=sha256_file(source),
        optimizer_state=payload.get("optimizer_state") if load_training_state else None,
        scheduler_state=payload.get("scheduler_state") if load_training_state else None,
        aggregator_state=payload.get("aggregator_state") if load_training_state else None,
        step=int(payload.get("training_step", metadata.get("step", 0)) or 0),
        torch_rng_state=payload.get("torch_rng_state") if load_training_state else None,
        python_rng_state=payload.get("python_rng_state") if load_training_state else None,
        checkpoint_type=classify_checkpoint(payload),
    )


def inspect_checkpoint(path: str | Path, include_state: bool = False, safe_only: bool = True) -> dict[str, Any]:
    source = Path(path)
    try:
        payload = torch.load(source, map_location="cpu", weights_only=True)
    except Exception as exc:
        if safe_only:
            raise ValueError("checkpoint cannot be inspected with safe tensor-only loading") from exc
        payload = torch.load(source, map_location="cpu", weights_only=False)
    report = {
        "path": str(source),
        "sha256": sha256_file(source),
        "size": source.stat().st_size,
        "format": payload.get("format"),
        "architecture_version": payload.get("architecture_version"),
        "config": payload.get("model_config"),
        "config_hash": payload.get("config_hash"),
        "vocab_sha256": payload.get("vocab_sha256"),
        "metadata": payload.get("metadata"),
        "has_optimizer_state": payload.get("optimizer_state") is not None,
        "has_scheduler_state": payload.get("scheduler_state") is not None,
        "has_aggregator_state": payload.get("aggregator_state") is not None,
        "training_step": payload.get("training_step"),
    }
    if include_state:
        report["state_keys"] = list(payload.get("model_state", {}).keys())
    return report


def parameter_state_digest(model: NexusFoundationModel) -> dict[str, float | str]:
    """Hash unique named parameters, not duplicated tied state_dict entries.

    ``state_dict()`` can contain one tensor under several keys (the text
    embedding is shared by the source encoder and decoder).  Summing state_dict
    entries therefore overstates the parameter count.  The digest and L2 use
    unique named parameters; ``state_dict_numel`` records the duplicated total
    for diagnostics only.
    """
    digest = hashlib.sha256()
    total_squared = 0.0
    total_abs = 0.0
    count = 0
    for name, parameter in sorted(model.named_parameters()):
        tensor = parameter.detach().float().cpu()
        digest.update(name.encode("utf-8"))
        digest.update(tensor.numpy().tobytes())
        total_squared += float(tensor.pow(2).sum())
        total_abs += float(tensor.abs().sum())
        count += tensor.numel()
    state_dict_numel = sum(tensor.numel() for tensor in model.state_dict().values())
    return {
        "sha256": digest.hexdigest(),
        "l2": total_squared ** 0.5,
        "mean_abs": total_abs / max(1, count),
        "parameters": count,
        "state_dict_numel": state_dict_numel,
        "scheme": "nexus.parameter-state/v2",
    }


def validate_initialization_checkpoint(
    path: str | Path,
    *,
    expected_config: FoundationConfig | None = None,
    expected_vocab: Vocabulary | None = None,
    expected_parameter_count: int | None = None,
    verify_from_scratch: bool = True,
) -> dict[str, Any]:
    """Validate a checkpoint as a random initialization, never as a resume.

    Legacy checkpoints without the normalized metadata are accepted only when
    their parameter state exactly matches a deterministic rebuild from the
    recorded seed.  This prevents missing metadata from being blindly trusted.
    """
    loaded = load_foundation_checkpoint(
        path,
        load_training_state=True,
        allow_pretrained=False,
    )
    metadata = dict(loaded.metadata or {})
    errors: list[str] = []
    if loaded.checkpoint_type != "INITIALIZATION":
        errors.append(f"checkpoint type is {loaded.checkpoint_type}, not INITIALIZATION")
    errors.extend(checkpoint_type_errors({"metadata": metadata, "optimizer_state": loaded.optimizer_state, "scheduler_state": loaded.scheduler_state}, "INITIALIZATION"))
    if loaded.optimizer_state is not None or loaded.scheduler_state is not None:
        errors.append("initialization checkpoint carries optimizer/scheduler state")
    if loaded.step != 0:
        errors.append(f"initialization checkpoint step is {loaded.step}, not 0")
    if metadata.get("training_started") not in {None, False}:
        errors.append("training_started must be false")
    if int(metadata.get("optimizer_steps", 0) or 0) != 0:
        errors.append("optimizer_steps must be 0")
    if metadata.get("pretrained_weights") not in {None, False} or metadata.get("pretrained_weights_loaded") is True:
        errors.append("checkpoint must not contain pretrained weights")
    if metadata.get("initialized_from_scratch") not in {None, True}:
        errors.append("initialized_from_scratch must be true")
    if expected_config is not None:
        if loaded.config.architecture_version != expected_config.architecture_version:
            errors.append("architecture version does not match the expected effective configuration")
        if loaded.config.config_hash() != expected_config.config_hash():
            errors.append("effective configuration hash does not match the expected configuration")
        if loaded.config.d_model != expected_config.d_model or loaded.config.n_heads != expected_config.n_heads:
            errors.append("model dimensions do not match the expected configuration")
    if expected_vocab is not None:
        expected_vocab_hash = expected_vocab.digest()
        stored = metadata.get("vocab_sha256") or metadata.get("tokenizer_sha256")
        if stored != expected_vocab_hash:
            errors.append("vocabulary hash does not match the frozen tokenizer")
        if loaded.vocab is not None and loaded.vocab.digest() != expected_vocab_hash:
            errors.append("checkpoint vocabulary object does not match the frozen tokenizer")
    expected_count = expected_parameter_count if expected_parameter_count is not None else metadata.get("parameter_count")
    if expected_count is not None and int(loaded.model.count_parameters()) != int(expected_count):
        errors.append("parameter count does not match checkpoint metadata")
    if verify_from_scratch:
        seed = metadata.get("initialization_seed", metadata.get("seed", loaded.config.seed))
        if seed is None:
            errors.append("no initialization seed is recorded; cannot verify from-scratch state")
        else:
            fresh = build_foundation_model(loaded.config, seed=int(seed))
            if parameter_state_digest(loaded.model)["sha256"] != parameter_state_digest(fresh)["sha256"]:
                errors.append("parameter state does not match a deterministic from-scratch rebuild for the recorded seed")
    if errors:
        raise ValueError("initialization checkpoint validation failed: " + "; ".join(errors))
    return {
        "status": "VALID_INITIALIZATION",
        "path": str(path),
        "sha256": loaded.sha256,
        "checkpoint_type": loaded.checkpoint_type,
        "parameter_count": loaded.model.count_parameters(),
        "architecture_version": loaded.config.architecture_version,
        "config_hash": loaded.config.config_hash(),
        "legacy_config_hash": loaded.config.legacy_config_hash(),
        "vocab_sha256": (loaded.vocab.digest() if loaded.vocab is not None else metadata.get("vocab_sha256")),
        "step": loaded.step,
        "initialization_seed": metadata.get("initialization_seed", metadata.get("seed")),
        "training_started": False,
        "optimizer_steps": 0,
        "pretrained_weights": False,
        "initialized_from_scratch": True,
        "from_scratch_state_verified": bool(verify_from_scratch),
        "metadata_schema": "normalized" if metadata.get("checkpoint_type") else "legacy-verified",
        "metadata": metadata,
    }
