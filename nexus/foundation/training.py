"""Multi-objective training engine for the unified NEXUS-BI foundation model.

The engine performs real random-initialized optimization, records per-objective
losses, supports gradient accumulation and clipping, validation, early stopping,
atomic checkpointing and exact resume, and writes reproducible run manifests.
"""

from __future__ import annotations

import json
import math
import os
import random
import resource
import time
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

from nexus.foundation.checkpoint import (
    load_foundation_checkpoint,
    parameter_state_digest,
    save_foundation_checkpoint,
)
from nexus.foundation.config import FoundationConfig, config_for_vocabulary, profile_config
from nexus.foundation.dataset import FoundationDataset
from nexus.foundation.model import NexusFoundationModel, build_foundation_model, parameter_report
from nexus.foundation.objectives import (
    OBJECTIVES,
    LossAggregator,
    ObjectiveResult,
    compute_objective_losses,
    get_objective,
    gradient_norms,
    objective_registry,
)
from nexus.foundation.representation import (
    mask_byte_batch,
    mask_evidence_batch,
    mask_instruction_batch,
)
from nexus.foundation.vocab import Vocabulary
from nexus.mission.policy import validate_training_run


class TrainingFailure(RuntimeError):
    pass


@dataclass
class FoundationTrainingConfig:
    run_name: str = "foundation-smoke"
    profile: str = "cpu_dev"
    batch_size: int = 2
    gradient_accumulation_steps: int = 1
    max_steps: int = 20
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    warmup_steps: int = 2
    min_lr_ratio: float = 0.1
    grad_clip: float = 1.0
    precision: str = "fp32"
    device: str = "cpu"
    seed: int = 20261103
    val_interval: int = 5
    checkpoint_interval: int = 10
    early_stopping_patience: int = 4
    output_dir: str = "artifacts/foundation_runs"
    objectives: list[str] = field(
        default_factory=lambda: [objective.objective_id for objective in OBJECTIVES if objective.default_enabled]
    )
    aggregation: str = "normalized"
    modality_dropout: float = 0.02
    equivalence_sampling_fraction: float = 0.5
    equivalence_groups_per_batch: int = 2
    equivalence_members_per_group: int = 2
    mask_instruction_probability: float = 0.15
    mask_byte_probability: float = 0.08
    gradient_diagnostics_interval: int = 5
    torch_threads: int = 1
    mission_objectives: list[str] = field(default_factory=list)
    dataset_categories: list[str] = field(default_factory=list)
    dataset_ids: list[str] = field(default_factory=list)
    run_mode: str = "pilot"
    experimental_justification: str = "Small CPU pilot of the unified architecture; not large-scale training."
    resume_from: str | None = None
    dataset_identity: dict[str, Any] = field(default_factory=dict)
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def mission_objectives_for_run(self) -> list[str]:
        if self.mission_objectives:
            return list(self.mission_objectives)
        names = []
        for objective_id in self.objectives:
            objective = get_objective(objective_id)
            if objective.mission_objective not in names:
                names.append(objective.mission_objective)
        return names


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _json_write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, default=str, sort_keys=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _jsonl_append(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, default=str) + "\n")


def _rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def _move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}


def _scheduler(optimizer: torch.optim.Optimizer, warmup: int, max_steps: int, min_ratio: float) -> torch.optim.lr_scheduler.LambdaLR:
    def multiplier(step: int) -> float:
        if warmup > 0 and step < warmup:
            return max(1e-3, (step + 1) / warmup)
        if max_steps <= warmup:
            return 1.0
        progress = min(1.0, (step - warmup) / max(1, max_steps - warmup))
        return min_ratio + (1.0 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)


def _autoregressive_batch(batch: dict[str, Any]) -> dict[str, Any]:
    ids = batch["instr_id"].clone()
    ar_ids = torch.zeros_like(ids)
    ar_ids[:, 0] = 3
    if ids.size(1) > 1:
        ar_ids[:, 1:] = ids[:, :-1]
    result = dict(batch)
    result["instr_id"] = ar_ids
    result["instr_ar_targets"] = batch["instr_id"]
    result["instr_ar_mask"] = batch["instr_mask"]
    return result


EVIDENCE_OBJECTIVES = {
    "ir_op_prediction",
    "register_effect_prediction",
    "memory_effect_prediction",
    "cfg_edge_prediction",
}


def _attach_evidence_outputs(model, batch: dict[str, Any], outputs: dict[str, torch.Tensor], seed: int, objectives: list[str]) -> None:
    if not (EVIDENCE_OBJECTIVES & set(objectives)):
        return
    evidence_batch = mask_evidence_batch(batch, seed=seed)
    evidence = model(evidence_batch)
    outputs["ir_mlm_logits"] = evidence["ir_logits"]
    outputs["register_read_masked_logits"] = evidence["register_read_logits"]
    outputs["register_write_masked_logits"] = evidence["register_write_logits"]
    outputs["memory_read_masked_logits"] = evidence["memory_read_logits"]
    outputs["memory_write_masked_logits"] = evidence["memory_write_logits"]
    outputs["memory_width_masked_logits"] = evidence["memory_width_logits"]
    outputs["cfg_masked_logits"] = evidence["cfg_logits"]
    batch["ir_targets"] = evidence_batch["ir_targets"]
    batch["ir_mlm_mask"] = evidence_batch["ir_mlm_mask"]


def _decoder_requested(objectives: list[str], batch: dict[str, Any], language: str) -> bool:
    objective_id = "c_reconstruction" if language == "c" else "java_jni_reconstruction"
    if objective_id not in objectives:
        return False
    key = "c_mask" if language == "c" else "java_mask"
    mask = batch.get(key)
    return bool(mask is not None and mask.any())


@torch.no_grad()
def evaluate_dataset(
    model: NexusFoundationModel,
    dataset: FoundationDataset,
    objectives: list[str],
    batch_size: int = 4,
    seed: int = 20261103,
    batch_transform: Any | None = None,
) -> dict[str, Any]:
    model.eval()
    totals: dict[str, float] = {}
    counts: dict[str, int] = {}
    metric_hits: dict[str, float] = {}
    metric_totals: dict[str, int] = {}
    samples = 0
    for batch in dataset.evaluation_batches(batch_size=batch_size):
        batch = mask_instruction_batch(batch, probability=0.15, seed=seed + samples)
        batch = mask_byte_batch(batch, probability=0.08, seed=seed + samples)
        if batch_transform is not None:
            batch = {**batch, "ir_target_source": batch["instr_ir"].clone()}
            batch = batch_transform(batch)
        needs_c = _decoder_requested(objectives, batch, "c")
        needs_java = _decoder_requested(objectives, batch, "java")
        if needs_c:
            outputs = model(batch, run_decoder=True, decoder_language="c")
        elif needs_java:
            outputs = model(batch, run_decoder=True, decoder_language="java")
        else:
            outputs = model(batch)
        _attach_evidence_outputs(model, batch, outputs, seed + samples, objectives)
        if "instruction_autoregressive" in objectives and batch.get("run_instruction_autoregressive", True):
            ar = _autoregressive_batch(batch)
            ar_outputs = model(ar, causal_instructions=True)
            outputs["instr_ar_logits"] = ar_outputs["instruction_logits"]
            batch = {**batch, "instr_ar_targets": ar["instr_ar_targets"], "instr_ar_mask": ar["instr_ar_mask"]}
        raw = compute_objective_losses(outputs, batch, objectives)
        for objective_id, (loss, count, _note) in raw.items():
            if count > 0:
                totals[objective_id] = totals.get(objective_id, 0.0) + float(loss.detach()) * count
                counts[objective_id] = counts.get(objective_id, 0) + count
        mask = batch["instr_mlm_mask"]
        if mask.any():
            prediction = outputs["instruction_logits"][mask].argmax(dim=-1)
            target = batch["instr_targets"][mask]
            metric_hits["instruction_mlm_accuracy"] = metric_hits.get("instruction_mlm_accuracy", 0.0) + float(
                (prediction == target).sum()
            )
            metric_totals["instruction_mlm_accuracy"] = metric_totals.get("instruction_mlm_accuracy", 0) + int(mask.sum())
        byte_mask = batch["byte_mlm_mask"]
        if byte_mask.any():
            prediction = outputs["byte_logits"][byte_mask].argmax(dim=-1)
            target = batch["byte_targets"][byte_mask]
            metric_hits["byte_mlm_accuracy"] = metric_hits.get("byte_mlm_accuracy", 0.0) + float(
                (prediction == target).sum()
            )
            metric_totals["byte_mlm_accuracy"] = metric_totals.get("byte_mlm_accuracy", 0) + int(byte_mask.sum())
        if "ir_mlm_mask" in batch and "ir_mlm_logits" in outputs:
            ir_mask = batch["ir_mlm_mask"]
            prediction = outputs["ir_mlm_logits"][ir_mask].argmax(dim=-1)
            target = batch["ir_targets"][ir_mask]
            metric_hits["ir_accuracy"] = metric_hits.get("ir_accuracy", 0.0) + float((prediction == target).sum())
            metric_totals["ir_accuracy"] = metric_totals.get("ir_accuracy", 0) + int(ir_mask.sum())
        block_mask = batch["block_mask"]
        valid = block_mask.unsqueeze(1) & block_mask.unsqueeze(2)
        diagonal = torch.eye(valid.size(1), dtype=torch.bool, device=valid.device).unsqueeze(0)
        valid = valid & ~diagonal
        if valid.any() and "cfg_masked_logits" in outputs:
            predicted = outputs["cfg_masked_logits"][valid] > 0
            target = batch["cfg_target"][valid] > 0.5
            tp = int((predicted & target).sum())
            fp = int((predicted & ~target).sum())
            fn = int((~predicted & target).sum())
            metric_hits["cfg_tp"] = metric_hits.get("cfg_tp", 0.0) + tp
            metric_hits["cfg_fp"] = metric_hits.get("cfg_fp", 0.0) + fp
            metric_hits["cfg_fn"] = metric_hits.get("cfg_fn", 0.0) + fn
            metric_totals["cfg_pairs"] = metric_totals.get("cfg_pairs", 0) + int(valid.sum())
        purpose_mask = batch["purpose_target"].ge(0)
        if purpose_mask.any():
            prediction = outputs["purpose_logits"][purpose_mask].argmax(dim=-1)
            target = batch["purpose_target"][purpose_mask]
            metric_hits["purpose_accuracy"] = metric_hits.get("purpose_accuracy", 0.0) + float(
                (prediction == target).sum()
            )
            metric_totals["purpose_accuracy"] = metric_totals.get("purpose_accuracy", 0) + int(purpose_mask.sum())
        if "c_logits" in outputs:
            target = batch["c_ids"][:, 1:]
            mask = batch["c_mask"][:, 1:]
            if mask.any():
                prediction = outputs["c_logits"][mask].argmax(dim=-1)
                target_values = target[mask]
                metric_hits["c_token_accuracy"] = metric_hits.get("c_token_accuracy", 0.0) + float(
                    (prediction == target_values).sum()
                )
                metric_totals["c_token_accuracy"] = metric_totals.get("c_token_accuracy", 0) + int(mask.sum())
        samples += batch["instr_id"].size(0)
    losses = {
        objective_id: {
            "mean": totals[objective_id] / max(1, counts[objective_id]),
            "count": counts[objective_id],
        }
        for objective_id in totals
    }
    metrics = {
        name: metric_hits[name] / max(1, metric_totals[name])
        for name in metric_hits
        if name not in {"cfg_tp", "cfg_fp", "cfg_fn"}
    }
    tp, fp, fn = metric_hits.get("cfg_tp", 0.0), metric_hits.get("cfg_fp", 0.0), metric_hits.get("cfg_fn", 0.0)
    metrics["cfg_precision"] = tp / max(1.0, tp + fp)
    metrics["cfg_recall"] = tp / max(1.0, tp + fn)
    metrics["cfg_f1"] = 2 * tp / max(1.0, 2 * tp + fp + fn)
    specs = {objective.objective_id: objective for objective in OBJECTIVES}
    weighted = sum(
        specs[objective_id].base_weight * value["mean"]
        for objective_id, value in losses.items()
        if objective_id in specs
    )
    return {
        "split": dataset.split,
        "samples": samples,
        "losses": losses,
        "metrics": metrics,
        "weighted_validation_score": weighted,
    }


class FoundationTrainer:
    def __init__(
        self,
        model_config: FoundationConfig,
        training_config: FoundationTrainingConfig,
        train_dataset: FoundationDataset,
        validation_dataset: FoundationDataset,
        vocab: Vocabulary,
        run_dir: str | Path | None = None,
    ):
        self.model_config = config_for_vocabulary(model_config, vocab)
        self.training_config = training_config
        self.train_dataset = train_dataset
        self.validation_dataset = validation_dataset
        self.vocab = vocab
        if train_dataset.split == "test" or validation_dataset.split == "test":
            raise ValueError("test data cannot enter training")
        categories = training_config.dataset_categories or sorted(
            {source.category for source in train_dataset.sources}
        )
        dataset_ids = training_config.dataset_ids or sorted(
            {source.dataset_id for source in train_dataset.sources}
        )
        objectives = training_config.mission_objectives_for_run()
        self.mission = validate_training_run(
            objectives,
            categories,
            dataset_ids,
            run_mode=training_config.run_mode,
            experimental_justification=training_config.experimental_justification,
        )
        self.training_config.mission_objectives = list(self.mission.objectives)
        self.training_config.dataset_categories = categories
        self.training_config.dataset_ids = dataset_ids
        torch.set_num_threads(max(1, training_config.torch_threads))
        self.device = torch.device(training_config.device if torch.cuda.is_available() or training_config.device == "cpu" else "cpu")
        seed = training_config.seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        self.model = build_foundation_model(self.model_config, seed=seed).to(self.device)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=training_config.learning_rate,
            weight_decay=training_config.weight_decay,
        )
        self.scheduler = _scheduler(
            self.optimizer,
            training_config.warmup_steps,
            training_config.max_steps,
            training_config.min_lr_ratio,
        )
        self.aggregator = LossAggregator(mode=training_config.aggregation, enabled=training_config.objectives)
        if training_config.aggregation == "uncertainty":
            self.optimizer.add_param_group({"params": list(self.aggregator.parameters()), "weight_decay": 0.0})
        self.step = 0
        self.data_cursor = 0
        self.history: list[dict[str, Any]] = []
        self.best_validation: float | None = None
        self.best_step: int | None = None
        self.stop_reason = "max_steps"
        self.batch_transform = None
        self.resource_guard: Any | None = None
        self.step_callback: Any | None = None
        self.failures = 0
        self.run_dir = Path(run_dir) if run_dir is not None else Path(training_config.output_dir) / training_config.run_name
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._prepare_initial_state()
        self.purpose_class_weights = self._purpose_class_weights()
        self.initial_digest = parameter_state_digest(self.model)
        self.started_at = _now()
        self.started_monotonic = time.perf_counter()
        self.manifest = self._manifest()
        _json_write(self.run_dir / "manifest.json", self.manifest)
        _json_write(self.run_dir / "objective_registry.json", objective_registry())
        _json_write(self.run_dir / "model_profile.json", parameter_report(self.model))
        _json_write(self.run_dir / "dataset_report.json", train_dataset.stats())

    def _purpose_class_weights(self) -> torch.Tensor | None:
        from nexus.foundation.representation import PURPOSE_CLASSES

        counts = [0] * len(PURPOSE_CLASSES)
        purpose_counts = getattr(self.train_dataset, "purpose_counts", None)
        if callable(purpose_counts):
            observed = purpose_counts()
            for index, value in enumerate(observed[: len(counts)]):
                counts[index] = int(value)
        else:
            for sample in self.train_dataset.samples:
                purpose = int(sample.get("purpose_id", -100))
                if 0 <= purpose < len(counts):
                    counts[purpose] += 1
        present = [index for index, count in enumerate(counts) if count > 0]
        if not present:
            return None
        total = sum(counts)
        weights = torch.ones(len(counts), dtype=torch.float32)
        for index in present:
            weights[index] = (total / (len(present) * counts[index])) ** 0.5
        weights = weights / weights[present].mean()
        return weights.to(self.device)

    def _manifest(self) -> dict[str, Any]:
        return {
            "schema_version": "0.1.0",
            "created_at": self.started_at,
            "run_name": self.training_config.run_name,
            "status": "initialized",
            "architecture_version": self.model_config.architecture_version,
            "model_config": self.model_config.to_dict(),
            "model_config_hash": self.model_config.config_hash(),
            "parameter_count": self.model.count_parameters(),
            "initialization": "random",
            "pretrained_weights_loaded": False,
            "vocabulary": {"sha256": self.vocab.digest(), "counts": self.vocab.counts},
            "training_config": self.training_config.to_dict(),
            "seeds": {"python": self.training_config.seed, "numpy": self.training_config.seed, "torch": self.training_config.seed},
            "objectives": [objective.to_dict() for objective in OBJECTIVES if objective.objective_id in self.training_config.objectives],
            "mission": self.mission.to_dict(),
            "train_dataset": self.train_dataset.stats(),
            "validation_dataset": self.validation_dataset.stats(),
            "dataset_splits": {
                "train": self.train_dataset.split,
                "validation": self.validation_dataset.split,
                "test_used_for_training": False,
            },
            "dataset_fingerprints": self._dataset_fingerprints(),
            "frozen_dataset": self.training_config.dataset_identity,
            "device": str(self.device),
            "host": {
                "torch_threads": torch.get_num_threads(),
                "cuda_available": torch.cuda.is_available(),
                "platform": os.uname().machine if hasattr(os, "uname") else "unknown",
            },
            "limitations": [
                "A CPU smoke run is not large-scale foundation-model training.",
                "Objective losses do not by themselves establish semantic understanding.",
                "Token reconstruction accuracy is not compilation or behavioral equivalence.",
            ],
        }

    def _dataset_fingerprints(self) -> list[dict[str, Any]]:
        result = []
        for source in self.train_dataset.sources:
            result.append(
                {
                    "dataset_id": source.dataset_id,
                    "category": source.category,
                    "classification": source.classification,
                    "manifest_sha256": source.sha256,
                    "path": source.path,
                }
            )
        return result

    def _checkpoint_metadata(self, status: str, validation: dict[str, Any] | None = None) -> dict[str, Any]:
        return {
            "run_name": self.training_config.run_name,
            "status": status,
            "initialization": "random",
            "pretrained_weights_loaded": False,
            "architecture_version": self.model_config.architecture_version,
            "parameter_count": self.model.count_parameters(),
            "training_config": self.training_config.to_dict(),
            "objective_config": [objective.to_dict() for objective in OBJECTIVES if objective.objective_id in self.training_config.objectives],
            "dataset_fingerprints": self._dataset_fingerprints(),
            "frozen_dataset": self.training_config.dataset_identity,
            "dataset_splits": {
                "train": self.train_dataset.split,
                "validation": self.validation_dataset.split,
                "test_used_for_training": False,
            },
            "seeds": {"python": self.training_config.seed, "numpy": self.training_config.seed, "torch": self.training_config.seed},
            "mission": self.mission.to_dict(),
            "progress": {"step": self.step, "max_steps": self.training_config.max_steps, "data_cursor": self.data_cursor},
            "aggregator_state": self.aggregator.export_state(),
            "scheduler_config": {
                "warmup_steps": self.training_config.warmup_steps,
                "max_steps": self.training_config.max_steps,
                "min_lr_ratio": self.training_config.min_lr_ratio,
            },
            "validation": validation,
            "initial_parameter_digest": self.initial_digest,
            "current_parameter_digest": parameter_state_digest(self.model),
            "training_history_tail": self.history[-10:],
        }

    def save_interrupt(self, status: str = "interrupted") -> dict[str, Any]:
        return self._save("interrupt.pt", status, None)

    def _save(self, name: str, status: str, validation: dict[str, Any] | None = None) -> dict[str, Any]:
        return save_foundation_checkpoint(
            self.run_dir / name,
            self.model,
            self.model_config,
            self.vocab,
            metadata=self._checkpoint_metadata(status, validation),
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            aggregator=self.aggregator,
            step=self.step,
            include_training_state=True,
        )

    def _next_batch(self, iterator, fixed_order: list[int] | None):
        try:
            return next(iterator)
        except StopIteration:
            return None

    def train(self, *, evaluate: bool = True) -> dict[str, Any]:
        config = self.training_config
        fixed_order = list(range(len(self.train_dataset)))
        random.Random(config.seed).shuffle(fixed_order)
        iterator = self.train_dataset.batches(
            batch_size=config.batch_size,
            shuffle=False,
            seed=config.seed,
            mask_instructions=True,
            mask_bytes=True,
            mask_probability=config.mask_instruction_probability,
            byte_mask_probability=config.mask_byte_probability,
            modality_dropout=config.modality_dropout,
            fixed_order=fixed_order,
            grouped_fraction=config.equivalence_sampling_fraction,
            groups_per_batch=config.equivalence_groups_per_batch,
            members_per_group=config.equivalence_members_per_group,
        )
        # Resume at the saved microbatch position for this fixed-order stream.
        # This is exact for the default accumulation=1 path and for checkpoints
        # saved at an optimizer-step boundary.
        skip = max(0, int(getattr(self, "data_cursor", 0)))
        while skip > 0:
            try:
                next(iterator)
                skip -= 1
            except StopIteration:
                iterator = self.train_dataset.batches(
                    batch_size=config.batch_size,
                    shuffle=False,
                    seed=config.seed + self.step,
                    mask_instructions=True,
                    mask_bytes=True,
                    mask_probability=config.mask_instruction_probability,
                    byte_mask_probability=config.mask_byte_probability,
                    modality_dropout=config.modality_dropout,
                    fixed_order=fixed_order,
                )
                self.data_cursor = 0
                skip = 0
                break
        accumulation = max(1, config.gradient_accumulation_steps)
        pending = 0
        validation: dict[str, Any] | None = None
        self.manifest["status"] = "running"
        _json_write(self.run_dir / "manifest.json", self.manifest)
        try:
            while self.step < config.max_steps:
                batch = self._next_batch(iterator, fixed_order)
                if batch is None:
                    iterator = self.train_dataset.batches(
                        batch_size=config.batch_size,
                        shuffle=False,
                        seed=config.seed + self.step,
                        mask_instructions=True,
                        mask_bytes=True,
                        mask_probability=config.mask_instruction_probability,
                        byte_mask_probability=config.mask_byte_probability,
                        modality_dropout=config.modality_dropout,
                        fixed_order=fixed_order,
                        grouped_fraction=config.equivalence_sampling_fraction,
                        groups_per_batch=config.equivalence_groups_per_batch,
                        members_per_group=config.equivalence_members_per_group,
                    )
                    batch = next(iterator)
                    self.data_cursor = 0
                self.data_cursor += 1
                batch = _move_batch(batch, self.device)
                self.model.train()
                total, results, outputs = self._losses(batch)
                if not torch.isfinite(total):
                    self.failures += 1
                    self.optimizer.zero_grad(set_to_none=True)
                    _jsonl_append(
                        self.run_dir / "failures.jsonl",
                        {"step": self.step, "event": "non_finite_loss", "failures": self.failures},
                    )
                    if self.failures >= 3:
                        raise TrainingFailure("three consecutive non-finite training losses")
                    continue
                diagnostics = {}
                if config.gradient_diagnostics_interval and self.step % config.gradient_diagnostics_interval == 0:
                    diagnostics = gradient_norms(self.model, self._raw_losses)
                scaled = total / accumulation
                scaled.backward()
                pending += 1
                if pending >= accumulation:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), config.grad_clip)
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad(set_to_none=True)
                    pending = 0
                    self.step += 1
                    record = {
                        "step": self.step,
                        "loss": float(total.detach()),
                        "objectives": [result.to_dict() for result in results],
                        "active_objectives": [result.objective_id for result in results if result.active],
                        "inactive_objectives": [result.objective_id for result in results if not result.active],
                        "gradient_norms": diagnostics,
                        "learning_rate": self.scheduler.get_last_lr()[0],
                        "rss_mb": round(_rss_mb(), 2),
                        "elapsed_seconds": round(time.perf_counter() - self.started_monotonic, 3),
                    }
                    self.history.append(record)
                    _jsonl_append(self.run_dir / "metrics.jsonl", record)
                    if self.step_callback is not None:
                        self.step_callback(record)
                    if config.val_interval and self.step % config.val_interval == 0:
                        validation = evaluate_dataset(
                            self.model,
                            self.validation_dataset,
                            config.objectives,
                            batch_size=config.batch_size,
                            seed=config.seed + self.step,
                            batch_transform=self.batch_transform,
                        )
                        _json_write(self.run_dir / "validation.json", validation)
                        score = validation["weighted_validation_score"]
                        if self.best_validation is None or score < self.best_validation:
                            self.best_validation = score
                            self.best_step = self.step
                            self.manifest["best_validation_score"] = score
                            self.manifest["best_step"] = self.step
                            self._save("best.pt", "best_validation", validation)
                        elif (
                            config.early_stopping_patience
                            and self.best_step is not None
                            and self.step - self.best_step >= config.early_stopping_patience * max(1, config.val_interval)
                        ):
                            self.manifest["status"] = "early_stopped"
                            self.stop_reason = "early_stopping"
                            break
                    if config.checkpoint_interval and self.step % config.checkpoint_interval == 0:
                        self._save("last.pt", "checkpoint", validation)
                    if self.resource_guard is not None:
                        guard_reason = self.resource_guard()
                        if guard_reason:
                            self.stop_reason = "sigint" if str(guard_reason) == "sigint" else "resource_limit"
                            self.manifest["status"] = "interrupted" if self.stop_reason == "sigint" else "resource_stopped"
                            self.manifest["resource_stop_reason"] = str(guard_reason)
                            self.save_interrupt(self.manifest["status"])
                            _json_write(self.run_dir / "manifest.json", self.manifest)
                            break
                    self.manifest["status"] = "running"
                    self.manifest["progress"] = {"step": self.step, "max_steps": config.max_steps}
                    if self.step % 5 == 0:
                        _json_write(self.run_dir / "manifest.json", self.manifest)
            final_validation = validation
            if final_validation is None and evaluate:
                final_validation = evaluate_dataset(
                    self.model,
                    self.validation_dataset,
                    config.objectives,
                    batch_size=config.batch_size,
                    seed=config.seed + self.step,
                    batch_transform=self.batch_transform,
                )
            _json_write(self.run_dir / "validation.json", final_validation)
            final_status = "completed" if self.stop_reason == "max_steps" else self.stop_reason
            checkpoint = self._save("final.pt", final_status, final_validation)
        except KeyboardInterrupt:
            self.manifest["status"] = "interrupted"
            self.manifest["interrupt"] = {"step": self.step}
            self.save_interrupt("interrupted")
            _json_write(self.run_dir / "manifest.json", self.manifest)
            raise
        except Exception as exc:
            self.manifest["status"] = "failed"
            self.manifest["failure"] = {"type": type(exc).__name__, "message": str(exc), "step": self.step}
            _json_write(self.run_dir / "manifest.json", self.manifest)
            raise
        self.model.eval()
        final_digest = parameter_state_digest(self.model)
        changed = sum(
            1
            for name, tensor in self.model.state_dict().items()
            if not torch.equal(tensor.detach().cpu(), self._initial_state[name])
        )
        delta_squared = 0.0
        for name, tensor in self.model.state_dict().items():
            delta_squared += float((tensor.detach().cpu().float() - self._initial_state[name].float()).pow(2).sum())
        resources = {
            "wall_seconds": round(time.perf_counter() - self.started_monotonic, 3),
            "peak_rss_mb": round(_rss_mb(), 2),
            "torch_threads": torch.get_num_threads(),
            "device": str(self.device),
            "cuda_available": torch.cuda.is_available(),
        }
        result = {
            "status": "completed" if self.stop_reason == "max_steps" else self.stop_reason,
            "stop_reason": self.stop_reason,
            "run_dir": str(self.run_dir),
            "run_name": config.run_name,
            "model_config": self.model_config.to_dict(),
            "parameter_count": self.model.count_parameters(),
            "steps": self.step,
            "history": self.history,
            "objective_summary": self._objective_summary(),
            "validation": final_validation,
            "checkpoint": checkpoint,
            "best_checkpoint": str(self.run_dir / "best.pt") if (self.run_dir / "best.pt").exists() else None,
            "best_validation_score": self.best_validation,
            "parameter_update": {
                "initial_digest": self.initial_digest,
                "final_digest": final_digest,
                "changed_parameter_tensors": changed,
                "total_parameter_tensors": len(self._initial_state),
                "delta_l2": delta_squared ** 0.5,
                "parameters_updated": delta_squared > 0.0,
            },
            "resources": resources,
            "mission": self.mission.to_dict(),
            "dataset_fingerprints": self._dataset_fingerprints(),
            "resumed_from": getattr(self, "resumed_from", None),
            "limitations": [
                "This run is a small CPU pilot, not large-scale foundation-model training.",
                "A finite loss or token accuracy is not semantic understanding or reconstruction quality.",
                "No test split was used for training or objective selection.",
            ],
        }
        self.manifest["status"] = result["status"]
        self.manifest["stop_reason"] = self.stop_reason
        self.manifest["completed_at"] = _now()
        self.manifest["validation"] = final_validation
        self.manifest["checkpoint"] = checkpoint
        self.manifest["parameter_update"] = result["parameter_update"]
        self.manifest["resources"] = resources
        _json_write(self.run_dir / "manifest.json", self.manifest)
        _json_write(self.run_dir / "result.json", result)
        return result

    def _losses(self, batch: dict[str, Any]) -> tuple[torch.Tensor, list[ObjectiveResult], dict[str, torch.Tensor]]:
        config = self.training_config
        if self.batch_transform is not None:
            batch = {**batch, "ir_target_source": batch["instr_ir"].clone()}
            batch = self.batch_transform(batch)
        objectives = config.objectives
        needs_c = _decoder_requested(objectives, batch, "c")
        needs_java = _decoder_requested(objectives, batch, "java")
        if needs_c:
            outputs = self.model(batch, run_decoder=True, decoder_language="c")
        elif needs_java:
            outputs = self.model(batch, run_decoder=True, decoder_language="java")
        else:
            outputs = self.model(batch)
        _attach_evidence_outputs(self.model, batch, outputs, self.step + config.seed, objectives)
        if "instruction_autoregressive" in objectives and batch.get("run_instruction_autoregressive", True):
            ar = _autoregressive_batch(batch)
            ar_outputs = self.model(ar, causal_instructions=True)
            outputs["instr_ar_logits"] = ar_outputs["instruction_logits"]
            batch = {**batch, "instr_ar_targets": ar["instr_ar_targets"], "instr_ar_mask": ar["instr_ar_mask"]}
        raw = compute_objective_losses(outputs, batch, objectives)
        self._raw_losses = raw
        total, results = self.aggregator(raw, self.step)
        return total, results, outputs

    def _objective_summary(self) -> dict[str, Any]:
        summary: dict[str, dict[str, Any]] = {}
        for objective in OBJECTIVES:
            if objective.objective_id not in self.training_config.objectives:
                continue
            values = [
                next((item for item in record["objectives"] if item["objective_id"] == objective.objective_id), None)
                for record in self.history
            ]
            values = [value for value in values if value and value["active"]]
            summary[objective.objective_id] = {
                "mission_objective": objective.mission_objective,
                "base_weight": objective.base_weight,
                "active_steps": len(values),
                "initial_loss": values[0]["raw"] if values else None,
                "final_loss": values[-1]["raw"] if values else None,
                "min_loss": min(value["raw"] for value in values) if values else None,
            }
        return summary

    def _prepare_initial_state(self) -> None:
        self._initial_state = {name: tensor.detach().cpu().clone() for name, tensor in self.model.state_dict().items()}

    def initialize_from(self, path: str | Path) -> dict[str, Any]:
        """Load verified random weights into a fresh run.

        Optimizer, scheduler and aggregator state are created new.  This method
        never accepts a trained/resumable checkpoint and never records an
        optimizer step.
        """
        from nexus.foundation.checkpoint import validate_initialization_checkpoint

        expected_config = config_for_vocabulary(self.model_config, self.vocab)
        validated = validate_initialization_checkpoint(
            path,
            expected_config=expected_config,
            expected_vocab=self.vocab,
        )
        loaded = load_foundation_checkpoint(
            path,
            map_location=str(self.device),
            expected_vocab_sha256=self.vocab.digest(),
            allow_pretrained=False,
            load_training_state=True,
        )
        if loaded.checkpoint_type != "INITIALIZATION":
            raise ValueError("initialize_from requires an INITIALIZATION checkpoint; use --resume-from for a training checkpoint")
        if loaded.config.config_hash() != expected_config.config_hash():
            raise ValueError("initialization checkpoint effective configuration does not match this run")
        if loaded.optimizer_state is not None or loaded.scheduler_state is not None:
            raise ValueError("initialization checkpoint unexpectedly contains optimizer/scheduler state")
        if self.training_config.dataset_identity and loaded.metadata.get("frozen_dataset"):
            if loaded.metadata.get("frozen_dataset") != self.training_config.dataset_identity:
                raise ValueError("initialization checkpoint frozen dataset identity does not match the current frozen dataset")
        self.model.load_state_dict(loaded.model.state_dict(), strict=True)
        self.model.to(self.device)
        self.model.eval()
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.training_config.learning_rate,
            weight_decay=self.training_config.weight_decay,
        )
        self.aggregator = LossAggregator(mode=self.training_config.aggregation, enabled=self.training_config.objectives)
        if self.training_config.aggregation == "uncertainty":
            self.optimizer.add_param_group({"params": list(self.aggregator.parameters()), "weight_decay": 0.0})
        self.scheduler = _scheduler(
            self.optimizer,
            self.training_config.warmup_steps,
            self.training_config.max_steps,
            self.training_config.min_lr_ratio,
        )
        self.step = 0
        self.data_cursor = 0
        self.history = []
        self.best_validation = None
        self.best_step = None
        self.failures = 0
        self.stop_reason = "max_steps"
        random.seed(self.training_config.seed)
        np.random.seed(self.training_config.seed)
        torch.manual_seed(self.training_config.seed)
        self._prepare_initial_state()
        self.initial_digest = parameter_state_digest(self.model)
        self.initialized_from = {
            "path": str(path),
            "sha256": loaded.sha256,
            "checkpoint_type": loaded.checkpoint_type,
            "step": 0,
            "parameter_count": self.model.count_parameters(),
            "config_hash": self.model.config.config_hash(),
            "vocab_sha256": self.vocab.digest(),
            "metadata_schema": validated.get("metadata_schema"),
            "from_scratch_state_verified": validated.get("from_scratch_state_verified"),
        }
        self.manifest = self._manifest()
        self.manifest["initialization"] = "verified_random_initialization_checkpoint"
        self.manifest["initialized_from"] = self.initialized_from
        _json_write(self.run_dir / "manifest.json", self.manifest)
        _json_write(self.run_dir / "model_profile.json", parameter_report(self.model))
        return self.initialized_from

    def resume(self, path: str | Path) -> dict[str, Any]:
        loaded = load_foundation_checkpoint(
            path,
            map_location=str(self.device),
            expected_vocab_sha256=self.vocab.digest(),
            allow_pretrained=False,
            load_training_state=True,
        )
        if loaded.config.config_hash() != self.model_config.config_hash():
            raise ValueError("checkpoint model configuration does not match this trainer")
        if loaded.optimizer_state is None or loaded.scheduler_state is None:
            raise ValueError(
                "checkpoint does not contain optimizer/scheduler state and cannot be resumed; "
                "it may be an initialization checkpoint, in which case use --init-from"
            )
        if self.training_config.dataset_identity and loaded.metadata.get("frozen_dataset") != self.training_config.dataset_identity:
            raise ValueError("checkpoint frozen dataset identity does not match the current frozen dataset")
        checkpoint_fingerprints = loaded.metadata.get("dataset_fingerprints")
        if checkpoint_fingerprints != self._dataset_fingerprints():
            raise ValueError("checkpoint dataset fingerprints do not match the current training data")
        self.model = loaded.model.to(self.device)
        self.model_config = loaded.config
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.training_config.learning_rate,
            weight_decay=self.training_config.weight_decay,
        )
        if self.training_config.aggregation == "uncertainty":
            self.optimizer.add_param_group({"params": list(self.aggregator.parameters()), "weight_decay": 0.0})
        self.optimizer.load_state_dict(loaded.optimizer_state)
        checkpoint_scheduler = (loaded.metadata or {}).get("scheduler_config") or {}
        if checkpoint_scheduler:
            for key in ("warmup_steps", "min_lr_ratio"):
                if key in checkpoint_scheduler and float(checkpoint_scheduler[key]) != float(getattr(self.training_config, key)):
                    raise ValueError(f"resume changes {key}: checkpoint={checkpoint_scheduler[key]} current={getattr(self.training_config, key)}")
        scheduler_config = {
            "warmup_steps": self.training_config.warmup_steps,
            "max_steps": self.training_config.max_steps,
            "min_lr_ratio": self.training_config.min_lr_ratio,
        }
        self.scheduler = _scheduler(
            self.optimizer,
            int(scheduler_config["warmup_steps"]),
            int(scheduler_config["max_steps"]),
            float(scheduler_config["min_lr_ratio"]),
        )
        self.scheduler.load_state_dict(loaded.scheduler_state)
        self.resumed_scheduler_config = dict(scheduler_config)
        self.checkpoint_scheduler_config = dict(checkpoint_scheduler)
        if loaded.aggregator_state is not None:
            self.aggregator.load_state_dict(loaded.aggregator_state)
        self.aggregator.import_state((loaded.metadata or {}).get("aggregator_state"))
        if loaded.torch_rng_state is not None:
            torch.set_rng_state(loaded.torch_rng_state.cpu())
        if loaded.python_rng_state is not None:
            import random as _random

            _random.setstate(loaded.python_rng_state)
        self.step = loaded.step
        progress = (loaded.metadata or {}).get("progress") or {}
        self.data_cursor = int(progress.get("data_cursor", (loaded.metadata or {}).get("data_cursor", 0)) or 0)
        self._prepare_initial_state()
        self.resumed_from = {"path": str(path), "sha256": loaded.sha256, "step": self.step}
        return {**self.resumed_from, "metadata": loaded.metadata}


def build_trainer(
    profile: str = "cpu_dev",
    training_config: FoundationTrainingConfig | None = None,
    train_dataset: FoundationDataset | None = None,
    validation_dataset: FoundationDataset | None = None,
    vocab: Vocabulary | None = None,
    **overrides: Any,
) -> FoundationTrainer:
    config = training_config or FoundationTrainingConfig(**overrides)
    model_config = profile_config(profile)
    if train_dataset is None or validation_dataset is None or vocab is None:
        raise ValueError("train_dataset, validation_dataset and vocab are required")
    trainer = FoundationTrainer(model_config, config, train_dataset, validation_dataset, vocab)
    trainer._prepare_initial_state()
    return trainer
