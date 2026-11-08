"""Capability-separated evaluation for the unified NEXUS-BI model.

The suite reports byte, instruction, structural, semantic, retrieval, purpose,
reconstruction and generalization categories separately.  Baselines, confidence
intervals, calibration, ablations, permutation tests and negative witnesses are
included.  Historical phase benchmarks are not reused as evidence.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from nexus.foundation.checkpoint import load_foundation_checkpoint
from nexus.foundation.dataset import FoundationDataset
from nexus.foundation.model import build_foundation_model
from nexus.foundation.objectives import compute_objective_losses
from nexus.foundation.reconstruction import UnifiedReconstructor
from nexus.foundation.representation import mask_evidence_batch

EVIDENCE_OBJECTIVES = {
    "ir_op_prediction",
    "register_effect_prediction",
    "memory_effect_prediction",
    "cfg_edge_prediction",
}


def _attach_evidence(model, batch: dict[str, Any], outputs: dict[str, Any], seed: int, objectives: list[str]) -> None:
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


def _bootstrap(values: list[float], iterations: int = 1000, seed: int = 20261103) -> dict[str, float] | None:
    if not values:
        return None
    array = np.asarray(values, dtype=np.float64)
    generator = np.random.default_rng(seed)
    means = []
    for _ in range(iterations):
        sample = array[generator.integers(0, len(array), size=len(array))]
        means.append(float(sample.mean()))
    return {
        "mean": float(array.mean()),
        "low": float(np.quantile(means, 0.025)),
        "high": float(np.quantile(means, 0.975)),
        "n": len(values),
    }


def _macro_f1(targets: list[int], predictions: list[int], number_of_classes: int) -> float:
    scores = []
    for label in range(number_of_classes):
        tp = sum(1 for target, prediction in zip(targets, predictions) if target == label and prediction == label)
        fp = sum(1 for target, prediction in zip(targets, predictions) if target != label and prediction == label)
        fn = sum(1 for target, prediction in zip(targets, predictions) if target == label and prediction != label)
        if tp + fp + fn == 0:
            continue
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        scores.append(2 * precision * recall / max(1e-9, precision + recall))
    return float(np.mean(scores)) if scores else 0.0


def _ece(confidences: list[float], correct: list[bool], bins: int = 10) -> dict[str, Any]:
    if not confidences:
        return {"status": "not_attempted", "ece": None}
    buckets = [[] for _ in range(bins)]
    for confidence, hit in zip(confidences, correct):
        index = min(bins - 1, int(confidence * bins))
        buckets[index].append((confidence, 1.0 if hit else 0.0))
    total = len(confidences)
    ece = 0.0
    reliability = []
    for index, bucket in enumerate(buckets):
        if not bucket:
            continue
        confidence = sum(item[0] for item in bucket) / len(bucket)
        accuracy = sum(item[1] for item in bucket) / len(bucket)
        ece += len(bucket) / total * abs(confidence - accuracy)
        reliability.append({"bin": index, "count": len(bucket), "confidence": confidence, "accuracy": accuracy})
    return {"status": "measured", "ece": round(ece, 6), "bins": reliability, "note": "Calibration is only meaningful where labels exist."}


@dataclass
class Collected:
    losses: dict[str, dict[str, float]]
    metrics: dict[str, float]
    instruction_correct: list[float]
    byte_correct: list[float]
    ir_correct: list[float]
    purpose_correct: list[float]
    c_correct: list[float]
    instruction_targets: list[int]
    byte_targets: list[int]
    ir_targets: list[int]
    purpose_targets: list[int]
    purpose_predictions: list[int]
    confidences: list[float]
    confidence_correct: list[bool]
    sample_metrics: list[dict[str, Any]]
    cfg_tp: int
    cfg_fp: int
    cfg_fn: int
    samples: int


@torch.no_grad()
def collect(model, dataset: FoundationDataset, objectives: list[str], mask_seed: int = 20261103) -> Collected:
    model.eval()
    losses: dict[str, list[tuple[float, int]]] = {}
    instruction_correct: list[float] = []
    byte_correct: list[float] = []
    ir_correct: list[float] = []
    purpose_correct: list[float] = []
    c_correct: list[float] = []
    instruction_targets: list[int] = []
    byte_targets: list[int] = []
    ir_targets: list[int] = []
    purpose_targets: list[int] = []
    purpose_predictions: list[int] = []
    confidences: list[float] = []
    confidence_correct: list[bool] = []
    sample_metrics: list[dict[str, Any]] = []
    cfg_tp = cfg_fp = cfg_fn = 0
    samples = 0
    for index, batch in enumerate(
        dataset.batches(
            batch_size=4,
            shuffle=False,
            seed=mask_seed,
            mask_instructions=True,
            mask_bytes=True,
            mask_probability=0.15,
            byte_mask_probability=0.08,
            modality_dropout=0.0,
            fixed_order=list(range(len(dataset.samples))),
        )
    ):
        needs_c = "c_reconstruction" in objectives and "c_mask" in batch and batch["c_mask"].any()
        outputs = model(batch, run_decoder=True, decoder_language="c") if needs_c else model(batch)
        _attach_evidence(model, batch, outputs, mask_seed + index, objectives)
        raw = compute_objective_losses(outputs, batch, objectives)
        for objective_id, (loss, count, _note) in raw.items():
            if count > 0:
                losses.setdefault(objective_id, []).append((float(loss.detach()), count))
        mask = batch["instr_mlm_mask"]
        for row in range(mask.size(0)):
            row_mask = mask[row]
            if row_mask.any():
                correct = (outputs["instruction_logits"][row, row_mask].argmax(-1) == batch["instr_targets"][row, row_mask]).float()
                instruction_correct.extend(correct.tolist())
                instruction_targets.extend(batch["instr_targets"][row, row_mask].tolist())
        bmask = batch["byte_mlm_mask"]
        for row in range(bmask.size(0)):
            row_mask = bmask[row]
            if row_mask.any():
                correct = (outputs["byte_logits"][row, row_mask].argmax(-1) == batch["byte_targets"][row, row_mask]).float()
                byte_correct.extend(correct.tolist())
                byte_targets.extend(batch["byte_targets"][row, row_mask].tolist())
        ir_mask = batch.get("ir_mlm_mask")
        ir_logits = outputs.get("ir_mlm_logits")
        if ir_mask is not None and ir_logits is not None:
            for row in range(ir_mask.size(0)):
                row_mask = ir_mask[row]
                if row_mask.any():
                    correct = (ir_logits[row, row_mask].argmax(-1) == batch["ir_targets"][row, row_mask]).float()
                    ir_correct.extend(correct.tolist())
                    ir_targets.extend(batch["ir_targets"][row, row_mask].tolist())
        purpose_mask = batch["purpose_target"].ge(0)
        purpose_logits = outputs["purpose_logits"]
        if purpose_mask.any():
            predictions = purpose_logits[purpose_mask].argmax(-1)
            purpose_predictions.extend(predictions.tolist())
            purpose_targets.extend(batch["purpose_target"][purpose_mask].tolist())
            purpose_correct.extend((predictions == batch["purpose_target"][purpose_mask]).float().tolist())
            probabilities = torch.softmax(purpose_logits[purpose_mask], dim=-1)
            confidence_values, _ = probabilities.max(dim=-1)
            confidences.extend(confidence_values.tolist())
            confidence_correct.extend((predictions == batch["purpose_target"][purpose_mask]).tolist())
        block_mask = batch["block_mask"]
        valid = block_mask.unsqueeze(1) & block_mask.unsqueeze(2)
        diagonal = torch.eye(valid.size(1), dtype=torch.bool, device=valid.device).unsqueeze(0)
        valid = valid & ~diagonal
        for row in range(valid.size(0)):
            row_valid = valid[row]
            if row_valid.any() and "cfg_masked_logits" in outputs:
                predicted = outputs["cfg_masked_logits"][row][row_valid] > 0
                target = batch["cfg_target"][row][row_valid] > 0.5
                cfg_tp += int((predicted & target).sum())
                cfg_fp += int((predicted & ~target).sum())
                cfg_fn += int((~predicted & target).sum())
        if "c_logits" in outputs:
            target = batch["c_ids"][:, 1:]
            c_mask = batch["c_mask"][:, 1:] if "c_mask" in batch else torch.ones_like(target, dtype=torch.bool)
            for row in range(target.size(0)):
                row_mask = c_mask[row]
                if row_mask.any():
                    correct = (outputs["c_logits"][row, row_mask].argmax(-1) == target[row, row_mask]).float()
                    c_correct.extend(correct.tolist())
        ir_mask_tensor = batch.get("ir_mlm_mask")
        for row, meta in enumerate(batch.get("sample_meta", [])):
            labels = meta.get("labels") or {}
            row_instruction_total = int(mask[row].sum())
            row_instruction_correct = int(
                (outputs["instruction_logits"][row, mask[row]].argmax(-1) == batch["instr_targets"][row, mask[row]]).sum()
            ) if row_instruction_total else 0
            row_ir_total = int(ir_mask_tensor[row].sum()) if ir_mask_tensor is not None else 0
            row_ir_correct = int(
                (outputs["ir_mlm_logits"][row, ir_mask_tensor[row]].argmax(-1) == batch["ir_targets"][row, ir_mask_tensor[row]]).sum()
            ) if row_ir_total and "ir_mlm_logits" in outputs else 0
            row_cfg_tp = row_cfg_fp = row_cfg_fn = 0
            if "cfg_masked_logits" in outputs and valid[row].any():
                predicted = outputs["cfg_masked_logits"][row][valid[row]] > 0
                target = batch["cfg_target"][row][valid[row]] > 0.5
                row_cfg_tp = int((predicted & target).sum())
                row_cfg_fp = int((predicted & ~target).sum())
                row_cfg_fn = int((~predicted & target).sum())
            row_purpose_total = int(purpose_mask[row].sum())
            row_purpose_correct = int(
                (outputs["purpose_logits"][row, purpose_mask[row]].argmax(-1) == batch["purpose_target"][row, purpose_mask[row]]).sum()
            ) if row_purpose_total else 0
            sample_metrics.append(
                {
                    "sample_id": meta.get("sample_id"),
                    "function_id": meta.get("function_id"),
                    "project_id": meta.get("project_id"),
                    "compiler": labels.get("compiler"),
                    "optimization": labels.get("optimization"),
                    "arch": labels.get("arch"),
                    "debug_info": labels.get("debug_info"),
                    "purpose_label": meta.get("purpose_label"),
                    "instruction_total": row_instruction_total,
                    "instruction_correct": row_instruction_correct,
                    "ir_total": row_ir_total,
                    "ir_correct": row_ir_correct,
                    "cfg_tp": row_cfg_tp,
                    "cfg_fp": row_cfg_fp,
                    "cfg_fn": row_cfg_fn,
                    "purpose_total": row_purpose_total,
                    "purpose_correct": row_purpose_correct,
                }
            )
        samples += batch["instr_id"].size(0)
    loss_report = {}
    for objective_id, values in losses.items():
        count = sum(value[1] for value in values)
        loss_report[objective_id] = {
            "mean_loss": sum(value[0] * value[1] for value in values) / max(1, count),
            "count": count,
        }
    metrics = {
        "instruction_mlm_accuracy": float(np.mean(instruction_correct)) if instruction_correct else None,
        "byte_mlm_accuracy": float(np.mean(byte_correct)) if byte_correct else None,
        "ir_accuracy": float(np.mean(ir_correct)) if ir_correct else None,
        "purpose_accuracy": float(np.mean(purpose_correct)) if purpose_correct else None,
        "purpose_macro_f1": _macro_f1(purpose_targets, purpose_predictions, 16) if purpose_targets else None,
        "c_token_accuracy": float(np.mean(c_correct)) if c_correct else None,
        "cfg_precision": cfg_tp / max(1.0, cfg_tp + cfg_fp),
        "cfg_recall": cfg_tp / max(1.0, cfg_tp + cfg_fn),
        "cfg_f1": 2 * cfg_tp / max(1.0, 2 * cfg_tp + cfg_fp + cfg_fn),
    }
    return Collected(
        losses=loss_report,
        metrics=metrics,
        instruction_correct=instruction_correct,
        byte_correct=byte_correct,
        ir_correct=ir_correct,
        purpose_correct=purpose_correct,
        c_correct=c_correct,
        instruction_targets=instruction_targets,
        byte_targets=byte_targets,
        ir_targets=ir_targets,
        purpose_targets=purpose_targets,
        purpose_predictions=purpose_predictions,
        confidences=confidences,
        confidence_correct=confidence_correct,
        sample_metrics=sample_metrics,
        cfg_tp=cfg_tp,
        cfg_fp=cfg_fp,
        cfg_fn=cfg_fn,
        samples=samples,
    )


@torch.no_grad()
def function_embeddings(model, dataset: FoundationDataset) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]:
    model.eval()
    embeddings = []
    histograms = []
    counts = []
    metadata = []
    for batch in dataset.evaluation_batches(batch_size=4):
        outputs = model(batch)
        embeddings.append(outputs["function_embedding"].cpu().numpy())
        for sample in dataset.samples[len(metadata) : len(metadata) + batch["instr_id"].size(0)]:
            histogram = np.zeros(512, dtype=np.float64)
            for identifier in sample.get("instr_mnemonic", []):
                if 0 <= identifier < len(histogram):
                    histogram[identifier] += 1
            histogram /= max(1.0, histogram.sum())
            histograms.append(histogram)
            counts.append([len(sample.get("instr_id", [])), len(sample.get("block_mask", []))])
            metadata.append(
                {
                    "sample_id": sample.get("sample_id"),
                    "function_id": sample.get("function_id"),
                    "project_id": sample.get("project_id"),
                    "source_sha256": sample.get("source_sha256"),
                    "binary_sha256": sample.get("binary_sha256"),
                    "compiler": (sample.get("labels") or {}).get("compiler"),
                    "optimization": (sample.get("labels") or {}).get("optimization"),
                }
            )
    if not embeddings:
        return np.zeros((0, 1)), np.zeros((0, 512)), np.zeros((0, 2)), []
    return np.concatenate(embeddings), np.asarray(histograms), np.asarray(counts, dtype=np.float64), metadata


def _cosine_scores(query: np.ndarray, gallery: np.ndarray) -> np.ndarray:
    query = query / np.clip(np.linalg.norm(query, axis=-1, keepdims=True), 1e-9, None)
    gallery = gallery / np.clip(np.linalg.norm(gallery, axis=-1, keepdims=True), 1e-9, None)
    return query @ gallery.T


def retrieval_metrics(
    embeddings: np.ndarray,
    metadata: list[dict[str, Any]],
    histograms: np.ndarray,
    counts: np.ndarray,
    permutations: int = 50,
    seed: int = 20261103,
) -> dict[str, Any]:
    usable = [index for index, item in enumerate(metadata) if item.get("source_sha256")]
    if len(usable) < 2:
        return {"status": "unavailable", "reason": "fewer than two source-labelled functions"}
    identities = [metadata[index]["source_sha256"] for index in usable]
    binaries = [metadata[index]["binary_sha256"] for index in usable]
    model_scores = _cosine_scores(embeddings[usable], embeddings[usable])
    mnemonic_scores = _cosine_scores(histograms[usable], histograms[usable])
    count_scores = _cosine_scores(counts[usable], counts[usable])

    def recall(scores: np.ndarray, labels: list[str], binary_ids: list[str]) -> dict[str, float]:
        ranks = []
        for row, label in enumerate(labels):
            order = np.argsort(-scores[row])
            filtered = [index for index in order if binary_ids[index] != binary_ids[row]]
            if not filtered:
                continue
            rank = next((position for position, index in enumerate(filtered) if labels[index] == label), None)
            ranks.append(rank if rank is not None else len(filtered))
        if not ranks:
            return {"recall_at_1": 0.0, "recall_at_5": 0.0, "mrr": 0.0, "queries": 0}
        return {
            "recall_at_1": sum(1 for rank in ranks if rank == 0) / len(ranks),
            "recall_at_5": sum(1 for rank in ranks if rank < 5) / len(ranks),
            "mrr": float(np.mean([1.0 / (rank + 1) for rank in ranks])),
            "queries": len(ranks),
        }

    model_result = recall(model_scores, identities, binaries)
    mnemonic_result = recall(mnemonic_scores, identities, binaries)
    count_result = recall(count_scores, identities, binaries)
    generator = np.random.default_rng(seed)
    permutation_recalls = []
    for _ in range(permutations):
        shuffled = list(identities)
        generator.shuffle(shuffled)
        permutation_recalls.append(recall(model_scores, shuffled, binaries)["recall_at_1"])
    actual = model_result["recall_at_1"]
    p_value = (1 + sum(1 for value in permutation_recalls if value >= actual)) / (permutations + 1)
    return {
        "status": "measured",
        "queries": model_result["queries"],
        "model": model_result,
        "baseline_mnemonic_histogram": mnemonic_result,
        "baseline_instruction_count": count_result,
        "permutation": {
            "iterations": permutations,
            "mean_recall_at_1": float(np.mean(permutation_recalls)),
            "p_value": p_value,
            "note": "Permutation shuffles function identities across the gallery.",
        },
        "identity": "source-function SHA-256 where exact name alignment exists; otherwise the sample is excluded",
        "limitations": [
            "This is function retrieval on a controlled corpus, not semantic equivalence.",
            "Exact source-name alignment may favour functions with distinctive source structure.",
        ],
    }


@torch.no_grad()
def ablation_metrics(model, dataset: FoundationDataset, objectives: list[str]) -> dict[str, Any]:
    variants = {
        "full": lambda batch: batch,
        "bytes_off": lambda batch: {**batch, "byte_mask": torch.zeros_like(batch["byte_mask"])},
        "ir_off": lambda batch: {**batch, "instr_ir": torch.zeros_like(batch["instr_ir"])},
        "graph_off": lambda batch: {
            **batch,
            "instruction_edge_mask": torch.zeros_like(batch["instruction_edge_mask"]),
            "block_edge_mask": torch.zeros_like(batch["block_edge_mask"]),
        },
        "metadata_off": lambda batch: {**batch, "function_features": torch.zeros_like(batch["function_features"])},
    }
    results = {}
    for name, transform in variants.items():
        instruction_correct = []
        ir_correct = []
        cfg_tp = cfg_fp = cfg_fn = 0
        for batch in dataset.batches(batch_size=4, shuffle=False, mask_instructions=True, mask_bytes=False, seed=20261103, fixed_order=list(range(len(dataset.samples)))):
            batch = transform(batch)
            evidence_batch = mask_evidence_batch(batch, seed=20261103)
            outputs = model(evidence_batch)
            mask = evidence_batch["instr_mlm_mask"]
            for row in range(mask.size(0)):
                row_mask = mask[row]
                if row_mask.any():
                    instruction_correct.extend(
                        (outputs["instruction_logits"][row, row_mask].argmax(-1) == batch["instr_targets"][row, row_mask]).float().tolist()
                    )
            ir_mask = evidence_batch["ir_mlm_mask"]
            for row in range(ir_mask.size(0)):
                row_mask = ir_mask[row]
                if row_mask.any():
                    ir_correct.extend(
                        (outputs["ir_logits"][row, row_mask].argmax(-1) == evidence_batch["ir_targets"][row, row_mask]).float().tolist()
                    )
            block_mask = batch["block_mask"]
            valid = block_mask.unsqueeze(1) & block_mask.unsqueeze(2)
            diagonal = torch.eye(valid.size(1), dtype=torch.bool, device=valid.device).unsqueeze(0)
            valid = valid & ~diagonal
            for row in range(valid.size(0)):
                row_valid = valid[row]
                if row_valid.any():
                    predicted = outputs["cfg_logits"][row][row_valid] > 0
                    target = evidence_batch["cfg_target"][row][row_valid] > 0.5
                    cfg_tp += int((predicted & target).sum())
                    cfg_fp += int((predicted & ~target).sum())
                    cfg_fn += int((~predicted & target).sum())
        results[name] = {
            "instruction_mlm_accuracy": float(np.mean(instruction_correct)) if instruction_correct else None,
            "ir_accuracy": float(np.mean(ir_correct)) if ir_correct else None,
            "cfg_f1": 2 * cfg_tp / max(1.0, 2 * cfg_tp + cfg_fp + cfg_fn),
        }
    return results


def _majority(values: list[int]) -> int | None:
    return Counter(values).most_common(1)[0][0] if values else None


def evaluate_checkpoint(
    checkpoint_path: str | Path,
    test_dataset: FoundationDataset,
    train_dataset: FoundationDataset,
    objectives: list[str] | None = None,
    reconstruction_samples: int = 2,
    permutations: int = 50,
    seed: int = 20261103,
    seed_repeats: int = 2,
    workspace: str | Path | None = None,
) -> dict[str, Any]:
    loaded = load_foundation_checkpoint(checkpoint_path, load_training_state=False)
    objectives = objectives or [
        "instruction_masked_modeling",
        "byte_masked_modeling",
        "ir_op_prediction",
        "cfg_edge_prediction",
        "purpose_prediction",
        "c_reconstruction",
    ]
    collected = collect(loaded.model, test_dataset, objectives, mask_seed=seed)
    seed_metrics = [collected.metrics]
    for repeat in range(max(0, seed_repeats - 1)):
        seed_metrics.append(collect(loaded.model, test_dataset, objectives, mask_seed=seed + 1000 * (repeat + 1)).metrics)
    seed_sensitivity = {
        "mask_seeds": [seed + 1000 * index for index in range(max(1, seed_repeats))],
        "instruction_mlm_accuracy": {
            "mean": float(np.mean([item["instruction_mlm_accuracy"] for item in seed_metrics if item["instruction_mlm_accuracy"] is not None])) if any(item["instruction_mlm_accuracy"] is not None for item in seed_metrics) else None,
            "values": [item["instruction_mlm_accuracy"] for item in seed_metrics],
        },
        "ir_accuracy": {
            "mean": float(np.mean([item["ir_accuracy"] for item in seed_metrics if item["ir_accuracy"] is not None])) if any(item["ir_accuracy"] is not None for item in seed_metrics) else None,
            "values": [item["ir_accuracy"] for item in seed_metrics],
        },
    }
    embeddings, histograms, counts, metadata = function_embeddings(loaded.model, test_dataset)
    retrieval = retrieval_metrics(embeddings, metadata, histograms, counts, permutations=permutations, seed=seed)
    train_instructions = [value for sample in train_dataset.samples for value in sample.get("instr_id", [])]
    train_bytes = [value for sample in train_dataset.samples for value in sample.get("byte_ids", [])]
    train_ir = [value for sample in train_dataset.samples for value in sample.get("instr_ir", [])]
    train_purpose = [sample.get("purpose_id") for sample in train_dataset.samples if sample.get("purpose_id", -100) >= 0]
    majority_instruction = _majority(train_instructions)
    majority_byte = _majority(train_bytes)
    majority_ir = _majority(train_ir)
    majority_purpose = _majority(train_purpose)
    baselines = {
        "instruction_majority": (
            sum(1 for value in collected.instruction_targets if value == majority_instruction) / max(1, len(collected.instruction_targets))
            if majority_instruction is not None
            else None
        ),
        "instruction_random": 1.0 / max(1, len(train_dataset.vocab.instruction)),
        "byte_majority": (
            sum(1 for value in collected.byte_targets if value == majority_byte) / max(1, len(collected.byte_targets))
            if majority_byte is not None
            else None
        ),
        "byte_random": 1.0 / 256.0,
        "ir_majority": (
            sum(1 for value in collected.ir_targets if value == majority_ir) / max(1, len(collected.ir_targets))
            if majority_ir is not None
            else None
        ),
        "purpose_majority": (
            sum(1 for value in collected.purpose_targets if value == majority_purpose) / max(1, len(collected.purpose_targets))
            if majority_purpose is not None and collected.purpose_targets
            else None
        ),
    }
    metrics = dict(collected.metrics)
    metrics["instruction_mlm_ci"] = _bootstrap(collected.instruction_correct, seed=seed)
    metrics["byte_mlm_ci"] = _bootstrap(collected.byte_correct, seed=seed + 1)
    metrics["ir_accuracy_ci"] = _bootstrap(collected.ir_correct, seed=seed + 2)
    metrics["purpose_accuracy_ci"] = _bootstrap(collected.purpose_correct, seed=seed + 3)
    metrics["c_token_accuracy_ci"] = _bootstrap(collected.c_correct, seed=seed + 4)
    calibration = _ece(collected.confidences, collected.confidence_correct)
    ablated = ablation_metrics(loaded.model, test_dataset, objectives)
    random_model = build_foundation_model(loaded.config, seed=seed + 999)
    negative = collect(random_model, test_dataset, objectives, mask_seed=seed)
    def _slice_metrics(subset: list[dict[str, Any]]) -> dict[str, Any]:
        instruction_total = sum(item["instruction_total"] for item in subset)
        ir_total = sum(item["ir_total"] for item in subset)
        purpose_total = sum(item["purpose_total"] for item in subset)
        cfg_tp = sum(item["cfg_tp"] for item in subset)
        cfg_fp = sum(item["cfg_fp"] for item in subset)
        cfg_fn = sum(item["cfg_fn"] for item in subset)
        return {
            "samples": len(subset),
            "instruction_mlm_accuracy": (
                sum(item["instruction_correct"] for item in subset) / instruction_total if instruction_total else None
            ),
            "ir_accuracy": sum(item["ir_correct"] for item in subset) / ir_total if ir_total else None,
            "cfg_f1": 2 * cfg_tp / max(1.0, 2 * cfg_tp + cfg_fp + cfg_fn),
            "purpose_accuracy": (
                sum(item["purpose_correct"] for item in subset) / purpose_total if purpose_total else None
            ),
        }

    cross_compiler = {}
    for compiler in sorted({item.get("compiler") for item in collected.sample_metrics if item.get("compiler")}):
        cross_compiler[compiler] = _slice_metrics([item for item in collected.sample_metrics if item.get("compiler") == compiler])
    cross_optimization = {}
    for optimization in sorted({item.get("optimization") for item in collected.sample_metrics if item.get("optimization")}):
        cross_optimization[optimization] = _slice_metrics([item for item in collected.sample_metrics if item.get("optimization") == optimization])
    architectures = sorted({item.get("arch") for item in collected.sample_metrics if item.get("arch")})
    reconstruction = _reconstruction_evaluation(
        checkpoint_path,
        test_dataset,
        loaded.model,
        loaded.vocab,
        loaded.config,
        limit=reconstruction_samples,
        workspace=workspace,
    )
    capability = {
        "byte_understanding": {"status": "measured", "metrics": {"accuracy": metrics.get("byte_mlm_accuracy"), "ci": metrics.get("byte_mlm_ci")}},
        "instruction_understanding": {"status": "measured", "metrics": {"accuracy": metrics.get("instruction_mlm_accuracy"), "ci": metrics.get("instruction_mlm_ci")}},
        "structural_understanding": {"status": "measured", "metrics": {"cfg_f1": metrics.get("cfg_f1")}, "note": "Function-boundary labels and module-level boundaries are not present in this test split."},
        "semantic_understanding": {"status": "measured_against_deterministic_ir_labels", "metrics": {"ir_accuracy": metrics.get("ir_accuracy"), "ci": metrics.get("ir_accuracy_ci")}, "note": "IR labels are deterministic-tool targets, not ground-truth semantics."},
        "function_retrieval": {"status": retrieval.get("status"), "metrics": retrieval if retrieval.get("status") == "measured" else None},
        "function_purpose_inference": {"status": "measured" if collected.purpose_targets else "unavailable", "metrics": {"accuracy": metrics.get("purpose_accuracy"), "macro_f1": metrics.get("purpose_macro_f1"), "ci": metrics.get("purpose_accuracy_ci")}},
        "c_reconstruction": {"status": reconstruction.get("status"), "metrics": reconstruction.get("summary")},
        "java_jni_reconstruction": {"status": "not_tested_in_this_corpus", "note": "The interface and JNI fixture are tested separately; no Java/JNI test split is present."},
        "cross_compiler_generalization": {"status": "measured_by_slice" if len(cross_compiler) > 1 else "unavailable", "slices": cross_compiler},
        "cross_optimization_generalization": {"status": "measured_by_slice" if len(cross_optimization) > 1 else "unavailable", "slices": cross_optimization},
        "cross_architecture_generalization": {"status": "unavailable", "architectures": architectures, "reason": "The registered corpus and decoder are x86-64 only; no cross-architecture execution or decoding is claimed."},
        "project_level_reconstruction": reconstruction.get("project_level"),
    }
    report = {
        "schema_version": "0.1.0",
        "checkpoint": {
            "path": str(checkpoint_path),
            "sha256": loaded.sha256,
            "parameter_count": loaded.model.count_parameters(),
            "metadata": loaded.metadata,
        },
        "datasets": {
            "train": train_dataset.stats(),
            "test": test_dataset.stats(),
            "test_used_for_training": False,
            "project_level_isolation": "manifest splits are project-level; test projects are excluded from training",
        },
        "seeds": {"evaluation": seed, "permutations": permutations, "bootstrap": 1000, "mask_seed_repeats": seed_repeats},
        "seed_sensitivity": seed_sensitivity,
        "samples": collected.samples,
        "objectives": objectives,
        "metrics": metrics,
        "losses": collected.losses,
        "baselines": baselines,
        "calibration": calibration,
        "ablations": ablated,
        "negative_witness_random_initialization": {
            "metrics": negative.metrics,
            "losses": negative.losses,
            "note": "A randomly initialized model is a negative witness, not a baseline competitor.",
        },
        "retrieval": retrieval,
        "reconstruction": reconstruction,
        "capability_categories": capability,
        "limitations": [
            "The test split is small and controlled; confidence intervals are correspondingly wide.",
            "Compilation and syntax are not behavioral agreement.",
            "No independent third-party human evaluation authorship is established.",
            "No cross-architecture claim is made.",
            "A finite loss or token accuracy is not a foundation-model capability claim.",
        ],
    }
    return report


def reconstruction_generation_budget(target_length: int, max_text_tokens: int, floor: int = 96, margin: int = 16) -> int:
    """Return a validated generation budget for one reconstruction target.

    The budget covers the target plus a small margin, is never below the
    historical floor, and never exceeds the decoder's configured maximum.
    """
    return max(1, min(max_text_tokens - 1, max(floor, int(target_length) + margin)))


def target_is_truncated(source_text: str, vocab, max_text_tokens: int) -> bool:
    return len(vocab.encode_text(source_text, 1_000_000)) > max_text_tokens


def _reconstruction_evaluation(
    checkpoint_path: str | Path,
    dataset: FoundationDataset,
    model,
    vocab,
    config,
    limit: int = 2,
    workspace: str | Path | None = None,
) -> dict[str, Any]:
    available = [sample for sample in dataset.samples if sample.get("source_text") and sample.get("binary_path")]
    truncated = [
        sample
        for sample in available
        if target_is_truncated(sample.get("source_text", ""), vocab, config.max_text_tokens)
    ]
    selected = [sample for sample in available if sample not in truncated][:limit]
    if not selected:
        return {
            "status": "unavailable",
            "reason": "no nontruncated source-labelled test functions with binary paths",
            "truncated_targets_excluded": len(truncated),
            "project_level": {"status": "unavailable", "reason": "no nontruncated source-labelled test functions"},
        }
    reconstructor = UnifiedReconstructor(
        checkpoint_path=checkpoint_path,
        model=model,
        vocab=vocab,
        config=config,
        workspace=workspace,
        allow_compilation=True,
    )
    candidates = []
    generation_budgets = []
    for sample in selected:
        target_length = len(sample.get("c_ids") or [])
        generation_budget = reconstruction_generation_budget(target_length, config.max_text_tokens)
        generation_budgets.append(generation_budget)
        result = reconstructor.reconstruct_function(
            sample["binary_path"],
            sample["function_id"],
            language="c",
            reference_source=sample.get("source_text"),
            max_new_tokens=generation_budget,
            include_deterministic=True,
            include_neural=True,
        )
        candidates.extend(result.get("candidates", []))
    status_counts: dict[str, Counter] = {}
    for candidate in candidates:
        for name, status in candidate.get("statuses", {}).items():
            status_counts.setdefault(name, Counter())[status] += 1
    by_method: dict[str, Counter] = {}
    for candidate in candidates:
        method = str(candidate.get("method", "unknown"))
        by_method.setdefault(method, Counter())[str(candidate.get("statuses", {}).get("compilation", "unknown"))] += 1
    summary = {
        "candidates": len(candidates),
        "functions": len(selected),
        "truncated_targets_excluded": len(truncated),
        "generation_budget_min": min(generation_budgets) if generation_budgets else None,
        "generation_budget_max": max(generation_budgets) if generation_budgets else None,
        "statuses": {name: dict(counts) for name, counts in status_counts.items()},
        "compilation_by_method": {method: dict(counts) for method, counts in by_method.items()},
        "neural_candidates": sum(1 for candidate in candidates if candidate.get("method") == "neural_unified_model"),
        "deterministic_candidates": sum(1 for candidate in candidates if candidate.get("method") in {"deterministic_semantic_ir", "ghidra_decompiler"}),
        "structural_ratio_mean": (
            float(np.mean([candidate["structural_metrics"]["ratio"] for candidate in candidates if candidate["structural_metrics"].get("ratio") is not None]))
            if any(candidate["structural_metrics"].get("ratio") is not None for candidate in candidates)
            else None
        ),
        "behavioral_verification": "not_attempted",
        "universal_equivalence": "not_claimed",
    }
    project_level: dict[str, Any] = {
        "status": "interface_tested",
        "candidates_considered": len(selected),
        "note": "Project-level interface aggregates per-function candidates; no project was executed or behaviorally verified.",
    }
    try:
        project_result = reconstructor.reconstruct_project(
            selected[0]["binary_path"],
            language="c",
            max_functions=2,
            include_neural=False,
        )
        project_level.update(
            {
                "status": "exercised",
                "functions_considered": project_result.get("functions_considered"),
                "functions_with_candidates": project_result.get("functions_with_candidates"),
                "candidate_status_counts": project_result.get("candidate_status_counts"),
                "verification_scope": project_result.get("verification_scope"),
            }
        )
    except Exception as exc:
        project_level.update({"status": "failed", "error": str(exc)})
    return {
        "status": "measured" if candidates else "unavailable",
        "summary": summary,
        "project_level": project_level,
        "candidates": candidates,
        "limitations": [
            "Model output is generated from a small CPU pilot checkpoint, not a trained foundation model.",
            "Syntax/compilation statuses are separate from behavioral agreement.",
            "Unresolved operations are retained.",
        ],
    }


def markdown_report(report: dict[str, Any]) -> str:
    lines = [
        "# NEXUS-BI foundation evaluation",
        "",
        f"Checkpoint: `{report['checkpoint']['path']}` (`{report['checkpoint']['sha256']}`)",
        f"Parameters: {report['checkpoint']['parameter_count']}",
        f"Test samples: {report['samples']}",
        "",
        "## Capability categories",
        "",
        "| Category | Status | Metrics |",
        "|---|---|---|",
    ]
    for name, value in report["capability_categories"].items():
        if not isinstance(value, dict):
            lines.append(f"| {name} | unavailable | no structured result |")
            continue
        metrics = value.get("metrics")
        summary = json.dumps(metrics, default=str)[:300] if metrics else value.get("note", value.get("reason", ""))
        lines.append(f"| {name} | {value.get('status')} | {summary} |")
    lines.extend(
        [
            "",
            "## Metrics",
            "",
            "```json",
            json.dumps(report["metrics"], indent=2, default=str),
            "```",
            "",
            "## Baselines",
            "",
            "```json",
            json.dumps(report["baselines"], indent=2, default=str),
            "```",
            "",
            "## Limitations",
            "",
            *[f"- {item}" for item in report["limitations"]],
            "",
        ]
    )
    return "\n".join(lines)
