"""Deterministic baselines for capability-separated evaluation.

These baselines are not NEXUS-BI intelligence.  They provide honest reference
points for masked-token prediction, retrieval, CFG density and deterministic C
reconstruction.  They are evaluated on the protected test split only.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch

from nexus.foundation.reconstruction import check_c_source, structural_similarity
from nexus.foundation.representation import mask_byte_batch, mask_instruction_batch
from nexus.program.bundle import analyze_binary
from nexus.reconstruction.c.structured import translate_structured


def _majority(values: Counter[int]) -> int:
    return values.most_common(1)[0][0] if values else 0


def _train_majority(dataset, field: str, mask_field: str, limit_batches: int = 200) -> int:
    counts: Counter[int] = Counter()
    for index, batch in enumerate(dataset.evaluation_batches(batch_size=8)):
        if index >= limit_batches:
            break
        batch = mask_instruction_batch(batch, probability=0.15, seed=7)
        batch = mask_byte_batch(batch, probability=0.08, seed=7)
        mask = batch[mask_field]
        values = batch[field][mask]
        counts.update(int(value) for value in values.tolist())
    return _majority(counts)


def masked_majority_baseline(train_dataset, test_dataset, field: str, mask_field: str, target_field: str) -> dict[str, Any]:
    majority = _train_majority(train_dataset, field, mask_field)
    correct = total = 0
    batches = 0
    for batch in test_dataset.evaluation_batches(batch_size=8):
        batch = mask_instruction_batch(batch, probability=0.15, seed=7)
        batch = mask_byte_batch(batch, probability=0.08, seed=7)
        mask = batch[mask_field]
        if mask.any():
            correct += int((batch[target_field][mask] == majority).sum())
            total += int(mask.sum())
        batches += 1
    return {
        "baseline": "majority_class",
        "field": field,
        "majority_token": majority,
        "correct": correct,
        "total": total,
        "accuracy": correct / total if total else None,
        "batches": batches,
        "status": "measured" if total else "not_available",
        "note": "Majority-class accuracy is a trivial lower bound; it does not demonstrate understanding.",
    }


def _sample_vector(sample: dict[str, Any], mode: str) -> Counter[str]:
    if mode == "mnemonic":
        return Counter(str(value) for value in sample.get("instr_mnemonic", []))
    if mode == "instruction":
        return Counter(str(value) for value in sample.get("instr_id", []))
    if mode == "byte":
        return Counter(str(value) for value in sample.get("byte_ids", []))
    return Counter(str(value) for value in sample.get("instr_ir", []))


def _cosine(left: Counter[str], right: Counter[str]) -> float:
    if not left or not right:
        return 0.0
    common = set(left) & set(right)
    numerator = sum(left[key] * right[key] for key in common)
    left_norm = math.sqrt(sum(value * value for value in left.values()))
    right_norm = math.sqrt(sum(value * value for value in right.values()))
    return numerator / max(1e-9, left_norm * right_norm)


def retrieval_baseline(dataset, mode: str = "mnemonic", topk: tuple[int, ...] = (1, 5)) -> dict[str, Any]:
    """Same-source retrieval across compiler/optimization variants."""
    items = []
    for sample in dataset.samples:
        group = str(sample.get("match_group") or "")
        if not group:
            continue
        items.append((sample, group, _sample_vector(sample, mode)))
    groups: dict[str, list[int]] = defaultdict(list)
    for index, (_sample, group, _vector) in enumerate(items):
        groups[group].append(index)
    queries = 0
    hits = {k: 0 for k in topk}
    cross_compiler = 0
    cross_compiler_hits = 0
    for index, (sample, group, vector) in enumerate(items):
        positives = [other for other in groups[group] if other != index]
        if not positives:
            continue
        scores = []
        for other_index, (other, other_group, other_vector) in enumerate(items):
            if other_index == index:
                continue
            scores.append((_cosine(vector, other_vector), other_index))
        scores.sort(reverse=True)
        query_compiler = (sample.get("labels") or {}).get("compiler")
        query_optimization = (sample.get("labels") or {}).get("optimization")
        queries += 1
        for k in topk:
            top = {item[1] for item in scores[:k]}
            if any(pos in top for pos in positives):
                hits[k] += 1
        if any(
            (items[pos][0].get("labels") or {}).get("compiler") != query_compiler
            or (items[pos][0].get("labels") or {}).get("optimization") != query_optimization
            for pos in positives
        ):
            cross_compiler += 1
            top = {item[1] for item in scores[:1]}
            if any(pos in top for pos in positives):
                cross_compiler_hits += 1
    return {
        "baseline": f"{mode}_histogram_cosine",
        "queries": queries,
        "grouped_samples": len(items),
        "groups": len(groups),
        "recall": {f"@{k}": hits[k] / queries if queries else None for k in topk},
        "cross_compiler_optimization": {
            "queries": cross_compiler,
            "recall@1": cross_compiler_hits / cross_compiler if cross_compiler else None,
        },
        "status": "measured" if queries else "not_available",
        "note": "Retrieval by lexical histograms is a weak baseline and does not establish semantic equivalence.",
    }


def cfg_density_baseline(dataset) -> dict[str, Any]:
    tp = fp = fn = tn = 0
    batches = 0
    for batch in dataset.evaluation_batches(batch_size=4):
        block_mask = batch["block_mask"]
        valid = block_mask.unsqueeze(1) & block_mask.unsqueeze(2)
        diagonal = torch.eye(valid.size(1), dtype=torch.bool).unsqueeze(0)
        valid = valid & ~diagonal
        target = batch["cfg_target"][valid] > 0.5
        predicted = torch.ones_like(target, dtype=torch.bool)
        tp += int((predicted & target).sum())
        fp += int((predicted & ~target).sum())
        fn += int((~predicted & target).sum())
        tn += int((~predicted & ~target).sum())
        batches += 1
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-9, precision + recall)
    return {
        "baseline": "all_edges_positive",
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "density": (tp + fp) / max(1, tp + fp + tn + fn),
        "batches": batches,
        "status": "measured" if tp + fp + fn + tn else "not_available",
        "note": "Density baseline exposes class imbalance; it is not a CFG recovery method.",
    }


def reconstruction_baseline(dataset, *, limit: int = 20) -> dict[str, Any]:
    samples = [sample for sample in dataset.samples if sample.get("source_text")]
    if limit:
        samples = samples[:limit]
    cache: dict[str, Any] = {}
    results = []
    for sample in samples:
        binary_path = sample.get("binary_path")
        function_id = sample.get("function_id")
        if not binary_path or not function_id or not Path(binary_path).exists():
            results.append({"sample_id": sample.get("sample_id"), "status": "not_available", "reason": "binary or function unavailable"})
            continue
        if binary_path not in cache:
            try:
                cache[binary_path] = analyze_binary(binary_path)
            except Exception as exc:
                cache[binary_path] = exc
        bundle = cache[binary_path]
        if isinstance(bundle, Exception):
            results.append({"sample_id": sample.get("sample_id"), "status": "not_available", "reason": str(bundle)})
            continue
        function = bundle.function_by_id(str(function_id))
        if function is None:
            results.append({"sample_id": sample.get("sample_id"), "status": "not_available", "reason": "function not recovered"})
            continue
        try:
            translated = translate_structured(bundle, function)
            if not translated.success or not translated.body:
                results.append({"sample_id": sample.get("sample_id"), "status": "not_available", "reason": translated.reason or "translation failed"})
                continue
            generated = translated.body
        except Exception as exc:
            results.append({"sample_id": sample.get("sample_id"), "status": "failed", "reason": str(exc)})
            continue
        similarity = structural_similarity(generated, sample.get("source_text"))
        compile_status = check_c_source(generated) if len(results) < 8 else {"syntax": {"status": "not_attempted"}, "compilation": {"status": "not_attempted"}}
        results.append(
            {
                "sample_id": sample.get("sample_id"),
                "status": "measured",
                "structural_similarity": similarity,
                "syntax": compile_status.get("syntax"),
                "compilation": compile_status.get("compilation"),
                "generated_chars": len(generated),
                "reference_chars": len(sample.get("source_text") or ""),
            }
        )
    measured = [item for item in results if item.get("status") == "measured"]
    ratios = [item["structural_similarity"]["ratio"] for item in measured if item["structural_similarity"].get("ratio") is not None]
    syntax_passed = sum(1 for item in measured if item.get("syntax", {}).get("status") == "passed")
    compile_passed = sum(1 for item in measured if item.get("compilation", {}).get("status") == "passed")
    return {
        "baseline": "deterministic_structured_c_translation",
        "attempted": len(samples),
        "measured": len(measured),
        "structural_similarity_mean": sum(ratios) / len(ratios) if ratios else None,
        "syntax_pass": syntax_passed,
        "compile_pass": compile_passed,
        "compiled_checked": sum(1 for item in measured if item.get("compilation", {}).get("status") in {"passed", "failed"}),
        "results": results,
        "status": "measured" if measured else "not_available",
        "note": "Deterministic translation is a baseline, not NEXUS-BI reconstruction. Structural similarity is not behavioral equivalence.",
    }


def run_baselines(train_dataset, test_dataset, *, reconstruction_limit: int = 20) -> dict[str, Any]:
    instruction = masked_majority_baseline(train_dataset, test_dataset, "instr_id", "instr_mlm_mask", "instr_targets")
    byte = masked_majority_baseline(train_dataset, test_dataset, "byte_ids", "byte_mlm_mask", "byte_targets")
    retrieval = {
        mode: retrieval_baseline(test_dataset, mode=mode) for mode in ("mnemonic", "byte")
    }
    return {
        "schema_version": "nexus.baselines/v1",
        "split": "test",
        "instruction_majority": instruction,
        "byte_majority": byte,
        "retrieval": retrieval,
        "cfg_density": cfg_density_baseline(test_dataset),
        "reconstruction": reconstruction_baseline(test_dataset, limit=reconstruction_limit),
        "test_isolation": "Only the protected test split was evaluated; no test labels were used for fitting these baselines except training-split majority statistics.",
        "limitations": [
            "Deterministic baselines do not establish NEXUS-BI capability.",
            "No behavioral execution or differential testing was performed.",
            "Compilation success is not behavioral equivalence.",
        ],
    }
