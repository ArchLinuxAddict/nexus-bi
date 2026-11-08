#!/usr/bin/env python3
"""Evaluate a residual checkpoint on held-out residual samples."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch

from nexus.foundation.config import FoundationConfig
from nexus.foundation.residual_data import collate_residual, load_residual_records
from nexus.foundation.residual_model import ResidualNexusModel
from nexus.foundation.vocab import Vocabulary


def _load_checkpoint(path: str | Path):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    config = FoundationConfig.from_dict(payload["config"])
    vocab = Vocabulary(**{key: value for key, value in payload["vocab"].items() if key in Vocabulary.__dataclass_fields__})
    model = ResidualNexusModel(config)
    incompatible = model.load_state_dict(payload["model_state"], strict=False)
    if "copy_query.weight" not in payload["model_state"]:
        model.use_copy = False
    model.eval()
    model.incompatible = incompatible
    return model, config, vocab, payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="artifacts/residual_runs/residual-v6/residual.pt")
    parser.add_argument("--test", default="artifacts/dataset/residual_v6/test.jsonl")
    parser.add_argument("--out", default=None)
    parser.add_argument("--batch-size", type=int, default=2)
    args = parser.parse_args()
    model, config, vocab, payload = _load_checkpoint(args.checkpoint)
    records = load_residual_records(args.test)
    totals = defaultdict(float)
    counts = defaultdict(int)
    groups = defaultdict(lambda: defaultdict(float))
    with torch.no_grad():
        for start in range(0, len(records), args.batch_size):
            batch_records = records[start : start + args.batch_size]
            batch = collate_residual(batch_records, vocab, config)
            input_ids = batch["repair_input_ids"][:, :-1]
            input_mask = batch["repair_mask"][:, :-1]
            labels = batch["repair_input_ids"][:, 1:]
            outputs = model(batch, repair_input_ids=input_ids, repair_input_mask=input_mask)
            candidate = (outputs["candidate_logit"] > 0).long()
            failure = outputs["failure_logits"].argmax(-1)
            confidence = outputs["confidence"]
            repair_logits = outputs.get("repair_log_probs", outputs["repair_logits"])
            predictions = repair_logits.argmax(-1)
            label_mask = batch["repair_mask"][:, 1:]
            token_correct = ((predictions == labels) & label_mask).sum().item()
            token_total = label_mask.sum().item()
            for index, record in enumerate(batch_records):
                key = record.get("language", "unknown")
                exact = bool((((predictions[index] == labels[index]) | ~label_mask[index]).all()).item())
                totals["exact"] += int(exact)
                groups[key]["exact"] += int(exact)
                totals["candidate_correct"] += int(candidate[index].item() == int(batch["candidate_label"][index] > 0.5))
                totals["failure_correct"] += int(failure[index].item() == int(batch["failure_label"][index]))
                totals["confidence_abs_error"] += abs(float(confidence[index]) - float(batch["confidence_label"][index]))
                counts["samples"] += 1
                counts[f"language:{key}"] += 1
                groups[key]["samples"] += 1
                groups[key]["confidence_abs_error"] += abs(float(confidence[index]) - float(batch["confidence_label"][index]))
                mask = label_mask[index]
                groups[key]["repair_correct"] += int(((predictions[index] == labels[index]) & mask).sum().item())
                groups[key]["repair_tokens"] += int(mask.sum().item())
            totals["repair_token_correct"] += token_correct
            totals["repair_tokens"] += token_total
            counts["batches"] += 1
    report = {
        "checkpoint": args.checkpoint,
        "test": args.test,
        "samples": counts["samples"],
        "checkpoint_metadata": payload.get("metadata"),
        "metrics": {
            "candidate_accuracy": totals["candidate_correct"] / max(1, counts["samples"]),
            "failure_accuracy": totals["failure_correct"] / max(1, counts["samples"]),
            "confidence_mae": totals["confidence_abs_error"] / max(1, counts["samples"]),
            "repair_token_accuracy": totals["repair_token_correct"] / max(1, totals["repair_tokens"]),
            "repair_exact_match": totals["exact"] / max(1, counts["samples"]),
            "repair_tokens": totals["repair_tokens"],
        },
        "by_language": {
            key: {
                "samples": value["samples"],
                "confidence_mae": value["confidence_abs_error"] / max(1, value["samples"]),
                "repair_token_accuracy": value["repair_correct"] / max(1, value["repair_tokens"]),
                "exact_matches": value["exact"],
            }
            for key, value in groups.items()
        },
        "limitations": [
            "Candidate and failure labels in the current corpus are mostly trivial.",
            "Repair token accuracy and exact match are not compile or behavioral equivalence.",
            "Whitespace is preserved through byte tokens for older checkpoints.",
        ],
    }
    out = Path(args.out) if args.out else Path(args.checkpoint).parent / f"evaluation_{Path(args.checkpoint).stem}.json"
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": "completed", "report": str(out), "metrics": report["metrics"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
