#!/usr/bin/env python3
"""Evaluate a residual checkpoint on selected JSONL samples."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from nexus.foundation.config import FoundationConfig
from nexus.foundation.residual_data import collate_residual
from nexus.foundation.residual_model import build_residual_model
from nexus.foundation.vocab import Vocabulary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--records", required=True)
    parser.add_argument("--filter", default="")
    parser.add_argument("--generate", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    args = parser.parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    vocab = Vocabulary(**checkpoint["vocab"])
    config = FoundationConfig.from_dict(checkpoint["config"])
    model = build_residual_model(config, seed=config.seed)
    incompatible = model.load_state_dict(checkpoint["model_state"], strict=False)
    if "copy_query.weight" not in checkpoint["model_state"]:
        model.use_copy = False
    if incompatible.missing_keys:
        print(json.dumps({"missing_keys": list(incompatible.missing_keys)}))
    model.eval()
    records = [
        json.loads(line)
        for line in Path(args.records).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if args.filter:
        records = [record for record in records if args.filter in str(record.get("sample_id", ""))]
    if not records:
        raise SystemExit("no matching records")
    with torch.no_grad():
        for record in records:
            batch = collate_residual([record], vocab, config)
            outputs = model(
                batch,
                repair_input_ids=batch["repair_input_ids"][:, :-1],
                repair_input_mask=batch["repair_mask"][:, :-1],
            )
            labels = batch["repair_input_ids"][:, 1:]
            mask = batch["repair_mask"][:, 1:]
            predictions = outputs.get("repair_log_probs", outputs["repair_logits"]).argmax(-1)
            correct = int(((predictions == labels) & mask).sum().item())
            total = int(mask.sum().item())
            teacher_forced_exact = bool(((predictions == labels) | ~mask).all().item())
            target_text = record.get("target_text", "").strip()
            result = {
                "sample_id": record.get("sample_id"),
                "source": record.get("source"),
                "protection": record.get("protection"),
                "repair_token_accuracy": correct / max(1, total),
                "tokens": total,
                "teacher_forced_exact": teacher_forced_exact,
            }
            if args.generate:
                generated = model.generate_repair(batch, max_new_tokens=args.max_new_tokens)
                generated_text = vocab.decode_text(generated[0].tolist()).strip()
                result["generation_exact"] = generated_text == target_text
            print(json.dumps(result))
            if args.generate:
                print("--- target ---")
                print(target_text)
                print("--- generated ---")
                print(generated_text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
