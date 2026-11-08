#!/usr/bin/env python3
"""Train the tool-aware residual model on CPU. No optimizer step in --dry-run."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from nexus.foundation.config import FoundationConfig, config_for_vocabulary, profile_config
from nexus.foundation.residual_data import collate_residual, load_residual_records, shift_for_teacher
from nexus.foundation.residual_model import build_residual_model, residual_parameter_report
from nexus.foundation.vocab import PAD, Vocabulary, build_vocabulary


def _batches(records: list[dict], batch_size: int, shuffle: bool, seed: int):
    order = list(range(len(records)))
    if shuffle:
        generator = torch.Generator().manual_seed(seed)
        order = torch.randperm(len(order), generator=generator).tolist()
    for start in range(0, len(order), batch_size):
        yield [records[index] for index in order[start : start + batch_size]]


def _cycle(records: list[dict], batch_size: int, seed: int):
    epoch = 0
    while True:
        yield from _batches(records, batch_size, True, seed + epoch)
        epoch += 1


def _loss(model, batch):
    input_ids, labels, input_mask, _ = shift_for_teacher(batch["repair_input_ids"], batch["repair_mask"])
    outputs = model(batch, repair_input_ids=input_ids, repair_input_mask=input_mask)
    loss_candidate = F.binary_cross_entropy_with_logits(outputs["candidate_logit"], batch["candidate_label"])
    loss_failure = F.cross_entropy(outputs["failure_logits"], batch["failure_label"])
    loss_confidence = F.binary_cross_entropy(outputs["confidence"], batch["confidence_label"])
    if "repair_log_probs" in outputs:
        log_probs = outputs["repair_log_probs"]
        loss_repair = F.nll_loss(log_probs.reshape(-1, log_probs.size(-1)), labels.reshape(-1), ignore_index=PAD)
    else:
        logits = outputs["repair_logits"]
        loss_repair = F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1), ignore_index=PAD)
    if "copy_logits" in outputs and "copy_labels" in batch:
        copy_labels = batch["copy_labels"][:, 1:]
        copy_logits = outputs["copy_logits"]
        if bool((copy_labels != -100).any()):
            loss_copy = F.cross_entropy(
                copy_logits.reshape(-1, copy_logits.size(-1)),
                copy_labels.reshape(-1),
                ignore_index=-100,
            )
        else:
            loss_copy = torch.zeros((), device=loss_repair.device)
    else:
        loss_copy = torch.zeros((), device=loss_repair.device)
    loss = 0.5 * loss_candidate + 0.5 * loss_failure + 0.2 * loss_confidence + loss_repair + 0.5 * loss_copy
    return loss, {
        "candidate": float(loss_candidate.detach()),
        "failure": float(loss_failure.detach()),
        "confidence": float(loss_confidence.detach()),
        "repair": float(loss_repair.detach()),
        "copy": float(loss_copy.detach()),
    }


@torch.no_grad()
def _validate(model, records: list[dict], vocab, config, batch_size: int) -> dict[str, float]:
    model.eval()
    token_correct = token_total = 0
    exact_correct = 0
    candidate_correct = failure_correct = 0
    confidence_error = 0.0
    for start in range(0, len(records), batch_size):
        subset = records[start : start + batch_size]
        batch = collate_residual(subset, vocab, config)
        outputs = model(batch, repair_input_ids=batch["repair_input_ids"][:, :-1], repair_input_mask=batch["repair_mask"][:, :-1])
        labels = batch["repair_input_ids"][:, 1:]
        mask = batch["repair_mask"][:, 1:]
        predictions = outputs.get("repair_log_probs", outputs["repair_logits"]).argmax(-1)
        token_correct += int(((predictions == labels) & mask).sum().item())
        token_total += int(mask.sum().item())
        exact_correct += int(((predictions == labels) | ~mask).all(dim=1).sum().item())
        candidate = (outputs["candidate_logit"] > 0).long()
        failure = outputs["failure_logits"].argmax(-1)
        for index in range(len(subset)):
            candidate_correct += int(candidate[index].item() == int(batch["candidate_label"][index] > 0.5))
            failure_correct += int(failure[index].item() == int(batch["failure_label"][index]))
            confidence_error += abs(float(outputs["confidence"][index]) - float(batch["confidence_label"][index]))
    count = max(1, len(records))
    return {
        "samples": len(records),
        "repair_token_accuracy": token_correct / max(1, token_total),
        "repair_exact_match": exact_correct / count,
        "candidate_accuracy": candidate_correct / count,
        "failure_accuracy": failure_correct / count,
        "confidence_mae": confidence_error / count,
    }


def _save(path: Path, model, config, vocab, metadata: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "schema_version": "nexus.residual-checkpoint/v1",
        "model_state": model.state_dict(),
        "config": config.to_dict(),
        "vocab": vocab.to_dict(),
        "metadata": metadata,
    }, path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default="artifacts/dataset/residual_v6/train.jsonl")
    parser.add_argument("--validation", default="artifacts/dataset/residual_v6/validation.jsonl")
    parser.add_argument("--profile", default="compact")
    parser.add_argument("--steps", type=int, default=12000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=2)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup", type=int, default=500)
    parser.add_argument("--val-interval", type=int, default=500)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--c-every", type=int, default=2, help="take a C batch every N steps when C data exists")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20261106)
    parser.add_argument("--output-dir", default="artifacts/residual_runs")
    parser.add_argument("--run-name", default="residual-v6-best")
    parser.add_argument("--require-c", action="store_true", help="abort if no C training samples are present")
    parser.add_argument("--resume", default=None, help="Checkpoint to continue from (config and vocabulary are taken from it)")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(max(1, min(args.threads, 16)))
    train = load_residual_records(args.train)
    validation = load_residual_records(args.validation)
    if not train:
        raise SystemExit("no residual training records")
    c_train = [record for record in train if record.get("language") == "c"]
    java_train = [record for record in train if record.get("language") != "c"]
    c_validation = [record for record in validation if record.get("language") == "c"]
    java_validation = [record for record in validation if record.get("language") != "c"]
    if args.require_c and not c_train:
        raise SystemExit("--require-c was set but the C training split is empty; finish the C residual build first")
    resume_payload = None
    if args.resume:
        resume_payload = torch.load(args.resume, map_location="cpu", weights_only=False)
        vocab = Vocabulary(**resume_payload["vocab"])
        config = FoundationConfig.from_dict(resume_payload["config"])
    else:
        vocab = build_vocabulary([], [record.get("tool_text", "") + "\n" + record.get("target_text", "") for record in train], text_vocab_size=16384)
        config = config_for_vocabulary(profile_config(args.profile), vocab)
    model = build_residual_model(config, seed=args.seed)
    if resume_payload is not None:
        incompatible = model.load_state_dict(resume_payload["model_state"], strict=False)
        print(json.dumps({
            "resume_missing": list(incompatible.missing_keys),
            "resume_unexpected": list(incompatible.unexpected_keys),
        }), flush=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    print(json.dumps({
        "records": len(train),
        "c_train": len(c_train),
        "java_train": len(java_train),
        "validation": len(validation),
        "parameters": residual_parameter_report(model),
        "threads": torch.get_num_threads(),
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "config_hash": config.config_hash(),
        "resume": args.resume,
        "profile": config.profile,
        "text_vocab": config.text_vocab,
    }, indent=2), flush=True)
    source = c_train if c_train else java_train
    sample_batch = next(_batches(source, min(args.batch_size, len(source)), False, args.seed))
    collated = collate_residual(sample_batch, vocab, config)
    loss, parts = _loss(model, collated)
    print(json.dumps({"dry_run": args.dry_run, "batch": len(sample_batch), "loss": float(loss.detach()), "parts": parts, "has_program": "program" in collated}, indent=2), flush=True)
    if args.dry_run:
        return 0
    run_dir = Path(args.output_dir) / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = run_dir / "metrics.jsonl"
    metrics_path.write_text("", encoding="utf-8")
    c_iter = _cycle(c_train, args.batch_size, args.seed) if c_train else None
    java_iter = _cycle(java_train, args.batch_size, args.seed + 1) if java_train else None
    best_score = -1.0
    best_step = 0
    stale = 0
    started = time.time()
    step = 0
    while step < args.steps:
        model.train()
        use_c = c_iter is not None and (not java_iter or step % max(1, args.c_every) == 0)
        batch_records = next(c_iter if use_c else java_iter)
        optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0
        part_totals = {}
        for _ in range(max(1, args.grad_accum)):
            collated = collate_residual(batch_records, vocab, config)
            loss, parts = _loss(model, collated)
            (loss / args.grad_accum).backward()
            total_loss += float(loss.detach()) / args.grad_accum
            for key, value in parts.items():
                part_totals[key] = part_totals.get(key, 0.0) + value / args.grad_accum
            batch_records = next(c_iter if use_c else java_iter)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        warmup = max(1, args.warmup)
        if step < warmup:
            lr_scale = step / warmup
        else:
            progress = min(1.0, (step - warmup) / max(1, args.steps - warmup))
            lr_scale = 0.5 * (1.0 + math.cos(math.pi * progress))
        for group in optimizer.param_groups:
            group["lr"] = args.lr * lr_scale
        optimizer.step()
        step += 1
        record = {"step": step, "loss": total_loss, "parts": part_totals, "elapsed": round(time.time() - started, 3), "lr": args.lr * lr_scale, "source": "c" if use_c else "java"}
        with metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
        if step % 10 == 0:
            print(json.dumps(record), flush=True)
        if step % args.val_interval == 0 or step == args.steps:
            metrics = {}
            if c_validation:
                metrics["c"] = _validate(model, c_validation, vocab, config, args.batch_size)
            if java_validation:
                metrics["java"] = _validate(model, java_validation, vocab, config, args.batch_size)
            score = sum(item["repair_token_accuracy"] for item in metrics.values()) / max(1, len(metrics))
            metadata = {"steps": step, "run_name": args.run_name, "train_records": len(train), "validation_records": len(validation), "best_score": score}
            _save(run_dir / "last.pt", model, config, vocab, metadata)
            if score > best_score:
                best_score = score
                best_step = step
                stale = 0
                _save(run_dir / "best.pt", model, config, vocab, {**metadata, "best_score": best_score, "best_step": best_step})
            else:
                stale += 1
            print(json.dumps({"step": step, "validation": metrics, "best_score": best_score, "best_step": best_step, "stale": stale}), flush=True)
            if stale >= args.patience:
                print(json.dumps({"early_stop": True, "step": step, "best_step": best_step, "best_score": best_score}), flush=True)
                break
    metadata = {"steps": step, "run_name": args.run_name, "train_records": len(train), "validation_records": len(validation), "best_score": best_score, "best_step": best_step}
    _save(run_dir / "final.pt", model, config, vocab, metadata)
    print(json.dumps({"status": "completed", "run_dir": str(run_dir), "checkpoint": str(run_dir / "final.pt"), "best": str(run_dir / "best.pt"), "steps": step, "best_step": best_step, "best_score": best_score}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
