"""Self-contained model sweep benchmark for the standalone runtime."""

from __future__ import annotations

import argparse
import json
import resource
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import torch

from nexus.foundation.config import FoundationConfig, config_for_vocabulary
from nexus.foundation.model import build_foundation_model, parameter_report
from nexus.foundation.objectives import compute_objective_losses
from nexus.foundation.representation import (
    collate_samples,
    mask_byte_batch,
    mask_evidence_batch,
    mask_instruction_batch,
)
from nexus_train.config import load_local_config, model_sweep
from nexus_train.data_validation import load_gate
from nexus_train.model_factory import model_config_from_dict

ROOT = Path(__file__).resolve().parents[1]
CORE_OBJECTIVES = ["instruction_masked_modeling", "byte_masked_modeling", "ir_op_prediction", "register_effect_prediction", "memory_effect_prediction", "cfg_edge_prediction", "c_reconstruction"]


def _pad(batch: dict[str, Any], length: int, blocks: int, bytes_length: int) -> dict[str, Any]:
    result = {key: value.clone() if isinstance(value, torch.Tensor) else value for key, value in batch.items()}
    bsz = result["instr_id"].size(0)
    for key in ("instr_mnemonic", "instr_shape", "instr_id", "instr_ir", "instr_block"):
        old = result[key]
        new = torch.zeros((bsz, length), dtype=old.dtype)
        new[:, : min(length, old.size(1))] = old[:, : min(length, old.size(1))]
        result[key] = new
    old = result["instr_mask"]
    mask = torch.zeros((bsz, length), dtype=torch.bool)
    mask[:, : min(length, old.size(1))] = old[:, : min(length, old.size(1))]
    result["instr_mask"] = mask
    for key in ("instr_targets", "instr_mlm_mask"):
        if key in result:
            old = result[key]
            new = torch.zeros((bsz, length), dtype=old.dtype)
            new[:, : min(length, old.size(1))] = old[:, : min(length, old.size(1))]
            result[key] = new
    for key in ("instr_regs", "instr_kinds", "instr_access", "instr_width"):
        old = result[key]
        new = torch.zeros((bsz, length, old.size(2)), dtype=old.dtype)
        new[:, : min(length, old.size(1)), :] = old[:, : min(length, old.size(1)), :]
        result[key] = new
    for key in ("instr_mem_read", "instr_mem_write", "instr_flags_read", "instr_flags_written", "instr_mem_width"):
        old = result[key]
        new = torch.zeros((bsz, length), dtype=old.dtype)
        new[:, : min(length, old.size(1))] = old[:, : min(length, old.size(1))]
        result[key] = new
    old = result["byte_ids"]
    new = torch.zeros((bsz, bytes_length), dtype=old.dtype)
    new[:, : min(bytes_length, old.size(1))] = old[:, : min(bytes_length, old.size(1))]
    result["byte_ids"] = new
    old = result["byte_mask"]
    mask = torch.zeros((bsz, bytes_length), dtype=torch.bool)
    mask[:, : min(bytes_length, old.size(1))] = old[:, : min(bytes_length, old.size(1))]
    result["byte_mask"] = mask
    for key in ("byte_targets", "byte_mlm_mask", "function_start_target"):
        if key in result:
            old = result[key]
            new = torch.zeros((bsz, bytes_length), dtype=old.dtype)
            new[:, : min(bytes_length, old.size(1))] = old[:, : min(bytes_length, old.size(1))]
            result[key] = new
    old = result["block_mask"]
    new = torch.zeros((bsz, blocks), dtype=torch.bool)
    new[:, : min(blocks, old.size(1))] = old[:, : min(blocks, old.size(1))]
    result["block_mask"] = new
    result["block_mask"][:, :blocks] = True
    for key in ("cfg_target",):
        old = result[key]
        new = torch.zeros((bsz, blocks, blocks), dtype=old.dtype)
        r, c = min(blocks, old.size(1)), min(blocks, old.size(2))
        new[:, :r, :c] = old[:, :r, :c]
        result[key] = new
    for key in ("read_targets", "write_targets"):
        old = result[key]
        new = torch.zeros((bsz, length, old.size(2)), dtype=old.dtype)
        new[:, : min(length, old.size(1)), :] = old[:, : min(length, old.size(1)), :]
        result[key] = new
    return result


def _representative_batch(gate, config: FoundationConfig, max_functions: int = 4) -> dict[str, Any]:
    dataset = gate.build_split("train", config, max_functions=max_functions)
    with_source = [sample for sample in dataset.samples if sample.get("source_text")]
    selected = sorted(with_source, key=lambda sample: len(sample.get("c_ids", [])), reverse=True)[:4]
    batch = collate_samples(selected, config)
    batch = mask_instruction_batch(batch, probability=0.15, seed=20261103)
    batch = mask_byte_batch(batch, probability=0.08, seed=20261103)
    return _pad(batch, 128, 32, 512)


def _single(config_path: Path, batch_path: Path, out_path: Path, threads: int, steps: int) -> dict[str, Any]:
    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    config = FoundationConfig.from_dict(payload)
    model = build_foundation_model(config, seed=20261103)
    batch = torch.load(batch_path, map_location="cpu", weights_only=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    model.train()
    times = []
    for index in range(1 + steps):
        started = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        outputs = model(batch, run_decoder=True, decoder_language="c")
        evidence_batch = mask_evidence_batch(batch, seed=100 + index)
        evidence = model(evidence_batch)
        outputs.update({"ir_mlm_logits": evidence["ir_logits"], "register_read_masked_logits": evidence["register_read_logits"], "register_write_masked_logits": evidence["register_write_logits"], "memory_read_masked_logits": evidence["memory_read_logits"], "memory_write_masked_logits": evidence["memory_write_logits"], "memory_width_masked_logits": evidence["memory_width_logits"], "cfg_masked_logits": evidence["cfg_logits"]})
        batch["ir_targets"] = evidence_batch["ir_targets"]
        batch["ir_mlm_mask"] = evidence_batch["ir_mlm_mask"]
        losses = compute_objective_losses(outputs, batch, CORE_OBJECTIVES)
        active = [loss for loss, count, _ in losses.values() if count > 0]
        loss = torch.stack(active).sum() if active else outputs["pooled"].sum() * 0
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if index > 0:
            times.append(time.perf_counter() - started)
    payload = {
        "parameter_count": model.count_parameters(),
        "parameter_groups": parameter_report(model)["groups"],
        "context": {"instructions": 128, "blocks": 32, "bytes": 512, "text": int(batch["c_ids"].size(1))},
        "threads": threads,
        "microbatch": int(batch["instr_id"].size(0)),
        "timed_steps": steps,
        "mean_step_seconds": sum(times) / len(times),
        "steps_per_hour": 3600.0 / (sum(times) / len(times)),
        "samples_per_second": int(batch["instr_id"].size(0)) / (sum(times) / len(times)),
        "peak_rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0, 2),
        "active_objectives": [key for key, (_, count, _) in losses.items() if count > 0],
    }
    out_path.write_text(json.dumps(payload, indent=2))
    return payload


def run_sweep(sweep_path: str | Path | None = None, local_path: str | Path | None = None, *, steps: int = 3, threads: int = 4, verbose: bool = True) -> dict[str, Any]:
    local = load_local_config(local_path or ROOT / "configs" / "local_cpu.yaml")
    gate = load_gate(local["dataset"]["frozen_dir"])
    base_config = model_config_from_dict(local["model"], gate.vocab)
    batch = _representative_batch(gate, base_config)
    sweep = model_sweep(sweep_path)
    context = sweep.get("benchmark_context", {})
    steps = int(context.get("steps", steps))
    threads = int(context.get("threads", threads))
    with tempfile.TemporaryDirectory(prefix="nexus-sweep-") as directory:
        temp = Path(directory)
        batch_path = temp / "batch.pt"
        torch.save(batch, batch_path)
        results = []
        for candidate in sweep["candidates"]:
            config = config_for_vocabulary(FoundationConfig.from_dict(candidate["model"]), gate.vocab)
            config_path = temp / f"{candidate['name']}.json"
            config_path.write_text(json.dumps(config.to_dict()), encoding="utf-8")
            out_path = temp / f"{candidate['name']}.json"
            command = [sys.executable, "-m", "nexus_train.benchmark", "--single", str(config_path), str(batch_path), str(out_path), "--threads", str(threads), "--steps", str(steps)]
            started = time.perf_counter()
            completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
            entry = {"name": candidate["name"], "command_returncode": completed.returncode, "wall_seconds": round(time.perf_counter() - started, 3)}
            if completed.returncode == 0:
                entry.update(json.loads(out_path.read_text(encoding="utf-8")))
            else:
                entry["error"] = completed.stderr[-2000:]
            results.append(entry)
            if verbose:
                print(json.dumps({key: entry.get(key) for key in ("name", "parameter_count", "mean_step_seconds", "samples_per_second", "peak_rss_mb", "command_returncode")}), flush=True)
    output = {"schema_version": "0.1.0", "method": {"context": "real frozen ProgramRecord batch, padded 128 instructions/32 blocks/512 bytes/160 text", "steps": steps, "threads": threads, "isolated_process_per_candidate": True}, "candidates": results, "recommended": {"name": local.get("run_name"), "reason": "local config is the 8.66M target; benchmark does not auto-select the largest candidate"}, "largest_successfully_benchmarked": max((item for item in results if item.get("parameter_count")), key=lambda item: item["parameter_count"], default=None)}
    return output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="NEXUS-BI model-size benchmark")
    parser.add_argument("--single", nargs=3, metavar=("CONFIG", "BATCH", "OUT"), help=argparse.SUPPRESS)
    parser.add_argument("--sweep-config")
    parser.add_argument("--local-config")
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--out", default=str(ROOT / "artifacts" / "training_runtime" / "model_benchmark.json"))
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    if args.single:
        _single(Path(args.single[0]), Path(args.single[1]), Path(args.single[2]), args.threads, args.steps)
        return 0
    result = run_sweep(args.sweep_config, args.local_config, steps=args.steps, threads=args.threads, verbose=not args.json)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"out": args.out, "candidates": len(result["candidates"])}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
