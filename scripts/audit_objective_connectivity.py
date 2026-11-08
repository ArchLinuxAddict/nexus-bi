"""Structural shared-model connectivity audit for every enabled objective.

Forward and autograd.grad only.  No optimizer step, no parameter mutation.
"""

from __future__ import annotations

import json
import math
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from nexus.foundation.model import build_foundation_model  # noqa: E402
from nexus.foundation.objectives import compute_objective_losses  # noqa: E402
from nexus.foundation.representation import collate_samples, mask_byte_batch, mask_instruction_batch  # noqa: E402
from nexus.foundation.training import _attach_evidence_outputs, _autoregressive_batch  # noqa: E402
from nexus_train.config import load_local_config  # noqa: E402
from nexus_train.data_validation import load_gate  # noqa: E402
from nexus_train.model_factory import model_config_from_dict  # noqa: E402

HEAD_PREFIXES = (
    "instruction_mlm",
    "byte_mlm",
    "ir_",
    "register_",
    "memory_",
    "cfg_",
    "function_entry",
    "purpose_",
    "tool_",
    "jni_",
    "security_",
    "confidence_",
    "c_head",
    "java_head",
    "source_projection",
    "pair_scorer",
)


def _target_batch(gate, config):
    groups: dict[str, list[str]] = defaultdict(list)
    for sample_id in gate.store.by_split["train"]:
        record = gate.store.index[sample_id]
        if record.has_source and record.match_group:
            groups[record.match_group].append(sample_id)
    candidates = [members for _group, members in sorted(groups.items()) if len(members) >= 2]
    chosen = []
    if candidates:
        chosen.extend(candidates[0][:2])
        for members in candidates[1:]:
            chosen.append(members[0])
            if len(chosen) >= max(4, config.batch_size if hasattr(config, "batch_size") else 4):
                break
    while len(chosen) < 4:
        chosen.append(gate.store.by_split["train"][len(chosen)])
    samples = [gate.store.get(sample_id) for sample_id in chosen]
    batch = collate_samples(samples, config)
    batch = mask_instruction_batch(batch, probability=0.15, seed=3)
    batch = mask_byte_batch(batch, probability=0.08, seed=3)
    return batch, samples


def main() -> int:
    local = load_local_config(ROOT / "artifacts" / "final_training_package" / "training" / "final_training.yaml")
    gate = load_gate(local["dataset"]["frozen_dir"])
    gate.require_clean()
    config = model_config_from_dict(local["model"], gate.vocab)
    enabled = list(local["training"]["objectives"])
    # Add the two v2 objectives for connectivity even if the v1 package config predates them.
    for objective_id in ("instruction_autoregressive", "function_start_prediction"):
        if objective_id not in enabled:
            enabled.append(objective_id)
    torch.manual_seed(int(config.seed))
    model = build_foundation_model(config, seed=int(config.seed))
    model.train()
    batch, samples = _target_batch(gate, config)
    outputs = model(batch, run_decoder=True)
    _attach_evidence_outputs(model, batch, outputs, 3, enabled)
    ar_outputs = model(_autoregressive_batch(batch), causal_instructions=True)
    outputs["instr_ar_logits"] = ar_outputs["instruction_logits"]
    named = list(model.named_parameters())
    shared = [(name, parameter) for name, parameter in named if not name.startswith(HEAD_PREFIXES)]
    results = {}
    for objective_id in enabled:
        objective_batch = batch
        if objective_id == "instruction_autoregressive":
            ar = _autoregressive_batch(batch)
            ar_outputs = model(ar, causal_instructions=True)
            outputs["instr_ar_logits"] = ar_outputs["instruction_logits"]
            objective_batch = {**batch, "instr_ar_targets": ar["instr_ar_targets"], "instr_ar_mask": ar["instr_ar_mask"]}
        losses = compute_objective_losses(outputs, objective_batch, [objective_id])
        loss, count, note = losses.get(objective_id, (None, 0, "not produced"))
        if loss is None or count <= 0:
            results[objective_id] = {"status": "FAIL", "reason": note, "active_labels": count, "shared_backbone_gradient": False, "head_gradient": False}
            continue
        grads = torch.autograd.grad(loss, [parameter for _name, parameter in named], allow_unused=True, retain_graph=True)
        grad_by_name = dict(zip([name for name, _parameter in named], grads))
        shared_finite = sum(1 for name, _p in shared if grad_by_name.get(name) is not None and torch.isfinite(grad_by_name[name]).all())
        head_finite = sum(1 for name, _p in named if name not in {item[0] for item in shared} and grad_by_name.get(name) is not None and torch.isfinite(grad_by_name[name]).all())
        shared_norm = math.sqrt(sum(float(grad_by_name[name].float().norm() ** 2) for name, _p in shared if grad_by_name.get(name) is not None))
        results[objective_id] = {
            "status": "PASS" if shared_finite > 0 and torch.isfinite(loss).all() else "FAIL",
            "active_labels": count,
            "loss": float(loss.detach()),
            "shared_parameters_with_finite_grad": shared_finite,
            "head_parameters_with_finite_grad": head_finite,
            "shared_gradient_norm": shared_norm,
            "shared_backbone_gradient": shared_finite > 0,
            "head_gradient": head_finite > 0,
            "note": note,
        }
    model.zero_grad(set_to_none=True)
    payload = {
        "schema_version": "nexus.objective-connectivity/v1",
        "model_config_hash": config.config_hash(),
        "parameter_count": model.count_parameters(),
        "batch_samples": len(samples),
        "enabled_objectives": enabled,
        "objectives": results,
        "all_enabled_shared_backbone_gradient": all(item.get("shared_backbone_gradient") for item in results.values()),
        "no_optimizer_step": True,
        "note": "Structural autograd.grad check only; no parameter update, no training and no behavioral claim.",
    }
    target = ROOT / "artifacts" / "objectives" / "objective_connectivity.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS" if payload["all_enabled_shared_backbone_gradient"] else "FAIL", "objectives": {key: value["status"] for key, value in results.items()}, "out": str(target)}, indent=2))
    return 0 if payload["all_enabled_shared_backbone_gradient"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
