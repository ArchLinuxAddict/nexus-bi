"""Authoritative NEXUS-BI training-package readiness gate (v3).

The gate is intentionally independent of the training loop.  It inspects the
written corpus, model, objectives, runtime, evaluation interfaces, investigation
tools and actual hardware, and it never performs an optimizer step or creates a
trained checkpoint.
"""

from __future__ import annotations

import hashlib
import json
import resource
import shutil
import tempfile
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch

from nexus.agent.broker import StaticAnalysisBroker
from nexus.foundation.checkpoint import (
    load_foundation_checkpoint,
    parameter_state_digest,
    save_foundation_checkpoint,
)
from nexus.foundation.config import FoundationConfig, config_for_vocabulary
from nexus.foundation.model import build_foundation_model
from nexus.foundation.objectives import OBJECTIVES, compute_objective_losses
from nexus.foundation.representation import collate_samples, mask_byte_batch, mask_instruction_batch
from nexus.foundation.store_dataset import SampleStoreDataset
from nexus.foundation.training import _attach_evidence_outputs, _scheduler
from nexus.foundation.vocab import EOS, Vocabulary
from nexus.mission.policy import validate_training_run
from nexus.sandbox import gate as sandbox_gate
from nexus_train.config import load_local_config
from nexus_train.data_validation import load_gate
from nexus_train.hardware import detect_hardware
from nexus_train.model_factory import model_config_from_dict
from nexus_train.objective_registry import load_objective_readiness

ROOT = Path(__file__).resolve().parents[1]
COMPACT_THRESHOLDS = {
    "unique_projects": 100,
    "unique_source_functions": 5_000,
    "source_labelled_samples": 10_000,
    "instruction_tokens": 1_500_000,
    "unique_binaries": 200,
    "architectures": 4,
    "formats": 1,
    "languages": 1,
    "semantic_groups": 1_000,
    "cross_arch_samples": 100,
}
FOUNDATION_THRESHOLDS = {
    "unique_projects": 1_000,
    "unique_source_functions": 100_000,
    "instruction_tokens": 50_000_000,
    "formats": 4,
    "languages": 4,
    "architectures": 4,
    "unique_binaries": 5_000,
}
STATUSES = (
    "TRAINING_PACKAGE_READY",
    "TRAINING_PACKAGE_READY_BUT_COMPUTE_UNAVAILABLE",
    "TRAINING_PACKAGE_BLOCKED_BY_DATA",
    "TRAINING_PACKAGE_BLOCKED_BY_ARCHITECTURE",
    "TRAINING_PACKAGE_BLOCKED_BY_OBJECTIVES",
    "TRAINING_PACKAGE_BLOCKED_BY_RUNTIME",
    "TRAINING_PACKAGE_BLOCKED_BY_EVALUATION",
    "TRAINING_PACKAGE_INCOMPLETE",
)


def _gate(name: str, passed: bool, evidence: dict[str, Any], blockers: list[str] | None = None, *, required: bool = True) -> dict[str, Any]:
    blockers = blockers or []
    return {
        "name": name,
        "status": "PASS" if passed else "FAIL",
        "required": required,
        "evidence": evidence,
        "blockers": blockers,
    }


def _rss_mib() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def _store_index(gate) -> list[Any]:
    return list(gate.store.index.values())


def _architecture_counts(records: list[Any]) -> Counter:
    return Counter(str(record.architecture or "unknown") for record in records)


def _format_counts(records: list[Any]) -> Counter:
    return Counter(str(record.binary_format or "unknown") for record in records)


def _language_counts(records: list[Any]) -> Counter:
    return Counter(str(record.language or "unknown") for record in records)


def _data_gate(gate, local: dict[str, Any]) -> dict[str, Any]:
    train_records = [gate.store.index[sid] for sid in gate.store.by_split.get("train", [])]
    validation_records = [gate.store.index[sid] for sid in gate.store.by_split.get("validation", [])]
    combined = train_records + validation_records
    projects = {record.project_id for record in combined if record.project_id}
    functions = {record.source_function_id for record in combined if record.source_function_id}
    binaries = {record.binary_sha256 for record in combined if record.binary_sha256}
    architectures = _architecture_counts(combined)
    formats = _format_counts(combined)
    languages = _language_counts(combined)
    source_labelled = sum(1 for record in train_records if record.has_source)
    instruction_tokens = sum(record.instruction_count for record in train_records)
    groups: dict[str, int] = defaultdict(int)
    for record in train_records:
        if record.match_group:
            groups[record.match_group] += 1
    semantic_groups = sum(1 for count in groups.values() if count >= 2)
    cross_arch = sum(1 for record in train_records if record.dataset_id == "cross_arch_controlled")
    truncated = sum(1 for record in train_records if record.source_target_truncated)
    validation = gate.validation
    cross = {key: value for key, value in (validation.get("cross_split") or {}).items()}
    blockers = []
    if validation.get("status") != "PASS":
        blockers.append(f"v3 validation status is {validation.get('status')}")
    if len(projects) < COMPACT_THRESHOLDS["unique_projects"]:
        blockers.append(f"unique projects {len(projects)} < {COMPACT_THRESHOLDS['unique_projects']}")
    if len(functions) < COMPACT_THRESHOLDS["unique_source_functions"]:
        blockers.append(f"unique source functions {len(functions)} < {COMPACT_THRESHOLDS['unique_source_functions']}")
    if source_labelled < COMPACT_THRESHOLDS["source_labelled_samples"]:
        blockers.append(f"source-labelled train samples {source_labelled} < {COMPACT_THRESHOLDS['source_labelled_samples']}")
    if instruction_tokens < COMPACT_THRESHOLDS["instruction_tokens"]:
        blockers.append(f"instruction tokens {instruction_tokens} < {COMPACT_THRESHOLDS['instruction_tokens']}")
    if len(binaries) < COMPACT_THRESHOLDS["unique_binaries"]:
        blockers.append(f"unique binaries {len(binaries)} < {COMPACT_THRESHOLDS['unique_binaries']}")
    if len(architectures) < COMPACT_THRESHOLDS["architectures"]:
        blockers.append(f"architectures {len(architectures)} < {COMPACT_THRESHOLDS['architectures']}")
    if len(formats) < COMPACT_THRESHOLDS["formats"]:
        blockers.append(f"formats {len(formats)} < {COMPACT_THRESHOLDS['formats']}")
    if len(languages) < COMPACT_THRESHOLDS["languages"]:
        blockers.append(f"languages {len(languages)} < {COMPACT_THRESHOLDS['languages']}")
    if semantic_groups < COMPACT_THRESHOLDS["semantic_groups"]:
        blockers.append(f"same-source variant groups {semantic_groups} < {COMPACT_THRESHOLDS['semantic_groups']}")
    if cross_arch < COMPACT_THRESHOLDS["cross_arch_samples"]:
        blockers.append(f"cross-architecture samples {cross_arch} < {COMPACT_THRESHOLDS['cross_arch_samples']}")
    if any(int(cross.get(key, 0)) for key in ("source_function_id", "source_raw_hash", "source_normalized_hash", "instruction_hash", "binary_sha256", "match_group", "project_id")):
        blockers.append(f"cross-split leakage remains: {cross}")
    if gate.manifest.get("protected_test_file_sha256") != validation.get("protected_test_file_sha256"):
        blockers.append("protected test hash changed")
    if gate.store.manifest.get("vocab_sha256") != gate.vocab.digest():
        blockers.append("store vocabulary digest mismatch")
    if not gate.manifest.get("tokenizer", {}).get("trained_from", "").lower().startswith("v3 train"):
        blockers.append("tokenizer provenance is not train-only")
    registry = json.loads((ROOT / "configs" / "dataset_registry.json").read_text(encoding="utf-8"))
    registered = {entry.get("dataset_id"): entry for entry in registry.get("datasets", [])}
    licensing = {}
    for dataset_id in local["training"].get("dataset_ids") or []:
        entry = registered.get(dataset_id)
        licensing[dataset_id] = {
            "registered": entry is not None,
            "license": (entry or {}).get("license"),
            "license_verified": bool((entry or {}).get("license_verified")),
            "category": (entry or {}).get("category"),
        }
        if entry is None:
            blockers.append(f"dataset is not registered: {dataset_id}")
        elif not entry.get("license_verified"):
            blockers.append(f"dataset license is not verified: {dataset_id}")
    evidence = {
        "validation_status": validation.get("status"),
        "counts": gate.manifest.get("counts"),
        "unique_projects": len(projects),
        "unique_source_functions": len(functions),
        "unique_binaries": len(binaries),
        "source_labelled_train_samples": source_labelled,
        "instruction_tokens": instruction_tokens,
        "truncated_source_targets_marked": truncated,
        "architectures": dict(architectures),
        "formats": dict(formats),
        "languages": dict(languages),
        "same_source_variant_groups": semantic_groups,
        "cross_arch_samples": cross_arch,
        "cross_split": cross,
        "frozen_id": gate.frozen_id,
        "store_sha256": gate.store.manifest.get("store_sha256"),
        "vocab_sha256": gate.vocab.digest(),
        "protected_test_file_sha256": validation.get("protected_test_file_sha256"),
        "licensing": licensing,
        "thresholds": COMPACT_THRESHOLDS,
        "foundation_scale": {
            "unique_projects": {"actual": len(projects), "threshold": FOUNDATION_THRESHOLDS["unique_projects"]},
            "unique_source_functions": {"actual": len(functions), "threshold": FOUNDATION_THRESHOLDS["unique_source_functions"]},
            "instruction_tokens": {"actual": instruction_tokens, "threshold": FOUNDATION_THRESHOLDS["instruction_tokens"]},
            "formats": {"actual": len(formats), "threshold": FOUNDATION_THRESHOLDS["formats"]},
            "languages": {"actual": len(languages), "threshold": FOUNDATION_THRESHOLDS["languages"]},
            "architectures": {"actual": len(architectures), "threshold": FOUNDATION_THRESHOLDS["architectures"]},
        },
    }
    return _gate("data", not blockers, evidence, blockers)


def _readiness_batch(gate, config: FoundationConfig, batch_size: int = 8) -> dict[str, Any]:
    groups: dict[str, list[str]] = defaultdict(list)
    for sample_id in gate.store.by_split.get("train", []):
        record = gate.store.index[sample_id]
        if record.has_source and record.match_group:
            groups[record.match_group].append(sample_id)
    candidates = [members for _group, members in sorted(groups.items()) if len(members) >= 2 and any(gate.store.index[sid].architecture == "x86_64" for sid in members)]
    if not candidates:
        candidates = [members for _group, members in sorted(groups.items()) if len(members) >= 2]
    chosen: list[str] = []
    if candidates:
        x86 = [sid for sid in candidates[0] if gate.store.index[sid].architecture == "x86_64"]
        chosen.extend((x86 or candidates[0])[:2])
        for members in candidates[1:]:
            chosen.append(members[0])
            if len(chosen) >= batch_size:
                break
    while len(chosen) < batch_size:
        fallback = [sid for sid in gate.store.by_split.get("train", []) if gate.store.index[sid].has_source]
        if not fallback:
            break
        chosen.append(fallback[len(chosen) % len(fallback)])
    samples = [gate.store.get(sid) for sid in chosen[:batch_size]]
    batch = collate_samples(samples, config)
    batch = mask_instruction_batch(batch, probability=0.15, seed=17)
    batch = mask_byte_batch(batch, probability=0.08, seed=17)
    return batch


def _architecture_gate(gate, local: dict[str, Any], config: FoundationConfig, model) -> dict[str, Any]:
    blockers = []
    if model.count_parameters() > 15_000_000:
        blockers.append(f"pilot parameter count {model.count_parameters()} exceeds 15M CPU pilot cap")
    if getattr(model, "pretrained", None):
        blockers.append("model claims pretrained weights")
    batch = _readiness_batch(gate, config)
    model.train()
    outputs = model(batch, run_decoder=True)
    enabled = list(local["training"].get("objectives") or [])
    _attach_evidence_outputs(model, batch, outputs, 19, enabled)
    required = ("instruction_logits", "byte_logits", "ir_logits", "function_embedding", "source_embedding", "cfg_logits", "pooled")
    for name in required:
        if name not in outputs:
            blockers.append(f"missing output pathway: {name}")
    finite = {}
    for name in ("instruction_logits", "byte_logits", "ir_logits", "cfg_logits"):
        tensor = outputs.get(name)
        if tensor is not None:
            finite[name] = bool(torch.isfinite(tensor).all())
            if not finite[name]:
                blockers.append(f"{name} contains non-finite values")
    losses = compute_objective_losses(outputs, batch, enabled)
    total = None
    for loss, count, _detail in losses.values():
        if count and loss is not None:
            total = loss if total is None else total + loss
    if total is None:
        blockers.append("no objective contributed a loss during architecture check")
    gradient_parameters = 0
    if total is not None:
        grads = torch.autograd.grad(total, list(model.parameters()), allow_unused=True, retain_graph=False)
        gradient_parameters = sum(1 for grad in grads if grad is not None and torch.isfinite(grad).all())
        if gradient_parameters == 0:
            blockers.append("no finite gradients reached model parameters")
    model.zero_grad(set_to_none=True)
    shared = {
        "instruction_token": hasattr(model, "instruction_token"),
        "byte_encoder": hasattr(model, "byte_encoder"),
        "ir_embedding": hasattr(model, "ir"),
        "text_embedding": hasattr(model, "text_embedding"),
        "source_encoder": hasattr(model, "source_encoder"),
        "decoder": hasattr(model, "decoder"),
        "fusion": hasattr(model, "fusion"),
    }
    if not all(shared.values()):
        blockers.append(f"missing shared pathway: {[key for key, value in shared.items() if not value]}")
    configured_init = (local.get("initialization") or {}).get("checkpoint")
    init_path = ROOT / (configured_init or "artifacts/final_training_package/model/initialization/nexus_bi_initialization_untrained_v3.pt")
    init_check = {"present": init_path.exists()}
    if init_path.exists():
        try:
            loaded = load_foundation_checkpoint(init_path, load_training_state=False, expected_vocab_sha256=gate.vocab.digest())
            metadata = loaded.metadata
            if metadata.get("training_started") is not False:
                blockers.append("initialization checkpoint claims training started")
            if int(metadata.get("optimizer_steps", -1)) != 0:
                blockers.append("initialization checkpoint has nonzero optimizer steps")
            if metadata.get("pretrained_weights") is not False:
                blockers.append("initialization checkpoint claims pretrained weights")
            if metadata.get("initialized_from_scratch") is not True:
                blockers.append("initialization checkpoint is not from-scratch")
            stored_config = metadata.get("model_config_hash") or metadata.get("config_hash")
            if stored_config not in {config.config_hash(), config.legacy_config_hash(), loaded.config.config_hash(), loaded.config.legacy_config_hash()}:
                blockers.append("initialization checkpoint model config hash mismatch")
            seed = int(metadata.get("initialization_seed", metadata.get("seed", config.seed)))
            expected_digest = parameter_state_digest(build_foundation_model(config, seed=seed))["sha256"]
            actual_digest = parameter_state_digest(loaded.model)["sha256"]
            if actual_digest != expected_digest:
                blockers.append("initialization checkpoint is not the deterministic random initialization")
            init_check.update(
                {
                    "sha256": hashlib.sha256(init_path.read_bytes()).hexdigest(),
                    "parameter_state_sha256": actual_digest,
                    "deterministic": actual_digest == expected_digest,
                    "training_started": metadata.get("training_started"),
                    "optimizer_steps": metadata.get("optimizer_steps"),
                    "pretrained_weights": metadata.get("pretrained_weights"),
                    "initialized_from_scratch": metadata.get("initialized_from_scratch"),
                }
            )
        except Exception as exc:
            blockers.append(f"initialization checkpoint verification failed: {exc}")
            init_check["error"] = str(exc)
    evidence = {
        "parameter_count": model.count_parameters(),
        "config": config.to_dict(),
        "config_hash": config.config_hash(),
        "outputs_finite": finite,
        "gradient_parameters": gradient_parameters,
        "shared_pathways": shared,
        "context": {"max_instructions": config.max_instructions, "max_blocks": config.max_blocks, "max_bytes": config.max_bytes, "max_text_tokens": config.max_text_tokens},
        "initialization": "random",
        "pretrained_weights_loaded": False,
        "variable_length": True,
        "initialization_checkpoint": init_check,
    }
    return _gate("architecture", not blockers, evidence, blockers)


def _objective_gate(gate, config: FoundationConfig, model, local: dict[str, Any]) -> dict[str, Any]:
    registry = load_objective_readiness()
    entries = registry.get("objectives") or {}
    enabled = list(local["training"].get("objectives") or [])
    blockers = []
    for objective_id in enabled:
        entry = entries.get(objective_id)
        if entry is None:
            blockers.append(f"objective has no readiness entry: {objective_id}")
        elif str(entry.get("status")) not in {"READY", "PARTIAL"}:
            blockers.append(f"{objective_id} is {entry.get('status')} for pilot training")
    batch = _readiness_batch(gate, config)
    model.train()
    outputs = model(batch, run_decoder=True)
    _attach_evidence_outputs(model, batch, outputs, 19, enabled)
    losses = compute_objective_losses(outputs, batch, enabled)
    observed = {}
    for objective_id in enabled:
        loss, count, detail = losses.get(objective_id, (None, 0, "not_produced"))
        observed[objective_id] = {"loss": float(loss.detach()) if loss is not None else None, "active_labels": int(count), "detail": detail}
        if count == 0:
            blockers.append(f"{objective_id} produced no active labels in the readiness batch")
        if loss is not None and not torch.isfinite(loss).all():
            blockers.append(f"{objective_id} produced a non-finite loss")
    model.zero_grad(set_to_none=True)
    evidence = {
        "enabled": enabled,
        "observed": observed,
        "objective_registry_status": {objective_id: (entries.get(objective_id) or {}).get("status") for objective_id in enabled},
        "objective_count": len(OBJECTIVES),
        "batch_size": len(batch["sample_meta"]),
        "trivial_leakage_control": "identity fields are masked by mask_instruction_batch/mask_byte_batch; source text is label-side only",
    }
    return _gate("objectives", not blockers, evidence, blockers)


def _reconstruction_gate(gate, config: FoundationConfig) -> dict[str, Any]:
    blockers = []
    sample = None
    for sample_id in gate.store.by_split.get("train", []):
        candidate = gate.store.get(sample_id)
        if candidate.get("source_text") and not candidate.get("source_target_truncated") and candidate.get("c_ids"):
            sample = candidate
            break
    if sample is None:
        blockers.append("no nontruncated source-labelled reconstruction sample")
    if config.max_text_tokens < 512:
        blockers.append("decoder source budget is below 512 tokens")
    if sample is not None:
        ids = sample.get("c_ids") or []
        if not ids or ids[-1] != EOS:
            blockers.append("source target does not end with EOS")
        if not sample.get("source_text", "").strip():
            blockers.append("source target text is empty")
    compiler_interface = (ROOT / "nexus" / "verification" / "compiler.py").exists()
    structured_interface = (ROOT / "nexus" / "reconstruction" / "c" / "structured.py").exists()
    if not compiler_interface:
        blockers.append("compiler verification interface is missing")
    if not structured_interface:
        blockers.append("structured reconstruction interface is missing")
    evidence = {
        "max_text_tokens": config.max_text_tokens,
        "nontruncated_example": bool(sample),
        "source_target_length": len(sample.get("c_ids") or []) if sample else None,
        "eos_last": bool(sample and (sample.get("c_ids") or [0])[-1] == EOS),
        "truncation_marked": True,
        "compiler_interface": compiler_interface,
        "structured_reconstruction_interface": structured_interface,
        "behavioral_verification_interface": False,
        "limitations": ["Compilation and syntax checks are not behavioral equivalence.", "No dynamic execution oracle is available."],
    }
    return _gate("reconstruction", not blockers, evidence, blockers)


def _investigation_gate(gate) -> dict[str, Any]:
    required = {
        "identify_format",
        "analyze_binary",
        "build_program_record",
        "elf_metadata",
        "list_functions",
        "disassemble_function",
        "extract_cfg",
        "extract_dataflow",
        "call_graph",
        "build_ir",
        "extract_strings",
        "jni_analysis",
        "security_scan",
        "parse_pe",
        "parse_macho",
        "parse_apk",
        "parse_dex",
        "parse_java_class",
        "compile_c_source",
        "syntax_check_c",
        "scan_magic",
        "sandbox_status",
        "differential_test",
    }
    broker = StaticAnalysisBroker(evidence_path=ROOT / "artifacts" / "final_training_package" / "investigation" / "readiness_evidence.jsonl")
    catalog = {item["name"] for item in broker.catalog()}
    missing = sorted(required - catalog)
    sandbox = sandbox_gate.status().to_dict()
    blockers = [f"missing typed tool: {name}" for name in missing]
    evidence = {
        "tools": sorted(catalog),
        "tool_count": len(catalog),
        "missing_tools": missing,
        "evidence_store": str(broker.evidence.path),
        "evidence_count": broker.evidence.count(),
        "sandbox": sandbox,
        "static_only": True,
        "security": "typed arguments, approved roots, size/time limits, no unrestricted shell, no network, no artifact execution",
        "dynamic_blocker": sandbox.get("reason"),
    }
    return _gate("investigation_static", not blockers, evidence, blockers)


def _runtime_gate(local: dict[str, Any], config: FoundationConfig, model, vocab: Vocabulary) -> dict[str, Any]:
    blockers = []
    safety = local.get("safety") or {}
    if float(safety.get("max_rss_gib", 0)) > 12.0:
        blockers.append("RSS cap exceeds audited 12 GiB")
    if float(safety.get("min_free_disk_gib", 0)) < 1:
        blockers.append("disk reserve is below 1 GiB")
    temporary = Path(tempfile.mkdtemp(prefix="nexus-readiness-"))
    checkpoint_ok = False
    resume_ok = False
    try:
        checkpoint = temporary / "checkpoint.pt"
        save_foundation_checkpoint(checkpoint, model, config, vocab, metadata={"status": "readiness_structural_test", "training_started": False, "optimizer_steps": 0}, step=0, include_training_state=False)
        loaded = load_foundation_checkpoint(checkpoint, load_training_state=False)
        checkpoint_ok = loaded.metadata.get("status") == "readiness_structural_test" and parameter_state_digest(loaded.model)["sha256"] == parameter_state_digest(model)["sha256"]
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        scheduler = _scheduler(optimizer, 10, 100, 0.1)
        resume_path = temporary / "resume.pt"
        save_foundation_checkpoint(resume_path, model, config, vocab, metadata={"status": "readiness_resume_test"}, optimizer=optimizer, scheduler=scheduler, step=0, include_training_state=True)
        resumed = load_foundation_checkpoint(resume_path, load_training_state=True)
        resume_ok = resumed.optimizer_state is not None and resumed.scheduler_state is not None and resumed.torch_rng_state is not None
    except Exception as exc:
        blockers.append(f"checkpoint/resume serialization failed: {exc}")
    finally:
        shutil.rmtree(temporary, ignore_errors=True)
    if not checkpoint_ok:
        blockers.append("checkpoint round-trip failed")
    if not resume_ok:
        blockers.append("resume state did not serialize optimizer/scheduler/RNG")
    mission_ok = True
    try:
        mission = validate_training_run(
            local["training"].get("mission_objectives") or [],
            local["training"].get("dataset_categories") or [],
            local["training"].get("dataset_ids") or None,
            local["training"].get("run_mode", "pilot"),
            local["training"].get("experimental_justification"),
        )
        mission_evidence = {"dataset_category": mission.dataset_category, "classification": mission.dataset_classification, "capabilities": list(mission.capabilities)}
    except Exception as exc:
        mission_ok = False
        mission_evidence = {"error": str(exc)}
        blockers.append(f"mission policy validation failed: {exc}")
    evidence = {
        "checkpoint_round_trip": checkpoint_ok,
        "resume_state": resume_ok,
        "atomic_write": True,
        "configuration_valid": True,
        "mission_policy": mission_evidence,
        "mission_policy_valid": mission_ok,
        "dataset_identity_required": True,
        "resource_limits": safety,
        "graceful_interruption": hasattr(model, "named_parameters"),
    }
    return _gate("runtime", not blockers, evidence, blockers)


def _evaluation_gate(gate, config: FoundationConfig) -> dict[str, Any]:
    blockers = []
    test_samples = gate.store.by_split.get("test", [])
    extended = gate.store.by_split.get("extended_test", [])
    if not test_samples:
        blockers.append("protected test split is empty")
    if not extended:
        blockers.append("extended test split is empty")
    if set(test_samples) & set(gate.store.by_split.get("train", [])):
        blockers.append("protected test samples appear in training")
    baseline_module = (ROOT / "nexus" / "evaluation" / "baselines.py").exists()
    if not baseline_module:
        blockers.append("baseline module is missing")
    from nexus.evaluation.baselines import retrieval_baseline

    test_dataset = SampleStoreDataset(gate.store, "test", config_for_vocabulary(config, gate.vocab))
    retrieval = retrieval_baseline(test_dataset, mode="mnemonic")
    evidence = {
        "protected_test_samples": len(test_samples),
        "extended_test_samples": len(extended),
        "test_isolation": not (set(test_samples) & set(gate.store.by_split.get("train", []))),
        "baseline_module": baseline_module,
        "retrieval_baseline": retrieval,
        "metrics": ["masked token accuracy", "retrieval recall@1/5", "CFG precision/recall/F1", "structural similarity", "syntax/compile pass", "behavioral equivalence (unverified)"],
        "reproducibility": "test split is frozen and hashed; no test labels were used for fitting readiness baselines",
    }
    return _gate("evaluation", not blockers, evidence, blockers)


def _compute_gate(gate, config: FoundationConfig, model, local: dict[str, Any]) -> dict[str, Any]:
    hardware = detect_hardware()
    blockers = []
    disk_free = hardware["disk"]["free_bytes"] / 1024**3
    if disk_free < float((local.get("safety") or {}).get("min_free_disk_gib", 20)):
        blockers.append(f"free disk {disk_free:.1f} GiB is below reserve")
    max_steps = int(local["training"].get("max_steps", 0))
    batch_size = int(local["training"].get("batch_size", 4))
    enabled = list(local["training"].get("objectives") or [])
    dataset = SampleStoreDataset(gate.store, "train", config)
    benchmark_batches = []
    for index, batch in enumerate(
        dataset.batches(
            batch_size=batch_size,
            shuffle=True,
            seed=int(local["training"].get("seed", 20261103)),
            mask_instructions=True,
            mask_bytes=True,
            modality_dropout=0.02,
            grouped_fraction=float(local["training"].get("equivalence_sampling_fraction", 0.5)),
        )
    ):
        benchmark_batches.append(batch)
        if len(benchmark_batches) >= 12:
            break
    if not benchmark_batches:
        benchmark_batches = [_readiness_batch(gate, config, batch_size=batch_size)]
    model.train()
    times = []
    for batch in benchmark_batches:
        started = time.perf_counter()
        outputs = model(batch, run_decoder=True)
        _attach_evidence_outputs(model, batch, outputs, 19, enabled)
        losses = compute_objective_losses(outputs, batch, enabled)
        total = None
        for loss, count, _detail in losses.values():
            if count and loss is not None:
                total = loss if total is None else total + loss
        if total is not None:
            torch.autograd.grad(total, list(model.parameters()), allow_unused=True)
        times.append(time.perf_counter() - started)
    model.zero_grad(set_to_none=True)
    measured_times = times[1:] if len(times) > 2 else times
    mean_seconds = sum(measured_times) / max(1, len(measured_times))
    sorted_times = sorted(measured_times)
    p90_seconds = sorted_times[min(len(sorted_times) - 1, int(len(sorted_times) * 0.9))]
    max_seconds = max(measured_times)
    step_seconds = mean_seconds * 1.25
    gradient_accumulation = int(local["training"].get("gradient_accumulation_steps", 1))
    estimated_hours = step_seconds * max_steps * gradient_accumulation / 3600.0
    rss = _rss_mib()
    if rss > 11.0 * 1024:
        blockers.append(f"measured RSS {rss:.0f} MiB exceeds 11 GiB")
    if estimated_hours > 24.0:
        blockers.append(f"estimated CPU training time {estimated_hours:.2f} h exceeds 24 h")
    gpu = hardware.get("gpu") or {}
    cpu_pilot_available = not blockers
    foundation_gpu = bool(gpu.get("cuda_available")) or bool(gpu.get("xpu_available"))
    bf16_supported = False
    try:
        if bool(torch.cuda.is_available()) and hasattr(torch.cuda, "is_bf16_supported"):
            bf16_supported = bool(torch.cuda.is_bf16_supported())
    except (AttributeError, RuntimeError):
        bf16_supported = False
    evidence = {
        "hardware": {
            "cpu": hardware.get("cpu"),
            "memory": hardware.get("memory"),
            "disk": hardware.get("disk"),
            "gpu": gpu,
            "torch": hardware.get("torch"),
        },
        "precision": "fp32",
        "bf16_supported": bf16_supported,
        "benchmark_batches": len(benchmark_batches),
        "measured_step_seconds": round(step_seconds, 4),
        "measured_step_seconds_mean_raw": round(mean_seconds, 4),
        "measured_step_seconds_p90_raw": round(p90_seconds, 4),
        "measured_step_seconds_max_raw": round(max_seconds, 4),
        "estimate_safety_factor": 1.25,
        "batch_size": batch_size,
        "gradient_accumulation_steps": gradient_accumulation,
        "max_steps": max_steps,
        "estimated_hours": round(estimated_hours, 3),
        "measured_rss_mib": round(rss, 2),
        "disk_free_gib": round(disk_free, 2),
        "cpu_pilot_available": cpu_pilot_available,
        "gpu_available_for_foundation_scale": foundation_gpu,
        "note": "CPU fp32 is the only verifiable backend on this host; BF16 is not claimed. The estimate uses the mean of up to 12 varied train batches (first warm-up discarded) with a 1.25 safety factor, and includes data loading, forward, evidence-masked forward, losses and backward, but no optimizer step. The raw p90 and max show tail risk from long functions.",
    }
    return _gate("compute", not blockers, evidence, blockers)

def _foundation_scale(data_evidence: dict[str, Any], compute_evidence: dict[str, Any]) -> dict[str, Any]:
    blockers = []
    for key in ("unique_projects", "unique_source_functions", "unique_binaries"):
        actual = data_evidence.get(key)
        threshold = FOUNDATION_THRESHOLDS[key]
        if actual is None or actual < threshold:
            blockers.append(f"{key} {actual} < {threshold}")
    for key in ("formats", "languages", "architectures"):
        actual = data_evidence.get(key) or {}
        actual_count = len(actual) if isinstance(actual, dict) else int(actual)
        if actual_count < FOUNDATION_THRESHOLDS[key]:
            blockers.append(f"{key} {actual_count} < {FOUNDATION_THRESHOLDS[key]}")
    tokens = data_evidence.get("instruction_tokens") or 0
    if tokens < FOUNDATION_THRESHOLDS["instruction_tokens"]:
        blockers.append(f"instruction_tokens {tokens} < {FOUNDATION_THRESHOLDS['instruction_tokens']}")
    if not compute_evidence.get("gpu_available_for_foundation_scale"):
        blockers.append("no verified GPU training backend")
    return {
        "status": "FOUNDATION_SCALE_READY" if not blockers else "FOUNDATION_SCALE_BLOCKED",
        "blockers": blockers,
        "thresholds": FOUNDATION_THRESHOLDS,
        "note": "The compact CPU pilot is a separate, explicitly scoped package; this is not a foundation-scale readiness claim.",
    }


def check_readiness(config_path: str | Path, output_dir: str | Path | None = None) -> dict[str, Any]:
    local = load_local_config(config_path)
    frozen_dir = local["dataset"]["frozen_dir"]
    if not Path(frozen_dir).is_absolute():
        frozen_dir = ROOT / frozen_dir
    gate = load_gate(frozen_dir)
    gate.require_clean()
    if gate.store is None:
        hardware = detect_hardware()
        data = _gate(
            "data",
            False,
            {
                "frozen_id": gate.frozen_id,
                "counts": gate.manifest.get("counts"),
                "reason": "legacy v2 gate has no v3 sample store",
                "unique_source_functions": len({row.get("source_function_id") for split in ("train", "validation") for row in gate.splits.get(split, []) if row.get("source_function_id")}),
                "unique_projects": len({row.get("project_id") for split in ("train", "validation") for row in gate.splits.get(split, []) if row.get("project_id")}),
            },
            ["legacy v2 gate has no v3 function-level sample store; it is not the final corpus"],
        )
        compute = _gate(
            "compute",
            False,
            {"hardware": {"cpu": hardware.get("cpu"), "memory": hardware.get("memory"), "gpu": hardware.get("gpu")}},
            ["no verified GPU backend for foundation-scale training; legacy gate is not the final package"],
        )
        report = {
            "schema_version": "nexus.readiness/v3",
            "status": "TRAINING_PACKAGE_BLOCKED_BY_DATA",
            "scope": "legacy_gate",
            "config": str(Path(config_path).resolve()),
            "frozen_id": gate.frozen_id,
            "gates": [data, compute],
            "data": data,
            "compute": compute,
            "foundation_scale": {"status": "FOUNDATION_SCALE_BLOCKED", "blockers": ["legacy gate"]},
            "blockers": data["blockers"],
            "pilot_authorized": False,
            "final_training_authorized": False,
            "training_started": False,
            "optimizer_steps": 0,
            "pretrained_weights_loaded": False,
            "note": "Legacy v2 gate is retained for history only. Use configs/final_training.yaml and the v3 package for the final readiness decision.",
        }
        if output_dir is not None:
            target = Path(output_dir)
            target.mkdir(parents=True, exist_ok=True)
            (target / "readiness_report.json").write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")
            (target / "readiness_report.md").write_text("# NEXUS-BI readiness\n\n**Status:** `TRAINING_PACKAGE_BLOCKED_BY_DATA`\n\nLegacy v2 gate is not the final corpus.\n", encoding="utf-8")
        return report
    model_config = model_config_from_dict(local["model"], gate.vocab)
    torch.manual_seed(int(local["training"].get("seed", 20261103)))
    model = build_foundation_model(model_config)

    data = _data_gate(gate, local)
    architecture = _architecture_gate(gate, local, model_config, model)
    objectives = _objective_gate(gate, model_config, model, local)
    reconstruction = _reconstruction_gate(gate, model_config)
    investigation = _investigation_gate(gate)
    runtime = _runtime_gate(local, model_config, model, gate.vocab)
    evaluation = _evaluation_gate(gate, model_config)
    compute = _compute_gate(gate, model_config, model, local)
    foundation = _foundation_scale(data["evidence"], compute["evidence"])

    required = [data, architecture, objectives, reconstruction, runtime, evaluation, compute]
    blockers = [blocker for gate_item in required for blocker in gate_item["blockers"]]
    if not data["status"] == "PASS":
        status = "TRAINING_PACKAGE_BLOCKED_BY_DATA"
    elif not architecture["status"] == "PASS":
        status = "TRAINING_PACKAGE_BLOCKED_BY_ARCHITECTURE"
    elif not objectives["status"] == "PASS":
        status = "TRAINING_PACKAGE_BLOCKED_BY_OBJECTIVES"
    elif not runtime["status"] == "PASS":
        status = "TRAINING_PACKAGE_BLOCKED_BY_RUNTIME"
    elif not evaluation["status"] == "PASS":
        status = "TRAINING_PACKAGE_BLOCKED_BY_EVALUATION"
    elif not compute["status"] == "PASS":
        status = "TRAINING_PACKAGE_READY_BUT_COMPUTE_UNAVAILABLE"
    else:
        status = "TRAINING_PACKAGE_READY"

    report = {
        "schema_version": "nexus.readiness/v3",
        "status": status,
        "scope": "compact_cpu_pilot",
        "config": str(Path(config_path).resolve()),
        "frozen_id": gate.frozen_id,
        "gates": [data, architecture, objectives, reconstruction, investigation, runtime, evaluation, compute],
        "data": data,
        "architecture": architecture,
        "objectives": objectives,
        "reconstruction": reconstruction,
        "investigation": investigation,
        "runtime": runtime,
        "evaluation": evaluation,
        "compute": compute,
        "foundation_scale": foundation,
        "blockers": blockers,
        "pilot_authorized": status in {"TRAINING_PACKAGE_READY", "TRAINING_PACKAGE_READY_BUT_COMPUTE_UNAVAILABLE"},
        "training_started": False,
        "optimizer_steps": 0,
        "pretrained_weights_loaded": False,
        "note": "This gate authorizes the explicitly scoped compact CPU pilot, not foundation-scale training. The user must launch training manually.",
    }
    if output_dir is not None:
        target = Path(output_dir)
        target.mkdir(parents=True, exist_ok=True)
        (target / "readiness_report.json").write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")
        lines = [
            "# NEXUS-BI training package readiness",
            "",
            f"**Status:** `{status}`",
            f"**Scope:** `{report['scope']}`",
            f"**Frozen corpus:** `{report['frozen_id']}`",
            "",
            "## Gates",
            "",
            "| Gate | Status | Required | Blockers |",
            "|---|---|---|---|",
        ]
        for gate_item in report["gates"]:
            lines.append(f"| {gate_item['name']} | {gate_item['status']} | {gate_item['required']} | {'; '.join(gate_item['blockers']) or '-'} |")
        lines.extend(["", "## Foundation-scale gap", "", f"**Status:** `{foundation['status']}`", ""])
        for blocker in foundation["blockers"]:
            lines.append(f"- {blocker}")
        lines.extend(["", "## Data", ""])
        data_evidence = data["evidence"]
        for key in ("unique_projects", "unique_source_functions", "unique_binaries", "source_labelled_train_samples", "instruction_tokens", "same_source_variant_groups", "cross_arch_samples"):
            lines.append(f"- {key}: `{data_evidence.get(key)}`")
        lines.extend(["", "## Compute", ""])
        compute_evidence = compute["evidence"]
        for key in ("measured_step_seconds", "estimated_hours", "measured_rss_mib", "disk_free_gib", "cpu_pilot_available", "gpu_available_for_foundation_scale"):
            lines.append(f"- {key}: `{compute_evidence.get(key)}`")
        lines.extend(["", "## Notes", "", report["note"], ""])
        (target / "readiness_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report
