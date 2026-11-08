"""Objective registry, losses, scheduling and loss aggregation.

Every objective is explicit: modalities, required labels, mission objective,
loss, default weight, schedule, evaluation method and failure modes.  The
training engine never silently enables a task whose labels are absent.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


@dataclass
class ObjectiveSpec:
    objective_id: str
    mission_objective: str
    description: str
    input_modalities: list[str]
    target: str
    required_labels: list[str]
    loss: str
    base_weight: float
    schedule: dict[str, Any] = field(default_factory=dict)
    evaluation: str = ""
    known_failure_modes: list[str] = field(default_factory=list)
    classification: str = "CORE"
    implemented: bool = True
    default_enabled: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


OBJECTIVES: list[ObjectiveSpec] = [
    ObjectiveSpec(
        "byte_masked_modeling",
        "binary_representation_masking",
        "Masked byte prediction over instruction byte streams.",
        ["bytes"],
        "masked byte ids",
        ["byte_mask"],
        "cross_entropy",
        0.50,
        {"warmup_steps": 0, "ramp_steps": 20, "min_scale": 0.1, "max_scale": 1.0},
        "byte accuracy and loss versus majority/random baselines",
        ["Byte statistics can dominate over semantics.", "Padding and instruction boundaries can leak position."],
    ),
    ObjectiveSpec(
        "instruction_masked_modeling",
        "binary_representation_masking",
        "Masked instruction-token prediction from normalized mnemonic/operand shapes.",
        ["instructions", "ir", "graph"],
        "masked instruction ids",
        ["instr_mask"],
        "cross_entropy",
        1.00,
        {"warmup_steps": 0, "ramp_steps": 10, "min_scale": 0.1, "max_scale": 1.0},
        "masked instruction accuracy and loss versus majority/random",
        ["A high score can reflect operand-shape statistics rather than semantics."],
    ),
    ObjectiveSpec(
        "instruction_autoregressive",
        "binary_representation_masking",
        "Causal next-instruction prediction with a causal instruction trunk.",
        ["instructions", "ir", "graph"],
        "next instruction id",
        ["instr_id"],
        "cross_entropy",
        0.20,
        {"warmup_steps": 5, "ramp_steps": 25, "min_scale": 0.0, "max_scale": 1.0},
        "perplexity/top-1 on held-out functions",
        ["Requires a second causal forward pass; disabled in the CPU smoke run."],
        default_enabled=False,
        classification="SUPPORTING",
    ),
    ObjectiveSpec(
        "ir_op_prediction",
        "instruction_semantics",
        "Predict semantic-IR operation classes for instructions whose IR input is masked.",
        ["instructions", "graph"],
        "IR op class",
        ["instr_ir"],
        "cross_entropy",
        0.50,
        {"warmup_steps": 0, "ramp_steps": 15, "min_scale": 0.1, "max_scale": 1.0},
        "IR-class accuracy versus mnemonic-mapping baseline",
        ["Targets are deterministic-tool labels, not ground-truth semantics."],
    ),
    ObjectiveSpec(
        "register_effect_prediction",
        "type_dataflow_inference",
        "Multi-label prediction of registers read and written by each instruction.",
        ["instructions", "ir"],
        "register read/write sets",
        ["instr_mask"],
        "binary_cross_entropy",
        0.35,
        {"warmup_steps": 0, "ramp_steps": 20, "min_scale": 0.1, "max_scale": 1.0},
        "micro/macro F1 on held-out functions",
        ["Generic read/write effects are tool-derived; aliasing and partial registers remain unresolved."],
    ),
    ObjectiveSpec(
        "memory_effect_prediction",
        "instruction_semantics",
        "Predict memory read/write presence and access width.",
        ["instructions", "ir"],
        "memory effect class",
        ["instr_mask"],
        "binary_cross_entropy+ce",
        0.25,
        {"warmup_steps": 0, "ramp_steps": 20, "min_scale": 0.1, "max_scale": 1.0},
        "memory-effect F1 and width accuracy",
        ["Stack-frame inference is heuristic; unresolved aliases are not fabricated."],
    ),
    ObjectiveSpec(
        "cfg_edge_prediction",
        "basic_block_correspondence",
        "Predict basic-block CFG adjacency from block representations.",
        ["instructions", "graph"],
        "block adjacency",
        ["block_mask"],
        "binary_cross_entropy",
        0.50,
        {"warmup_steps": 0, "ramp_steps": 15, "min_scale": 0.1, "max_scale": 1.0},
        "edge precision/recall/F1 on held-out functions",
        ["Sparse positive edges; class imbalance can collapse to all-negative."],
    ),
    ObjectiveSpec(
        "function_start_prediction",
        "function_boundary_prediction",
        "Per-byte function-entry classification when aligned boundary labels are supplied.",
        ["bytes", "metadata"],
        "function start flag",
        ["byte_mask"],
        "binary_cross_entropy",
        0.15,
        {"warmup_steps": 10, "ramp_steps": 30, "min_scale": 0.0, "max_scale": 1.0},
        "precision/recall/F1 versus entry-only and random baselines",
        ["Function-level samples only label the first byte; do not claim boundary capability from this objective."],
        default_enabled=False,
        classification="SUPPORTING",
    ),
    ObjectiveSpec(
        "cross_modal_alignment",
        "binary_representation_masking",
        "Symmetric contrastive alignment between instruction, byte and IR pooled spaces.",
        ["bytes", "instructions", "ir"],
        "shared representation",
        ["*"],
        "infonce",
        0.15,
        {"warmup_steps": 0, "ramp_steps": 20, "min_scale": 0.1, "max_scale": 1.0},
        "retrieval recall across modalities",
        ["Small batches create false negatives; alignment can reward length/compiler shortcuts."],
    ),
    ObjectiveSpec(
        "binary_source_alignment",
        "source_binary_alignment",
        "Contrastive alignment between binary function representation and source-function encoding (training only).",
        ["instructions", "ir", "graph", "source_target"],
        "binary/source correspondence",
        ["source target"],
        "infonce",
        0.35,
        {"warmup_steps": 0, "ramp_steps": 20, "min_scale": 0.1, "max_scale": 1.0},
        "binary-to-source retrieval on held-out projects",
        ["Source target is label-side only and must never enter binary-only inference."],
    ),
    ObjectiveSpec(
        "function_matching",
        "cross_compiler_function_retrieval",
        "Supervised contrastive function matching across compiler/optimization variants.",
        ["instructions", "ir", "graph"],
        "function identity group",
        ["match_group"],
        "supervised_contrastive",
        0.30,
        {"warmup_steps": 5, "ramp_steps": 20, "min_scale": 0.0, "max_scale": 1.0},
        "Recall@1/@5 and MRR versus mnemonic/byte baselines",
        ["Batch-composition shortcuts and compiler fingerprints can create spurious matches."],
        default_enabled=False,
        classification="SUPPORTING",
    ),
    ObjectiveSpec(
        "semantic_equivalence",
        "cross_compiler_function_retrieval",
        "Supervised contrastive learning of semantic equivalence across compiler, optimization, build and architecture variants of the same source function.",
        ["instructions", "ir", "graph", "bytes"],
        "semantic equivalence group",
        ["match_group"],
        "supervised_contrastive",
        0.50,
        {"warmup_steps": 0, "ramp_steps": 20, "min_scale": 0.1, "max_scale": 1.0},
        "Recall@1/@5 and MRR across compiler/optimization variants; never a behavioral-equivalence claim",
        ["Random batches may not contain two variants of one function; grouped sampling is required.", "Compiler fingerprints can create spurious matches."],
    ),
    ObjectiveSpec(
        "purpose_prediction",
        "function_purpose_inference",
        "Function-purpose classification from binary-only function representations.",
        ["instructions", "ir", "graph", "metadata"],
        "purpose class",
        ["purpose label"],
        "cross_entropy",
        0.40,
        {"warmup_steps": 0, "ramp_steps": 20, "min_scale": 0.1, "max_scale": 1.0},
        "accuracy and macro-F1 versus majority baseline",
        ["Labels may be sparse or subjective; no causal behavioral claim follows."],
        default_enabled=False,
        classification="SUPPORTING",
    ),
    ObjectiveSpec(
        "c_reconstruction",
        "c_reconstruction",
        "Conditional autoregressive C reconstruction from binary function evidence.",
        ["instructions", "ir", "graph", "bytes"],
        "C token sequence",
        ["source target"],
        "cross_entropy",
        1.00,
        {"warmup_steps": 0, "ramp_steps": 20, "min_scale": 0.1, "max_scale": 1.0},
        "syntax, compilation, structural and behavioral statuses separately",
        ["Token accuracy is not compilation or behavioral equivalence.", "Source-derived targets must not leak into inputs."],
    ),
    ObjectiveSpec(
        "java_jni_reconstruction",
        "java_jni_reconstruction",
        "Java/JNI reconstruction and JNI signature prediction.",
        ["instructions", "ir", "jni_metadata"],
        "Java token sequence and JNI signature",
        ["java target", "jni metadata"],
        "cross_entropy",
        0.50,
        {"warmup_steps": 5, "ramp_steps": 25, "min_scale": 0.0, "max_scale": 1.0},
        "syntax, compilation and JNI signature statuses",
        ["JNI threading, long-lived references and callbacks remain unverified."],
        default_enabled=False,
        classification="SUPPORTING",
    ),
    ObjectiveSpec(
        "evidence_grounded_planning",
        "investigation_planning",
        "Evidence-conditioned tool/plan selection and confidence estimation.",
        ["instructions", "metadata", "evidence"],
        "tool class and confidence",
        ["plan label"],
        "cross_entropy+calibration",
        0.15,
        {"warmup_steps": 10, "ramp_steps": 30, "min_scale": 0.0, "max_scale": 1.0},
        "plan accuracy against deterministic planner and calibration error",
        ["Deterministic planner labels are a distillation target, not autonomous reasoning evidence."],
        default_enabled=False,
    ),
    ObjectiveSpec(
        "confidence_calibration",
        "binary_representation_masking",
        "Calibrated confidence for task predictions where labels exist.",
        ["*"],
        "confidence",
        ["confidence label"],
        "binary_cross_entropy",
        0.10,
        {"warmup_steps": 10, "ramp_steps": 30, "min_scale": 0.0, "max_scale": 1.0},
        "expected calibration error and reliability diagram",
        ["Unlabeled defaults must never be treated as calibrated confidence."],
        default_enabled=False,
        classification="SUPPORTING",
    ),
]


_CLASSIFICATIONS = {
    "instruction_autoregressive": "SUPPORTING",
    "function_start_prediction": "SUPPORTING",
    "function_matching": "SUPPORTING",
    "java_jni_reconstruction": "SUPPORTING",
    "evidence_grounded_planning": "SUPPORTING",
    "confidence_calibration": "SUPPORTING",
}
for _objective in OBJECTIVES:
    if _objective.objective_id in _CLASSIFICATIONS:
        _objective.classification = _CLASSIFICATIONS[_objective.objective_id]


def objective_registry() -> dict[str, Any]:
    return {
        "schema_version": "0.1.0",
        "architecture_version": "nexus-bi/foundation-1",
        "objectives": [objective.to_dict() for objective in OBJECTIVES],
        "default_weights": {objective.objective_id: objective.base_weight for objective in OBJECTIVES},
        "notes": [
            "Weights are starting points, not tuned acceptance thresholds.",
            "Objectives with absent labels are inactive and reported as such.",
            "Test splits are never used for objective selection.",
        ],
    }


def save_objective_registry(path: str | Path) -> dict[str, Any]:
    import hashlib

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = objective_registry()
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    target.write_text(text, encoding="utf-8")
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    payload["registry_sha256"] = digest
    return {"path": str(target), "sha256": digest}


def get_objective(objective_id: str) -> ObjectiveSpec:
    for objective in OBJECTIVES:
        if objective.objective_id == objective_id:
            return objective
    raise KeyError(f"unknown objective: {objective_id}")


@dataclass
class ObjectiveResult:
    objective_id: str
    raw: float
    weight: float
    weighted: float
    active: bool
    count: int
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _masked_cross_entropy(logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, int]:
    if logits.dim() == 3:
        logits = logits.reshape(-1, logits.size(-1))
        targets = targets.reshape(-1)
        mask = mask.reshape(-1)
    valid = mask & targets.ge(0) & targets.lt(logits.size(-1))
    count = int(valid.sum())
    if count == 0:
        return logits.sum() * 0.0, 0
    return F.cross_entropy(logits[valid], targets[valid]), count


def _binary(logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, int]:
    if logits.dim() > targets.dim():
        logits = logits.squeeze(-1)
    valid = mask
    count = int(valid.sum())
    if count == 0:
        return logits.sum() * 0.0, 0
    return F.binary_cross_entropy_with_logits(logits[valid], targets[valid].to(logits.dtype)), count


def _infonce(left: torch.Tensor, right: torch.Tensor, temperature: float = 0.1) -> tuple[torch.Tensor, int]:
    if left.size(0) < 2:
        return left.sum() * 0.0, 0
    left = F.normalize(left, dim=-1)
    right = F.normalize(right, dim=-1)
    logits = left @ right.transpose(0, 1) / temperature
    labels = torch.arange(left.size(0), device=left.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.transpose(0, 1), labels)), left.size(0)


def _supervised_contrastive(embeddings: torch.Tensor, groups: list[str], temperature: float = 0.07) -> tuple[torch.Tensor, int]:
    if len(groups) < 2 or len(set(groups)) < 2:
        return embeddings.sum() * 0.0, 0
    embeddings = F.normalize(embeddings, dim=-1)
    logits = embeddings @ embeddings.transpose(0, 1) / temperature
    index = torch.arange(len(groups), device=embeddings.device)
    logits = logits.masked_fill(index.unsqueeze(0) == index.unsqueeze(1), -1e9)
    same = torch.tensor([[groups[i] == groups[j] for j in range(len(groups))] for i in range(len(groups))], device=embeddings.device)
    same = same & (index.unsqueeze(0) != index.unsqueeze(1))
    positives = same.sum(dim=1)
    if int(positives.sum()) == 0:
        return embeddings.sum() * 0.0, 0
    log_probabilities = F.log_softmax(logits, dim=-1)
    anchor_loss = -(log_probabilities * same).sum(dim=-1) / positives.clamp(min=1)
    valid = positives > 0
    return anchor_loss[valid].mean(), int(valid.sum())


def compute_objective_losses(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, Any],
    enabled: list[str] | None = None,
) -> dict[str, tuple[torch.Tensor, int, str]]:
    """Return raw loss, count and note for every enabled objective with labels."""
    enabled = enabled or [objective.objective_id for objective in OBJECTIVES if objective.default_enabled]
    results: dict[str, tuple[torch.Tensor, int, str]] = {}
    instruction_mask = batch.get("instr_mask")
    if instruction_mask is None:
        instruction_mask = torch.ones(batch["instr_id"].shape, dtype=torch.bool)
    for objective_id in enabled:
        if objective_id == "byte_masked_modeling":
            if "byte_targets" in batch and "byte_mlm_mask" in batch:
                loss, count = _masked_cross_entropy(outputs["byte_logits"], batch["byte_targets"], batch["byte_mlm_mask"])
                results[objective_id] = (loss, count, "")
            else:
                results[objective_id] = (outputs["byte_logits"].sum() * 0.0, 0, "no masked byte labels")
        elif objective_id == "instruction_masked_modeling":
            if "instr_targets" in batch and "instr_mlm_mask" in batch:
                loss, count = _masked_cross_entropy(outputs["instruction_logits"], batch["instr_targets"], batch["instr_mlm_mask"])
                results[objective_id] = (loss, count, "")
            else:
                results[objective_id] = (outputs["instruction_logits"].sum() * 0.0, 0, "no masked instruction labels")
        elif objective_id == "instruction_autoregressive":
            if "instr_ar_logits" in outputs and "instr_ar_targets" in batch:
                loss, count = _masked_cross_entropy(outputs["instr_ar_logits"], batch["instr_ar_targets"], batch["instr_ar_mask"])
                results[objective_id] = (loss, count, "")
            else:
                results[objective_id] = (outputs["instruction_logits"].sum() * 0.0, 0, "autoregressive forward not run")
        elif objective_id == "ir_op_prediction":
            if "ir_mlm_logits" in outputs and "ir_targets" in batch and "ir_mlm_mask" in batch:
                loss, count = _masked_cross_entropy(outputs["ir_mlm_logits"], batch["ir_targets"], batch["ir_mlm_mask"])
                results[objective_id] = (loss, count, "evidence-masked IR prediction")
            else:
                results[objective_id] = (outputs["ir_logits"].sum() * 0.0, 0, "evidence-masked forward not run")
        elif objective_id == "register_effect_prediction":
            if "register_read_masked_logits" not in outputs or "register_write_masked_logits" not in outputs:
                results[objective_id] = (outputs["pooled"].sum() * 0.0, 0, "evidence-masked forward not run")
            else:
                read_loss, read_count = F.binary_cross_entropy_with_logits(
                    outputs["register_read_masked_logits"][instruction_mask], batch["read_targets"][instruction_mask]
                ), int(instruction_mask.sum())
                write_loss, write_count = F.binary_cross_entropy_with_logits(
                    outputs["register_write_masked_logits"][instruction_mask], batch["write_targets"][instruction_mask]
                ), int(instruction_mask.sum())
                results[objective_id] = ((read_loss + write_loss) * 0.5, read_count + write_count, "masked read+write BCE")
        elif objective_id == "memory_effect_prediction":
            if "memory_read_masked_logits" not in outputs:
                results[objective_id] = (outputs["pooled"].sum() * 0.0, 0, "evidence-masked forward not run")
            else:
                read_loss, read_count = _binary(outputs["memory_read_masked_logits"], batch["instr_mem_read"], instruction_mask)
                write_loss, write_count = _binary(outputs["memory_write_masked_logits"], batch["instr_mem_write"], instruction_mask)
                width_mask = instruction_mask & batch["instr_mem_width"].gt(0)
                width_loss, width_count = _masked_cross_entropy(outputs["memory_width_masked_logits"], batch["instr_mem_width"], width_mask)
                results[objective_id] = (read_loss + write_loss + width_loss, read_count + write_count + width_count, "masked read+write+width")
        elif objective_id == "cfg_edge_prediction":
            if "cfg_masked_logits" not in outputs:
                results[objective_id] = (outputs["pooled"].sum() * 0.0, 0, "evidence-masked forward not run")
                continue
            block_mask = batch["block_mask"]
            valid = block_mask.unsqueeze(1) & block_mask.unsqueeze(2)
            diagonal = torch.eye(valid.size(1), dtype=torch.bool, device=valid.device).unsqueeze(0)
            valid = valid & ~diagonal
            count = int(valid.sum())
            if count:
                loss = F.binary_cross_entropy_with_logits(
                    outputs["cfg_masked_logits"][valid],
                    batch["cfg_target"][valid].to(outputs["cfg_masked_logits"].dtype),
                    pos_weight=torch.tensor(4.0, device=outputs["cfg_masked_logits"].device),
                )
            else:
                loss = outputs["cfg_masked_logits"].sum() * 0.0
            results[objective_id] = (loss, count, "masked block-pair BCE, diagonal excluded")
        elif objective_id == "function_start_prediction":
            logits = outputs.get("function_start_logits")
            mask = outputs.get("function_start_mask")
            if logits is None or mask is None:
                results[objective_id] = (outputs["pooled"].sum() * 0.0, 0, "no byte-level start head")
            else:
                loss, count = _binary(logits, batch["function_start_target"], mask & batch["function_start_target"].ge(0))
                results[objective_id] = (loss, count, "per-byte entry classification")
        elif objective_id == "cross_modal_alignment":
            first, count = _infonce(outputs["instruction_embedding"], outputs["byte_embedding"])
            second, _ = _infonce(outputs["instruction_embedding"], outputs["ir_embedding"])
            results[objective_id] = (first + second, count, "instruction-byte + instruction-IR InfoNCE")
        elif objective_id == "binary_source_alignment":
            source = outputs.get("source_embedding")
            if source is None:
                results[objective_id] = (outputs["pooled"].sum() * 0.0, 0, "no source targets")
            else:
                loss, count = _infonce(outputs["function_embedding"], source)
                results[objective_id] = (loss, count, "function-source InfoNCE")
        elif objective_id in {"function_matching", "semantic_equivalence"}:
            groups = [str(meta.get("match_group") or "") for meta in batch.get("sample_meta", [])]
            groups = [group if group else f"singleton:{index}" for index, group in enumerate(groups)]
            loss, count = _supervised_contrastive(outputs["function_embedding"], groups)
            results[objective_id] = (loss, count, "supervised contrastive by source/function equivalence group")
        elif objective_id == "purpose_prediction":
            valid = batch["purpose_target"].ge(0)
            count = int(valid.sum())
            if count == 0:
                loss = outputs["purpose_logits"].sum() * 0.0
            else:
                weights = batch.get("purpose_class_weights")
                loss = F.cross_entropy(outputs["purpose_logits"][valid], batch["purpose_target"][valid], weight=weights)
            results[objective_id] = (loss, count, "class-weighted CE" if batch.get("purpose_class_weights") is not None else "")
        elif objective_id == "c_reconstruction":
            if "c_logits" in outputs and "c_ids" in batch:
                target = batch["c_ids"][:, 1:]
                mask = batch["c_mask"][:, 1:] if "c_mask" in batch else torch.ones_like(target, dtype=torch.bool)
                loss, count = _masked_cross_entropy(outputs["c_logits"], target, mask)
                results[objective_id] = (loss, count, "C token cross-entropy")
            else:
                results[objective_id] = (outputs["pooled"].sum() * 0.0, 0, "decoder not run / no C target")
        elif objective_id == "java_jni_reconstruction":
            java_loss = outputs["pooled"].sum() * 0.0
            java_count = 0
            if "java_logits" in outputs and "java_ids" in batch and batch["java_ids"].any():
                target = batch["java_ids"][:, 1:]
                mask = batch["java_mask"][:, 1:] if "java_mask" in batch else torch.ones_like(target, dtype=torch.bool)
                java_loss, java_count = _masked_cross_entropy(outputs["java_logits"], target, mask)
            jni_valid = batch["jni_targets"].ge(0)
            jni_count = int(jni_valid.sum())
            if jni_count:
                logits = outputs["jni_logits"][jni_valid]
                target = batch["jni_targets"][jni_valid]
                jni_loss = F.cross_entropy(logits, target)
                java_loss = java_loss + jni_loss
            results[objective_id] = (java_loss, java_count + jni_count, "Java tokens + JNI types")
        elif objective_id == "evidence_grounded_planning":
            tool_loss, tool_count = _masked_cross_entropy(
                outputs["tool_logits"], batch["tool_target"], torch.ones_like(batch["tool_target"], dtype=torch.bool)
            )
            security_loss, security_count = _masked_cross_entropy(
                outputs["security_logits"], batch["security_target"], torch.ones_like(batch["security_target"], dtype=torch.bool)
            )
            results[objective_id] = (tool_loss + security_loss, tool_count + security_count, "tool + security planning classes")
        elif objective_id == "confidence_calibration":
            mask = batch.get("confidence_mask")
            if mask is None or not bool(mask.any()):
                results[objective_id] = (outputs["confidence"].sum() * 0.0, 0, "no confidence labels")
            else:
                loss, count = _binary(outputs["confidence"], batch["confidence_target"], mask)
                results[objective_id] = (loss, count, "")
        else:
            results[objective_id] = (outputs["pooled"].sum() * 0.0, 0, "unknown objective")
    return results


class ObjectiveSchedule:
    def __init__(self, spec: ObjectiveSpec):
        self.spec = spec

    def scale(self, step: int) -> float:
        schedule = self.spec.schedule or {}
        warmup = int(schedule.get("warmup_steps", 0))
        ramp = int(schedule.get("ramp_steps", 0))
        minimum = float(schedule.get("min_scale", 0.0))
        maximum = float(schedule.get("max_scale", 1.0))
        if step < warmup:
            return 0.0
        if ramp <= 0:
            return maximum
        progress = min(1.0, (step - warmup) / max(1, ramp))
        return minimum + (maximum - minimum) * progress


class LossAggregator(torch.nn.Module):
    """Configurable multi-objective aggregation with per-objective diagnostics."""

    def __init__(self, mode: str = "normalized", enabled: list[str] | None = None):
        super().__init__()
        if mode not in {"fixed", "normalized", "uncertainty"}:
            raise ValueError(f"unknown aggregation mode: {mode}")
        self.mode = mode
        self.enabled = enabled or [objective.objective_id for objective in OBJECTIVES if objective.default_enabled]
        self.log_sigma = torch.nn.ParameterDict(
            {objective_id: torch.nn.Parameter(torch.zeros(())) for objective_id in self.enabled}
        )
        self.register_buffer("_unused", torch.zeros(1), persistent=False)
        self.running: dict[str, float] = {}

    def export_state(self) -> dict[str, Any]:
        return {"mode": self.mode, "running": dict(self.running)}

    def import_state(self, payload: dict[str, Any] | None) -> None:
        if not payload:
            return
        mode = payload.get("mode")
        if mode is not None and mode != self.mode:
            raise ValueError(f"loss aggregator mode mismatch: {mode} vs {self.mode}")
        self.running = {str(key): float(value) for key, value in (payload.get("running") or {}).items()}

    def forward(
        self,
        losses: dict[str, tuple[torch.Tensor, int, str]],
        step: int,
    ) -> tuple[torch.Tensor, list[ObjectiveResult]]:
        specs = {objective.objective_id: objective for objective in OBJECTIVES}
        results: list[ObjectiveResult] = []
        terms: list[torch.Tensor] = []
        for objective_id, (raw, count, note) in losses.items():
            spec = specs[objective_id]
            schedule_scale = ObjectiveSchedule(spec).scale(step)
            weight = spec.base_weight * schedule_scale
            active = count > 0 and float(raw.detach().abs()) >= 0.0
            if not active:
                results.append(ObjectiveResult(objective_id, float(raw.detach()), weight, 0.0, False, count, note or "inactive"))
                continue
            if self.mode == "uncertainty" and objective_id in self.log_sigma:
                sigma = self.log_sigma[objective_id].clamp(-4.0, 4.0)
                term = torch.exp(-sigma) * raw + sigma
                weighted_value = float(term.detach())
            else:
                scale = 1.0
                if self.mode == "normalized":
                    previous = self.running.get(objective_id, max(float(raw.detach()), 1e-4))
                    scale = max(previous, 1e-4)
                    self.running[objective_id] = 0.9 * previous + 0.1 * max(float(raw.detach()), 1e-6)
                term = weight * raw / scale
                weighted_value = float(term.detach())
            terms.append(term)
            results.append(ObjectiveResult(objective_id, float(raw.detach()), weight, weighted_value, True, count, note))
        if not terms:
            total = self._unused.sum() * 0.0
        else:
            total = torch.stack(terms).sum()
        return total, results


def gradient_norms(
    model: torch.nn.Module,
    losses: dict[str, tuple[torch.Tensor, int, str]],
    parameters: list[torch.nn.Parameter] | None = None,
) -> dict[str, float]:
    parameters = parameters or [parameter for parameter in model.parameters() if parameter.requires_grad]
    result = {}
    for objective_id, (loss, count, _note) in losses.items():
        if count <= 0 or not loss.requires_grad:
            continue
        grads = torch.autograd.grad(loss, parameters, retain_graph=True, allow_unused=True)
        squared = 0.0
        for gradient in grads:
            if gradient is not None:
                squared += float(gradient.detach().pow(2).sum())
        result[objective_id] = squared ** 0.5
    return result
