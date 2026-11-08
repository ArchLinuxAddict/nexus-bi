"""Load and run the residual checkpoint for API and local checks."""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from nexus.foundation.config import FoundationConfig
from nexus.foundation.java_types import canonicalize_generated
from nexus.foundation.residual_data import collate_residual, load_residual_records
from nexus.foundation.residual_model import ResidualNexusModel, residual_parameter_report
from nexus.foundation.vocab import Vocabulary

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TEST = ROOT / "artifacts" / "dataset" / "jni_joint_v1" / "validation.jsonl"
if not DEFAULT_TEST.exists():
    DEFAULT_TEST = ROOT / "artifacts" / "dataset" / "residual_v6" / "test.jsonl"


def find_residual_checkpoint() -> Path:
    root = ROOT / "artifacts" / "residual_runs"
    active = root / "ACTIVE"
    if active.exists():
        target = Path(active.read_text(encoding="utf-8").strip())
        if target.exists():
            return target
    preferred = root / "residual-v15-joint" / "best.pt"
    if preferred.exists():
        return preferred
    candidates = sorted(root.glob("*/best.pt"), key=lambda path: path.stat().st_mtime, reverse=True)
    candidates += sorted(root.glob("*/final.pt"), key=lambda path: path.stat().st_mtime, reverse=True)
    candidates += sorted(root.glob("*/residual.pt"), key=lambda path: path.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else root / "residual-v6" / "residual.pt"


DEFAULT_CHECKPOINT = find_residual_checkpoint()
_LOCK = threading.Lock()
_RUNTIME: "ResidualRuntime | None" = None


class ResidualRuntime:
    def __init__(self, checkpoint: str | Path = DEFAULT_CHECKPOINT):
        self.checkpoint = Path(checkpoint)
        payload = torch.load(self.checkpoint, map_location="cpu", weights_only=False)
        self.payload = payload
        self.config = FoundationConfig.from_dict(payload["config"])
        self.vocab = Vocabulary(**{key: value for key, value in payload["vocab"].items() if key in Vocabulary.__dataclass_fields__})
        self.model = ResidualNexusModel(self.config)
        incompatible = self.model.load_state_dict(payload["model_state"], strict=False)
        if "copy_query.weight" not in payload["model_state"]:
            self.model.use_copy = False
        self.incompatible = {
            "missing": list(incompatible.missing_keys),
            "unexpected": list(incompatible.unexpected_keys),
        }
        self.heads_available = all(
            key in payload["model_state"]
            for key in ("candidate_encoder.token.weight", "head_fusion.0.weight")
        )
        self.core_incompatible = [
            key
            for key in self.incompatible["missing"]
            if not key.startswith(("candidate_encoder.", "head_fusion."))
        ]
        self.compatible = not self.core_incompatible and not self.incompatible["unexpected"]
        self.model.eval()
        self.metadata = payload.get("metadata") or {}

    def status(self) -> dict[str, Any]:
        return {
            "checkpoint": str(self.checkpoint),
            "exists": self.checkpoint.exists(),
            "size": self.checkpoint.stat().st_size if self.checkpoint.exists() else None,
            "metadata": self.metadata,
            "parameters": residual_parameter_report(self.model),
            "compatible": self.compatible,
            "core_missing_keys": len(self.core_incompatible),
            "missing_keys": len(self.incompatible["missing"]),
            "unexpected_keys": len(self.incompatible["unexpected"]),
            "auxiliary_heads_available": self.heads_available,
            "architecture_version": self.config.architecture_version,
            "profile": self.config.profile,
            "config_hash": self.config.config_hash(),
            "evaluation": json.loads((self.checkpoint.parent / f"evaluation_{self.checkpoint.stem}.json").read_text(encoding="utf-8")) if (self.checkpoint.parent / f"evaluation_{self.checkpoint.stem}.json").exists() else (json.loads((self.checkpoint.parent / "evaluation.json").read_text(encoding="utf-8")) if (self.checkpoint.parent / "evaluation.json").exists() else None),
        }

    def infer(
        self,
        tool_text: str,
        target_text: str = "",
        language: str = "java",
        tool: str = "jadx",
        tool_status: str = "completed",
        compile_status: str = "unknown",
        warning_count: int = 0,
        cached: bool = False,
        max_new_tokens: int = 128,
        candidate_text: str = "",
        jni_method: str = "",
        jni_descriptor: str = "",
        jni_instance: bool = False,
    ) -> dict[str, Any]:
        if not candidate_text and "CANDIDATE:" in tool_text:
            tail = tool_text.split("CANDIDATE:", 1)[1]
            candidate_text = tail.split("TASK:", 1)[0].strip().splitlines()[0] if tail.strip() else ""
        record = {
            "sample_id": "runtime",
            "language": language,
            "tool": tool,
            "tool_status": tool_status,
            "compile_status": compile_status,
            "warning_count": warning_count,
            "cached": cached,
            "tool_text": tool_text,
            "candidate_text": candidate_text,
            "target_text": target_text,
            "failure_label": 0,
            "candidate_label": 1.0,
            "confidence_label": 1.0,
        }
        batch = collate_residual([record], self.vocab, self.config)
        with torch.no_grad():
            outputs = self.model(batch)
            generated = self.model.generate_repair(batch, max_new_tokens=max_new_tokens)
        generated_text = self.vocab.decode_text(generated[0].tolist())
        canonical_text = None
        if jni_method and jni_descriptor:
            canonical_text = canonicalize_generated(
                generated_text,
                jni_method,
                jni_descriptor,
                instance=jni_instance,
            )
        candidate_score = float(torch.sigmoid(outputs["candidate_logit"])[0]) if self.heads_available else None
        failure_probability = float(F.softmax(outputs["failure_logits"], dim=-1)[0, 1]) if self.heads_available else None
        confidence = float(outputs["confidence"][0]) if self.heads_available else None
        result = {
            "candidate_score": candidate_score,
            "candidate_exact": (candidate_text.strip() == target_text.strip()) if (candidate_text and target_text) else None,
            "auxiliary_scores_reliable": False,
            "failure_probability": failure_probability,
            "confidence": confidence,
            "generated_text": generated_text,
            "canonical_text": canonical_text,
            "generated_tokens": int(generated.shape[1]),
            "tool": tool,
            "language": language,
            "limitations": [
                "The checkpoint is trained on JNI/Java repair pairs; generation is not exact code synthesis.",
                "Whitespace is preserved through byte tokens for older vocabularies.",
                "Candidate/failure/confidence heads are experimental; when unavailable they are returned as null. Use candidate_exact when a reference is supplied.",
                "Generation quality is not compile or behavioral verification.",
            ],
        }
        if target_text:
            target_ids = torch.tensor([self.vocab.encode_text(target_text, self.config.max_text_tokens)], dtype=torch.long)
            target_mask = torch.ones(target_ids.shape, dtype=torch.bool)
            input_ids = target_ids[:, :-1]
            input_mask = target_mask[:, :-1]
            with torch.no_grad():
                repaired = self.model(batch, repair_input_ids=input_ids, repair_input_mask=input_mask)
            logits = repaired.get("repair_log_probs", repaired["repair_logits"])
            labels = target_ids[:, 1:]
            mask = target_mask[:, 1:]
            predictions = logits.argmax(-1)
            result["teacher_forced_token_accuracy"] = float(((predictions == labels) & mask).sum().item() / max(1, int(mask.sum().item())))
            result["teacher_forced_exact"] = bool(((predictions == labels) | ~mask).all().item())
        return result

    def evaluate(self, test_path: str | Path = DEFAULT_TEST, batch_size: int = 2) -> dict[str, Any]:
        records = load_residual_records(test_path)
        token_correct = token_total = 0
        exact_correct = 0
        candidate_correct = failure_correct = 0
        confidence_error = 0.0
        with torch.no_grad():
            for start in range(0, len(records), batch_size):
                batch_records = records[start : start + batch_size]
                batch = collate_residual(batch_records, self.vocab, self.config)
                outputs = self.model(batch, repair_input_ids=batch["repair_input_ids"][:, :-1], repair_input_mask=batch["repair_mask"][:, :-1])
                labels = batch["repair_input_ids"][:, 1:]
                mask = batch["repair_mask"][:, 1:]
                predictions = outputs.get("repair_log_probs", outputs["repair_logits"]).argmax(-1)
                token_correct += int(((predictions == labels) & mask).sum().item())
                token_total += int(mask.sum().item())
                exact_correct += int(((predictions == labels) | ~mask).all(dim=1).sum().item())
                candidate = (outputs["candidate_logit"] > 0).long()
                failure = outputs["failure_logits"].argmax(-1)
                for index in range(len(batch_records)):
                    candidate_correct += int(candidate[index].item() == int(batch["candidate_label"][index] > 0.5))
                    failure_correct += int(failure[index].item() == int(batch["failure_label"][index]))
                    confidence_error += abs(float(outputs["confidence"][index]) - float(batch["confidence_label"][index]))
        return {
            "checkpoint": str(self.checkpoint),
            "test": str(test_path),
            "samples": len(records),
            "auxiliary_heads_available": self.heads_available,
            "metrics": {
                "candidate_accuracy": (candidate_correct / max(1, len(records))) if self.heads_available else None,
                "failure_accuracy": (failure_correct / max(1, len(records))) if self.heads_available else None,
                "confidence_mae": (confidence_error / max(1, len(records))) if self.heads_available else None,
                "repair_token_accuracy": token_correct / max(1, token_total),
                "repair_exact_match": exact_correct / max(1, len(records)),
            },
            "limitations": [
                "Candidate/failure labels are mostly trivial.",
                "Exact match is teacher-forced, not generated decoding.",
                "C was not in this checkpoint's training split.",
                "Token accuracy is not compile or behavioral verification.",
            ],
        }


def get_runtime(checkpoint: str | Path = DEFAULT_CHECKPOINT) -> ResidualRuntime:
    global _RUNTIME
    with _LOCK:
        if _RUNTIME is None or Path(checkpoint) != _RUNTIME.checkpoint:
            _RUNTIME = ResidualRuntime(checkpoint)
        return _RUNTIME
