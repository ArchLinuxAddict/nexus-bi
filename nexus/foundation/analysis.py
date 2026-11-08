"""Foundation-model inference over a binary without executing it.

Deterministic analysis supplies function boundaries and structured evidence; the
model supplies task predictions. Source/name/DWARF evidence is not an input.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from nexus.foundation.checkpoint import load_foundation_checkpoint
from nexus.foundation.representation import (
    PURPOSE_CLASSES,
    RepresentationOptions,
    collate_samples,
    encode_function_sample,
)
from nexus.program.bundle import analyze_binary
from nexus.program.record import build_program_record


@torch.no_grad()
def analyze_with_foundation(
    binary_path: str | Path,
    checkpoint_path: str | Path,
    max_functions: int = 8,
    max_instructions: int | None = None,
) -> dict[str, Any]:
    loaded = load_foundation_checkpoint(checkpoint_path, load_training_state=False)
    if loaded.vocab is None:
        raise ValueError("checkpoint does not contain a vocabulary")
    bundle = analyze_binary(binary_path)
    record = build_program_record(bundle)
    functions = [
        function
        for function in record.get("functions", [])
        if function.get("confidence", 0) >= 0.5 and len(function.get("blocks", [])) > 0
    ][: max_functions]
    samples = []
    for function in functions:
        sample = encode_function_sample(
            record,
            function,
            loaded.vocab,
            loaded.config,
            identity={
                "sample_id": f"analysis:{function.get('function_id')}",
                "function_id": function.get("function_id"),
                "project_id": "analysis",
                "dataset_id": "analysis",
                "split": "inference",
                "binary_sha256": record.get("program", {}).get("sha256", ""),
                "source_family": "",
                "match_group": "",
                "binary_path": str(binary_path),
            },
            options=RepresentationOptions(
                include_source_target=False,
                include_purpose_target=False,
                max_instructions=max_instructions,
            ),
        )
        if sample is not None:
            samples.append(sample)
    if not samples:
        return {
            "status": "no_functions",
            "binary_sha256": record.get("program", {}).get("sha256"),
            "checkpoint": {"path": str(checkpoint_path), "sha256": loaded.sha256},
            "functions": [],
            "limitations": ["No encodable functions were found; no model prediction is claimed."],
        }
    batch = collate_samples(samples, loaded.config)
    outputs = loaded.model(batch)
    predictions = []
    for index, sample in enumerate(samples):
        logits = outputs["purpose_logits"][index]
        probabilities = torch.softmax(logits, dim=-1)
        purpose_id = int(logits.argmax())
        ir_logits = outputs["ir_logits"][index, : len(sample["instr_ir"])]
        ir_predictions = ir_logits.argmax(dim=-1).tolist()
        cfg_probabilities = torch.sigmoid(outputs["cfg_logits"][index]).detach().cpu()
        predictions.append(
            {
                "function_id": sample["function_id"],
                "instructions": len(sample["instr_id"]),
                "blocks": len(sample["block_mask"]),
                "predicted_purpose": PURPOSE_CLASSES[purpose_id] if 0 <= purpose_id < len(PURPOSE_CLASSES) else "unknown",
                "purpose_probability": float(probabilities[purpose_id]),
                "confidence": float(outputs["confidence"][index]),
                "predicted_ir_classes": ir_predictions,
                "cfg_predicted_edge_rate": float((cfg_probabilities > 0.5).float().mean()),
                "retrieval_embedding_norm": float(outputs["function_embedding"][index].norm()),
                "input_provenance": sample["input_provenance"],
            }
        )
    return {
        "status": "completed",
        "binary_sha256": record.get("program", {}).get("sha256"),
        "checkpoint": {
            "path": str(checkpoint_path),
            "sha256": loaded.sha256,
            "step": loaded.metadata.get("step"),
            "status": loaded.metadata.get("status"),
            "parameter_count": loaded.model.count_parameters(),
        },
        "functions": predictions,
        "deterministic_function_count": len(record.get("functions", [])),
        "encoded_function_count": len(samples),
        "limitations": [
            "Predictions are model outputs, not verified semantic facts.",
            "Function boundaries and CFG evidence come from deterministic analysis.",
            "No behavioral execution or equivalence proof is performed.",
        ],
    }
