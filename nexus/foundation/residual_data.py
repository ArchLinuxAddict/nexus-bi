"""Collation for tool-output residual training records."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from nexus.foundation.config import FoundationConfig
from nexus.foundation.representation import collate_samples
from nexus.foundation.vocab import PAD, Vocabulary

TOOL_IDS = {"none": 0, "jadx": 1, "cfr": 2, "ghidra_headless": 3, "llvm_objdump": 4, "neural": 5, "compiler": 6, "jni_table": 7, "jni_register": 8}
LANGUAGE_IDS = {"c": 0, "java": 1, "jni": 2, "unknown": 3}
STATUS_IDS = {"unknown": 0, "completed": 1, "failed": 2, "timeout": 3, "missing": 4, "no_functions": 5}
COMPILE_IDS = {"unknown": 0, "not_attempted": 1, "passed": 2, "failed": 3}


def load_residual_records(path: str | Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def _ids(vocab: Vocabulary, text: str, limit: int) -> tuple[list[int], list[int]]:
    ids = vocab.encode_text(text or "", limit)
    return ids, [1] * len(ids)


def collate_residual(records: list[dict[str, Any]], vocab: Vocabulary, config: FoundationConfig, include_program: bool = True) -> dict[str, Any]:
    if not records:
        raise ValueError("cannot collate an empty residual batch")
    tool_text = [_ids(vocab, record.get("tool_text", ""), config.max_text_tokens) for record in records]
    target_text = [_ids(vocab, record.get("target_text", ""), config.max_text_tokens) for record in records]
    candidate_text = [
        _ids(vocab, record.get("candidate_text", ""), min(config.max_text_tokens, 512))
        if record.get("candidate_text")
        else ([PAD], [0])
        for record in records
    ]
    tool_length = max(1, max(len(item[0]) for item in tool_text))
    target_length = max(1, max(len(item[0]) for item in target_text))
    candidate_length = max(1, max(len(item[0]) for item in candidate_text))
    batch: dict[str, Any] = {
        "tool_ids": torch.tensor([item[0] + [PAD] * (tool_length - len(item[0])) for item in tool_text], dtype=torch.long),
        "tool_mask": torch.tensor([item[1] + [0] * (tool_length - len(item[1])) for item in tool_text], dtype=torch.bool),
        "candidate_ids": torch.tensor([item[0] + [PAD] * (candidate_length - len(item[0])) for item in candidate_text], dtype=torch.long),
        "candidate_mask": torch.tensor([item[1] + [0] * (candidate_length - len(item[1])) for item in candidate_text], dtype=torch.bool),
        "repair_input_ids": torch.tensor([item[0] + [PAD] * (target_length - len(item[0])) for item in target_text], dtype=torch.long),
        "repair_mask": torch.tensor([item[1] + [0] * (target_length - len(item[1])) for item in target_text], dtype=torch.bool),
        "tool_id": torch.tensor([TOOL_IDS.get(str(record.get("tool", "none")), 0) for record in records], dtype=torch.long),
        "language_id": torch.tensor([LANGUAGE_IDS.get(str(record.get("language", "unknown")), 3) for record in records], dtype=torch.long),
        "status_id": torch.tensor([STATUS_IDS.get(str(record.get("tool_status", "unknown")), 0) for record in records], dtype=torch.long),
        "compile_id": torch.tensor([COMPILE_IDS.get(str(record.get("compile_status", "unknown")), 0) for record in records], dtype=torch.long),
        "warning_count": torch.tensor([int(record.get("warning_count") or 0) for record in records], dtype=torch.long),
        "cached": torch.tensor([bool(record.get("cached")) for record in records], dtype=torch.bool),
        "failure_label": torch.tensor([int(record.get("failure_label") or 0) for record in records], dtype=torch.long),
        "candidate_label": torch.tensor([float(record.get("candidate_label") or 0.0) for record in records], dtype=torch.float32),
        "confidence_label": torch.tensor([float(record.get("confidence_label") or 0.0) for record in records], dtype=torch.float32),
    }
    target_ids = batch["repair_input_ids"]
    tool_ids = batch["tool_ids"]
    tool_mask = batch["tool_mask"]
    copy_labels = torch.full_like(target_ids, -100)
    for row in range(target_ids.size(0)):
        positions = tool_mask[row].nonzero(as_tuple=False).squeeze(1)
        if positions.numel() == 0:
            continue
        source = tool_ids[row, positions]
        for column in range(target_ids.size(1)):
            token = int(target_ids[row, column])
            if token == PAD:
                continue
            matches = (source == token).nonzero(as_tuple=False)
            if matches.numel():
                copy_labels[row, column] = int(positions[matches[0, 0]])
    batch["copy_labels"] = copy_labels
    if include_program and any(record.get("program") for record in records):
        program_records = [record.get("program") for record in records if record.get("program")]
        if len(program_records) == len(records):
            batch["program"] = collate_samples(program_records, config)
    return batch


def shift_for_teacher(ids: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    input_ids = ids[:, :-1].contiguous()
    labels = ids[:, 1:].contiguous()
    input_mask = mask[:, :-1].contiguous()
    label_mask = mask[:, 1:].contiguous()
    return input_ids, labels, input_mask, label_mask
