"""Write function-level samples into a sharded NEXUS-BI sample store."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from nexus.data.sample_store import SPLITS, SampleIndexRecord, SampleStoreWriter
from nexus.foundation.vocab import Vocabulary


def _index_record(sample: dict[str, Any], split: str) -> SampleIndexRecord:
    labels = sample.get("labels") or {}
    source_identity = sample.get("source_identity") or {}
    return SampleIndexRecord(
        sample_id=str(sample.get("sample_id") or ""),
        split=split,
        shard="",
        offset=0,
        length=0,
        project_id=str(sample.get("project_id") or labels.get("project_id") or ""),
        source_function_id=source_identity.get("source_function_id") or labels.get("source_function_id"),
        source_file_id=source_identity.get("source_file_id"),
        source_archive_id=source_identity.get("source_archive_id"),
        source_raw_hash=source_identity.get("raw_source_hash"),
        source_normalized_hash=source_identity.get("normalized_source_hash"),
        instruction_hash=sample.get("instruction_hash") or (sample.get("labels") or {}).get("instruction_hash"),
        binary_sha256=str(sample.get("binary_sha256") or ""),
        group_id=source_identity.get("source_file_id"),
        match_group=sample.get("match_group"),
        has_source=bool(sample.get("source_text")),
        source_target_truncated=bool(sample.get("source_target_truncated")),
        purpose_id=int(sample.get("purpose_id", -100)),
        instruction_count=len(sample.get("instr_id") or []),
        source_token_count=len(sample.get("c_ids") or []),
        language=str(labels.get("language") or "c"),
        architecture=str(labels.get("arch") or "unknown"),
        binary_format=str(labels.get("binary_format") or "unknown"),
        compiler=labels.get("compiler"),
        optimization=labels.get("optimization"),
        dataset_id=str(labels.get("dataset_id") or ""),
    )


def assemble_store(
    root: Path,
    samples_by_split: dict[str, list[dict[str, Any]]],
    vocab: Vocabulary,
    *,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    root.mkdir(parents=True, exist_ok=True)
    writer = SampleStoreWriter(root)
    seen: dict[str, str] = {}
    for split in SPLITS:
        samples = sorted(samples_by_split.get(split, []), key=lambda sample: str(sample.get("sample_id") or ""))
        for sample in samples:
            sample_id = str(sample.get("sample_id") or "")
            if not sample_id:
                raise ValueError(f"sample without id in split {split}")
            if sample_id in seen:
                raise ValueError(f"duplicate sample id across splits: {sample_id} ({seen[sample_id]} vs {split})")
            seen[sample_id] = split
            record = _index_record(sample, split)
            writer.append(sample, split, record)
    writer.finalize({"vocab_sha256": vocab.digest(), **metadata})
    return json.loads((root / "manifest.json").read_text(encoding="utf-8"))
