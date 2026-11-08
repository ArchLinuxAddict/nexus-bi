"""Independent validator for the v3 corpus and lazy sample store.

The validator does not call the corpus builder.  It reads the written
manifests, store index, vocabulary and protected-test snapshot and checks the
properties the training gate depends on.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from nexus.data.sample_store import SampleStore
from nexus.foundation.vocab import Vocabulary

SPLITS = ("train", "validation", "test", "extended_test", "quarantine")


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json_hash(payload: Any) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _cross_split(store: SampleStore, field: str, *, skip_trivial_instruction: bool = False, skip_dataset: str | None = None) -> dict[str, Any]:
    owners: dict[str, set[str]] = defaultdict(set)
    examples: dict[str, list[str]] = defaultdict(list)
    for split in ("train", "validation", "test", "extended_test"):
        for sample_id in store.by_split.get(split, []):
            record = store.index[sample_id]
            if skip_dataset and record.dataset_id == skip_dataset:
                continue
            if skip_trivial_instruction and int(record.instruction_count or 0) < 16:
                continue
            value = getattr(record, field, None)
            if not value:
                continue
            owners[str(value)].add(split)
            if len(examples[str(value)]) < 4:
                examples[str(value)].append(f"{split}:{sample_id}")
    conflicts = {key: sorted(value) for key, value in owners.items() if len(value) > 1}
    return {
        "count": len(conflicts),
        "examples": [{"value": key, "splits": value, "samples": examples[key]} for key, value in list(conflicts.items())[:10]],
    }


def validate_corpus_v3(root: str | Path) -> dict[str, Any]:
    base = Path(root).resolve()
    problems: list[str] = []
    checks: list[dict[str, Any]] = []

    def check(name: str, passed: bool, detail: Any) -> None:
        checks.append({"check": name, "status": "PASS" if passed else "FAIL", "detail": detail})
        if not passed:
            problems.append(name)

    manifest_path = base / "manifest.json"
    if not manifest_path.exists():
        return {"schema_version": "nexus.corpus-v3-validation/v1", "root": str(base), "status": "FAIL", "problems": ["manifest.json missing"], "checks": []}
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    check("schema_version", manifest.get("schema_version") == "nexus.corpus-v3/v1", manifest.get("schema_version"))
    check("identity_schema", manifest.get("identity_schema") == "nexus.function-identity/v2", manifest.get("identity_schema"))
    store_root = base / "samples"
    try:
        store = SampleStore.load(store_root)
    except Exception as exc:
        return {
            "schema_version": "nexus.corpus-v3-validation/v1",
            "root": str(base),
            "status": "FAIL",
            "problems": [f"sample store failed to load: {exc}"],
            "checks": checks,
        }
    check("store_schema", store.manifest.get("schema_version") == "nexus.sample-store/v1", store.manifest.get("schema_version"))
    check("store_counts_match_manifest", store.manifest.get("counts") == manifest.get("counts"), {"store": store.manifest.get("counts"), "manifest": manifest.get("counts")})
    check("store_sha256_present", bool(manifest.get("store", {}).get("store_sha256")), manifest.get("store"))
    check("store_manifest_hash_matches", manifest.get("store", {}).get("manifest_sha256") == _file_hash(store_root / "manifest.json"), {"expected": manifest.get("store", {}).get("manifest_sha256"), "actual": _file_hash(store_root / "manifest.json")})

    vocab = Vocabulary.load(base / "vocab.json")
    check("vocab_hash_matches_store", vocab.digest() == store.manifest.get("vocab_sha256"), {"vocab": vocab.digest(), "store": store.manifest.get("vocab_sha256")})
    check("vocab_hash_matches_manifest", vocab.digest() == manifest.get("vocab_sha256"), manifest.get("vocab_sha256"))
    check("vocab_train_only_declared", "train" in str(manifest.get("tokenizer", {}).get("trained_from", "")).lower(), manifest.get("tokenizer"))

    protected_path = base / "protected_test.json"
    protected = json.loads(protected_path.read_text(encoding="utf-8")) if protected_path.exists() else {}
    protected_ids = {str(item.get("sample_id")) for item in protected.get("samples", [])}
    test_path = base / "test.jsonl"
    test_file_hash = _file_hash(test_path) if test_path.exists() else None
    check("protected_test_present", bool(protected_ids), {"protected_manifests": len(protected_ids)})
    check("protected_test_file_hash", protected.get("test_file_sha256") == test_file_hash, {"expected": protected.get("test_file_sha256"), "actual": test_file_hash})
    check("protected_test_declared_hash", manifest.get("protected_test_file_sha256") == test_file_hash, {"expected": manifest.get("protected_test_file_sha256"), "actual": test_file_hash})
    check("protected_test_source_declared", manifest.get("protected_test_source") == "v2_controlled/test.jsonl", manifest.get("protected_test_source"))

    test_manifest_ids: set[str] = set()
    bad_test = []
    for sample_id in store.by_split.get("test", []):
        sample = store.get(sample_id)
        manifest_id = str((sample.get("labels") or {}).get("sample_id") or sample_id.split(":")[0])
        if manifest_id:
            test_manifest_ids.add(manifest_id)
        if manifest_id not in protected_ids:
            bad_test.append({"sample_id": sample_id, "manifest_id": manifest_id})
    check("test_ids_exactly_protected", not bad_test and test_manifest_ids == protected_ids, {"missing": sorted(protected_ids - test_manifest_ids)[:20], "extra": sorted(test_manifest_ids - protected_ids)[:20], "bad": bad_test[:20]})

    # Extended test is disjoint from train/validation by source identity.
    cross = {
        "source_function_id": _cross_split(store, "source_function_id"),
        "source_file_id": _cross_split(store, "source_file_id", skip_dataset="cross_arch_controlled"),
        "source_archive_id": _cross_split(store, "source_archive_id", skip_dataset="cross_arch_controlled"),
        "source_raw_hash": _cross_split(store, "source_raw_hash"),
        "source_normalized_hash": _cross_split(store, "source_normalized_hash"),
        "instruction_hash": _cross_split(store, "instruction_hash", skip_trivial_instruction=True),
        "binary_sha256": _cross_split(store, "binary_sha256", skip_dataset="cross_arch_controlled"),
        "match_group": _cross_split(store, "match_group"),
        "project_id": _cross_split(store, "project_id"),
    }
    check("no_cross_split_source_identities", all(value["count"] == 0 for key, value in cross.items() if key.startswith("source_")), {key: value["count"] for key, value in cross.items() if key.startswith("source_")})
    check("no_cross_split_instruction_hashes", cross["instruction_hash"]["count"] == 0, cross["instruction_hash"])
    check("no_cross_split_binaries", cross["binary_sha256"]["count"] == 0, cross["binary_sha256"])
    check("no_cross_split_match_groups", cross["match_group"]["count"] == 0, cross["match_group"])
    check("no_cross_split_projects", cross["project_id"]["count"] == 0, cross["project_id"])

    # Target integrity: truncated targets must be marked and excluded from the decoder.
    truncated = 0
    truncated_with_decoder = 0
    source_labelled = 0
    unmarked_empty = 0
    for split in ("train", "validation", "extended_test"):
        for sample_id in store.by_split.get(split, [])[:20000]:
            sample = store.get(sample_id)
            has_source = bool(sample.get("source_text"))
            truncated_flag = bool(sample.get("source_target_truncated"))
            has_decoder = bool(sample.get("c_ids") or sample.get("java_ids"))
            if truncated_flag:
                truncated += 1
                if has_decoder:
                    truncated_with_decoder += 1
            if has_source:
                source_labelled += 1
                if not has_decoder and not truncated_flag:
                    unmarked_empty += 1
    check("truncated_targets_are_marked_and_excluded", truncated_with_decoder == 0 and unmarked_empty == 0, {"truncated": truncated, "truncated_with_decoder": truncated_with_decoder, "unmarked_empty": unmarked_empty, "source_labelled_checked": source_labelled})

    # Sample ID uniqueness is already enforced by the store, but recheck across splits.
    seen: dict[str, str] = {}
    duplicates = []
    for split in SPLITS:
        for sample_id in store.by_split.get(split, []):
            if sample_id in seen:
                duplicates.append({"sample_id": sample_id, "first": seen[sample_id], "second": split})
            seen[sample_id] = split
    check("no_duplicate_sample_ids", not duplicates, duplicates[:20])

    # Identity row files are provenance records; their counts must cover the store.
    row_counts = {}
    for split in SPLITS:
        path = base / f"{split}.jsonl"
        row_counts[split] = len(_rows(path))
    check("store_not_larger_than_identity_rows", all(int(manifest.get("counts", {}).get(split, 0)) <= row_counts.get(split, 0) for split in SPLITS), {"store": manifest.get("counts"), "rows": row_counts})

    result = {
        "schema_version": "nexus.corpus-v3-validation/v1",
        "root": str(base),
        "status": "PASS" if not problems else "FAIL",
        "problems": problems,
        "checks": checks,
        "counts": {split: len(store.by_split.get(split, [])) for split in SPLITS},
        "cross_split": {key: value["count"] for key, value in cross.items()},
        "protected_test_samples": len(protected_ids),
        "protected_test_file_sha256": test_file_hash,
        "store_sha256": store.manifest.get("store_sha256"),
        "vocab_sha256": vocab.digest(),
    }
    return result
