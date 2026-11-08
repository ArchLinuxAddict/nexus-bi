"""Independent validator for split v2.

This module deliberately does not call the split builder.  It validates the
written manifests and protected-test snapshot, so a defect in split generation
cannot silently validate itself.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

REQUIRED_FIELDS = (
    "identity_version",
    "function_uid",
    "project_id",
    "binary_sha256",
    "instruction_hash",
    "build_id",
    "split",
)
SPLITS = ("train", "validation", "test", "quarantine")


def _json_hash(payload: Any) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _jaccard(left: list[int], right: list[int]) -> float:
    left_set, right_set = set(left), set(right)
    union = left_set | right_set
    return len(left_set & right_set) / len(union) if union else 0.0


def _near_cross_split(rows: dict[str, list[dict[str, Any]]], threshold: float = 0.85) -> list[dict[str, Any]]:
    """Independent LSH + exact-Jaccard check over cross-split representatives."""
    reps: dict[str, dict[str, Any]] = {}
    for split in ("train", "validation", "test"):
        for row in rows[split]:
            if row.get("source_available") and row.get("normalized_source_hash"):
                reps.setdefault(str(row["normalized_source_hash"]), row)
    buckets: dict[tuple[int, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in sorted(reps.values(), key=lambda item: str(item["function_uid"])):
        shingles = tuple(sorted(int(value) for value in (row.get("source_shingle_hashes") or [])))
        if not shingles:
            continue
        buckets[shingles[:2]].append(row)
    findings = []
    seen: set[tuple[str, str]] = set()
    for bucket in buckets.values():
        if len(bucket) < 2:
            continue
        for index, left in enumerate(bucket):
            for right in bucket[index + 1 :]:
                if left.get("split") == right.get("split"):
                    continue
                pair = tuple(sorted((str(left["function_uid"]), str(right["function_uid"]))))
                if pair in seen:
                    continue
                seen.add(pair)
                score = _jaccard(left.get("source_shingle_hashes") or [], right.get("source_shingle_hashes") or [])
                if score >= threshold:
                    findings.append(
                        {
                            "left": pair[0],
                            "right": pair[1],
                            "left_split": left.get("split"),
                            "right_split": right.get("split"),
                            "jaccard": round(score, 6),
                        }
                    )
    return findings


def validate_split_v2(root: str | Path) -> dict[str, Any]:
    base = Path(root).resolve()
    manifest_path = base / "manifest.json"
    problems: list[str] = []
    if not manifest_path.exists():
        return {"status": "FAIL", "root": str(base), "problems": ["manifest.json missing"], "checks": []}
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    checks: list[dict[str, Any]] = []

    def check(name: str, passed: bool, detail: Any) -> None:
        checks.append({"check": name, "status": "PASS" if passed else "FAIL", "detail": detail})
        if not passed:
            problems.append(name)

    check("schema_version", manifest.get("schema_version") == "nexus.split-v2/v1", manifest.get("schema_version"))
    check("identity_schema", manifest.get("identity_schema") == "nexus.function-identity/v2", manifest.get("identity_schema"))
    rows = {split: _rows(base / f"{split}.jsonl") for split in SPLITS}
    counts = {split: len(rows[split]) for split in SPLITS}
    check("counts_match_manifest", counts == manifest.get("counts"), {"actual": counts, "manifest": manifest.get("counts")})

    seen_uids: dict[str, str] = {}
    duplicate_uids: list[str] = []
    seen_samples: dict[str, str] = {}
    duplicate_samples: list[dict[str, str]] = []
    malformed: list[dict[str, Any]] = []
    for split in SPLITS:
        for row in rows[split]:
            for field in REQUIRED_FIELDS:
                if not row.get(field):
                    malformed.append({"split": split, "function_uid": row.get("function_uid"), "field": field})
            if row.get("split") != split:
                malformed.append({"split": split, "function_uid": row.get("function_uid"), "field": "split_membership"})
            uid = str(row.get("function_uid") or "")
            if uid in seen_uids:
                duplicate_uids.append(uid)
            seen_uids[uid] = split
            sample_id = str(row.get("sample_id") or "")
            if sample_id in seen_samples:
                duplicate_samples.append({"sample_id": sample_id, "first": seen_samples[sample_id], "second": split})
            seen_samples[sample_id] = split
            if row.get("source_available"):
                if not all(row.get(field) for field in ("raw_source_hash", "normalized_source_hash", "normalized_function_hash")):
                    malformed.append({"split": split, "function_uid": uid, "field": "source_hashes"})
    check("no_duplicate_function_uids", not duplicate_uids, duplicate_uids[:20])
    check("no_duplicate_sample_ids_across_splits", not duplicate_samples, duplicate_samples[:20])
    check("identity_fields_complete", not malformed, malformed[:20])

    cross: dict[str, int] = {}
    for field in (
        "source_function_id",
        "raw_source_hash",
        "normalized_source_hash",
        "instruction_hash",
        "binary_sha256",
        "project_id",
        "repository_id",
        "source_file_id",
        "source_archive_id",
    ):
        owners: dict[str, set[str]] = defaultdict(set)
        for split in ("train", "validation", "test"):
            for row in rows[split]:
                value = row.get(field)
                if field == "instruction_hash" and int(row.get("instruction_count") or 0) < 16:
                    continue
                if value:
                    owners[str(value)].add(split)
        cross[field] = sum(1 for value in owners.values() if len(value) > 1)
    check("no_cross_split_identities", all(value == 0 for value in cross.values()), cross)

    near = _near_cross_split(rows)
    check("no_cross_split_near_duplicates", not near, near[:20])

    protected_path = base / "protected_test.json"
    protected = json.loads(protected_path.read_text(encoding="utf-8")) if protected_path.exists() else {}
    protected_ids = {str(item["sample_id"]) for item in protected.get("samples", [])}
    test_manifest_ids = {
        str((row.get("provenance") or {}).get("manifest_sample_id") or "")
        for row in rows["test"]
    }
    test_manifest_ids.discard("")
    check("protected_test_present", bool(protected_ids), {"protected": len(protected_ids)})
    check(
        "test_ids_exactly_protected",
        test_manifest_ids == protected_ids,
        {"missing": sorted(protected_ids - test_manifest_ids)[:20], "extra": sorted(test_manifest_ids - protected_ids)[:20]},
    )
    expected_by_manifest = {str(item["sample_id"]): item for item in protected.get("samples", [])}
    protected_hashes_match = True
    protected_detail: list[dict[str, Any]] = []
    for row in rows["test"]:
        manifest_id = str((row.get("provenance") or {}).get("manifest_sample_id") or "")
        item = expected_by_manifest.get(manifest_id)
        if item is None:
            protected_hashes_match = False
            protected_detail.append({"sample_id": row.get("sample_id"), "issue": "unprotected_test_row"})
            continue
        if item.get("binary_sha256") and row.get("binary_sha256") != item["binary_sha256"]:
            protected_hashes_match = False
            protected_detail.append({"sample_id": row.get("sample_id"), "issue": "binary_hash_changed"})
    check("protected_test_hashes_unchanged", protected_hashes_match, protected_detail[:20])

    current_test_file_hash = _file_hash(base / "test.jsonl") if (base / "test.jsonl").exists() else None
    check(
        "protected_test_manifest_hash",
        protected.get("test_file_sha256") == current_test_file_hash,
        {"expected": protected.get("test_file_sha256"), "actual": current_test_file_hash},
    )

    assignments = {split: [row.get("function_uid") for row in rows[split]] for split in SPLITS}
    check("assignments_hash", manifest.get("assignments_sha256") == _json_hash(assignments), manifest.get("assignments_sha256"))

    result = {
        "schema_version": "nexus.split-v2-validation/v1",
        "root": str(base),
        "status": "PASS" if not problems else "FAIL",
        "problems": problems,
        "checks": checks,
        "counts": counts,
        "cross_split": cross,
        "near_duplicate_pairs": len(near),
        "protected_test_samples": len(protected_ids),
        "protected_test_file_sha256": current_test_file_hash,
    }
    return result


def validate_legacy_v1_against_identities(
    v1_root: str | Path,
    identities: list[Any],
) -> dict[str, Any]:
    """Reject the historical v1 split at function granularity.

    This is intentionally separate from split v2 generation.  It maps each
    function identity back to the v1 manifest split and reports exact
    source-function leakage.
    """
    base = Path(v1_root).resolve()
    split_by_manifest: dict[str, str] = {}
    for split in ("train", "validation", "test"):
        for row in _rows(base / f"{split}.jsonl"):
            split_by_manifest[str(row.get("sample_id"))] = split
    owners: dict[str, set[str]] = defaultdict(set)
    counts: dict[str, dict[str, int]] = defaultdict(lambda: {"train": 0, "validation": 0, "test": 0})
    for identity in identities:
        manifest_id = str((identity.provenance or {}).get("manifest_sample_id") or "")
        split = split_by_manifest.get(manifest_id)
        if split is None or not identity.raw_source_hash:
            continue
        owners[identity.raw_source_hash].add(split)
        counts[identity.raw_source_hash][split] += 1
    cross = {key: sorted(value) for key, value in owners.items() if len(value) > 1}
    train_validation_pairs = sum(value["train"] * value["validation"] for value in counts.values())
    test_involved = any("test" in value for value in cross.values())
    problems = []
    if cross:
        problems.append("legacy_v1_has_cross_split_source_functions")
    return {
        "schema_version": "nexus.legacy-v1-function-audit/v1",
        "root": str(base),
        "status": "FAIL" if cross else "PASS",
        "problems": problems,
        "manifests_by_split": {
            split: sum(1 for value in split_by_manifest.values() if value == split)
            for split in ("train", "validation", "test")
        },
        "cross_split_source_functions": len(cross),
        "train_validation_sample_pairs": train_validation_pairs,
        "test_involved": test_involved,
        "leaked_hashes": sorted(cross)[:20],
    }
