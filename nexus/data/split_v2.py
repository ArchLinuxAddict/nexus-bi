"""Function-level, lineage-aware split v2.

The split is generated from identity groups, never from sample counts.  A
protected test sample always wins a group; non-test members of that group are
quarantined rather than moved into test.  Train/validation collisions are moved
to validation as a whole group, so no function identity crosses the boundary.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Iterable

from nexus.data.identity import FunctionIdentityV2

SPLIT_NAMES = ("train", "validation", "test", "quarantine")
NEAR_DUPLICATE_THRESHOLD = 0.85


class UnionFind:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}
        self.rank: dict[str, int] = {}

    def add(self, value: str) -> None:
        self.parent.setdefault(value, value)
        self.rank.setdefault(value, 0)

    def find(self, value: str) -> str:
        self.add(value)
        root = value
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[value] != value:
            next_value = self.parent[value]
            self.parent[value] = root
            value = next_value
        return root

    def union(self, left: str, right: str) -> bool:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return False
        if self.rank[left_root] < self.rank[right_root]:
            left_root, right_root = right_root, left_root
        self.parent[right_root] = left_root
        if self.rank[left_root] == self.rank[right_root]:
            self.rank[left_root] += 1
        return True


def _digest(payload: Any) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _jaccard(left: tuple[int, ...], right: tuple[int, ...]) -> float:
    if not left or not right:
        return 0.0
    left_set, right_set = set(left), set(right)
    intersection = len(left_set & right_set)
    union = len(left_set | right_set)
    return intersection / union if union else 0.0


def _minhash(shingles: tuple[int, ...], size: int = 12) -> tuple[int, ...]:
    if not shingles:
        return tuple([0] * size)
    values = []
    for seed in range(size):
        values.append(min(((value ^ (seed * 0x9E3779B1)) & 0xFFFFFFFF) for value in shingles))
    return tuple(values)


def _near_duplicate_pairs(identities: list[FunctionIdentityV2], threshold: float) -> list[tuple[str, str, float]]:
    """LSH candidates followed by exact Jaccard verification on shingle sets."""
    representatives: dict[str, FunctionIdentityV2] = {}
    for identity in identities:
        if not identity.source_available or not identity.source_shingle_hashes:
            continue
        key = identity.normalized_source_hash or identity.function_uid
        representatives.setdefault(key, identity)
    ordered = sorted(representatives.values(), key=lambda item: item.function_uid)
    signatures = {item.function_uid: _minhash(item.source_shingle_hashes) for item in ordered}
    buckets: dict[tuple[int, int], list[FunctionIdentityV2]] = defaultdict(list)
    bands, rows = 4, 3
    for item in ordered:
        signature = signatures[item.function_uid]
        for band in range(bands):
            key = (band, hash(signature[band * rows : (band + 1) * rows]))
            buckets[key].append(item)
    pairs: set[tuple[str, str]] = set()
    results: list[tuple[str, str, float]] = []
    for bucket in buckets.values():
        if len(bucket) < 2:
            continue
        for left_index in range(len(bucket)):
            for right_index in range(left_index + 1, len(bucket)):
                left = bucket[left_index]
                right = bucket[right_index]
                pair = tuple(sorted((left.function_uid, right.function_uid)))
                if pair in pairs:
                    continue
                pairs.add(pair)
                score = _jaccard(left.source_shingle_hashes, right.source_shingle_hashes)
                if score >= threshold:
                    results.append((pair[0], pair[1], round(score, 6)))
    return sorted(results)


@dataclass
class SplitV2Result:
    train: list[dict[str, Any]]
    validation: list[dict[str, Any]]
    test: list[dict[str, Any]]
    quarantine: list[dict[str, Any]]
    groups: list[dict[str, Any]]
    report: dict[str, Any]

    def rows(self) -> dict[str, list[dict[str, Any]]]:
        return {"train": self.train, "validation": self.validation, "test": self.test, "quarantine": self.quarantine}


def _identity_is_valid(identity: FunctionIdentityV2) -> tuple[bool, list[str]]:
    problems: list[str] = []
    if not identity.function_uid:
        problems.append("missing_function_uid")
    if not identity.binary_sha256:
        problems.append("missing_binary_sha256")
    if not identity.project_id or identity.project_id == "unknown":
        problems.append("missing_project_id")
    if identity.source_available and not identity.raw_source_hash:
        problems.append("source_available_without_hash")
    if identity.identity_version != "nexus.function-identity/v2":
        problems.append("unsupported_identity_version")
    return not problems, problems


def _group_split(members: list[FunctionIdentityV2], new_group_split: str) -> str:
    if any(member.split == "test" or member.protected_test for member in members):
        return "test"
    if any(member.split == "validation" for member in members):
        return "validation"
    if any(member.split == "train" for member in members):
        return "train"
    return new_group_split


def _new_group_split(members: list[FunctionIdentityV2]) -> str:
    key = _digest(sorted({member.project_id for member in members}))
    return "validation" if int(key[:8], 16) % 7 == 0 else "train"


def build_split_v2(
    identities: Iterable[FunctionIdentityV2],
    *,
    protected_test_ids: set[str] | None = None,
    near_duplicate_threshold: float = NEAR_DUPLICATE_THRESHOLD,
    allow_new_test: bool = False,
) -> SplitV2Result:
    values = list(identities)
    protected = set(protected_test_ids or set())
    union = UnionFind()
    by_uid: dict[str, FunctionIdentityV2] = {}
    invalid: list[dict[str, Any]] = []
    invalid_uids: set[str] = set()
    for identity in values:
        if identity.function_uid in by_uid:
            raise ValueError(f"duplicate function_uid: {identity.function_uid}")
        by_uid[identity.function_uid] = identity
        union.add(identity.function_uid)
        valid, problems = _identity_is_valid(identity)
        if not valid:
            invalid.append({"function_uid": identity.function_uid, "sample_id": identity.sample_id, "reasons": problems})
            invalid_uids.add(identity.function_uid)
            continue
        group_keys = [
            ("source_archive", identity.source_archive_id),
            ("source_file", identity.source_file_id),
            ("source_function", identity.source_function_id),
            ("source_raw", identity.raw_source_hash),
            ("source_normalized", identity.normalized_source_hash),
            ("instruction", identity.instruction_hash if identity.instruction_count >= 16 else None),
            ("binary", identity.binary_sha256),
        ]
        for kind, value in group_keys:
            if value:
                union.union(identity.function_uid, f"{kind}:{value}")
    near_pairs = _near_duplicate_pairs(values, near_duplicate_threshold)
    for left, right, _score in near_pairs:
        union.union(left, right)

    project_members: dict[str, list[FunctionIdentityV2]] = defaultdict(list)
    for identity in values:
        if identity.function_uid not in invalid_uids:
            project_members[identity.repository_id or identity.project_id].append(identity)
    project_split = {
        key: _group_split(members, _new_group_split(members))
        for key, members in project_members.items()
    }

    grouped: dict[str, list[FunctionIdentityV2]] = defaultdict(list)
    for identity in values:
        if identity.function_uid not in invalid_uids:
            grouped[union.find(identity.function_uid)].append(identity)

    rows: dict[str, list[dict[str, Any]]] = {name: [] for name in SPLIT_NAMES}
    groups: list[dict[str, Any]] = []
    priority = {"test": 3, "validation": 2, "train": 1}
    for root, members in sorted(grouped.items(), key=lambda item: item[0]):
        members = sorted(members, key=lambda item: item.function_uid)
        candidate_splits = {
            project_split.get(member.repository_id or member.project_id, "train") for member in members
        }
        split = max(candidate_splits, key=lambda value: priority.get(value, 0))
        if split == "test" and not allow_new_test:
            test_members = [member for member in members if member.protected_test or member.split == "test"]
            if not test_members:
                split = "validation" if "validation" in candidate_splits else "train"
        group_id = "grp_" + _digest(sorted(member.function_uid for member in members))[:24]
        group_record = {
            "group_id": group_id,
            "split": split,
            "member_count": len(members),
            "projects": sorted({member.project_id for member in members}),
            "source_functions": len({member.source_function_id for member in members if member.source_function_id}),
            "source_files": len({member.source_file_id for member in members if member.source_file_id}),
            "project_splits": sorted(candidate_splits),
            "reasons": [],
        }
        for member in members:
            member_project_split = project_split.get(member.repository_id or member.project_id, "train")
            if split == "test" and not (member.protected_test or member.split == "test"):
                row = member.to_dict()
                row["split"] = "quarantine"
                row["quarantine_reason"] = "non_test_member_of_protected_test_group"
                row["group_id"] = group_id
                rows["quarantine"].append(row)
                group_record["reasons"].append("test_group_quarantine")
            elif member_project_split != split:
                row = member.to_dict()
                row["split"] = "quarantine"
                row["quarantine_reason"] = "function_group_spans_project_splits"
                row["group_id"] = group_id
                rows["quarantine"].append(row)
                group_record["reasons"].append("project_split_quarantine")
            else:
                row = member.to_dict()
                row["source_split"] = member.split
                row["split"] = split
                row["group_id"] = group_id
                rows[split].append(row)
        groups.append(group_record)
    for split in rows:
        rows[split].sort(key=lambda row: row["function_uid"])

    cross = {
        "exact_source_functions": _cross_split_count(rows, "source_function_id"),
        "raw_source_hashes": _cross_split_count(rows, "raw_source_hash"),
        "normalized_source_hashes": _cross_split_count(rows, "normalized_source_hash"),
        "instruction_hashes": _cross_split_count(
            rows, "instruction_hash", predicate=lambda row: int(row.get("instruction_count") or 0) >= 16
        ),
        "trivial_instruction_hashes": _cross_split_count(rows, "instruction_hash"),
        "binary_hashes": _cross_split_count(rows, "binary_sha256"),
        "projects": _cross_split_count(rows, "project_id"),
    }
    report = {
        "schema_version": "nexus.split-v2-report/v1",
        "counts": {split: len(rows[split]) for split in SPLIT_NAMES},
        "unique": {
            "function_uids": len({row["function_uid"] for split in rows for row in rows[split]}),
            "sample_ids": len({row["sample_id"] for split in rows for row in rows[split]}),
            "projects": len({row["project_id"] for split in rows for row in rows[split]}),
            "source_functions": len(
                {row["source_function_id"] for split in rows for row in rows[split] if row.get("source_function_id")}
            ),
            "binary_hashes": len({row["binary_sha256"] for split in rows for row in rows[split] if row.get("binary_sha256")}),
        },
        "cross_split": cross,
        "near_duplicate_threshold": near_duplicate_threshold,
        "near_duplicate_pairs": len(near_pairs),
        "groups": len(groups),
        "protected_test_samples": len(protected),
        "protected_test_present": sum(1 for row in rows["test"] if row.get("protected_test")),
        "test_contains_only_protected": all(row.get("protected_test") or row.get("source_split") == "test" for row in rows["test"]),
        "quarantine_reasons": _quarantine_reason_counts(rows["quarantine"]),
        "assignments_sha256": _digest({split: [row["function_uid"] for row in rows[split]] for split in SPLIT_NAMES}),
    }
    return SplitV2Result(rows["train"], rows["validation"], rows["test"], rows["quarantine"], groups, report)


def _cross_split_count(
    rows: dict[str, list[dict[str, Any]]],
    field: str,
    predicate: Any | None = None,
) -> int:
    owners: dict[str, set[str]] = defaultdict(set)
    for split in ("train", "validation", "test"):
        for row in rows[split]:
            if predicate is not None and not predicate(row):
                continue
            value = row.get(field)
            if value:
                owners[str(value)].add(split)
    return sum(1 for splits in owners.values() if len(splits) > 1)


def _quarantine_reason_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for row in rows:
        counts[str(row.get("quarantine_reason") or "unknown")] += 1
    return dict(sorted(counts.items()))
