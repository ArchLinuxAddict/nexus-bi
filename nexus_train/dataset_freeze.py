"""Create and independently validate the test-protective NEXUS-BI source split.

Original dataset manifests are immutable inputs. The freeze command writes a
new immutable snapshot; it never moves or edits original samples.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable

from nexus.core.hashing import sha256_file
from nexus.mission.policy import load_registry

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FREEZE_ROOT = ROOT / "artifacts" / "training_runtime" / "frozen_dataset" / "v4"
FREEZE_VERSION = "nexus-frozen-dataset-v1"
TOKEN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|0[xX][0-9a-fA-F]+|\d+|[^\s]")
SPLITS = ("train", "validation", "test", "quarantine")
DEFAULT_DATASET_IDS = (
    "9c4fdc0d2744c4384ffb8fa0a626c093e77e9861a61bdeac6edab97677937c6e",
    "418ff61423fdcf659d16bfae2f3ada838bc1faf48e6ae609209a84979b87b210",
    "b152c8186340a44de4f7b4e479ddaba35b11dc60e2a11f0658daafb158b473c9",
)


def _sha_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha_json(payload: Any) -> str:
    return _sha_bytes(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8"))


def _tokens(text: str) -> list[str]:
    return TOKEN.findall(text)


def _shingles(text: str, width: int = 5) -> set[str]:
    tokens = _tokens(text)
    return {" ".join(tokens[index : index + width]) for index in range(max(0, len(tokens) - width + 1))}


def _normalized_hash(text: str) -> str:
    return _sha_bytes(" ".join(_tokens(text)).encode("utf-8"))


def _manifest_rows(dataset_ids: Iterable[str] | None = None) -> tuple[list[dict[str, Any]], dict[str, Path]]:
    registry = load_registry()
    allowed = set(dataset_ids or DEFAULT_DATASET_IDS)
    rows: list[dict[str, Any]] = []
    roots: dict[str, Path] = {}
    for entry in registry["datasets"]:
        if entry.get("dataset_id") not in allowed:
            continue
        path = Path(entry["path"])
        if not path.is_absolute():
            path = ROOT / path
        path = path.resolve()
        manifests_path = path / "manifests.jsonl"
        if not manifests_path.exists():
            continue
        dataset_id = entry["dataset_id"]
        roots[dataset_id] = path
        for line in manifests_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            manifest = json.loads(line)
            manifest["_dataset_id"] = dataset_id
            manifest["_dataset_root"] = str(path)
            manifest["_project_key"] = f"{dataset_id}:{manifest.get('project_id')}"
            rows.append(manifest)
    if not rows:
        raise RuntimeError("no registered manifests were found")
    return rows, roots


def _source_info(row: dict[str, Any]) -> dict[str, Any]:
    root = Path(row["_dataset_root"])
    files = [artifact for artifact in row.get("artifacts", []) if str(artifact.get("path", "")).endswith(".c")]
    hashes = []
    normalized = []
    shingles: set[str] = set()
    for artifact in files:
        path = root / artifact["path"]
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        hashes.append(artifact["sha256"])
        normalized.append(_normalized_hash(text))
        shingles |= _shingles(text)
    return {
        "source_archive_sha256": (row.get("provenance") or {}).get("source_archive_sha256"),
        "source_file_sha256": hashes,
        "normalized_source_sha256": normalized,
        "shingles": shingles,
    }


class _UnionFind:
    def __init__(self, keys: Iterable[str]):
        self.parent = {key: key for key in keys}

    def find(self, key: str) -> str:
        while self.parent[key] != key:
            self.parent[key] = self.parent[self.parent[key]]
            key = self.parent[key]
        return key

    def union(self, left: str, right: str) -> None:
        a, b = self.find(left), self.find(right)
        if a != b:
            self.parent[b] = a


def build_proposed_split(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Recompute source identities and the test-protective assignment."""
    infos = {row["_project_key"]: _source_info(row) for row in rows}
    uf = _UnionFind(infos)
    by_archive: dict[str, list[str]] = defaultdict(list)
    by_file: dict[str, list[str]] = defaultdict(list)
    by_normalized: dict[str, list[str]] = defaultdict(list)
    for key, info in infos.items():
        archive = info["source_archive_sha256"]
        if archive:
            by_archive[archive].append(key)
        for value in info["source_file_sha256"]:
            by_file[value].append(key)
        for value in info["normalized_source_sha256"]:
            by_normalized[value].append(key)
    for grouping in (by_archive, by_file, by_normalized):
        for members in grouping.values():
            for member in members[1:]:
                uf.union(members[0], member)
    # Near-duplicate 5-gram shingle Jaccard >= 0.90.
    inverted: dict[str, set[str]] = defaultdict(set)
    for key, info in infos.items():
        for shingle in info["shingles"]:
            inverted[shingle].add(key)
    candidates: set[tuple[str, str]] = set()
    for keys in inverted.values():
        ordered = sorted(keys)
        for index, left in enumerate(ordered):
            for right in ordered[index + 1 :]:
                candidates.add((left, right))
    near_pairs = 0
    for left, right in sorted(candidates):
        if uf.find(left) == uf.find(right):
            continue
        left_shingles, right_shingles = infos[left]["shingles"], infos[right]["shingles"]
        score = len(left_shingles & right_shingles) / max(1, len(left_shingles | right_shingles))
        if score >= 0.90:
            uf.union(left, right)
            near_pairs += 1
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[uf.find(row["_project_key"])].append(row)
    assignments: dict[str, str] = {}
    quarantine_reasons: dict[str, str] = {}
    group_rows: dict[str, list[str]] = {}
    contaminated_groups = 0
    for group_key, members in groups.items():
        splits = sorted({str(member.get("split")) for member in members})
        project_keys = sorted({member["_project_key"] for member in members})
        group_id = _sha_json({"projects": project_keys, "source_archive_hashes": sorted({infos[k]["source_archive_sha256"] for k in project_keys if infos[k]["source_archive_sha256"]}), "splits": splits})
        group_rows[group_id] = sorted(str(member.get("sample_id")) for member in members)
        if len(splits) == 1:
            for member in members:
                assignments[str(member["sample_id"])] = str(member["split"])
        else:
            contaminated_groups += 1
            if "test" in splits:
                for member in members:
                    if member.get("split") == "test":
                        assignments[str(member["sample_id"])] = "test"
                    else:
                        assignments[str(member["sample_id"])] = "quarantine"
                        quarantine_reasons[str(member.get("sample_id"))] = "source_clone_group_contains_protected_test"
            else:
                for member in members:
                    assignments[str(member.get("sample_id"))] = "quarantine"
                    quarantine_reasons[str(member.get("sample_id"))] = "source_clone_group_crosses_train_validation"
    counts = Counter(assignments.values())
    return {
        "assignments": assignments,
        "quarantine_reasons": quarantine_reasons,
        "groups": group_rows,
        "counts": dict(counts),
        "contaminated_groups": contaminated_groups,
        "near_duplicate_union_pairs": near_pairs,
        "source_identity": {
            "archive_groups": len(by_archive),
            "file_groups": len(by_file),
            "normalized_groups": len(by_normalized),
        },
    }


def _row_for_freeze(row: dict[str, Any], frozen_split: str, reason: str | None) -> dict[str, Any]:
    info = _source_info(row)
    project_key = row["_project_key"]
    group_key = _sha_json({"project": project_key, "archive": info["source_archive_sha256"], "files": sorted(info["source_file_sha256"])})
    return {
        "sample_id": str(row.get("sample_id")),
        "dataset_id": row["_dataset_id"],
        "dataset_root": str(Path(row["_dataset_root"]).relative_to(ROOT)),
        "project_id": row.get("project_id"),
        "source_family": row.get("source_family"),
        "original_split": row.get("split"),
        "frozen_split": frozen_split,
        "binary_sha256": (row.get("binary") or {}).get("sha256"),
        "source_archive_sha256": info["source_archive_sha256"],
        "source_file_sha256": sorted(info["source_file_sha256"]),
        "normalized_source_sha256": sorted(info["normalized_source_sha256"]),
        "compiler": (row.get("build") or {}).get("compiler"),
        "optimization": (row.get("build") or {}).get("optimization"),
        "debug_info": (row.get("build") or {}).get("debug_info"),
        "arch": (row.get("build") or {}).get("arch"),
        "license": (row.get("license") or {}).get("spdx"),
        "provenance_repository": (row.get("provenance") or {}).get("repository"),
        "source_group_id": group_key,
        "quarantine_reason": reason,
    }


def freeze_dataset(output: str | Path = DEFAULT_FREEZE_ROOT, *, verify_artifacts: bool = True) -> dict[str, Any]:
    target = Path(output).resolve()
    if target.exists() and any(target.glob("*.jsonl")):
        raise FileExistsError(f"frozen dataset already exists: {target}; refusing to overwrite immutable data")
    rows, roots = _manifest_rows()
    if verify_artifacts:
        for row in rows:
            root = Path(row["_dataset_root"])
            for artifact in row.get("artifacts", []):
                path = root / artifact["path"]
                if not path.exists() or sha256_file(path) != artifact["sha256"]:
                    raise ValueError(f"artifact integrity failure: {path}")
    proposed = build_proposed_split(rows)
    assignments = proposed["assignments"]
    target.mkdir(parents=True, exist_ok=True)
    files = {}
    for split in SPLITS:
        entries = []
        for row in rows:
            if assignments.get(str(row.get("sample_id"))) != split:
                continue
            entries.append(_row_for_freeze(row, split, proposed["quarantine_reasons"].get(str(row.get("sample_id")))))
        entries.sort(key=lambda item: item["sample_id"])
        path = target / f"{split}.jsonl"
        path.write_text("".join(json.dumps(entry, sort_keys=True) + "\n" for entry in entries), encoding="utf-8")
        files[split] = {"path": path.name, "sha256": sha256_file(path), "count": len(entries)}
    # Vocabulary is data, not model weights; extract the v5 vocabulary only, never weights.
    import torch

    checkpoint_path = ROOT / "artifacts" / "final_training_package" / "runs" / "nexus-bi-unified-first-run" / "final.pt"
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    vocab_payload = dict(payload["vocab"])
    vocab_payload["vocab_sha256"] = payload.get("vocab_sha256") or _sha_json({key: value for key, value in vocab_payload.items() if key != "vocab_sha256"})
    vocab_path = target / "vocab.json"
    vocab_path.write_text(json.dumps(vocab_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    files["vocab"] = {"path": vocab_path.name, "sha256": sha256_file(vocab_path), "vocab_sha256": vocab_payload.get("vocab_sha256")}
    source_manifest_hashes = {dataset_id: sha256_file(root / "manifests.jsonl") for dataset_id, root in roots.items()}
    manifest_payload = {
        "schema_version": FREEZE_VERSION,
        "created_at": datetime.now(UTC).isoformat(),
        "rule": "source identity = exact archive/file/normalized hash union + 5-gram shingle Jaccard >= 0.90; test-protective retention",
        "source_manifests": source_manifest_hashes,
        "dataset_roots": {dataset_id: str(root.relative_to(ROOT)) for dataset_id, root in roots.items()},
        "counts": {split: files[split]["count"] for split in SPLITS},
        "files": files,
        "contaminated_groups": proposed["contaminated_groups"],
        "near_duplicate_union_pairs": proposed["near_duplicate_union_pairs"],
        "source_identity": proposed["source_identity"],
        "vocab_sha256": vocab_payload.get("vocab_sha256"),
        "test_retention": "all original test samples are retained; train/validation members of source groups containing test are quarantined",
        "quarantine_reasons": dict(Counter(proposed["quarantine_reasons"].values())),
    }
    payload_for_id = dict(manifest_payload)
    frozen_id = _sha_json(payload_for_id)
    manifest_payload["frozen_id"] = frozen_id
    manifest_path = target / "manifest.json"
    manifest_path.write_text(json.dumps(manifest_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    # Immutable snapshot: read-only files.
    for path in target.iterdir():
        if path.is_file():
            os.chmod(path, 0o444)
    return {
        "frozen_dir": str(target),
        "frozen_id": frozen_id,
        "manifest_sha256": sha256_file(manifest_path),
        "counts": manifest_payload["counts"],
        "files": files,
        "proposed_counts": proposed["counts"],
        "contaminated_groups": proposed["contaminated_groups"],
    }


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def validate_frozen_dataset(root: str | Path = DEFAULT_FREEZE_ROOT, *, recompute: bool = True) -> dict[str, Any]:
    base = Path(root).resolve()
    manifest_path = base / "manifest.json"
    problems: list[str] = []
    if not manifest_path.exists():
        return {"status": "FAIL", "problems": ["missing manifest.json"], "root": str(base)}
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != FREEZE_VERSION:
        problems.append("unknown freeze schema version")
    # Original manifests must still match the hashes captured at freeze time.
    for dataset_id, expected in (manifest.get("source_manifests") or {}).items():
        root_rel = (manifest.get("dataset_roots") or {}).get(dataset_id)
        if not root_rel:
            problems.append(f"missing dataset root for {dataset_id}")
            continue
        manifests_path = ROOT / root_rel / "manifests.jsonl"
        if not manifests_path.exists():
            problems.append(f"original manifest missing: {dataset_id}")
        elif sha256_file(manifests_path) != expected:
            problems.append(f"original manifest changed: {dataset_id}")
    for name, info in (manifest.get("files") or {}).items():
        path = base / info["path"]
        if not path.exists():
            problems.append(f"missing frozen file: {name}")
        elif sha256_file(path) != info["sha256"]:
            problems.append(f"hash mismatch: {name}")
    rows = {split: _load_jsonl(base / f"{split}.jsonl") for split in SPLITS}
    counts = {split: len(rows[split]) for split in SPLITS}
    if counts != manifest.get("counts"):
        problems.append(f"count mismatch: actual={counts} expected={manifest.get('counts')}")
    # Verify the proposed counts independently; do not force them.
    if counts != {"train": 548, "validation": 137, "test": 138, "quarantine": 93}:
        problems.append(f"proposed-count discrepancy: {counts}")
    sample_sets = {split: {row["sample_id"] for row in rows[split]} for split in SPLITS}
    for left in SPLITS:
        for right in SPLITS:
            if left >= right:
                continue
            overlap = sample_sets[left] & sample_sets[right]
            if overlap:
                problems.append(f"sample overlap {left}/{right}: {len(overlap)}")
    # Test retention: all original test sample IDs must remain in frozen test.
    original_rows, _ = _manifest_rows()
    original_test = {str(row["sample_id"]) for row in original_rows if row.get("split") == "test"}
    missing_test = original_test - sample_sets["test"]
    if missing_test:
        problems.append(f"protected test samples missing: {len(missing_test)}")
    # Frozen rows must not reassign a test sample to train/validation/quarantine.
    for split in ("train", "validation", "quarantine"):
        bad = {row["sample_id"] for row in rows[split] if row.get("original_split") == "test"}
        if bad:
            problems.append(f"test sample assigned to {split}: {len(bad)}")
    if recompute:
        proposed = build_proposed_split(original_rows)
        expected = {split: proposed["counts"].get(split, 0) for split in SPLITS}
        if counts != expected:
            problems.append(f"recomputed assignment mismatch: actual={counts} recomputed={expected}")
        if sample_sets["test"] != {str(row["sample_id"]) for row in original_rows if proposed["assignments"].get(str(row["sample_id"])) == "test"}:
            problems.append("frozen test IDs do not match recomputed test assignment")
    # Cross-split identity: no source archive/file/normalized identity or project/binary in more than one protected/usable split.
    identity_fields = ("project_id", "binary_sha256", "source_archive_sha256", "source_file_sha256", "normalized_source_sha256")
    cross = {}
    for field in identity_fields:
        values: dict[str, set[str]] = defaultdict(set)
        for split in ("train", "validation", "test"):
            for row in rows[split]:
                raw = row.get(field)
                items = raw if isinstance(raw, list) else [raw]
                for item in items:
                    if item:
                        values[str(item)].add(split)
        overlap = {key: sorted(value) for key, value in values.items() if len(value) > 1}
        if overlap:
            cross[field] = len(overlap)
            problems.append(f"cross-split {field} identity: {len(overlap)}")
    return {
        "status": "PASS" if not problems else "FAIL",
        "root": str(base),
        "frozen_id": manifest.get("frozen_id"),
        "manifest_sha256": sha256_file(manifest_path),
        "counts": counts,
        "expected_counts": manifest.get("counts"),
        "protected_test_retained": not missing_test,
        "cross_split_identity": cross,
        "problems": problems,
        "limitations": ["Near-duplicate detection is a 5-gram shingle Jaccard approximation.", "This validation does not prove source semantics; it proves identity/partition hygiene."],
    }


def load_frozen_vocab(root: str | Path = DEFAULT_FREEZE_ROOT):
    from nexus.foundation.vocab import Vocabulary

    path = Path(root).resolve() / "vocab.json"
    return Vocabulary.load(path)


def frozen_manifest(root: str | Path = DEFAULT_FREEZE_ROOT) -> dict[str, Any]:
    return json.loads((Path(root).resolve() / "manifest.json").read_text(encoding="utf-8"))
