import hashlib
import json
from dataclasses import replace
from pathlib import Path

from nexus.data.identity import FunctionIdentityV2
from nexus.data.split_v2 import build_split_v2
from nexus.data.validate_v2 import validate_legacy_v1_against_identities, validate_split_v2


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _identity(
    name: str,
    *,
    split: str = "train",
    protected: bool = False,
    project: str = "project_a",
    raw: str | None = None,
    normalized: str | None = None,
    instruction: str | None = None,
    count: int = 24,
    manifest: str | None = None,
) -> FunctionIdentityV2:
    raw = raw or _digest(name)
    normalized = normalized or raw
    instruction = instruction or _digest("ins:" + name)
    source_function = _digest("sf:" + name)
    return FunctionIdentityV2(
        function_uid=_digest("uid:" + name),
        project_id=project,
        repository_id=project,
        source_archive_id=_digest("archive:" + project),
        source_file_id=_digest("file:" + project),
        source_function_id=source_function,
        sample_id=name,
        function_id=name,
        source_path="source.c",
        source_line_start=1,
        source_line_end=10,
        raw_source_hash=raw,
        normalized_source_hash=normalized,
        normalized_function_hash=normalized,
        source_shingle_hashes=tuple(
            int.from_bytes(hashlib.sha256((name + str(index)).encode()).digest()[:4], "big")
            for index in range(5)
        ),
        source_available=True,
        source_alignment="exact_source_function_name",
        source_alignment_confidence=0.99,
        binary_sha256=_digest("binary:" + project),
        instruction_hash=instruction,
        instruction_count=count,
        build_id=_digest("build:" + project),
        compiler_id="gcc",
        compiler_version="13",
        optimization_level="O2",
        architecture="x86_64",
        language="c",
        binary_format="ELF",
        debug_info=True,
        split=split,
        protected_test=protected,
        license_id="MIT",
        provenance={"manifest_sample_id": manifest or project, "sample_id": name},
    )


def _v2_fixture(root: Path, rows: dict[str, list[FunctionIdentityV2]], protected: list[FunctionIdentityV2]):
    root.mkdir(parents=True, exist_ok=True)
    for split in ("train", "validation", "test", "quarantine"):
        (root / f"{split}.jsonl").write_text(
            "".join(json.dumps(row.to_dict(), sort_keys=True) + "\n" for row in rows.get(split, [])),
            encoding="utf-8",
        )
    protected_payload = {
        "schema_version": "nexus.protected-test/v2",
        "samples": [
            {"sample_id": row.provenance["manifest_sample_id"], "binary_sha256": row.binary_sha256}
            for row in protected
        ],
    }
    protected_payload["test_file_sha256"] = hashlib.sha256((root / "test.jsonl").read_bytes()).hexdigest()
    (root / "protected_test.json").write_text(json.dumps(protected_payload, sort_keys=True), encoding="utf-8")
    assignments = {
        split: [row.function_uid for row in rows.get(split, [])]
        for split in ("train", "validation", "test", "quarantine")
    }
    manifest = {
        "schema_version": "nexus.split-v2/v1",
        "identity_schema": "nexus.function-identity/v2",
        "counts": {split: len(rows.get(split, [])) for split in ("train", "validation", "test", "quarantine")},
        "assignments_sha256": hashlib.sha256(
            json.dumps(assignments, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
    }
    (root / "manifest.json").write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")


def test_protected_test_wins_group_and_train_clone_is_quarantined():
    train = _identity("train_clone")
    test = _identity("test_original", split="test", protected=True, raw=train.raw_source_hash)
    result = build_split_v2([train, test], protected_test_ids={"project_a"})
    assert [row["function_uid"] for row in result.test] == [test.function_uid]
    assert [row["function_uid"] for row in result.quarantine] == [train.function_uid]
    assert result.quarantine[0]["quarantine_reason"] == "non_test_member_of_protected_test_group"


def test_train_validation_collision_quarantines_train_clone_without_crossing():
    train = _identity("shared_train", project="p1")
    validation = _identity("shared_validation", project="p2", split="validation", raw=train.raw_source_hash)
    result = build_split_v2([train, validation])
    assert {row["function_uid"] for row in result.validation} == {validation.function_uid}
    assert {row["function_uid"] for row in result.quarantine} == {train.function_uid}
    assert result.quarantine[0]["quarantine_reason"] == "function_group_spans_project_splits"
    assert result.report["cross_split"]["raw_source_hashes"] == 0


def test_near_duplicate_grouping_is_deterministic():
    left = _identity("left")
    right = replace(
        _identity("right", project="p2"),
        raw_source_hash=_digest("different raw"),
        normalized_source_hash=_digest("different normalized"),
        source_shingle_hashes=left.source_shingle_hashes + (left.source_shingle_hashes[0] + 1,),
    )
    result = build_split_v2([left, right], near_duplicate_threshold=0.8)
    assert len(result.report["counts"]) == 4
    assert result.report["near_duplicate_pairs"] >= 1


def test_legacy_validator_rejects_shared_train_validation_function(tmp_path):
    v1 = tmp_path / "v1"
    v1.mkdir()
    for split in ("train", "validation", "test"):
        (v1 / f"{split}.jsonl").write_text(
            json.dumps({"sample_id": f"{split}_manifest"}) + "\n", encoding="utf-8"
        )
    shared = _digest("shared")
    train = _identity("train_fn", manifest="train_manifest", raw=shared)
    validation = _identity("validation_fn", manifest="validation_manifest", split="validation", raw=shared)
    test = _identity("test_fn", manifest="test_manifest", split="test", protected=True, raw=_digest("test"))
    report = validate_legacy_v1_against_identities(v1, [train, validation, test])
    assert report["status"] == "FAIL"
    assert report["cross_split_source_functions"] == 1
    assert report["train_validation_sample_pairs"] == 1
    assert report["test_involved"] is False


def test_independent_validator_accepts_clean_and_rejects_tampered_split(tmp_path):
    train = _identity("train_clean", project="p1", raw=_digest("clean_train"))
    test = _identity("test_clean", project="p2", split="test", protected=True, raw=_digest("clean_test"))
    rows = {"train": [train], "validation": [], "test": [test], "quarantine": []}
    _v2_fixture(tmp_path / "v2", rows, [test])
    assert validate_split_v2(tmp_path / "v2")["status"] == "PASS"

    tampered = replace(train, raw_source_hash=test.raw_source_hash, normalized_source_hash=test.raw_source_hash)
    rows["train"] = [tampered]
    _v2_fixture(tmp_path / "v2", rows, [test])
    report = validate_split_v2(tmp_path / "v2")
    assert report["status"] == "FAIL"
    assert "no_cross_split_identities" in report["problems"]
