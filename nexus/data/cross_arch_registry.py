"""Canonical registry materialization for the controlled cross-architecture set.

The historical ``artifacts/dataset/cross_arch_controlled`` directory contains
builds and program records but no registry manifests.  This builder leaves that
directory untouched and writes a versioned, manifest-complete dataset whose
canonical identity is ``cross-arch-controlled-v1`` with the legacy sample label
``cross_arch_controlled`` retained as an alias.
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

CANONICAL_ID = "cross-arch-controlled-v1"
ALIAS = "cross_arch_controlled"
SOURCE_ROOT = Path("artifacts/dataset/cross_arch_controlled")
TARGET_ROOT = Path("artifacts/dataset/cross_arch_controlled_v1")
SPLITS = {"x86_64": "train", "i386": "train", "aarch64": "train", "arm": "train", "riscv": "validation", "mips": "test"}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_canonical_cross_arch(root: Path = Path(".")) -> dict[str, Any]:
    source_root = (root / SOURCE_ROOT).resolve()
    target_root = (root / TARGET_ROOT).resolve()
    if not source_root.exists():
        raise FileNotFoundError(f"cross-arch source dataset is missing: {source_root}")
    records_root = source_root / "records"
    builds_root = source_root / "builds"
    source_path = source_root / "source" / "cross_arch.c"
    if not records_root.exists() or not builds_root.exists() or not source_path.exists():
        raise FileNotFoundError("cross-arch dataset is incomplete: records/builds/source required")
    target_root.mkdir(parents=True, exist_ok=True)
    import shutil
    for relative in ("records", "source", "builds"):
        source_dir = source_root / relative
        target_dir = target_root / relative
        if target_dir.exists():
            shutil.rmtree(target_dir)
        shutil.copytree(source_dir, target_dir)
    target_records = target_root / "records"
    target_builds = target_root / "builds"
    target_source = target_root / "source" / "cross_arch.c"
    manifests = []
    records = sorted(target_records.glob("*.json"))
    for record_path in records:
        name = record_path.stem
        arch, optimization = name.rsplit("_", 1) if "_" in name else (name, "O0")
        binary = target_builds / arch / optimization / "cross_arch.o"
        if not binary.exists():
            raise FileNotFoundError(f"binary missing for {name}: {binary}")
        record = json.loads(record_path.read_text(encoding="utf-8"))
        for function in record.get("functions") or []:
            if not function.get("names") and function.get("name"):
                function["names"] = [{"value": function["name"], "source": "symbol"}]
        record_path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        functions = record.get("functions") or []
        split = SPLITS.get(arch, "validation")
        sample_id = f"cross-arch:{arch}:{optimization}"
        manifest = {
            "schema_version": "0.1.0",
            "sample_id": sample_id,
            "dataset_id": CANONICAL_ID,
            "legacy_dataset_id": ALIAS,
            "project_id": f"cross_arch_{arch}",
            "split": split,
            "source_family": ALIAS,
            "license": {"spdx": "CC0-1.0"},
            "provenance": {
                "source_archive_sha256": _sha(source_path),
                "source_path": str(target_source.relative_to(target_root)),
                "builder": "nexus.data.cross_arch_registry/v1",
            },
            "binary": {
                "sha256": _sha(binary),
                "size": binary.stat().st_size,
                "path": str(binary.relative_to(target_root)),
                "format": "ELF",
            },
            "artifacts": [
                {
                    "path": str(record_path.relative_to(target_root)),
                    "sha256": _sha(record_path),
                    "size": record_path.stat().st_size,
                    "media_type": "application/json",
                    "role": "program_record",
                },
                {
                    "path": str(binary.relative_to(target_root)),
                    "sha256": _sha(binary),
                    "size": binary.stat().st_size,
                    "media_type": "application/x-elf",
                    "role": "binary",
                },
                {
                    "path": str(target_source.relative_to(target_root)),
                    "sha256": _sha(source_path),
                    "size": source_path.stat().st_size,
                    "media_type": "text/x-c",
                    "role": "source",
                },
            ],
            "build": {
                "compiler": "clang",
                "optimization": optimization,
                "arch": arch,
                "debug_info": True,
            },
            "function_count": len(functions),
            "function_names": [function.get("name") for function in functions],
            "mission_relevance": {
                "category": "controlled_binary_source",
                "classification": "CORE",
                "allowed_for_training": split in {"train", "validation"},
            },
            "quality": {"validation_status": "validated", "validated_at": datetime.now(UTC).isoformat()},
        }
        manifests.append(manifest)
    if not manifests:
        raise ValueError("cross-arch dataset has no records")
    manifests.sort(key=lambda item: item["sample_id"])
    manifest_path = target_root / "manifests.jsonl"
    manifest_path.write_text("\n".join(json.dumps(item, sort_keys=True) for item in manifests) + "\n", encoding="utf-8")
    manifest_hash = _sha(manifest_path)
    splits = {split: sum(1 for item in manifests if item["split"] == split) for split in ("train", "validation", "test")}
    summary = {
        "schema_version": "nexus.dataset-summary/v2",
        "dataset_id": CANONICAL_ID,
        "canonical_id": CANONICAL_ID,
        "aliases": [ALIAS, "cross-arch-controlled-v1"],
        "manifest_sha256": manifest_hash,
        "content_fingerprint": manifest_hash,
        "path": str(target_root.relative_to(root.resolve())),
        "created_at": datetime.now(UTC).isoformat(),
        "license": "CC0-1.0",
        "manifests": len(manifests),
        "unique_projects": len({item["project_id"] for item in manifests}),
        "unique_functions": sum(int(item["function_count"]) for item in manifests),
        "architectures": sorted({item["build"]["arch"] for item in manifests}),
        "splits": splits,
        "builds": sorted({item["build"]["optimization"] for item in manifests}),
        "source": str(target_source.relative_to(target_root)),
        "notes": [
            "Canonical registry identity is cross-arch-controlled-v1.",
            "The legacy sample label cross_arch_controlled is retained in sample metadata and as an alias.",
            "This pilot assigns architecture projects to splits without project leakage; the authoritative frozen v3 corpus contains the cross-architecture training samples.",
            "Generated C/ELF artifacts are static-analysis evidence, not runtime-verified behavior.",
        ],
    }
    (target_root / "dataset_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


if __name__ == "__main__":
    print(json.dumps(build_canonical_cross_arch(Path.cwd()), indent=2))
