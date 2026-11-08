import hashlib
import json
from pathlib import Path
from typing import Any

from nexus.mission.policy import load_policy, mission_for_dataset


def backfill_dataset(root: str | Path, write_sidecar: bool = True) -> dict[str, Any]:
    path = Path(root).resolve()
    entry = mission_for_dataset(path)
    policy = load_policy()
    category = policy["dataset_categories"][entry["category"]]
    manifests_path = path / "manifests.jsonl"
    manifests = [json.loads(line) for line in manifests_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    samples = []
    quarantined = []
    for manifest in manifests:
        sample_id = manifest.get("sample_id")
        existing = manifest.get("mission_relevance")
        provenance = manifest.get("provenance", {})
        license_info = manifest.get("license", {})
        binary = any(artifact.get("media_type") == "application/x-elf" for artifact in manifest.get("artifacts", []))
        source = bool(provenance.get("source_archive_sha256")) and bool(provenance.get("source_paths"))
        function_labels = any(item.get("kind") == "functions" for item in manifest.get("label_manifests", []))
        cfg_labels = any(item.get("kind") == "cfg" for item in manifest.get("label_manifests", []))
        mapping = any(item.get("kind") == "instruction_source_map" for item in manifest.get("label_manifests", []))
        if existing:
            mission = {
                "category": existing.get("category"),
                "classification": existing.get("classification"),
                "capabilities": existing.get("capabilities", []),
                "objectives": existing.get("objectives", []),
                "backfill_source": "manifest",
                "confidence": 1.0,
            }
        else:
            mission = {
                "category": entry["category"],
                "classification": entry["classification"],
                "capabilities": category.get("capabilities", []),
                "objectives": entry.get("objectives", []),
                "backfill_source": "registry_dataset_level",
                "confidence": 0.6,
            }
        reasons = []
        if not provenance.get("source_archive_sha256"):
            reasons.append("missing source archive provenance")
        if not license_info.get("spdx"):
            reasons.append("missing license identifier")
        if not binary:
            reasons.append("missing binary artifact")
        if not source:
            reasons.append("missing source evidence")
        if reasons:
            quarantined.append({"sample_id": sample_id, "reasons": reasons})
            continue
        samples.append({
            "sample_id": sample_id,
            "project_id": manifest.get("project_id"),
            "split": manifest.get("split"),
            "mission": mission,
            "provenance": {
                "repository": provenance.get("repository"),
                "commit": provenance.get("commit"),
                "source_archive_sha256": provenance.get("source_archive_sha256"),
                "source_paths": provenance.get("source_paths", []),
                "license_spdx": license_info.get("spdx"),
                "license_file_sha256": provenance.get("upstream_license_file_sha256"),
            },
            "evidence": {
                "binary_present": binary,
                "source_present": source,
                "function_labels": function_labels,
                "cfg_labels": cfg_labels,
                "instruction_source_mapping": mapping,
            },
            "alignment_confidence": "partial" if mapping else "none",
            "alignment_limitations": [] if mapping else ["No instruction/source mapping artifact exists."],
        })
    report = {
        "schema_version": "0.1.0",
        "dataset_id": entry.get("dataset_id"),
        "dataset_path": str(path),
        "source_manifest_sha256": hashlib.sha256(manifests_path.read_bytes()).hexdigest(),
        "samples": len(samples),
        "quarantined": len(quarantined),
        "quarantine": quarantined,
        "backfill_policy": "historical manifests are not mutated; dataset-level registry classification is copied into a versioned sidecar",
        "samples_detail": samples,
        "limitations": [
            "Registry-level backfill is dataset-level classification, not independent per-sample proof.",
            "No instruction/source mapping exists, so alignment confidence is none.",
            "Provenance is copied from existing manifest fields; missing fields are quarantined, not invented.",
        ],
    }
    if write_sidecar:
        sidecar = path / "mission_backfill.json"
        sidecar.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def backfill_all(paths: list[str | Path], output: str | Path) -> dict[str, Any]:
    results = [backfill_dataset(path) for path in paths]
    payload = {
        "status": "completed",
        "datasets": results,
        "summary": {
            "datasets": len(results),
            "samples": sum(item["samples"] for item in results),
            "quarantined": sum(item["quarantined"] for item in results),
        },
    }
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload
