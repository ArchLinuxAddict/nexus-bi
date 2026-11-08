import json
from pathlib import Path
from typing import Any

from nexus.mission.policy import load_policy, load_registry, mission_for_dataset


def _artifact(manifest: dict[str, Any], media_type: str | None = None, suffix: str | None = None) -> dict[str, Any] | None:
    for artifact in manifest.get("artifacts", []):
        if media_type and artifact.get("media_type") == media_type:
            return artifact
        if suffix and artifact.get("path", "").endswith(suffix):
            return artifact
    return None


def _function_count(root: Path, manifest: dict[str, Any]) -> int:
    label = next((item.get("artifact") for item in manifest.get("label_manifests", []) if item.get("kind") == "functions"), None)
    if not label:
        return 0
    try:
        payload = json.loads((root / label["path"]).read_text(encoding="utf-8"))
        return len(payload.get("functions", []))
    except Exception:
        return 0


def audit_dataset_relevance(root: str | Path) -> dict[str, Any]:
    path = Path(root).resolve()
    policy = load_policy()
    entry = mission_for_dataset(path)
    category = policy["dataset_categories"].get(entry["category"])
    if category is None:
        return {"status": "FAIL", "reason": "unclassified category", "path": str(path)}
    manifests = [json.loads(line) for line in (path / "manifests.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    binary_count = sum(1 for manifest in manifests if _artifact(manifest, media_type="application/x-elf"))
    source_count = sum(1 for manifest in manifests if manifest.get("provenance", {}).get("source_archive_sha256"))
    license_count = sum(1 for manifest in manifests if manifest.get("license", {}).get("spdx"))
    mission_coverage = sum(1 for manifest in manifests if manifest.get("mission_relevance"))
    debug_count = sum(1 for manifest in manifests if manifest.get("build", {}).get("debug_info"))
    dwarf_count = sum(1 for manifest in manifests if any("dwarf" in artifact.get("path", "") for artifact in manifest.get("artifacts", [])))
    ir_count = sum(1 for manifest in manifests if any(artifact.get("path", "").endswith(".ll") for artifact in manifest.get("artifacts", [])))
    cfg_count = sum(1 for manifest in manifests if any(item.get("kind") == "cfg" for item in manifest.get("label_manifests", [])))
    function_observations = sum(_function_count(path, manifest) for manifest in manifests)
    source_files = sum(1 for manifest in manifests for source_path in manifest.get("provenance", {}).get("source_paths", []))
    sample_rows = []
    false_core = []
    for manifest in manifests:
        has_binary = _artifact(manifest, media_type="application/x-elf") is not None
        has_source = bool(manifest.get("provenance", {}).get("source_archive_sha256")) and bool(manifest.get("provenance", {}).get("source_paths"))
        mission = manifest.get("mission_relevance") or {}
        classification = mission.get("classification", entry.get("classification"))
        if classification == "CORE" and (not has_binary or not has_source):
            false_core.append(manifest.get("sample_id"))
        sample_rows.append({
            "sample_id": manifest.get("sample_id"),
            "project_id": manifest.get("project_id"),
            "split": manifest.get("split"),
            "classification": classification,
            "has_binary": has_binary,
            "has_source": has_source,
            "has_cfg_labels": any(item.get("kind") == "cfg" for item in manifest.get("label_manifests", [])),
            "has_function_labels": any(item.get("kind") == "functions" for item in manifest.get("label_manifests", [])),
            "functions": _function_count(path, manifest),
            "has_license": bool(manifest.get("license", {}).get("spdx")),
            "has_provenance": bool(manifest.get("provenance", {}).get("source_archive_sha256")),
            "has_mission": bool(mission),
        })
    declared_capabilities = set(entry.get("capabilities", []))
    actual_support = {
        "native_binary_understanding": binary_count > 0,
        "instruction_semantics": function_observations > 0,
        "function_identification": function_observations > 0,
        "control_flow_reasoning": cfg_count > 0,
        "source_binary_alignment": False,
        "c_reconstruction": False,
        "cpp_reconstruction": False,
        "java_jni_reconstruction": False,
        "library_analysis": binary_count > 0,
        "security_investigation": False,
        "cross_compiler_matching": len({manifest.get("build", {}).get("compiler") for manifest in manifests}) > 1,
    }
    unsupported_declared = sorted(capability for capability in declared_capabilities if not actual_support.get(capability, False))
    objective_capability_mismatch = []
    for objective_name in entry.get("objectives", []):
        objective = policy["objectives"].get(objective_name, {})
        for capability in objective.get("capabilities", []):
            if capability not in declared_capabilities:
                objective_capability_mismatch.append({"objective": objective_name, "capability": capability})
    source_only = binary_count == 0 and source_count > 0
    text_only = binary_count == 0 and source_count == 0
    if text_only or false_core or (category.get("classification") == "CORE" and source_only):
        verdict = "FAIL"
    elif unsupported_declared or objective_capability_mismatch:
        verdict = "PARTIAL"
    else:
        verdict = "PASS"
    if source_only and category.get("classification") == "CORE":
        verdict = "FAIL"
    return {
        "dataset_id": entry.get("dataset_id"),
        "path": str(path),
        "category": entry["category"],
        "classification": entry["classification"],
        "samples": len(manifests),
        "binary_samples": binary_count,
        "source_samples": source_count,
        "license_samples": license_count,
        "mission_relevance_samples": mission_coverage,
        "debug_samples": debug_count,
        "dwarf_samples": dwarf_count,
        "llvm_ir_samples": ir_count,
        "cfg_label_samples": cfg_count,
        "source_files": source_files,
        "function_observations": function_observations,
        "declared_capabilities": sorted(declared_capabilities),
        "verified_capability_support": {name: value for name, value in actual_support.items() if value},
        "unverified_or_unsupported_capabilities": unsupported_declared,
        "objective_capability_mismatches": objective_capability_mismatch,
        "false_core_samples": false_core[:20],
        "false_core_count": len(false_core),
        "source_only": source_only,
        "text_only": text_only,
        "verdict": verdict,
        "limitations": [
            "Capability support checks presence of artifacts and labels, not their correctness.",
            "Source-to-binary alignment is not verified without per-function source/address mappings.",
            "C/C++/Java reconstruction and security capability are not credited from file presence alone.",
            "This audit cannot understand semantic relevance; human review remains required for novel categories.",
        ],
        "samples_detail": sample_rows,
    }


def audit_all_registered() -> dict[str, Any]:
    registry = load_registry()
    results = []
    for entry in registry["datasets"]:
        if entry.get("path") == "synthetic":
            results.append({"dataset_id": entry.get("dataset_id"), "verdict": "UNVERIFIED", "reason": "synthetic infrastructure dataset has no files"})
            continue
        try:
            results.append(audit_dataset_relevance(Path(entry["path"])))
        except Exception as exc:
            results.append({"dataset_id": entry.get("dataset_id"), "path": entry.get("path"), "verdict": "BLOCKED", "reason": str(exc)})
    return {
        "status": "passed" if all(item.get("verdict") in {"PASS", "PARTIAL", "UNVERIFIED"} for item in results) else "failed",
        "datasets": results,
        "summary": {
            "pass": sum(1 for item in results if item.get("verdict") == "PASS"),
            "partial": sum(1 for item in results if item.get("verdict") == "PARTIAL"),
            "fail": sum(1 for item in results if item.get("verdict") == "FAIL"),
            "unverified": sum(1 for item in results if item.get("verdict") == "UNVERIFIED"),
            "blocked": sum(1 for item in results if item.get("verdict") == "BLOCKED"),
        },
    }
