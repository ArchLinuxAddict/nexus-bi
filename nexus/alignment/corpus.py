import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from nexus.alignment.pipeline import extract_alignment, transfer_alignment, write_alignment
from nexus.core.hashing import sha256_file


def _variant_key(variant: dict[str, Any]) -> tuple[str, str, bool, bool, str]:
    return (variant.get("compiler"), variant.get("optimization"), bool(variant.get("debug")), bool(variant.get("pie")), variant.get("link") or "executable")


def _select_debug_variants(builds: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected = []
    seen = set()
    for build in builds:
        if build.get("status") != "built":
            continue
        variant = build.get("variant") or {}
        if not variant.get("debug"):
            continue
        if variant.get("optimization") not in {"O0", "O2"}:
            continue
        key = (variant.get("compiler"), variant.get("optimization"))
        if key in seen:
            continue
        seen.add(key)
        selected.append(build)
    return selected


def _derived_views(index: dict[str, Any], target: Path) -> dict[str, str]:
    target.mkdir(parents=True, exist_ok=True)
    functions = {
        "schema_version": "0.1.0",
        "binary_sha256": index.get("binary", {}).get("sha256"),
        "alignment_index_status": index.get("status"),
        "functions": [
            {
                "function_id": item.get("function_id"),
                "name": item.get("name"),
                "entry_va": item.get("entry_va"),
                "ranges": item.get("ranges"),
                "source_file": item.get("source_file"),
                "source_line": item.get("source_line"),
                "classification": item.get("classification"),
                "confidence": item.get("confidence"),
                "instruction_count": len(item.get("instruction_ids", [])),
                "block_ids": item.get("block_ids", []),
            }
            for item in index.get("functions", [])
        ],
    }
    cfg = {
        "schema_version": "0.1.0",
        "binary_sha256": index.get("binary", {}).get("sha256"),
        "classification": "HEURISTIC",
        "functions": [
            {
                "function_id": item.get("function_id"),
                "name": item.get("name"),
                "blocks": [
                    {
                        "block_id": block.get("block_id"),
                        "start_va": block.get("start_va"),
                        "end_va": block.get("end_va"),
                        "instruction_ids": block.get("instruction_ids", []),
                        "successors": block.get("successors", []),
                        "classification": block.get("classification"),
                    }
                    for block in item.get("blocks", [])
                ],
            }
            for item in index.get("functions", [])
        ],
    }
    callgraph = {"schema_version": "0.1.0", "binary_sha256": index.get("binary", {}).get("sha256"), **index.get("call_graph", {})}
    payloads = {"functions.json": functions, "cfg.json": cfg, "callgraph.json": callgraph}
    paths = {}
    for name, payload in payloads.items():
        path = target / name
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        paths[name] = str(path)
    return paths


def run_corpus_alignment(
    dataset_root: str | Path,
    *,
    max_projects: int | None = None,
    transfer_stripped: bool = True,
    resume: bool = True,
) -> dict[str, Any]:
    root = Path(dataset_root).resolve()
    catalog = json.loads((root / "catalog.json").read_text(encoding="utf-8"))
    accepted = [project for project in catalog if project.get("status") == "accepted"]
    if max_projects is not None:
        accepted = accepted[:max_projects]
    report_path = root / "build_matrix_report.json"
    if report_path.exists():
        build_report = json.loads(report_path.read_text(encoding="utf-8"))
        builds_by_project = {project["project_id"]: project.get("builds", []) for project in build_report.get("projects", [])}
    else:
        builds_by_project = {}
    results = []
    for project in accepted:
        project_id = project["project_id"]
        builds = builds_by_project.get(project_id, [])
        source_root = (root / project["source_root"]).resolve()
        project_dir = root / "alignments" / project_id
        derived_dir = root / "derived" / project_id
        debug_builds = _select_debug_variants(builds)
        debug_indexes = {}
        project_result = {
            "project_id": project_id,
            "name": project.get("name"),
            "category": project.get("category"),
            "source_root": str(source_root),
            "debug_builds": len(debug_builds),
            "indexes": [],
            "stripped": [],
            "status": "completed",
        }
        for build in debug_builds:
            variant = build.get("variant") or {}
            binary = (root / build["artifact_path"]).resolve()
            if not binary.exists():
                project_result["indexes"].append({"variant_id": variant.get("variant_id"), "status": "unavailable", "reason": "binary missing"})
                continue
            variant_dir = project_dir / str(variant.get("variant_id"))
            index_path = variant_dir / "alignment_index.json"
            index = None
            if resume and index_path.exists():
                try:
                    existing = json.loads(index_path.read_text(encoding="utf-8"))
                    if existing.get("binary", {}).get("sha256") == sha256_file(binary):
                        index = existing
                except (OSError, json.JSONDecodeError):
                    index = None
            if index is None:
                index = extract_alignment(
                    binary,
                    source_root,
                    project_id=project_id,
                    build_variant=variant,
                    record={
                        "source_archive_sha256": project.get("source_archive_sha256"),
                        "commit": project.get("commit"),
                        "license": project.get("license"),
                    },
                )
                write_alignment(index, index_path)
            views = _derived_views(index, derived_dir / str(variant.get("variant_id")))
            item = {
                "variant_id": variant.get("variant_id"),
                "status": index.get("status"),
                "dwarf_status": index.get("dwarf_status"),
                "binary_sha256": index.get("binary", {}).get("sha256"),
                "alignment_index": str(index_path.relative_to(root)),
                "alignment_index_sha256": sha256_file(index_path),
                "derived_views": {name: str(Path(path).relative_to(root)) for name, path in views.items()},
                "coverage": index.get("coverage", {}),
                "classifications": index.get("classifications", {}),
            }
            project_result["indexes"].append(item)
            debug_indexes[(variant.get("compiler"), variant.get("optimization"), bool(variant.get("pie")), variant.get("link") or "executable")] = (build, index, index_path)
        if transfer_stripped:
            for build in builds:
                variant = build.get("variant") or {}
                if build.get("status") != "built" or variant.get("debug"):
                    continue
                key = (variant.get("compiler"), variant.get("optimization"), bool(variant.get("pie")), variant.get("link") or "executable")
                pair = debug_indexes.get(key)
                if pair is None:
                    project_result["stripped"].append({"variant_id": variant.get("variant_id"), "status": "unavailable", "reason": "no debug counterpart"})
                    continue
                debug_build, debug_index, _ = pair
                debug_binary = (root / debug_build["artifact_path"]).resolve()
                stripped_binary = (root / build["artifact_path"]).resolve()
                variant_dir = project_dir / str(variant.get("variant_id"))
                index_path = variant_dir / "alignment_index.json"
                transferred = None
                reused_transfer = False
                if resume and index_path.exists():
                    try:
                        existing = json.loads(index_path.read_text(encoding="utf-8"))
                        if existing.get("binary", {}).get("sha256") == sha256_file(stripped_binary) and existing.get("transfer"):
                            transferred = existing
                            reused_transfer = True
                    except (OSError, json.JSONDecodeError):
                        transferred = None
                if transferred is None:
                    transferred = transfer_alignment(debug_index, stripped_binary, debug_binary)
                if transferred.get("status") == "available":
                    if not reused_transfer:
                        write_alignment(transferred, index_path)
                    views = _derived_views(transferred, derived_dir / str(variant.get("variant_id")))
                    project_result["stripped"].append(
                        {
                            "variant_id": variant.get("variant_id"),
                            "status": "available",
                            "method": "text-identity-transfer",
                            "alignment_index": str(index_path.relative_to(root)),
                            "alignment_index_sha256": sha256_file(index_path),
                            "derived_views": {name: str(Path(path).relative_to(root)) for name, path in views.items()},
                            "coverage": transferred.get("coverage", {}),
                            "classifications": transferred.get("classifications", {}),
                        }
                    )
                else:
                    project_result["stripped"].append({"variant_id": variant.get("variant_id"), "status": "unavailable", "reason": transferred.get("reason"), "debug_text_sha256": transferred.get("debug_text_sha256"), "target_text_sha256": transferred.get("target_text_sha256")})
        results.append(project_result)
    available_indexes = [item for project in results for item in project["indexes"] if item.get("status") == "available"]
    available_transfers = [item for project in results for item in project["stripped"] if item.get("status") == "available"]
    total_functions = sum(item["coverage"].get("functions", 0) for item in available_indexes)
    functions_with_source = sum(item["coverage"].get("functions_with_source", 0) for item in available_indexes)
    instructions = sum(item["coverage"].get("instructions", 0) for item in available_indexes)
    mapped = sum(item["coverage"].get("instructions_mapped", 0) for item in available_indexes)
    summary = {
        "schema_version": "0.1.0",
        "created_at": datetime.now(UTC).isoformat(),
        "dataset_root": str(root),
        "projects_total": len(results),
        "projects_with_alignment": sum(1 for project in results if any(item.get("status") == "available" for item in project["indexes"])),
        "debug_indexes": len(available_indexes),
        "stripped_transfers": len(available_transfers),
        "functions": total_functions,
        "functions_with_source": functions_with_source,
        "function_source_coverage": round(functions_with_source / max(1, total_functions), 4),
        "instructions": instructions,
        "instructions_mapped": mapped,
        "instruction_mapping_coverage": round(mapped / max(1, instructions), 4),
        "projects": results,
        "limitations": [
            "Real-project instruction coverage includes startup/runtime code without source mapping; the denominator is all decoded instructions.",
            "Stripped-transfer is claimed only when the .text section is byte-identical to the debug build.",
            "Symbol-derived functions are counted as function-level alignment but never as instruction-level source alignment.",
        ],
    }
    alignment_root = root / "alignments"
    alignment_root.mkdir(parents=True, exist_ok=True)
    (alignment_root / "corpus_alignment_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary
