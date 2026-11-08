import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from nexus.core.hashing import sha256_file

RAW_REPRESENTATIONS = (
    "source_archive",
    "source_tree",
    "binary",
    "build_log",
    "symbols",
    "disassembly",
    "elf_metadata",
    "assembly",
    "object_files",
    "llvm_ir",
    "dwarf_dump",
)
DERIVED_REPRESENTATIONS = (
    "functions",
    "cfg",
    "call_graph",
    "instruction_alignment",
    "basic_block_alignment",
    "variables",
    "data_objects",
)


def _file_check(path: Path, expected_sha: str | None = None) -> dict[str, Any]:
    if not path.exists() or not path.is_file():
        return {"status": "missing", "path": str(path)}
    digest = sha256_file(path)
    status = "passed" if not expected_sha or digest == expected_sha else "hash_mismatch"
    return {"status": status, "path": str(path), "sha256": digest, "size": path.stat().st_size}


def collect_representation_coverage(dataset_root: str | Path, reports_root: str | Path) -> dict[str, Any]:
    root = Path(dataset_root).resolve()
    reports = Path(reports_root).resolve()
    catalog = json.loads((root / "catalog.json").read_text(encoding="utf-8"))
    accepted = [project for project in catalog if project.get("status") == "accepted"]
    build_report_path = root / "build_matrix_report.json"
    build_report = json.loads(build_report_path.read_text(encoding="utf-8")) if build_report_path.exists() else {"projects": []}
    builds_by_project = {item["project_id"]: item for item in build_report.get("projects", [])}
    project_rows = []
    raw_totals = {name: 0 for name in RAW_REPRESENTATIONS}
    derived_totals = {name: 0 for name in DERIVED_REPRESENTATIONS}
    build_count = 0
    for project in accepted:
        project_id = project["project_id"]
        source_archive = (root / project["source_archive_path"]).resolve()
        source_tree = (root / project["source_root"]).resolve()
        inventory_path = source_tree.parent / "inventory.json"
        build_project = builds_by_project.get(project_id, {})
        rows = []
        for build in build_project.get("builds", []):
            if build.get("status") != "built":
                continue
            build_count += 1
            variant_id = (build.get("variant") or {}).get("variant_id", "unknown")
            build_dir = (root / "builds" / project_id / variant_id).resolve()
            extras = build.get("extras") or {}
            raw = {
                "source_archive": _file_check(source_archive, project.get("source_archive_sha256")),
                "source_tree": {"status": "passed" if source_tree.exists() and inventory_path.exists() else "missing", "path": str(source_tree)},
                "binary": _file_check((root / build["artifact_path"]).resolve(), build.get("artifact_sha256")),
                "build_log": _file_check(build_dir / "build.log"),
                "symbols": _file_check(build_dir / "symbols.txt"),
                "disassembly": _file_check(build_dir / "disassembly.txt"),
                "elf_metadata": {"status": "passed" if (build.get("elf") or {}).get("status") == "ok" else "unavailable", "path": str(build_dir / "build_metadata.json")},
                "assembly": {"status": "passed" if extras.get("assembly") else "unavailable", "count": len(extras.get("assembly") or [])},
                "object_files": {"status": "passed" if extras.get("objects") else "unavailable", "count": len(extras.get("objects") or [])},
                "llvm_ir": {"status": "passed" if extras.get("ir") else "unavailable", "count": len(extras.get("ir") or [])},
                "dwarf_dump": {"status": "passed" if extras.get("dwarf") else "unavailable", "count": len(extras.get("dwarf") or [])},
            }
            alignment_index = root / "alignments" / project_id / variant_id / "alignment_index.json"
            derived = {name: {"status": "unavailable"} for name in DERIVED_REPRESENTATIONS}
            if alignment_index.exists():
                try:
                    index = json.loads(alignment_index.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    index = {}
                functions = index.get("functions", [])
                blocks = index.get("basic_block_alignments", [])
                instructions = index.get("instruction_alignments", [])
                variables = index.get("variables", [])
                data_objects = index.get("data_objects", [])
                call_graph = index.get("call_graph", {})
                derived = {
                    "functions": {"status": "passed" if functions else "unavailable", "count": len(functions), "classification": index.get("project", {}).get("classification")},
                    "cfg": {"status": "passed" if blocks else "unavailable", "count": len(blocks), "classification": "HEURISTIC"},
                    "call_graph": {"status": "passed" if call_graph.get("edges") is not None else "unavailable", "nodes": len(call_graph.get("nodes", [])), "edges": len(call_graph.get("edges", [])), "classification": call_graph.get("classification")},
                    "instruction_alignment": {"status": "passed" if instructions else "unavailable", "count": len(instructions), "classification": "DEBUG_DERIVED"},
                    "basic_block_alignment": {"status": "passed" if any(item.get("classification") != "UNAVAILABLE" for item in blocks) else "unavailable", "count": sum(1 for item in blocks if item.get("classification") != "UNAVAILABLE")},
                    "variables": {"status": "passed" if variables else "unavailable", "count": len(variables)},
                    "data_objects": {"status": "passed" if data_objects else "unavailable", "count": len(data_objects)},
                }
            for name in RAW_REPRESENTATIONS:
                if raw[name].get("status") in {"passed", "hash_mismatch"}:
                    raw_totals[name] += 1
            for name in DERIVED_REPRESENTATIONS:
                if derived[name].get("status") == "passed":
                    derived_totals[name] += 1
            rows.append({"variant_id": variant_id, "raw": raw, "derived": derived, "alignment_index": str(alignment_index.relative_to(root)) if alignment_index.exists() else None})
        project_rows.append(
            {
                "project_id": project_id,
                "name": project.get("name"),
                "category": project.get("category"),
                "builds": len(rows),
                "raw": {name: sum(1 for row in rows if row["raw"][name].get("status") in {"passed", "hash_mismatch"}) for name in RAW_REPRESENTATIONS},
                "derived": {name: sum(1 for row in rows if row["derived"][name].get("status") == "passed") for name in DERIVED_REPRESENTATIONS},
                "rows": rows,
            }
        )
    raw_ratio = {name: round(raw_totals[name] / max(1, build_count), 4) for name in RAW_REPRESENTATIONS}
    derived_ratio = {name: round(derived_totals[name] / max(1, build_count), 4) for name in DERIVED_REPRESENTATIONS}
    missing_raw = {name: round(1.0 - raw_ratio[name], 4) for name in RAW_REPRESENTATIONS}
    missing_derived = {name: round(1.0 - derived_ratio[name], 4) for name in DERIVED_REPRESENTATIONS}
    report = {
        "schema_version": "0.1.0",
        "created_at": datetime.now(UTC).isoformat(),
        "dataset_root": str(root),
        "projects": len(project_rows),
        "builds": build_count,
        "raw_representations": raw_ratio,
        "derived_representations": derived_ratio,
        "missing_raw": missing_raw,
        "missing_derived": missing_derived,
        "raw_counts": raw_totals,
        "derived_counts": derived_totals,
        "status": "passed" if raw_ratio["source_archive"] >= 0.99 and raw_ratio["binary"] >= 0.99 and raw_ratio["symbols"] >= 0.5 and derived_ratio["functions"] > 0 else "partial",
        "policy": "Raw artifacts and derived representations are recorded separately. Derived representations are never substituted for raw evidence, and an unavailable representation is reported as unavailable rather than inferred.",
        "projects_detail": project_rows,
        "limitations": [
            "IR is available only for clang builds; GCC builds report IR unavailable.",
            "DWARF dumps and object files are generated only for debug builds.",
            "CFG and call-graph edges are heuristic/derived; they do not establish runtime reachability.",
            "Source-tree inventory is shared across builds and is not a per-build artifact.",
        ],
    }
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "representation_coverage_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report
