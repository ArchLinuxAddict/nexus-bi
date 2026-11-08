import hashlib
import json
from pathlib import Path

from nexus.tasks.builder import build_task_datasets


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _function(name: str, line: int, entry: int, binary: str) -> dict:
    return {
        "function_id": f"fn_{binary[:6]}_{entry:x}",
        "name": name,
        "entry_va": f"0x{entry:x}",
        "ranges": [{"start_va": f"0x{entry:x}", "end_va": f"0x{entry + 16:x}", "size": 16}],
        "size": 16,
        "source_file": "demo.c",
        "source_line": line,
        "classification": "EXACT",
        "confidence": 0.99,
        "instruction_ids": [f"ins_{entry:x}"],
        "instruction_count": 1,
        "mnemonic_histogram": {"mov": 1, "ret": 1},
        "code_sha256": _sha(f"{name}-{binary}"),
        "blocks": [{"block_id": f"bb_{entry:x}", "instruction_ids": [f"ins_{entry:x}"]}],
    }


def _index(binary: str, functions: list[dict], transfer: dict | None = None) -> dict:
    return {
        "schema_version": "0.2.0",
        "status": "available",
        "binary": {"path": f"/tmp/{binary}.elf", "sha256": _sha(binary), "machine": "x86_64", "stripped": transfer is not None},
        "build_variant": {"variant_id": binary, "compiler": binary.split("_")[0], "optimization": binary.split("_")[1]},
        "project": {"classification": "DEBUG_DERIVED", "confidence": 0.95},
        "functions": functions,
        "line_mappings": [],
        "instruction_alignments": [],
        "basic_block_alignments": [],
        "variables": [],
        "data_objects": [],
        "call_graph": {"nodes": [], "edges": [], "classification": "HEURISTIC"},
        "coverage": {"functions": len(functions), "functions_with_source": len(functions), "instructions": 0, "instructions_mapped": 0},
        "classifications": {"EXACT": len(functions)},
        "transfer": transfer,
    }


def test_task_builder_creates_eligible_tasks_and_purpose_unavailable(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    reports = tmp_path / "reports"
    source = root / "source"
    source.mkdir(parents=True)
    (source / "demo.c").write_text("int add(int a, int b) {\n    return a + b;\n}\n\nint main(void) {\n    return add(1, 2);\n}\n", encoding="utf-8")
    (root / "catalog.json").write_text(
        json.dumps([{"project_id": "demo", "name": "demo/demo", "category": "utilities", "status": "accepted", "source_root": "source", "license": "MIT", "commit": "a" * 40, "source_archive_sha256": _sha("archive")}]),
        encoding="utf-8",
    )
    alignments = root / "alignments"
    variants = {
        "gcc_O0_debug_pie_executable": _index("gcc_O0_debug_pie_executable", [_function("add", 1, 0x1000, "gcc_O0_debug_pie_executable"), _function("main", 5, 0x1100, "gcc_O0_debug_pie_executable")]),
        "gcc_O2_debug_pie_executable": _index("gcc_O2_debug_pie_executable", [_function("add", 1, 0x2000, "gcc_O2_debug_pie_executable"), _function("main", 5, 0x2100, "gcc_O2_debug_pie_executable")]),
        "clang_O0_debug_pie_executable": _index("clang_O0_debug_pie_executable", [_function("add", 1, 0x3000, "clang_O0_debug_pie_executable"), _function("main", 5, 0x3100, "clang_O0_debug_pie_executable")]),
        "clang_O2_debug_pie_executable": _index("clang_O2_debug_pie_executable", [_function("add", 1, 0x4000, "clang_O2_debug_pie_executable"), _function("main", 5, 0x4100, "clang_O2_debug_pie_executable")]),
        "gcc_O2_stripped_pie_executable": _index("gcc_O2_stripped_pie_executable", [_function("add", 1, 0x2000, "gcc_O2_stripped_pie_executable"), _function("main", 5, 0x2100, "gcc_O2_stripped_pie_executable")], {"method": "text-identity"}),
    }
    entries = {"indexes": [], "stripped": []}
    for variant, payload in variants.items():
        path = alignments / "demo" / variant / "alignment_index.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload), encoding="utf-8")
        entry = {"variant_id": variant, "status": "available", "alignment_index": str(path.relative_to(root))}
        if "_stripped_" in variant:
            entries["stripped"].append(entry)
        else:
            entries["indexes"].append(entry)
    (alignments / "corpus_alignment_summary.json").write_text(json.dumps({"projects": [{"project_id": "demo", **entries}]}), encoding="utf-8")
    disassembly = root / "builds" / "demo" / "gcc_O0_debug_pie_executable" / "disassembly.txt"
    disassembly.parent.mkdir(parents=True, exist_ok=True)
    disassembly.write_text("\n".join(f"  {index:04x}:\t48 89 e5\tmov %rsp,%rbp" for index in range(40)), encoding="utf-8")
    report = build_task_datasets(root, reports)
    assert report["tasks"]["cross_compiler_function_matching"]["samples"] > 0
    assert report["tasks"]["cross_optimization_function_matching"]["samples"] > 0
    assert report["tasks"]["source_to_binary_alignment"]["samples"] == 8
    assert report["tasks"]["function_boundary_detection"]["samples"] == 1
    assert report["tasks"]["instruction_semantics"]["samples"] > 0
    assert report["tasks"]["type_dataflow_inference"]["eligibility"] == "PARTIAL"
    assert report["tasks"]["function_purpose_inference"]["eligibility"] == "UNAVAILABLE"
    assert report["tasks"]["function_purpose_inference"]["samples"] == 0

