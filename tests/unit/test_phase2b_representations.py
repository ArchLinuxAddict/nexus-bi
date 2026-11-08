import hashlib
import json
from pathlib import Path

from nexus.representations.coverage import collect_representation_coverage


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def test_representation_coverage_separates_raw_and_derived(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    reports = tmp_path / "reports"
    source = root / "corpus" / "demo" / "source"
    source.mkdir(parents=True)
    (source / "demo.c").write_text("int main(void){return 0;}\n", encoding="utf-8")
    archive = root / "corpus" / "demo" / "source.tar.gz"
    archive.write_bytes(b"archive")
    build_dir = root / "builds" / "demo" / "gcc_O0_debug_pie_executable"
    build_dir.mkdir(parents=True)
    binary = build_dir / "demo"
    binary.write_bytes(b"binary")
    for name in ("build.log", "symbols.txt", "disassembly.txt"):
        (build_dir / name).write_text("x", encoding="utf-8")
    (build_dir / "build_metadata.json").write_text("{}", encoding="utf-8")
    alignment = root / "alignments" / "demo" / "gcc_O0_debug_pie_executable" / "alignment_index.json"
    alignment.parent.mkdir(parents=True)
    alignment.write_text(json.dumps({"functions": [{"name": "main"}], "basic_block_alignments": [{"classification": "HEURISTIC"}], "instruction_alignments": [{}], "call_graph": {"nodes": [], "edges": []}, "variables": [{}], "data_objects": [{}]}), encoding="utf-8")
    (root / "catalog.json").write_text(json.dumps([{"project_id": "demo", "name": "demo", "category": "utilities", "status": "accepted", "source_archive_path": "corpus/demo/source.tar.gz", "source_root": "corpus/demo/source", "source_archive_sha256": _sha(b"archive")}]), encoding="utf-8")
    (root / "build_matrix_report.json").write_text(
        json.dumps(
            {
                "projects": [
                    {
                        "project_id": "demo",
                        "builds": [
                            {
                                "status": "built",
                                "artifact_path": "builds/demo/gcc_O0_debug_pie_executable/demo",
                                "artifact_sha256": _sha(b"binary"),
                                "variant": {"variant_id": "gcc_O0_debug_pie_executable"},
                                "elf": {"status": "ok"},
                                "extras": {"assembly": ["a.s"], "objects": [{"object": "a.o"}], "ir": [], "dwarf": ["dwarf.txt"]},
                            }
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    report = collect_representation_coverage(root, reports)
    assert report["projects"] == 1
    assert report["builds"] == 1
    assert report["raw_representations"]["binary"] == 1.0
    assert report["raw_representations"]["source_archive"] == 1.0
    assert report["derived_representations"]["functions"] == 1.0
    assert report["derived_representations"]["cfg"] == 1.0
    assert report["derived_representations"]["call_graph"] == 1.0
    assert report["status"] == "passed"
