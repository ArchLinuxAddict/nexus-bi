import shutil
import subprocess
from pathlib import Path

import pytest

from nexus.alignment.pipeline import extract_alignment, transfer_alignment
from nexus.elf.parser import parse_elf

SOURCE = """int g_counter = 9;
static __attribute__((noinline)) int helper(int x) {
    return x * 3;
}
__attribute__((noinline)) int classify(int value) {
    if (value < 0) {
        return -1;
    }
    return helper(value);
}
int main(void) {
    return classify(g_counter);
}
"""


def _compile(tmp_path: Path, debug: bool, name: str = "fixture") -> Path:
    source = tmp_path / "fixture.c"
    source.write_text(SOURCE, encoding="utf-8")
    binary = tmp_path / name
    command = ["gcc", "-O0", "-std=gnu11", "-fno-omit-frame-pointer", "-fno-pie", "-no-pie", "-g3" if debug else "-g0", str(source), "-o", str(binary)]
    result = subprocess.run(command, capture_output=True, text=True, timeout=60)
    if result.returncode != 0:
        pytest.skip(f"gcc unavailable or failed: {result.stderr[-200:]}")
    return binary


def _text_sha(path: Path) -> str | None:
    elf = parse_elf(path)
    section = elf.section_by_name(".text")
    if section is None:
        return None
    import hashlib

    return hashlib.sha256(elf.raw[section.offset : section.offset + section.size]).hexdigest()


def test_dwarf_alignment_levels_and_honesty(tmp_path: Path) -> None:
    binary = _compile(tmp_path, debug=True)
    index = extract_alignment(binary, tmp_path, project_id="fixture")
    assert index["status"] == "available"
    assert index["dwarf_status"] == "available"
    functions = {item["name"]: item for item in index["functions"]}
    for name in ("helper", "classify", "main"):
        assert name in functions
        assert functions[name]["source_file"] == "fixture.c"
        assert functions[name]["source_line"] is not None
        assert functions[name]["classification"] in {"EXACT", "DEBUG_DERIVED"}
    assert index["coverage"]["instructions_mapped"] > 0
    assert index["coverage"]["functions_with_source"] >= 3
    assert all(item["classification"] == "DEBUG_DERIVED" for item in index["instruction_alignments"])
    assert all(item["classification"] != "EXACT" for item in index["instruction_alignments"])
    assert any(item["name"] == "g_counter" for item in index["data_objects"])
    assert any(item["name"] == "value" for item in index["variables"])
    assert index["call_graph"]["edges"]


def test_stripped_transfer_is_text_identity_only(tmp_path: Path) -> None:
    debug_binary = _compile(tmp_path, debug=True)
    stripped_binary = tmp_path / "fixture_stripped"
    shutil.copy2(debug_binary, stripped_binary)
    subprocess.run(["strip", "--strip-all", str(stripped_binary)], check=True, timeout=30)
    debug_index = extract_alignment(debug_binary, tmp_path, project_id="fixture")
    transferred = transfer_alignment(debug_index, stripped_binary, debug_binary)
    assert transferred["status"] == "available"
    assert transferred["transfer"]["method"] == "text-identity"
    assert transferred["transfer"]["verified"] is True
    assert any(item["name"] == "classify" for item in transferred["functions"])
    assert all(item["classification"] == "DEBUG_DERIVED" for item in transferred["instruction_alignments"])
    assert _text_sha(debug_binary) == _text_sha(stripped_binary)


def test_symbol_only_does_not_claim_source_alignment(tmp_path: Path) -> None:
    binary = _compile(tmp_path, debug=False)
    index = extract_alignment(binary, tmp_path, project_id="fixture")
    assert index["status"] == "available"
    assert index["coverage"]["instructions_mapped"] == 0
    assert not index["instruction_alignments"]
    assert all(item["source_file"] is None for item in index["functions"])
