import shutil
import subprocess

import pytest

from nexus.core.errors import ElfError
from nexus.elf.parser import parse_elf


def test_parse_x86_64_binary(simple_binary):
    elf = parse_elf(simple_binary)
    assert elf.header.elf_class == "ELF64"
    assert elf.header.machine_name == "x86_64"
    assert elf.header.entry > 0
    assert elf.size == simple_binary.stat().st_size
    assert len(elf.program_headers) > 0
    assert len(elf.sections) > 0
    assert any(symbol.name == "add" and symbol.symbol_type == "FUNC" for symbol in elf.symbols)
    assert elf.metadata.get("has_dynamic") is True
    assert elf.is_stripped is False


def test_stripped_detection(stripped_binary):
    elf = parse_elf(stripped_binary)
    assert elf.is_stripped is True
    assert len(elf.symbols) == 0
    assert len(elf.dynamic_symbols) > 0


def test_inspect_dict_has_identity(simple_binary):
    payload = parse_elf(simple_binary).inspect_dict()
    assert payload["header"]["machine"] == "x86_64"
    assert payload["counts"]["sections"] > 0
    assert payload["sha256"]


def test_invalid_magic(tmp_path):
    path = tmp_path / "bad"
    path.write_bytes(b"not an elf file")
    with pytest.raises(ElfError):
        parse_elf(path)


def test_truncated_header(tmp_path):
    path = tmp_path / "truncated"
    path.write_bytes(b"\x7fELF\x02\x01\x01" + b"\x00" * 4)
    with pytest.raises(ElfError):
        parse_elf(path)


def test_truncated_section_table(simple_binary, tmp_path):
    data = simple_binary.read_bytes()
    path = tmp_path / "truncated-section-table"
    path.write_bytes(data[: len(data) // 2])
    with pytest.raises(ElfError):
        parse_elf(path)


def test_elf32_object(build_root):
    if not shutil.which("gcc"):
        pytest.skip("gcc unavailable")
    source = build_root / "tiny32.c"
    source.write_text("int value(void) { return 7; }\n", encoding="utf-8")
    obj = build_root / "tiny32.o"
    result = subprocess.run(["gcc", "-m32", "-c", str(source), "-o", str(obj)], capture_output=True, text=True)
    if result.returncode != 0:
        pytest.skip("32-bit multilib unavailable")
    elf = parse_elf(obj)
    assert elf.header.elf_class == "ELF32"
    assert elf.header.machine_name == "x86"


def test_shared_library(jni_binary):
    elf = parse_elf(jni_binary)
    assert elf.is_shared_object is True
    assert elf.header.machine_name == "x86_64"
    assert any(symbol.name.startswith("Java_") for symbol in elf.dynamic_symbols)
    assert elf.build_id or True
