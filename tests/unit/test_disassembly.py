import pytest

from nexus.core.errors import DisassemblyError
from nexus.disassembly.objdump import disassemble_elf
from nexus.elf.parser import parse_elf


def test_disassemble_add(simple_binary):
    elf = parse_elf(simple_binary)
    disassembly = disassemble_elf(elf)
    assert disassembly.instructions
    assert "add" in disassembly.functions
    add_instructions = [item for item in disassembly.instructions if item.function_label == "add"]
    assert add_instructions
    assert all(item.length == len(item.bytes_hex) // 2 for item in add_instructions)
    moves = [item for item in add_instructions if item.mnemonic == "mov" and "DWORD PTR" in item.text]
    assert moves
    assert any(effect.width == 32 for item in moves for effect in item.memory)
    assert any(item.mnemonic == "add" and "rax" in item.writes for item in add_instructions)


def test_call_target(simple_binary):
    elf = parse_elf(simple_binary)
    disassembly = disassemble_elf(elf)
    calls = [item for item in disassembly.instructions if item.mnemonic == "call" and item.targets and item.targets[0].va]
    assert calls
    assert any(item.function_label == "main" for item in calls)


def test_unsupported_architecture(build_root):
    import shutil
    import subprocess

    if not shutil.which("clang"):
        pytest.skip("clang unavailable")
    source = build_root / "cross.c"
    source.write_text("int f(int x) { return x + 1; }\n", encoding="utf-8")
    obj = build_root / "cross-aarch64.o"
    result = subprocess.run(["clang", "-target", "aarch64-linux-gnu", "-O2", "-c", str(source), "-o", str(obj)], capture_output=True, text=True)
    if result.returncode != 0:
        pytest.skip("cross target unavailable")
    elf = parse_elf(obj)
    with pytest.raises(DisassemblyError):
        disassemble_elf(elf)
