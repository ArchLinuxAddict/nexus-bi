import shutil
import subprocess

import pytest

from nexus.architectures import adapter_for_elf
from nexus.architectures.abi import infer_abi_signature
from nexus.elf.parser import parse_elf
from nexus.program.bundle import analyze_binary
from nexus.reconstruction.c.generator import reconstruct_c
from nexus.verification.compiler import compile_c_project

SOURCE = """
int add(int a, int b) { return a + b; }
int mul(int a, int b) { return a * b; }
int sub(int a, int b) { return a - b; }
int call_add(int a, int b) { return add(a, b); }
""".strip()

TARGETS = {
    "aarch64": "aarch64-linux-gnu",
    "armv7": "armv7-linux-gnueabihf",
    "riscv": "riscv64-linux-gnu",
    "riscv32": "riscv32-linux-gnu",
    "x86": "i386-linux-gnu",
}


def _compile(tmp_path, triple):
    if not shutil.which("clang") or not shutil.which("llvm-objdump"):
        pytest.skip("clang or llvm-objdump is unavailable")
    source = tmp_path / "multi.c"
    source.write_text(SOURCE + "\n", encoding="utf-8")
    output = tmp_path / f"{triple.split('-')[0]}.o"
    result = subprocess.run(
        ["clang", f"--target={triple}", "-O1", "-fno-omit-frame-pointer", "-c", str(source), "-o", str(output)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        pytest.skip(f"cross compilation failed: {result.stderr[:300]}")
    return output


@pytest.mark.parametrize("expected,triple", TARGETS.items())
def test_cross_architecture_analysis(tmp_path, expected, triple):
    binary = _compile(tmp_path, triple)
    elf = parse_elf(binary)
    adapter = adapter_for_elf(elf)
    assert adapter.name == expected
    assert adapter.calling_convention_register_count if False else adapter.argument_registers
    bundle = analyze_binary(binary)
    assert bundle.disassembly is not None
    assert bundle.disassembly.instructions
    assert len(bundle.functions) >= 2
    names = {function.name for function in bundle.functions}
    assert {"add", "mul", "sub", "call_add"} <= names
    add = bundle.function_by_name("add")
    assert add is not None
    signature = bundle.signatures[add.function_id]
    assert signature["calling_convention"] == adapter.abi
    assert len(signature["parameters"]) >= 1
    assert signature["return_type"] in {"uint32_t", "uint64_t"}
    # The generic translator recovers at least simple arithmetic for ARM/AArch64/RISC-V.
    if expected in {"aarch64", "armv7", "riscv"}:
        result = reconstruct_c(bundle, out_dir=tmp_path / "project", project_name="multi")
        assert any(item["classification"] in {"partial_behavioral_reconstruction", "ghidra_decompiler_reconstruction"} for item in result["functions"])
        assert compile_c_project(tmp_path / "project")["status"] == "passed"


def test_abi_signature_reads_only_incoming_registers(tmp_path):
    binary = _compile(tmp_path, "aarch64-linux-gnu")
    bundle = analyze_binary(binary)
    function = bundle.function_by_name("add")
    assert function is not None
    signature = infer_abi_signature(function, adapter_for_elf(bundle.elf))
    assert signature["parameters"]
    for parameter in signature["parameters"]:
        assert parameter["register"] in {"a64_x0", "a64_x1", "a64_x2", "a64_x3", "a64_x4", "a64_x5", "a64_x6", "a64_x7"}
