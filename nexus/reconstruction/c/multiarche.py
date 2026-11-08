"""Conservative multi-architecture C reconstruction.

This is intentionally narrower than the x86-64 translator.  It recovers
straight-line register arithmetic, register moves, simple loads/stores and a
single return value for AAPCS32/AAPCS64/RISC-V functions.  Anything involving
flags, calls, indirect memory, vectors, PC-relative addressing or unresolved
registers becomes an explicit unresolved stub with evidence.

The generated project compiles on the host because the output is plain C.  It
does not prove behavioral equivalence or the original source structure.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from nexus.architectures import get_adapter
from nexus.disassembly.llvm_objdump import _canonical_reg
from nexus.program.bundle import AnalysisBundle
from nexus.program.model import FunctionCandidate
from nexus.reconstruction.c.ghidra import file_prefix, support_header
from nexus.toolchain.adapters import GhidraAdapter


@dataclass
class MultiTranslation:
    success: bool
    return_expr: str | None
    statements: list[tuple[str, int]]
    reason: str | None
    width: int


def _sanitize(name: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_]", "_", name or "reconstructed")
    if not value:
        value = "reconstructed"
    if value[0].isdigit():
        value = f"fn_{value}"
    return value


def _adapter_name(bundle: AnalysisBundle) -> str:
    return str((bundle.architecture or {}).get("name") or bundle.elf.header.machine_name)


def _register(adapter_name: str, text: str) -> str | None:
    arch = "arm" if adapter_name == "armv7" else adapter_name
    return _canonical_reg(arch, text.strip())


def _immediate(text: str) -> int | None:
    value = text.strip().rstrip(",")
    value = value.lstrip("#").strip()
    try:
        return int(value, 0)
    except ValueError:
        return None


def _address_expression(text: str, regs: dict[str, str], adapter_name: str, width: int) -> str | None:
    # x86 AT&T/SIB or bracket forms.
    match = re.match(r"\s*([^()]*?)\(([^()]*)\)\s*$", text.replace("[", "(").replace("]", ")"))
    if match:
        displacement_text = match.group(1).strip()
        components = [part.strip() for part in match.group(2).split(",")]
        base_text = components[0] if components else ""
        index_text = components[1] if len(components) > 1 else ""
        scale_text = components[2] if len(components) > 2 else "1"
        base = _register(adapter_name, base_text) if base_text else None
        index = _register(adapter_name, index_text) if index_text else None
        if base is None and index is None:
            return None
        terms = []
        if base and base in regs:
            terms.append(regs[base])
        if index and index in regs:
            try:
                scale = int(scale_text, 0)
            except ValueError:
                scale = 1
            terms.append(f"({regs[index]} * {scale})" if scale != 1 else regs[index])
        if displacement_text:
            immediate = _immediate(displacement_text)
            if immediate:
                terms.append(str(immediate))
        if not terms:
            return None
        return _cast(" + ".join(terms), width)
    # Bracket form [base, #disp].
    memory = _memory(text)
    if memory:
        base_name, displacement = memory
        base = _register(adapter_name, base_name)
        if base is None or base not in regs:
            return None
        terms = [regs[base]]
        if displacement:
            terms.append(str(displacement))
        return _cast(" + ".join(terms), width)
    return None


def _memory(text: str) -> tuple[str, int] | None:
    match = re.search(r"\[([^\]]+)\]", text)
    parenthesized = re.search(r"(-?(?:0x[0-9a-fA-F]+|\d+))?\s*\(\s*([a-zA-Z0-9$%]+)\s*\)", text)
    if not match and parenthesized:
        displacement = int(parenthesized.group(1), 0) if parenthesized.group(1) else 0
        return parenthesized.group(2), displacement
    if not match:
        return None
    parts = [part.strip() for part in match.group(1).split(",")]
    if not parts:
        return None
    base = parts[0]
    displacement = 0
    for part in parts[1:]:
        immediate = _immediate(part)
        if immediate is not None:
            displacement += immediate
        elif part:
            return None
    return base, displacement


def _width_for_register(name: str, default: int) -> int:
    text = name.lower()
    if re.match(r"(?:w|r(?:[0-9]|[12][0-9]|3[01])|arm_r|a64_x|rv_|x)\d+", text):
        if text.startswith("w"):
            return 32
        if text.startswith("arm_r") or text.startswith("rv_"):
            return default
        if re.match(r"x\d+", text):
            return 64
        return default
    return default


def _cast(expr: str, width: int) -> str:
    return f"((uint{width}_t)({expr}))"


def _binary(left: str, operator: str, right: str, width: int) -> str:
    return _cast(f"({left} {operator} {right})", width)


def _normalize_mnemonic(adapter_name: str, mnemonic: str) -> str:
    mn = mnemonic.lower()
    if adapter_name in {"x86", "x86_64"}:
        base = mn[:-1] if len(mn) > 3 and mn[-1] in {"b", "w", "l", "q"} else mn
        if base in {"mov", "add", "sub", "and", "or", "xor", "imul", "shl", "shr", "sar", "lea", "inc", "dec", "neg", "not", "cmp", "test", "push", "pop", "ret", "call", "jmp"}:
            return base
        if mn == "retq":
            return "ret"
    return mn


def _translate_instruction(
    adapter_name: str,
    mnemonic: str,
    operands: list[str],
    regs: dict[str, str],
    pointer_width: int,
    address: int,
    statements: list[tuple[str, int]],
) -> str | None:
    """Return a failure reason, or None when the instruction was handled."""
    mn = mnemonic.lower()
    try:
        adapter = get_adapter(adapter_name)
        stack_registers = {adapter.stack_pointer, adapter.frame_pointer, adapter.link_register, *adapter.callee_saved}
        stack_registers.discard(None)
    except Exception:
        stack_registers = set()
    if mn.startswith("nop") or mn in {"bti", "pacibsp", "endbr64", "endbr32", "hint", "autibsp"}:
        return None
    if mn in {"ret", "retq", "bx", "jr", "jr ra"}:
        return None
    if mn in {"push", "pop", "pushq", "popq", "pushl", "popl"}:
        return None
    if mn in {"ldr", "ld", "lw", "lh", "lb", "lbu", "lhu", "lwu", "str", "st", "sw", "sh", "sb", "sd"} and len(operands) >= 2:
        destination = _register(adapter_name, operands[0])
        memory = _memory(operands[1])
        if memory:
            base = _register(adapter_name, memory[0])
            if base in stack_registers and (destination in stack_registers or destination is None):
                return None
    if mn in {"stp", "ldp", "sub", "subi", "add", "addi", "mov"} and operands:
        # Stack-frame setup is ignored only when it touches the stack/frame/link
        # register and does not define a value used by the computation.
        first = _register(adapter_name, operands[0]) if operands else None
        stack_like = {"a64_sp", "a64_fp", "a64_lr", "arm_sp", "arm_r11", "arm_lr", "rv_sp", "rv_s0", "rv_ra"}
        if first in stack_like:
            return None
        if mn == "stp" and any(_memory(operand) for operand in operands):
            return None
    if mn in {"add", "addi", "sub", "subi", "mov"} and len(operands) >= 2:
        destination = _register(adapter_name, operands[0])
        source = _register(adapter_name, operands[1])
        if destination and source and destination in {adapter.frame_pointer if 'adapter' in locals() else None, adapter.link_register if 'adapter' in locals() else None} and source == (adapter.stack_pointer if 'adapter' in locals() else None):
            return None
    if mn == "lea" and len(operands) == 2:
        destination = _register(adapter_name, operands[0])
        if destination is None:
            return "unsupported lea destination"
        width = _width_for_register(destination, pointer_width)
        expression = _address_expression(operands[1], regs, adapter_name, width)
        if expression is None:
            return f"lea address at 0x{address:x} is unresolved"
        regs[destination] = expression
        return None
    if mn in {"cmp", "cmn", "tst", "teq", "ccmp", "fcmp"}:
        return None
    if mn in {"b", "bl", "br", "blr", "blx", "jal", "jalr", "call", "callq", "jmp", "jmpq"} or mn.startswith(("b.", "cb", "tb", "beq", "bne", "blt", "bge", "ble", "bgt", "bhi", "bls", "bcc", "bcs", "bmi", "bpl", "bvs", "bvc", "bhs", "blo", "loop")):
        return f"control transfer {mn} is not translated"
    if mn in {"svc", "swi", "ecall", "syscall", "int3", "ud2", "ebreak"}:
        return f"system instruction {mn} is not translated"
    if mn.startswith(("ldp", "stp")) or "vector" in mn or mn.startswith("v"):
        return f"vector/paired operation {mn} is not translated"
    if mn in {"adrp", "adr", "auipc", "lui", "lui_"}:  # PC-relative or upper-immediate
        return f"PC-relative immediate {mn} is not translated"

    if mn in {"mov", "mv", "movz", "li"} or mn.startswith("mov"):
        if len(operands) < 2:
            return f"unsupported {mn} form"
        destination = _register(adapter_name, operands[0])
        if destination is None:
            return f"unsupported destination in {mn}"
        source = _register(adapter_name, operands[1])
        if source is not None:
            value = regs.get(source)
            if value is None:
                return f"source register {source} is unresolved"
        else:
            immediate = _immediate(operands[1])
            if immediate is None:
                return f"unsupported source in {mn}"
            value = str(immediate)
        width = _width_for_register(destination, pointer_width)
        regs[destination] = _cast(value, width) if source is not None else f"((uint{width}_t){value})"
        return None

    if mn in {"add", "addi", "addw", "sub", "subi", "subw", "mul", "imul", "mulw", "madd", "and", "andi", "orr", "or", "ori", "eor", "xor", "xori", "lsl", "sll", "slli", "sllw", "lsr", "srl", "srli", "srlw", "asr", "sra", "srai", "sraw", "shl", "shr", "sar"}:
        if len(operands) < 3:
            if adapter_name in {"x86", "x86_64"} and len(operands) == 2:
                operands = [operands[0], operands[0], operands[1]]
            else:
                return f"unsupported {mn} form"
        destination = _register(adapter_name, operands[0])
        left_register = _register(adapter_name, operands[1])
        if destination is None or left_register is None:
            return f"unsupported {mn} operands"
        left = regs.get(left_register)
        if left is None:
            return f"source register {left_register} is unresolved"
        right_register = _register(adapter_name, operands[2])
        if right_register is not None:
            right = regs.get(right_register)
            if right is None:
                return f"source register {right_register} is unresolved"
        else:
            immediate = _immediate(operands[2])
            if immediate is None:
                return f"unsupported {mn} source"
            right = str(immediate)
        operator = {
            "add": "+", "addi": "+", "addw": "+", "sub": "-", "subi": "-", "subw": "-",
            "mul": "*", "imul": "*", "mulw": "*", "madd": "*",
            "and": "&", "andi": "&", "orr": "|", "or": "|", "ori": "|", "eor": "^", "xor": "^", "xori": "^",
            "lsl": "<<", "sll": "<<", "slli": "<<", "sllw": "<<", "shl": "<<", "lsr": ">>", "srl": ">>", "srli": ">>",
            "srlw": ">>", "asr": ">>", "sra": ">>", "srai": ">>", "sraw": ">>", "shr": ">>", "sar": ">>",
        }[mn]
        width = _width_for_register(destination, pointer_width)
        regs[destination] = _binary(left, operator, right, width)
        return None

    if mn in {"addi"} and len(operands) == 3:
        return None

    if mn in {"ldr", "ldrb", "ldrh", "ldrsb", "ldrsh", "ldrsw", "ld", "lw", "lh", "lb", "lbu", "lhu", "lwu"}:
        if len(operands) < 2:
            return f"unsupported {mn} form"
        destination = _register(adapter_name, operands[0])
        if destination is None:
            return f"unsupported load destination in {mn}"
        memory = _memory(operands[1])
        if memory is None:
            return f"load address at 0x{address:x} is not a simple base+displacement"
        base_name, displacement = memory
        base = _register(adapter_name, base_name)
        if base is None or base not in regs:
            return f"load base register is unresolved at 0x{address:x}"
        width = {"ldrb": 8, "ldrh": 16, "ldrsb": 8, "ldrsh": 16, "ldrsw": 32, "lb": 8, "lh": 16, "lw": 32, "ld": 64, "lbu": 8, "lhu": 16, "lwu": 32}.get(mn, _width_for_register(destination, pointer_width))
        expression = f"(*((uint{width}_t *)({regs[base]} + {displacement})))"
        regs[destination] = _cast(expression, width)
        return None

    if mn in {"str", "strb", "strh", "st", "sw", "sh", "sb"}:
        if len(operands) < 2:
            return f"unsupported {mn} form"
        source = _register(adapter_name, operands[0])
        memory = _memory(operands[1])
        if source is None or memory is None:
            return f"unsupported store form at 0x{address:x}"
        base_name, displacement = memory
        base = _register(adapter_name, base_name)
        if base is None or base not in regs or source not in regs:
            return f"store operand is unresolved at 0x{address:x}"
        width = {"strb": 8, "strh": 16, "sb": 8, "sh": 16, "sw": 32, "st": 64}.get(mn, _width_for_register(source, pointer_width))
        expression = f"(*((uint{width}_t *)({regs[base]} + {displacement})))"
        statements.append((f"{expression} = {regs[source]}", address))
        return None

    if mn in {"sxtw", "uxtw", "sext.w", "zext.w", "sxth", "uxth", "sxtb", "uxtb"}:
        if len(operands) < 2:
            return f"unsupported {mn} form"
        destination = _register(adapter_name, operands[0])
        source = _register(adapter_name, operands[1])
        if destination is None or source is None or source not in regs:
            return f"unresolved extension at 0x{address:x}"
        width = _width_for_register(destination, pointer_width)
        regs[destination] = _cast(regs[source], width)
        return None

    return f"instruction {mn} at 0x{address:x} is not translated"


def translate_function(bundle: AnalysisBundle, function: FunctionCandidate) -> MultiTranslation:
    adapter_name = _adapter_name(bundle)
    try:
        adapter = get_adapter(adapter_name)
    except Exception:
        return MultiTranslation(False, None, [], f"no ABI adapter for {adapter_name}", 64)
    signature = (bundle.signatures or {}).get(function.function_id, {})
    pointer_width = int(signature.get("pointer_width") or adapter.pointer_width)
    regs: dict[str, str] = {}
    for index, parameter in enumerate(signature.get("parameters") or []):
        register = parameter.get("register")
        if register:
            regs[register] = f"arg{index}"
    statements: list[tuple[str, int]] = []
    ordered: dict[int, Any] = {}
    for block in function.blocks:
        for instruction in block.instructions:
            ordered[instruction.address] = instruction
    if not ordered:
        for instruction in function.instructions:
            ordered[instruction.address] = instruction
    instructions = [ordered[address] for address in sorted(ordered)]
    if any(instruction.control_flow not in {"fallthrough", "return"} for instruction in instructions):
        return MultiTranslation(False, None, statements, "branches or calls are not supported by the simple multi-architecture translator", pointer_width)
    for instruction in instructions:
        operands = instruction.operands
        mnemonic = _normalize_mnemonic(adapter_name, instruction.mnemonic)
        if adapter_name in {"x86", "x86_64"} and mnemonic in {"mov", "add", "sub", "and", "or", "xor", "imul", "shl", "shr", "sar", "lea"} and len(operands) == 2:
            operands = [operands[1], operands[0]]
        reason = _translate_instruction(
            adapter_name,
            mnemonic,
            operands,
            regs,
            pointer_width,
            instruction.address,
            statements,
        )
        if reason:
            return MultiTranslation(False, None, statements, reason, pointer_width)
    return_register = adapter.return_register
    return_expression = regs.get(return_register) if return_register else None
    if signature.get("return_type") == "void":
        return MultiTranslation(True, None, statements, None, pointer_width)
    if return_expression is None:
        return MultiTranslation(False, None, statements, "return value is not proven", pointer_width)
    return MultiTranslation(True, return_expression, statements, None, pointer_width)


def _comment(function: FunctionCandidate, bundle: AnalysisBundle) -> str:
    architecture = _adapter_name(bundle)
    evidence = ", ".join(f"0x{instruction.address:x}" for instruction in function.instructions[:10])
    return (
        f"/* NEXUS-BI approximate {architecture} reconstruction of {function.name}.\n"
        f" * Entry: 0x{function.entry:x}; ABI: {(bundle.architecture or {}).get('abi')}.\n"
        f" * Evidence: {evidence or 'no instructions'}.\n"
        f" * Straight-line translation only; not behaviorally verified.\n"
        f" */\n"
    )


def reconstruct_c_multiarche(
    bundle: AnalysisBundle,
    function_ids: list[str] | None = None,
    out_dir: str | Path | None = None,
    project_name: str = "nexus_reconstruction",
) -> dict[str, Any]:
    selected = bundle.functions
    if function_ids:
        ids = set(function_ids)
        selected = [function for function in bundle.functions if function.function_id in ids or function.name in ids]
    else:
        selected = [function for function in bundle.functions if not function.name.startswith("_")]
    if not selected:
        selected = bundle.functions[:1]
    names: dict[str, str] = {}
    used: set[str] = set()
    for function in selected:
        name = _sanitize(function.name)
        if name == "main":
            name = "reconstructed_main"
        candidate = name
        suffix = 2
        while candidate in used:
            candidate = f"{name}_{suffix}"
            suffix += 1
        used.add(candidate)
        names[function.function_id] = candidate
    ghidra_result = GhidraAdapter().decompile(bundle.path) if getattr(bundle, "path", None) else None
    ghidra = {
        "status": ghidra_result.status if ghidra_result else "unavailable",
        "functions": (ghidra_result.artifacts.get("functions") or []) if ghidra_result else [],
        "tool_evidence_id": ghidra_result.to_dict().get("evidence_id") if ghidra_result else None,
        "elapsed_seconds": ghidra_result.artifacts.get("elapsed_seconds") if ghidra_result else None,
        "cached": ghidra_result.cached if ghidra_result else None,
        "reason": ghidra_result.error if ghidra_result else "no binary path",
    }
    ghidra_functions = ghidra.get("functions") or []
    ghidra_by_entry = {int(item["entry"]): item for item in ghidra_functions}
    header = [
        "#ifndef NEXUS_RECONSTRUCTED_H",
        "#define NEXUS_RECONSTRUCTED_H",
        "",
        "#include <stdint.h>",
        "#include <stddef.h>",
        "",
        f"/* Generated by NEXUS-BI for {(bundle.architecture or {}).get('name')} ({(bundle.architecture or {}).get('abi')}). */",
        f"/* Reconstruction backend: {'ghidra' if ghidra_functions else 'multiarche'} */",
        "",
    ]
    if ghidra_functions:
        header.extend(support_header(ghidra_functions)[2:-2])
    files: dict[str, str] = {}
    results = []
    evidence = {
        "binary_sha256": bundle.elf.sha256,
        "architecture": bundle.architecture,
        "project": project_name,
        "backend": "ghidra" if ghidra_functions else "multiarche",
        "ghidra": {key: ghidra.get(key) for key in ("status", "elapsed_seconds", "cached", "reason", "tool_evidence_id")},
        "functions": [],
        "limitations": [
            "Multi-architecture reconstruction covers straight-line arithmetic, moves and simple loads/stores only.",
            "Flags, calls, branches, vectors, PC-relative addressing and indirect memory are unresolved.",
            "Generated C compiles on the host but has not been executed against original behavior.",
        ],
    }
    for function in selected:
        signature = (bundle.signatures or {}).get(function.function_id, {})
        return_type = str(signature.get("return_type") or "void")
        if return_type == "unknown":
            return_type = "uintptr_t"
        parameters = ", ".join(
            f"{parameter.get('type') or 'uintptr_t'} {parameter.get('name') or f'arg{index}'}"
            for index, parameter in enumerate(signature.get("parameters") or [])
        ) or "void"
        name = names[function.function_id]
        ghidra_function = ghidra_by_entry.get(function.entry)
        if ghidra_function is not None:
            ghidra_source = str(ghidra_function.get("source") or "")
            body = "\n".join(file_prefix(str(ghidra_function.get("name") or ""), ghidra_functions, f"{project_name}.h", ghidra_source)) + "\n" + ghidra_source + "\n"
            files[f"src/{name}.c"] = body
            results.append(
                {
                    "function_id": function.function_id,
                    "name": function.name,
                    "generated_name": name,
                    "classification": "ghidra_decompiler_reconstruction",
                    "confidence": 0.7,
                    "reason": None,
                    "parameter_count": len(signature.get("parameters") or []),
                    "evidence": [f"ev_{bundle.elf.sha256[:12]}_func_{function.entry:x}", "ghidra_decompiler"],
                }
            )
            evidence["functions"].append(
                {
                    "function_id": function.function_id,
                    "name": function.name,
                    "source_file": f"src/{name}.c",
                    "classification": "ghidra_decompiler_reconstruction",
                    "confidence": 0.7,
                    "abi": signature.get("calling_convention"),
                    "register_model": signature.get("register_model"),
                    "ranges": [{"start": f"0x{start:x}", "end": f"0x{end:x}"} for start, end in function.ranges],
                    "backend": "ghidra",
                    "ghidra_entry": ghidra_function.get("entry_va"),
                }
            )
            continue
        header.append(f"{return_type} {name}({parameters});")
        translation = translate_function(bundle, function)
        if translation.success:
            lines = [_comment(function, bundle), f"{return_type} {name}({parameters}) {{"]
            for statement, address in translation.statements:
                lines.append(f"    /* 0x{address:x} */ {statement};")
            if return_type == "void":
                lines.append("    return;")
            else:
                lines.append(f"    return ({return_type})({translation.return_expr});")
            lines.append("}")
            body = "\n".join(lines) + "\n"
            classification = "partial_behavioral_reconstruction"
            confidence = 0.4
        else:
            lines = [_comment(function, bundle), f"{return_type} {name}({parameters}) {{"]
            lines.append(f"    /* Unresolved: {translation.reason}. */")
            lines.append("    return" + (" 0;" if return_type != "void" else ";"))
            lines.append("}")
            lines.append("")
            lines.append("/* Disassembly evidence:")
            for instruction in function.instructions[:80]:
                lines.append(f" *   0x{instruction.address:x}: {instruction.text}")
            lines.append(" */")
            body = "\n".join(lines) + "\n"
            classification = "unresolved"
            confidence = 0.12
        files[f"src/{name}.c"] = f'#include "{project_name}.h"\n\n' + body
        results.append(
            {
                "function_id": function.function_id,
                "name": function.name,
                "generated_name": name,
                "classification": classification,
                "confidence": confidence,
                "reason": translation.reason,
                "parameter_count": len(signature.get("parameters") or []),
                "evidence": [f"ev_{bundle.elf.sha256[:12]}_func_{function.entry:x}"],
            }
        )
        evidence["functions"].append(
            {
                "function_id": function.function_id,
                "name": function.name,
                "source_file": f"src/{name}.c",
                "classification": classification,
                "confidence": confidence,
                "abi": signature.get("calling_convention"),
                "register_model": signature.get("register_model"),
                "ranges": [{"start": f"0x{start:x}", "end": f"0x{end:x}"} for start, end in function.ranges],
            }
        )
    header.extend(["", "#endif", ""])
    files[f"include/{project_name}.h"] = "\n".join(header)
    files["Makefile"] = (
        "CC ?= gcc\nCFLAGS ?= -std=c11 -Wall -Wextra -Iinclude\n"
        f"SOURCES := $(wildcard src/*.c)\nOBJECTS := $(SOURCES:.c=.o)\n"
        f"all: lib{project_name}.a\nlib{project_name}.a: $(OBJECTS)\n\t$(AR) rcs $@ $^\n"
        "src/%.o: src/%.c\n\t$(CC) $(CFLAGS) -c $< -o $@\nclean:\n\trm -f src/*.o\n"
    )
    files["CMakeLists.txt"] = (
        "cmake_minimum_required(VERSION 3.16)\n"
        f"project({project_name} C)\nset(CMAKE_C_STANDARD 11)\n"
        'file(GLOB SOURCES "src/*.c")\n'
        f"add_library({project_name} STATIC ${{SOURCES}})\n"
        f"target_include_directories({project_name} PUBLIC include)\n"
    )
    files["README.md"] = (
        f"# {project_name}\n\n"
        f"C reconstruction for {bundle.elf.header.machine_name} using {'Ghidra' if ghidra_functions else 'the conservative multi-architecture translator'}. "
        "Ghidra output uses inferred types and names; fallback bodies cover only a straight-line subset. "
        "Compilation and structural checks are not behavioral equivalence.\n"
    )
    files["evidence_map.json"] = json.dumps(evidence, indent=2) + "\n"
    if out_dir is not None:
        root = Path(out_dir)
        for relative, content in files.items():
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
    return {
        "language": "c",
        "project_name": project_name,
        "architecture": bundle.architecture,
        "classification": "ghidra_decompiler_reconstruction" if any(item["classification"] == "ghidra_decompiler_reconstruction" for item in results) else ("partial_behavioral_reconstruction" if any(item["classification"] == "partial_behavioral_reconstruction" for item in results) else "unresolved"),
        "functions": results,
        "files": files,
        "evidence_map": evidence,
        "limitations": evidence["limitations"],
    }
