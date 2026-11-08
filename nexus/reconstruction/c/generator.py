import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from nexus.program.bundle import AnalysisBundle
from nexus.program.model import FunctionCandidate


@dataclass
class Expr:
    text: str
    width: int = 64
    evidence: list[int] = field(default_factory=list)

    def cast(self, width: int) -> "Expr":
        if width == self.width:
            return Expr(self.text, width, list(self.evidence))
        return Expr(f"((uint{width}_t)({self.text}))", width, list(self.evidence))


@dataclass
class Translation:
    success: bool
    return_expr: Expr | None
    statements: list[tuple[str, int]]
    reason: str | None = None
    width: int = 64
    evidence: list[int] = field(default_factory=list)


def _sanitize(name: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_]", "_", name)
    if not value:
        value = "reconstructed"
    if value[0].isdigit():
        value = f"fn_{value}"
    return value


def _immediate(text: str) -> int | None:
    value = text.strip()
    value = value.removeprefix("$")
    try:
        return int(value, 0)
    except ValueError:
        try:
            return int(value, 10)
        except ValueError:
            return None


def _slot(operand: str) -> tuple[str, int] | None:
    match = re.search(r"\[(rbp|rsp)([+-]0x[0-9a-fA-F]+|[+-]\d+)?\]", operand)
    if not match:
        return None
    base = match.group(1)
    displacement = int(match.group(2), 0) if match.group(2) else 0
    return base, displacement


def _reg(operand: str) -> str | None:
    from nexus.disassembly.objdump import canonical_register
    if "[" in operand:
        return None
    return canonical_register(operand.strip())


def _width_for_reg(operand: str) -> int:
    from nexus.disassembly.objdump import operand_register_width
    return operand_register_width(operand) or 64


def _expr_for_operand(operand: str, regs: dict[str, Expr], stack: dict[tuple[str, int], Expr], address: int, width: int) -> Expr | None:
    register = _reg(operand)
    if register:
        value = regs.get(register)
        return value.cast(width) if value else Expr(f"unknown_{register}", width, [address])
    immediate = _immediate(operand)
    if immediate is not None:
        return Expr(str(immediate), width, [address])
    slot = _slot(operand)
    if slot:
        value = stack.get(slot)
        if value is None:
            return Expr(f"stack_{slot[0]}_{slot[1]}", width, [address])
        return value.cast(width)
    if "rip" in operand:
        return None
    return None


def _address_expr(operand: str, regs: dict[str, Expr], address: int, width: int) -> Expr | None:
    if "[" not in operand:
        return None
    inside = operand[operand.find("[") + 1:operand.find("]")].replace(" ", "")
    if "rip" in inside:
        return None
    terms: list[tuple[int, str]] = []
    sign = 1
    for part in re.split(r"([+-])", inside):
        if part == "+":
            sign = 1
            continue
        if part == "-":
            sign = -1
            continue
        if not part:
            continue
        if "*" in part:
            register, scale_text = part.split("*", 1)
            try:
                scale = int(scale_text, 0)
            except ValueError:
                return None
            value = regs.get(register)
            if value is None:
                return None
            terms.append((sign * scale, value.text))
            continue
        if part in regs:
            terms.append((sign, regs[part].text))
            continue
        try:
            terms.append((sign * int(part, 0), ""))
        except ValueError:
            return None
    parts = []
    for coefficient, text in terms:
        if text:
            if coefficient == 1:
                parts.append(text)
            elif coefficient == -1:
                parts.append(f"-{text}")
            else:
                parts.append(f"{coefficient}*{text}")
        else:
            parts.append(str(coefficient))
    if not parts:
        return None
    result = parts[0]
    for part in parts[1:]:
        if part.startswith("-"):
            result += part
        else:
            result += "+" + part
    return Expr(result, width, [address])


def _binary_expr(dst: Expr, src: Expr, operator: str, width: int, address: int) -> Expr:
    if operator in {"shl", "shr", "sar"} and _immediate(src.text) is not None:
        text = f"({dst.text} { '<<' if operator == 'shl' else '>>' } {src.text})"
    else:
        symbol = {"add": "+", "sub": "-", "imul": "*", "and": "&", "or": "|", "xor": "^", "shl": "<<", "shr": ">>", "sar": ">>"}.get(operator, "+")
        text = f"({dst.text} {symbol} {src.text})"
    return Expr(text, width, list(dict.fromkeys(dst.evidence + src.evidence + [address])))


def translate_function(bundle: AnalysisBundle, function: FunctionCandidate) -> Translation:
    types = bundle.type_candidates.get(function.function_id, [])
    params = sorted([item for item in types if item.get("role") == "parameter"], key=lambda item: item.get("parameter_index", 0))
    return_type = next((item for item in types if item.get("role") == "return"), None)
    return_width = (return_type or {}).get("size", 8) * 8 if return_type else 64
    argument_regs = ["rdi", "rsi", "rdx", "rcx", "r8", "r9"]
    regs: dict[str, Expr] = {}
    for index, parameter in enumerate(params):
        register = parameter.get("register") or (argument_regs[index] if index < len(argument_regs) else None)
        if register:
            width = (parameter.get("size") or 8) * 8
            abi_index = argument_regs.index(register) if register in argument_regs else index
            regs[register] = Expr(f"arg{abi_index}", width, [function.entry])
    stack: dict[tuple[str, int], Expr] = {}
    statements: list[tuple[str, int]] = []
    reasons: list[str] = []
    for instruction in function.instructions:
        mnemonic = instruction.mnemonic
        operands = instruction.operands
        address = instruction.address
        if mnemonic in {"endbr64", "endbr32", "nop", "push", "pop", "leave", "sub", "add"} and (mnemonic in {"push", "pop", "leave", "sub"} and (not operands or operands[0] in {"rbp", "rsp"} or (mnemonic == "sub" and operands[0] == "rsp"))):
            continue
        if mnemonic == "mov" and len(operands) == 2 and operands[0] == "rbp" and operands[1] == "rsp":
            continue
        if mnemonic in {"ret", "retq"}:
            if "rax" not in regs:
                reasons.append(f"return value at 0x{address:x} is not proven")
                break
            value = regs["rax"].cast(return_width)
            if "stack_" in value.text or "unknown_" in value.text:
                reasons.append(f"return expression at 0x{address:x} contains an unresolved local")
                break
            return Translation(True, value, statements, None, return_width, value.evidence)
        if mnemonic in {"call", "callq", "jmp", "jmpq", "je", "jne", "jz", "jnz", "jl", "jle", "jg", "jge", "ja", "jb", "jae", "jbe", "js", "jns", "jo", "jno", "loop", "loope", "loopne"}:
            reasons.append(f"control transfer {mnemonic} at 0x{address:x} is not translated")
            break
        if mnemonic.startswith("set") or mnemonic.startswith("cmov"):
            reasons.append(f"flag-dependent instruction {mnemonic} at 0x{address:x} is not translated")
            break
        if mnemonic in {"mov", "movabs", "movzx", "movsx", "movsxd", "movbe"}:
            if len(operands) != 2:
                reasons.append(f"unsupported mov form at 0x{address:x}")
                break
            destination, source = operands
            source_width = _width_for_reg(source) if _reg(source) else (_width_for_reg(destination) if _reg(destination) else 64)
            value = _expr_for_operand(source, regs, stack, address, source_width)
            if value is None:
                reasons.append(f"unsupported source operand at 0x{address:x}")
                break
            destination_register = _reg(destination)
            if destination_register:
                width = _width_for_reg(destination)
                regs[destination_register] = value.cast(width)
                statements.append((f"{destination_register} = {regs[destination_register].text};", address))
            else:
                slot = _slot(destination)
                if slot is None:
                    reasons.append(f"unsupported destination memory at 0x{address:x}")
                    break
                stack[slot] = value
                statements.append((f"stack_{slot[0]}_{slot[1]} = {value.text};", address))
            continue
        if mnemonic == "lea":
            if len(operands) != 2:
                reasons.append(f"unsupported lea at 0x{address:x}")
                break
            destination = _reg(operands[0])
            if not destination:
                reasons.append(f"unsupported lea destination at 0x{address:x}")
                break
            value = _address_expr(operands[1], regs, address, _width_for_reg(operands[0]))
            if value is None:
                reasons.append(f"unresolved address expression at 0x{address:x}")
                break
            regs[destination] = value
            statements.append((f"{destination} = {value.text};", address))
            continue
        if mnemonic in {"add", "sub", "imul", "and", "or", "xor", "shl", "shr", "sar", "inc", "dec", "neg", "not"}:
            if mnemonic in {"inc", "dec", "neg", "not"}:
                if len(operands) != 1:
                    reasons.append(f"unsupported unary form at 0x{address:x}")
                    break
                destination = operands[0]
                register = _reg(destination)
                if not register or register not in regs:
                    reasons.append(f"unsupported unary destination at 0x{address:x}")
                    break
                width = _width_for_reg(destination)
                current = regs[register]
                if mnemonic == "inc":
                    regs[register] = _binary_expr(current, Expr("1", width, [address]), "add", width, address)
                elif mnemonic == "dec":
                    regs[register] = _binary_expr(current, Expr("1", width, [address]), "sub", width, address)
                elif mnemonic == "neg":
                    regs[register] = Expr(f"(-{current.text})", width, current.evidence + [address])
                else:
                    regs[register] = Expr(f"(~{current.text})", width, current.evidence + [address])
                statements.append((f"{register} = {regs[register].text};", address))
                continue
            if len(operands) < 2:
                reasons.append(f"unsupported arithmetic form at 0x{address:x}")
                break
            destination = operands[0]
            register = _reg(destination)
            if not register or register not in regs:
                reasons.append(f"unsupported arithmetic destination at 0x{address:x}")
                break
            width = _width_for_reg(destination)
            current = regs[register]
            if mnemonic == "xor" and len(operands) == 2 and _reg(operands[1]) == register:
                regs[register] = Expr("0", width, current.evidence + [address])
            else:
                source = _expr_for_operand(operands[1], regs, stack, address, width)
                if source is None:
                    reasons.append(f"unsupported arithmetic source at 0x{address:x}")
                    break
                regs[register] = _binary_expr(current, source, mnemonic, width, address)
            statements.append((f"{register} = {regs[register].text};", address))
            continue
        if mnemonic in {"cmp", "test"}:
            continue
        reasons.append(f"instruction {mnemonic} at 0x{address:x} is not translated")
        break
    return Translation(False, None, statements, "; ".join(reasons) or "unsupported control flow", return_width, [])


def _abi_input_registers(function: FunctionCandidate) -> set[int]:
    abi = {"rdi": 0, "rsi": 1, "rdx": 2, "rcx": 3, "r8": 4, "r9": 5}
    raw_blocks = list(function.blocks)
    if not raw_blocks:
        return set()
    block_ids = [getattr(block, "block_id", f"block_{index}") for index, block in enumerate(raw_blocks)]
    blocks = {block_id: block for block_id, block in zip(block_ids, raw_blocks)}
    entry = getattr(function, "entry", getattr(raw_blocks[0], "start", 0))
    entry_id = next((block_id for block_id, block in blocks.items() if getattr(block, "start", None) == entry), block_ids[0])
    incoming: dict[str, list[str]] = {block_id: [] for block_id in blocks}
    for block_id, block in blocks.items():
        for successor in getattr(block, "successors", []):
            if successor in incoming:
                incoming[successor].append(block_id)
    entry_state = set(abi)
    in_states: dict[str, set[str]] = {block_id: set() for block_id in blocks}
    out_states: dict[str, set[str]] = {block_id: set() for block_id in blocks}
    inputs: set[int] = set()
    for _ in range(max(1, len(blocks) * 3)):
        changed = False
        for block_id, block in blocks.items():
            if block_id == entry_id:
                state = set(entry_state)
            else:
                predecessors = incoming.get(block_id, [])
                state = set().union(*(out_states[pred] for pred in predecessors)) if predecessors else set()
            if state != in_states[block_id]:
                changed = True
                in_states[block_id] = state
            current = set(state)
            for instruction in getattr(block, "instructions", []):
                for register in instruction.reads:
                    if register in current and register in abi:
                        inputs.add(abi[register])
                for register in instruction.writes:
                    current.discard(register)
            if current != out_states[block_id]:
                changed = True
                out_states[block_id] = current
        if not changed:
            break
    return inputs


def _function_prototype(function: FunctionCandidate, bundle: AnalysisBundle) -> tuple[str, str, int]:
    abi_signature = (getattr(bundle, "signatures", None) or {}).get(function.function_id)
    if abi_signature:
        return_type = str(abi_signature.get("return_type") or "void")
        return_width = int(abi_signature.get("return_width") or 0) or 64
        rendered = []
        for index, parameter in enumerate(abi_signature.get("parameters") or []):
            parameter_type = str(parameter.get("type") or "uintptr_t")
            rendered.append(f"{parameter_type} {parameter.get('name') or f'arg{index}'}")
        return return_type, ", ".join(rendered) if rendered else "void", return_width
    types = bundle.type_candidates.get(function.function_id, [])
    params = sorted([item for item in types if item.get("role") == "parameter"], key=lambda item: item.get("parameter_index", 0))
    inferred = _abi_input_registers(function)
    if not params:
        params = [{"role": "parameter", "parameter_index": index, "size": 8, "confidence": 0.3, "evidence": ["abi_register_read"]} for index in sorted(inferred)]
    else:
        present = {int(item.get("parameter_index", 0)) for item in params}
        missing = sorted(inferred - present)
        params.extend(
            {"role": "parameter", "parameter_index": index, "size": 8, "confidence": 0.3, "evidence": ["abi_register_read"]}
            for index in missing
        )
        params.sort(key=lambda item: item.get("parameter_index", 0))
    return_type = next((item for item in types if item.get("role") == "return"), None)
    return_width = (return_type or {}).get("size", 8) * 8 if return_type else 64
    if return_type is None or return_type.get("confidence", 0.0) < 0.6:
        return_width = 64
    return_name = f"uint{return_width}_t"
    parameters = []
    for index, parameter in enumerate(params):
        # ABI registers are 64-bit.  Narrower source types are recoverable only
        # through instruction-level truncation, not from ABI register width.
        parameter_width = 64 if parameter.get("register") in {None, "rdi", "rsi", "rdx", "rcx", "r8", "r9"} else (parameter.get("size") or 8) * 8
        parameters.append(f"uint{parameter_width}_t arg{index}")
    return return_name, ", ".join(parameters) if parameters else "void", return_width


def _stub_source(function: FunctionCandidate, bundle: AnalysisBundle, reason: str, generated_name: str | None = None) -> str:
    return_name, params, _ = _abi_prototype(function, bundle)
    name = generated_name or _sanitize(function.name)
    evidence = ", ".join(f"0x{instruction.address:x}" for instruction in function.instructions[:8])
    return (
        f"/* NEXUS-BI unresolved reconstruction for {function.name}.\n"
        f" * Reason: {reason}\n"
        f" * Evidence: {evidence or 'no instructions'}\n"
        f" * This stub is not claimed to be behaviorally equivalent.\n"
        f" */\n"
        f"{return_name} {name}({params}) {{\n"
        + ("    return 0;\n" if return_name != "void" else "    return;\n")
        + "}\n"
    )


def _abi_prototype(function: FunctionCandidate, bundle: AnalysisBundle) -> tuple[str, str, int]:
    return_name, _, return_width = _function_prototype(function, bundle)
    parameters = [f"uint64_t arg{index}" for index in range(6)]
    return return_name, ", ".join(parameters), return_width


def reconstruct_c(bundle: AnalysisBundle, function_ids: list[str] | None = None, out_dir: str | Path | None = None, project_name: str = "nexus_reconstruction", strategy: str = "legacy_hybrid") -> dict[str, Any]:
    architecture = (bundle.architecture or {}).get("name") or bundle.elf.header.machine_name
    if architecture != "x86_64" or (bundle.disassembly is not None and bundle.disassembly.backend == "llvm-objdump"):
        from nexus.reconstruction.c.multiarche import reconstruct_c_multiarche
        return reconstruct_c_multiarche(bundle, function_ids=function_ids, out_dir=out_dir, project_name=project_name)
    selected = bundle.functions
    if function_ids:
        ids = set(function_ids)
        selected = [function for function in bundle.functions if function.function_id in ids or function.name in ids]
    else:
        selected = [
            function for function in bundle.functions
            if function.confidence >= 0.5 and not function.name.startswith("_") and function.name not in {"frame_dummy", "deregister_tm_clones", "register_tm_clones", "__do_global_dtors_aux"}
        ]
    if not selected:
        selected = bundle.functions[:1]
    header_lines = [
        "#ifndef NEXUS_RECONSTRUCTED_H",
        "#define NEXUS_RECONSTRUCTED_H",
        "",
        "#include <stdint.h>",
        "#include <stddef.h>",
        "#include <stdlib.h>",
        "#include <string.h>",
        "#include <stdio.h>",
        "",
        "/* Generated by NEXUS-BI. Signatures and bodies are evidence-scored. */",
        "",
    ]
    prototypes = []
    source_files: dict[str, str] = {}
    function_results = []
    reserved_names = {"abort", "abs", "atexit", "atof", "atoi", "atol", "bsearch", "calloc", "div", "exit", "free", "getenv", "labs", "ldiv", "malloc", "mblen", "mbstowcs", "mbtowc", "qsort", "rand", "realloc", "srand", "strtod", "strtol", "strtoul", "system", "wcstombs", "wctomb", "memchr", "memcmp", "memcpy", "memmove", "memset", "strcat", "strchr", "strcmp", "strcoll", "strcpy", "strcspn", "strerror", "strlen", "strncat", "strncmp", "strncpy", "strpbrk", "strrchr", "strspn", "strstr", "strtok", "strxfrm", "strndup", "printf", "fprintf", "sprintf", "snprintf", "puts", "fputs", "putchar", "fputc"}
    name_map: dict[str, str] = {}
    used_names: set[str] = set()
    for function in selected:
        name = _sanitize(function.name)
        if name == "main":
            name = "reconstructed_main"
        if name in reserved_names:
            name = f"nexus_local_{name}"
        if name in used_names:
            suffix = 2
            while f"{name}_{suffix}" in used_names:
                suffix += 1
            name = f"{name}_{suffix}"
        used_names.add(name)
        name_map[function.function_id] = name
    evidence_map = {"binary_sha256": bundle.elf.sha256, "project": project_name, "functions": [], "claims": []}
    for function in selected:
        return_name, params, return_width = _abi_prototype(function, bundle)
        name = name_map[function.function_id]
        prototypes.append(f"{return_name} {name}({params});")
        translation = translate_function(bundle, function)
        structured_body = None
        if strategy != "legacy_direct" and (not translation.success or strategy in {"hybrid_v2", "legacy_structured"}):
            from nexus.reconstruction.c.structured import translate_structured
            structured = translate_structured(bundle, function, generated_name=name)
            if structured.success:
                structured_body = structured.body
        semantic_result = None
        if strategy in {"semantic_ir", "ssa", "ssa_informed", "region", "hybrid_v2"}:
            from nexus.reconstruction.c.semantic_emitter import emit_function
            semantic_result = emit_function(bundle, function, name, name_map=name_map, strategy=strategy)
        if semantic_result is not None and semantic_result.success:
            body = (
                f"/* Semantic {strategy} reconstruction of {function.name} at 0x{function.entry:x}.\n"
                f" * Classification: semantic_ir_reconstruction (not dynamically verified by this stage).\n"
                f" * Unsupported operations: {len(semantic_result.stats.get('unsupported', []))}.\n"
                f" */\n"
                + (semantic_result.body or "")
            )
            classification = "semantic_ir_reconstruction"
            confidence = 0.5
        elif strategy in {"semantic_ir", "ssa", "ssa_informed", "region"}:
            reason = semantic_result.reason if semantic_result is not None else "semantic emitter unavailable"
            body = _stub_source(function, bundle, reason, generated_name=name)
            classification = "unresolved"
            confidence = 0.15
        elif strategy in {"hybrid_v2", "legacy_structured"} and structured_body is not None:
            body = (
                f"/* Semantic {'hybrid' if strategy == 'hybrid_v2' else 'structured'} fallback for {function.name} at 0x{function.entry:x}.\n"
                f" * Classification: structural_approximation (not dynamically verified).\n"
                f" */\n"
                + structured_body
            )
            classification = "structural_approximation"
            confidence = 0.35
        elif translation.success and translation.return_expr is not None:
            evidence = sorted(set(translation.return_expr.evidence))
            body = (
                f"/* Evidence-linked reconstruction of {function.name} at 0x{function.entry:x}.\n"
                f" * Classification: partial_behavioral_reconstruction (not dynamically verified).\n"
                f" * Evidence: {', '.join(f'0x{item:x}' for item in evidence)}\n"
                f" */\n"
                f"{return_name} {name}({params}) {{\n"
                f"    return {translation.return_expr.text};\n"
                f"}}\n"
            )
            classification = "partial_behavioral_reconstruction"
            confidence = 0.55
        elif structured_body is not None:
            body = (
                f"/* Structural CFG reconstruction of {function.name} at 0x{function.entry:x}.\n"
                f" * Classification: structural_approximation (not dynamically verified).\n"
                f" * Direct CFG translation uses labels and goto; unsupported instructions are rejected.\n"
                f" */\n"
                + structured_body
            )
            classification = "structural_approximation"
            confidence = 0.35
        else:
            body = _stub_source(function, bundle, translation.reason or "unsupported function shape", generated_name=name)
            classification = "unresolved"
            confidence = 0.15
        source_files[f"src/{name}.c"] = f'#include "{project_name}.h"\n\n' + body
        function_results.append({
            "function_id": function.function_id,
            "name": function.name,
            "generated_name": name,
            "classification": classification,
            "confidence": confidence,
            "reason": (
                semantic_result.reason if semantic_result is not None and not semantic_result.success
                else (None if translation.success else (None if structured_body is not None else (translation.reason or "unsupported function shape")))
            ),
            "parameter_count": len([item for item in bundle.type_candidates.get(function.function_id, []) if item.get("role") == "parameter"]),
            "semantic_stats": semantic_result.stats if semantic_result is not None else None,
            "evidence": [f"ev_{bundle.elf.sha256[:12]}_func_{function.entry:x}"],
        })
        evidence_map["functions"].append({
            "function_id": function.function_id,
            "name": function.name,
            "source_file": f"src/{name}.c",
            "classification": classification,
            "binary_ranges": [{"start_va": f"0x{start:x}", "end_va": f"0x{end:x}"} for start, end in function.ranges],
            "evidence": [f"ev_{bundle.elf.sha256[:12]}_func_{function.entry:x}"],
            "confidence": confidence,
        })
    header_lines.extend(prototypes)
    header_lines.extend(["", "#endif", ""])
    header_name = f"include/{project_name}.h"
    source_files[header_name] = "\n".join(header_lines)
    smoke_lines = [f'#include "{project_name}.h"', "", "int main(void) {"]
    for result in function_results:
        if result["generated_name"] == "main":
            continue
        arguments = ", ".join("0" for _ in range(6))
        smoke_lines.append(f"    (void){result['generated_name']}({arguments});")
    smoke_lines.append("    return 0;")
    smoke_lines.append("}")
    smoke_lines.append("")
    source_files["tests/smoke.c"] = "\n".join(smoke_lines)
    source_files["CMakeLists.txt"] = (
        "cmake_minimum_required(VERSION 3.16)\n"
        f"project({project_name} C)\n"
        "set(CMAKE_C_STANDARD 11)\n"
        "file(GLOB SOURCES \"src/*.c\")\n"
        f"add_library({project_name} STATIC ${{SOURCES}})\n"
        f"target_include_directories({project_name} PUBLIC include)\n"
    )
    source_files["Makefile"] = (
        f"CC ?= gcc\nCFLAGS ?= -std=c11 -Wall -Wextra -Iinclude\n"
        f"SOURCES := $(wildcard src/*.c)\nOBJECTS := $(SOURCES:.c=.o)\n"
        f"all: lib{project_name}.a\nlib{project_name}.a: $(OBJECTS)\n\t$(AR) rcs $@ $^\n"
        f"src/%.o: src/%.c\n\t$(CC) $(CFLAGS) -c $< -o $@\nclean:\n\trm -f src/*.o lib{project_name}.a\n"
    )
    source_files["README.md"] = (
        f"# {project_name}\n\n"
        "Generated by NEXUS-BI from a native binary. Reconstructions are evidence-scored; "
        "unresolved functions are explicit stubs and are not claimed to be behaviorally equivalent. "
        "Dynamic verification is disabled until a verified sandbox is available.\n"
    )
    source_files["evidence_map.json"] = json.dumps(evidence_map, indent=2) + "\n"
    if out_dir is not None:
        root = Path(out_dir)
        for relative, content in source_files.items():
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
    return {
        "language": "c",
        "project_name": project_name,
        "classification": "partial_behavioral_reconstruction" if any(item["classification"] == "partial_behavioral_reconstruction" for item in function_results) else "unresolved",
        "functions": function_results,
        "files": source_files,
        "evidence_map": evidence_map,
        "limitations": [
            "Generated C is compiled but not executed or differentially verified.",
            "Type widths and signedness are heuristic; signedness is not proven.",
            "Functions with branches, calls, flags, or unsupported instructions are explicit unresolved stubs.",
        ],
    }
