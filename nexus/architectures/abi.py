"""Evidence-based ABI signature inference for analyzed functions.

The inference is deliberately conservative.  It observes which incoming
argument registers are read before being overwritten, whether the ABI return
register is written, and whether the function adjusts the stack.  It does not
claim original parameter names, signedness, struct layout or a verified type
system.
"""

from __future__ import annotations

import re
from typing import Any

from nexus.architectures.base import ArchitectureAdapter
from nexus.disassembly.model import Instruction
from nexus.program.model import FunctionCandidate


def _instruction_order(function: FunctionCandidate) -> list[Instruction]:
    ordered: dict[int, Instruction] = {}
    for block in function.blocks:
        for instruction in block.instructions:
            ordered[instruction.address] = instruction
    if not ordered:
        for instruction in function.instructions:
            ordered[instruction.address] = instruction
    return [ordered[address] for address in sorted(ordered)]


def _observations(instructions: list[Instruction], registers: tuple[str, ...]) -> tuple[set[str], set[str], set[str]]:
    read_before_write: set[str] = set()
    written: set[str] = set()
    touched: set[str] = set()
    watched = set(registers)
    for instruction in instructions:
        for register in instruction.reads:
            if register in watched:
                touched.add(register)
                if register not in written:
                    read_before_write.add(register)
        for register in instruction.writes:
            if register in watched:
                touched.add(register)
                written.add(register)
        for effect in instruction.memory:
            for register in (effect.base, effect.index):
                if register and register in watched:
                    touched.add(register)
                    if register not in written:
                        read_before_write.add(register)
    return read_before_write, written, touched


def _width_type(adapter: ArchitectureAdapter) -> str:
    return "uint64_t" if adapter.pointer_width == 64 else "uint32_t"


def infer_abi_signature(
    function: FunctionCandidate,
    adapter: ArchitectureAdapter,
    *,
    existing: dict[str, Any] | None = None,
) -> dict[str, Any]:
    instructions = _instruction_order(function)
    if function.name in {"_start", "_init", "_fini", "_dl_relocate_static_pie"}:
        return {
            "return_type": "void",
            "return_width": 0,
            "return_written": False,
            "parameters": [],
            "parameter_count": 0,
            "calling_convention": "process-or-runtime-entry",
            "architecture": adapter.name,
            "pointer_width": adapter.pointer_width,
            "variadic": None,
            "noreturn": None,
            "stack_adjusted": False,
            "callee_saved": list(adapter.callee_saved),
            "confidence": 0.7,
            "provenance": "runtime_entry_no_c_abi",
            "limitations": ["Runtime entry points do not follow the C argument-register ABI."],
            "evidence": [f"ev_abi:entry:{function.function_id}"],
        }
    arguments = adapter.argument_registers
    read_before_write, written, touched = _observations(instructions, (*arguments, adapter.return_register or ""))
    used_arguments = [register for register in arguments if register in read_before_write]
    # Some compilers copy an argument to another register without reading the
    # memory form directly.  Reads are already captured by the decoder.
    return_written = bool(adapter.return_register and adapter.return_register in written)
    stack_adjusted = any(
        adapter.stack_pointer and adapter.stack_pointer in instruction.writes for instruction in instructions
    )
    existing = existing or {}
    existing_parameters = existing.get("parameters") or []
    stack_parameters: list[dict[str, Any]] = []
    if adapter.name == "x86" and not used_arguments:
        offsets: set[int] = set()
        for instruction in instructions:
            for effect in instruction.memory:
                if effect.base in {"rbp", "rsp"} and effect.displacement is not None:
                    displacement = int(effect.displacement)
                    base_offset = 8 if effect.base == "rbp" else 4
                    if displacement >= base_offset and displacement < 4096 and (displacement - base_offset) % 4 == 0:
                        offsets.add(displacement)
        for index, offset in enumerate(sorted(offsets)):
            stack_parameters.append(
                {
                    "index": index,
                    "name": f"arg{index}",
                    "register": f"stack+{offset}",
                    "type": "uint32_t",
                    "width": 32,
                    "confidence": 0.42,
                    "evidence": [f"ev_abi_stack_argument:{function.function_id}:{offset}"],
                    "provenance": "statically_inferred_cdecl_stack",
                }
            )
    parameters = []
    parameter_sources = stack_parameters if stack_parameters else used_arguments
    for index, register in enumerate(parameter_sources):
        if isinstance(register, dict):
            parameters.append(register)
            continue
        parameter: dict[str, Any] = {
            "index": index,
            "name": f"arg{index}",
            "register": register,
            "type": _width_type(adapter),
            "width": adapter.pointer_width,
            "confidence": 0.62 if function.source == "symbol" else 0.45,
            "evidence": [f"ev_abi_read_before_write:{function.function_id}:{register}"],
            "provenance": "statically_inferred",
        }
        if index < len(existing_parameters):
            candidate = existing_parameters[index]
            if candidate.get("type") and candidate.get("type", {}).get("name") not in {None, "unknown"}:
                parameter["type"] = candidate["type"].get("name", parameter["type"])
                parameter["width"] = candidate["type"].get("width") or parameter["width"]
                parameter["confidence"] = max(parameter["confidence"], float(candidate.get("confidence", 0.0)))
                parameter["evidence"].extend(candidate.get("evidence", []))
                parameter["provenance"] = "type_candidate_merged"
        parameters.append(parameter)
    existing_return = existing.get("return_type") or {}
    return_name = existing_return.get("name") if isinstance(existing_return, dict) else None
    if not return_name or return_name == "unknown":
        return_name = _width_type(adapter) if return_written else "void"
    confidence = 0.58
    if function.source == "symbol":
        confidence += 0.08
    if not instructions:
        confidence = 0.1
    argument_names = [
        (str(item.get("register")) if isinstance(item, dict) else str(item)).replace("+", "_").replace("-", "_")
        for item in parameter_sources
    ]
    return {
        "return_type": return_name,
        "return_width": adapter.pointer_width if return_written else 0,
        "return_written": return_written,
        "parameters": parameters,
        "parameter_count": len(parameters),
        "calling_convention": adapter.abi,
        "architecture": adapter.name,
        "pointer_width": adapter.pointer_width,
        "variadic": None,
        "noreturn": None,
        "stack_adjusted": stack_adjusted,
        "callee_saved": list(adapter.callee_saved),
        "confidence": round(confidence, 3),
        "provenance": "statically_inferred_abi_registers",
        "limitations": [
            "Parameter count is inferred from read-before-write argument registers, not debug information.",
            "Signedness, struct layout, floating-point arguments and variadic calls are not recovered.",
        ],
        "evidence": [
            f"ev_abi:{adapter.name}:{function.function_id}",
            f"ev_abi_arguments:{function.function_id}:{'_'.join(argument_names) or 'none'}",
            f"ev_abi_return:{function.function_id}:{adapter.return_register if return_written else 'void'}",
        ],
    }


def render_c_signature(name: str, signature: dict[str, Any]) -> str:
    safe = re.sub(r"[^A-Za-z0-9_]", "_", name or "reconstructed")
    if not safe or safe[0].isdigit():
        safe = f"fn_{safe}"
    return_type = str(signature.get("return_type") or "void")
    if return_type == "unknown":
        return_type = "void"
    parameters = signature.get("parameters") or []
    rendered = ", ".join(
        f"{parameter.get('type') or 'uintptr_t'} {parameter.get('name') or f'arg{index}'}"
        for index, parameter in enumerate(parameters)
    )
    return f"{return_type} {safe}({rendered if rendered else 'void'})"


def abi_payload_for_record(function: FunctionCandidate, adapter: ArchitectureAdapter, existing: dict[str, Any] | None = None) -> dict[str, Any]:
    signature = infer_abi_signature(function, adapter, existing=existing)
    return {
        "return_type": signature["return_type"],
        "parameters": signature["parameters"],
        "calling_convention": signature["calling_convention"],
        "variadic": None,
        "noreturn": None,
        "confidence": signature["confidence"],
        "architecture": signature["architecture"],
        "register_model": {
            "argument_registers": list(adapter.argument_registers),
            "return_register": adapter.return_register,
            "callee_saved": list(adapter.callee_saved),
            "stack_pointer": adapter.stack_pointer,
            "link_register": adapter.link_register,
        },
        "limitations": signature["limitations"],
        "evidence": signature["evidence"],
    }
