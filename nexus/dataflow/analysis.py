from typing import Any

from nexus.disassembly.model import Instruction
from nexus.program.model import FunctionCandidate


def _instruction_id(instruction: Instruction) -> str:
    return f"ins_{instruction.address:x}"


def analyze_function_dataflow(
    function: FunctionCandidate,
    binary_sha256: str,
    *,
    argument_registers: list[str] | None = None,
    return_register: str = "rax",
    stack_pointer: str = "rsp",
    frame_pointer: str = "rbp",
    instruction_pointer: str = "rip",
    architecture: str = "x86_64",
) -> dict[str, Any]:
    chains: list[dict[str, Any]] = []
    last_writer: dict[str, str] = {}
    memory_accesses: list[dict[str, Any]] = []
    stack_slots: dict[tuple[str, int, int | None], dict[str, Any]] = {}
    globals_: dict[tuple[int, int | None], dict[str, Any]] = {}
    argument_candidates: list[dict[str, Any]] = []
    return_written = False
    written: set[str] = set()
    argument_registers = argument_registers or ["rdi", "rsi", "rdx", "rcx", "r8", "r9"]

    for instruction in function.instructions:
        instruction_id = _instruction_id(instruction)
        for register in instruction.reads:
            if register in last_writer:
                chains.append({
                    "def_instruction": last_writer[register],
                    "use_instruction": instruction_id,
                    "register": register,
                    "confidence": 0.5,
                })
        if instruction.writes:
            for register in instruction.writes:
                last_writer[register] = instruction_id
            if return_register and return_register in instruction.writes:
                return_written = True
        for register in instruction.reads:
            if register in argument_registers and register not in written and register not in [item["register"] for item in argument_candidates]:
                argument_candidates.append({
                    "register": register,
                    "index": argument_registers.index(register),
                    "confidence": 0.45,
                    "evidence": [f"ev_{binary_sha256[:12]}_arg_{instruction.address:x}"],
                })
        written.update(instruction.writes)
        for memory in instruction.memory:
            record = {
                "instruction_id": instruction_id,
                "address": f"0x{instruction.address:x}",
                "kind": memory.kind,
                "width": memory.width,
                "base": memory.base,
                "index": memory.index,
                "displacement": memory.displacement,
                "operand": memory.address,
                "confidence": 0.45,
            }
            memory_accesses.append(record)
            if memory.base in {frame_pointer, stack_pointer} and memory.displacement is not None:
                key = (memory.base, memory.displacement, memory.width)
                slot = stack_slots.setdefault(key, {
                    "base": memory.base,
                    "offset": memory.displacement,
                    "width": memory.width,
                    "reads": 0,
                    "writes": 0,
                    "instructions": [],
                    "confidence": 0.4,
                })
                if memory.kind in {"read", "read_write"}:
                    slot["reads"] += 1
                if memory.kind in {"write", "read_write"}:
                    slot["writes"] += 1
                slot["instructions"].append(instruction_id)
            elif memory.base == instruction_pointer and memory.displacement is not None:
                target = instruction.end + memory.displacement
                key = (target, memory.width)
                global_record = globals_.setdefault(key, {
                    "address": f"0x{target:x}",
                    "width": memory.width,
                    "reads": 0,
                    "writes": 0,
                    "instructions": [],
                    "confidence": 0.35,
                })
                if memory.kind in {"read", "read_write"}:
                    global_record["reads"] += 1
                if memory.kind in {"write", "read_write"}:
                    global_record["writes"] += 1
                global_record["instructions"].append(instruction_id)
    return {
        "def_use_chains": chains,
        "memory_accesses": memory_accesses,
        "stack_slots": sorted(stack_slots.values(), key=lambda item: (item["base"], item["offset"] or 0)),
        "globals": sorted(globals_.values(), key=lambda item: item["address"]),
        "argument_candidates": sorted(argument_candidates, key=lambda item: item["index"]),
        "return_written": return_written,
        "architecture": architecture,
        "confidence": 0.4,
        "limitations": [
            "Def-use chains are linear and do not model control-flow merges.",
            "Stack slots and globals are access-pattern candidates, not verified variables.",
            "Pointer aliasing and indirect memory effects are not resolved.",
        ],
    }
