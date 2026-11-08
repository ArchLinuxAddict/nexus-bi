from typing import Any

from nexus.disassembly.objdump import _canonical_register, operand_register_width
from nexus.program.model import FunctionCandidate


def _width_for_register(function: FunctionCandidate, register: str) -> int | None:
    widths = []
    for instruction in function.instructions:
        for operand in instruction.operands:
            if _canonical_register(operand) == register:
                width = operand_register_width(operand)
                if width:
                    widths.append(width)
    if not widths:
        return None
    return min(widths)


def infer_types(function: FunctionCandidate, dataflow: dict[str, Any], binary_sha256: str) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for argument in dataflow.get("argument_candidates", []):
        register = argument["register"]
        width = _width_for_register(function, register)
        type_name = f"uint{width}_t" if width else "uint64_t"
        candidates.append({
            "type_id": f"ty_{binary_sha256[:12]}_{function.entry:x}_arg{argument['index']}",
            "name": type_name,
            "kind": "integer",
            "size": width // 8 if width else 8,
            "signed": False,
            "confidence": 0.35,
            "evidence": [f"ev_{binary_sha256[:12]}_type_arg_{function.entry:x}"],
            "role": "parameter",
            "parameter_index": argument["index"],
            "register": register,
            "signedness_unknown": True,
        })
    if dataflow.get("return_written"):
        width = _width_for_register(function, "rax")
        candidates.append({
            "type_id": f"ty_{binary_sha256[:12]}_{function.entry:x}_return",
            "name": f"uint{width}_t" if width else "uint64_t",
            "kind": "integer",
            "size": width // 8 if width else 8,
            "signed": False,
            "confidence": 0.35,
            "evidence": [f"ev_{binary_sha256[:12]}_type_return_{function.entry:x}"],
            "role": "return",
            "signedness_unknown": True,
        })
    for index, slot in enumerate(dataflow.get("stack_slots", [])):
        width = slot.get("width")
        candidates.append({
            "type_id": f"ty_{binary_sha256[:12]}_{function.entry:x}_stack{index}",
            "name": f"uint{width}_t" if width else "uint64_t",
            "kind": "integer",
            "size": width // 8 if width else 8,
            "signed": False,
            "confidence": 0.3,
            "evidence": [f"ev_{binary_sha256[:12]}_type_stack_{function.entry:x}"],
            "role": "stack_slot",
            "base": slot.get("base"),
            "offset": slot.get("offset"),
        })
    for index, global_record in enumerate(dataflow.get("globals", [])):
        width = global_record.get("width")
        candidates.append({
            "type_id": f"ty_{binary_sha256[:12]}_{function.entry:x}_global{index}",
            "name": f"uint{width}_t" if width else "uint64_t",
            "kind": "integer",
            "size": width // 8 if width else 8,
            "signed": False,
            "confidence": 0.25,
            "evidence": [f"ev_{binary_sha256[:12]}_type_global_{function.entry:x}"],
            "role": "global",
            "address": global_record.get("address"),
        })
    return candidates
