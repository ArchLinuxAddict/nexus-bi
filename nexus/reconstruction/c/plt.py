"""Resolve PLT/GOT stubs without executing the binary.

Static and static-PIE glibc images retain PLT-style stubs whose GOT slot is
described by a relocation (often R_X86_64_IRELATIV with an IFUNC addend).  The
resolver maps a one-instruction indirect-jump function to the external symbol
named by its relocation.  Unknown or malformed stubs remain unresolved.
"""

import re
from typing import Any

from nexus.program.bundle import AnalysisBundle
from nexus.program.model import FunctionCandidate


def _got_address(instruction) -> int | None:
    if instruction.mnemonic not in {"jmp", "jmpq"} or not instruction.operands:
        return None
    match = re.search(r"\[\s*rip\s*([+-]\s*0x[0-9a-fA-F]+|\+\s*0x[0-9a-fA-F]+)\s*\]", instruction.operands[0])
    if not match:
        match = re.search(r"\[\s*rip\s*([+-]\s*\d+)\s*\]", instruction.operands[0])
    if not match:
        return None
    text = match.group(1).replace(" ", "")
    try:
        displacement = int(text, 0)
    except ValueError:
        return None
    if displacement >= (1 << 63):
        displacement -= 1 << 64
    elif displacement >= (1 << 31) and displacement < (1 << 32):
        displacement -= 1 << 32
    return instruction.end + displacement


def resolve_plt_stub(bundle: AnalysisBundle, function: FunctionCandidate) -> dict[str, Any]:
    instructions = [instruction for block in function.blocks for instruction in block.instructions]
    if len(instructions) != 1:
        return {"resolved": False, "reason": "not a single-instruction stub"}
    instruction = instructions[0]
    comment = instruction.comment or ""
    comment_match = re.search(r"<([^<>@+\s]+)(?:@[^<>]*)?(?:[+-][^<>]*)?>", comment)
    if comment_match:
        comment_name = comment_match.group(1)
        if comment_name not in {"_GLOBAL_OFFSET_TABLE_", "_GLOBAL_OFFSET_TABLE", ".plt", ".got", ".got.plt"} and not comment_name.startswith(".L"):
            return {
                "resolved": True,
                "symbol": comment_name,
                "got": None,
                "source": "instruction_comment",
                "instruction": instruction.to_dict(),
            }
    got = _got_address(instruction)
    if got is None:
        return {"resolved": False, "reason": "not an indirect rip-relative jump"}
    relocations = [relocation for relocation in bundle.elf.relocations if relocation.offset == got]
    for relocation in relocations:
        name = relocation.symbol_name
        if relocation.addend:
            candidates = [symbol.name for symbol in bundle.elf.symbols + bundle.elf.dynamic_symbols if symbol.value == relocation.addend and symbol.name]
            public = [candidate for candidate in candidates if not candidate.startswith("__") and "ifunc" not in candidate.lower()]
            if public:
                name = public[0]
            elif not name and candidates:
                name = candidates[0]
        if name and name.startswith("__new_") and name.endswith("_ifunc"):
            name = name[len("__new_"):-len("_ifunc")]
        if name:
            return {
                "resolved": True,
                "symbol": name,
                "got": f"0x{got:x}",
                "source": "relocation",
                "relocation": relocation.to_dict(),
                "instruction": instruction.to_dict(),
            }
    return {
        "resolved": False,
        "reason": "no named relocation at GOT slot",
        "got": f"0x{got:x}",
        "relocations": [relocation.to_dict() for relocation in relocations],
        "instruction": instruction.to_dict(),
    }
