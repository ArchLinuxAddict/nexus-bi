"""Generic architecture disassembly through llvm-objdump.

This parser covers object files for x86-64, i386, AArch64, ARM32, RISC-V and
MIPS.  It produces the same structural record shape used by the foundation
representation encoder.  It is a deterministic static parser, not a verified
decompiler, and it does not execute the object.
"""

from __future__ import annotations

import hashlib
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def _run(command: list[str], timeout: float = 120) -> str:
    result = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
    if result.returncode != 0:
        raise RuntimeError(f"command failed ({result.returncode}): {' '.join(command)}\n{result.stderr[-2000:]}")
    return result.stdout


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _architecture(file_format: str) -> str:
    text = file_format.lower()
    if "aarch64" in text:
        return "aarch64"
    if "arm" in text:
        return "arm"
    if "riscv" in text:
        return "riscv"
    if "mips" in text:
        return "mips"
    if "x86-64" in text or "x86_64" in text:
        return "x86_64"
    if "i386" in text:
        return "i386"
    if "powerpc" in text:
        return "powerpc"
    return "unknown"


def _canonical_reg(arch: str, value: str) -> str:
    text = str(value or "").strip().lower().lstrip("%$")
    if not text:
        return ""
    if arch == "aarch64":
        if text in {"sp", "wsp"}:
            return "a64_sp"
        if text in {"xzr", "wzr"}:
            return "a64_xzr"
        if text in {"lr", "x30"}:
            return "a64_lr"
        if text == "fp" or text == "x29":
            return "a64_fp"
        if text == "pc":
            return "a64_pc"
        match = re.fullmatch(r"[xw](\d+)", text)
        if match and 0 <= int(match.group(1)) <= 30:
            return f"a64_x{int(match.group(1))}"
        return ""
    if arch == "arm":
        if text == "sp":
            return "arm_sp"
        if text == "lr":
            return "arm_lr"
        if text == "pc":
            return "arm_pc"
        match = re.fullmatch(r"r(\d+)", text)
        if match and 0 <= int(match.group(1)) <= 15:
            return f"arm_r{int(match.group(1))}"
        return ""
    if arch == "riscv":
        aliases = {
            "zero": "rv_x0", "ra": "rv_ra", "sp": "rv_sp", "gp": "rv_gp", "tp": "rv_tp",
            "s0": "rv_s0", "fp": "rv_s0",
            **{f"a{i}": f"rv_a{i}" for i in range(8)},
            **{f"s{i}": f"rv_s{i}" for i in range(12)},
            **{f"t{i}": f"rv_t{i}" for i in range(7)},
        }
        if text in aliases:
            return aliases[text]
        match = re.fullmatch(r"x(\d+)", text)
        if match and 0 <= int(match.group(1)) <= 31:
            return f"rv_x{int(match.group(1))}"
        return ""
    if arch == "mips":
        aliases = {"zero": "mips_zero", "at": "mips_at", "v0": "mips_v0", "v1": "mips_v1", "gp": "mips_gp", "sp": "mips_sp", "fp": "mips_fp", "ra": "mips_ra"}
        if text in aliases:
            return aliases[text]
        if re.fullmatch(r"[astk]\d+", text):
            return f"mips_{text}"
        if re.fullmatch(r"\d+", text):
            return f"mips_{text}"
        return ""
    from nexus.foundation.vocab import canonical_register_name

    return canonical_register_name(text)


@dataclass
class ParsedFunction:
    name: str
    entry: int
    instructions: list[dict[str, Any]]
    end: int


def _split_operands(text: str) -> list[str]:
    parts = []
    current = []
    depth = 0
    for character in text:
        if character in "[(":
            depth += 1
        elif character in "])":
            depth = max(0, depth - 1)
        if character == "," and depth == 0:
            value = "".join(current).strip()
            if value:
                parts.append(value)
            current = []
        else:
            current.append(character)
    value = "".join(current).strip()
    if value:
        parts.append(value)
    return parts


def _memory_operand(arch: str, text: str) -> dict[str, Any] | None:
    raw = text.strip()
    base = index = None
    displacement = None
    width = None
    if "[" in raw and "]" in raw:
        inner = raw[raw.find("[") + 1 : raw.rfind("]")]
        pieces = [piece.strip() for piece in inner.split(",") if piece.strip()]
        if pieces:
            base = _canonical_reg(arch, pieces[0])
            for piece in pieces[1:]:
                value = piece.lstrip("#").strip()
                if re.fullmatch(r"[-+]?0x[0-9a-f]+|[-+]?\d+", value, re.IGNORECASE):
                    displacement = int(value, 0)
                else:
                    index = _canonical_reg(arch, piece)
    sib = re.search(r"([-+]?0x[0-9a-f]+|[-+]?\d+)?\s*\(([^()]*)\)", raw, re.IGNORECASE)
    if sib:
        if sib.group(1):
            displacement = int(sib.group(1), 0)
        components = [part.strip() for part in sib.group(2).split(",") if part.strip()]
        if components:
            base = _canonical_reg(arch, components[0])
        if len(components) > 1:
            index = _canonical_reg(arch, components[1])
    if base is None and index is None:
        return None
    match = re.search(r"\b(byte|half|word|dword|qword)\b", raw, re.IGNORECASE)
    if match:
        width = {"byte": 8, "half": 16, "word": 32, "dword": 32, "qword": 64}[match.group(1).lower()]
    return {"base": base, "index": index, "displacement": displacement, "width": width}


def _is_memory(text: str) -> bool:
    return "[" in text or ("(" in text and ")" in text)


def _operand(arch: str, text: str, access: str) -> dict[str, Any]:
    raw = text.strip()
    memory = _memory_operand(arch, raw)
    if memory is not None:
        return {"kind": "memory", "value": raw, "access": access, "width": memory.get("width"), "memory": memory}
    register = _canonical_reg(arch, raw.lstrip("#").strip())
    if register:
        return {"kind": "register", "register": register, "value": register, "access": access, "width": None}
    if re.fullmatch(r"[-+]?(?:0x[0-9a-f]+|\d+)", raw.lstrip("#$").strip(), re.IGNORECASE):
        return {"kind": "immediate", "value": raw, "access": "read", "width": None}
    return {"kind": "unknown", "value": raw, "access": access, "width": None}


def _flow(arch: str, mnemonic: str, operands: list[str], labels: dict[str, int]) -> dict[str, Any]:
    mn = mnemonic.lower()
    targets: list[dict[str, Any]] = []

    def add_target(text: str) -> None:
        # Prefer an explicit hexadecimal address, then a symbolic label.  Bare
        # decimal immediates must not be mistaken for branch destinations.
        explicit = re.search(r"(?<![\w#])0x([0-9a-f]+)", text, re.IGNORECASE)
        if explicit:
            targets.append({"va": int(explicit.group(1), 16), "kind": "direct", "confidence": 0.85})
            return
        named = re.search(r"<([^>]+)>", text)
        if named:
            name = named.group(1).split("+")[0]
            if name in labels:
                targets.append({"va": labels[name], "kind": "direct", "confidence": 0.8})
            else:
                targets.append({"va": None, "kind": "external", "confidence": 0.5, "name": name})
            return
        bare = re.fullmatch(r"\s*#?([0-9a-f]+)\s*", text.strip(), re.IGNORECASE)
        if bare and not text.strip().startswith("#"):
            targets.append({"va": int(bare.group(1), 16), "kind": "direct", "confidence": 0.6})

    if mn in {"ret", "retq", "eret", "iret"} or (mn == "bx" and operands and operands[0].strip().lower() == "lr") or (mn == "jr" and operands and operands[0].strip().lower() in {"ra", "$ra"}):
        return {"kind": "return", "targets": targets}
    if mn in {"bl", "blx", "call", "callq", "jal"}:
        for operand in operands:
            add_target(operand)
        return {"kind": "call", "targets": targets}
    if mn in {"blr", "br", "jalr", "jr", "bctr"}:
        for operand in operands:
            add_target(operand)
        return {"kind": "indirect_call" if mn in {"blr", "jalr"} else "indirect_branch", "targets": targets}
    if mn in {"jmp", "jmpq", "b", "br"}:
        for operand in operands:
            add_target(operand)
        return {"kind": "unconditional_branch", "targets": targets}
    conditional_prefixes = ("j", "b", "cb", "tb", "loop", "beq", "bne", "blt", "bge", "ble", "bgt", "bhi", "bls", "bcc", "bcs", "bmi", "bpl", "bvs", "bvc", "bhs", "blo")
    if mn.startswith(conditional_prefixes) and mn not in {"b", "bl", "bx", "br", "blr"}:
        for operand in operands:
            add_target(operand)
        return {"kind": "conditional_branch", "targets": targets}
    if mn in {"svc", "swi", "ud2", "ebreak", "ecall", "syscall", "int3"}:
        return {"kind": "trap", "targets": targets}
    return {"kind": "fallthrough", "targets": targets}


def _mnemonic(text: str) -> tuple[str, str]:
    parts = text.strip().split(None, 1)
    if not parts:
        return "unknown", ""
    return parts[0].lower(), parts[1] if len(parts) > 1 else ""


def _read_write(arch: str, mnemonic: str, operands: list[dict[str, Any]], flow: dict[str, Any]) -> tuple[list[str], list[str], list[dict[str, Any]]]:
    reads: list[str] = []
    writes: list[str] = []
    memory: list[dict[str, Any]] = []
    mn = mnemonic.lower()
    is_store = mn.startswith(("str", "st", "sb", "sh", "sw", "push", "swl", "swr"))
    for index, operand in enumerate(operands):
        if operand.get("kind") == "register":
            register = operand.get("register")
            access = operand.get("access")
            if access in {"write", "read_write"}:
                if register not in writes:
                    writes.append(register)
            if access in {"read", "read_write"}:
                if register not in reads:
                    reads.append(register)
        elif operand.get("kind") == "memory":
            info = operand.get("memory") or {}
            for key in ("base", "index"):
                register = info.get(key)
                if register and register not in reads:
                    reads.append(register)
            if arch in {"x86", "x86_64", "i386"}:
                if index == len(operands) - 1 and mn not in {"cmp", "test", "push", "call", "jmp", "ret"}:
                    kind = "read_write" if mn in {"add", "sub", "and", "or", "xor", "shl", "shr", "sar"} else "write"
                else:
                    kind = "read"
            else:
                kind = "write" if is_store else "read"
            memory.append({"kind": kind, "base": info.get("base"), "index": info.get("index"), "displacement": info.get("displacement"), "width": info.get("width")})
    if not reads and flow.get("kind") == "return":
        reads.append(_canonical_reg(arch, "ra") or "unknown")
    return reads, writes, memory


def _parse_sections(text: str) -> list[dict[str, Any]]:
    sections = []
    for line in text.splitlines():
        match = re.match(r"\s*\d+\s+(\S*)\s+([0-9a-f]+)\s+([0-9a-f]+)\s*(.*)$", line)
        if match and match.group(1):
            sections.append(
                {
                    "index": len(sections),
                    "name": match.group(1),
                    "size": int(match.group(2), 16),
                    "vma": int(match.group(3), 16),
                    "file_offset": None,
                    "type": match.group(4).strip(),
                }
            )
    return sections


def parse_llvm_object(path: str | Path, *, timeout: float = 120) -> dict[str, Any]:
    target = Path(path)
    output = _run(["llvm-objdump", "-d", str(target)], timeout=timeout)
    header_match = re.search(r"file format (\S+)", output)
    file_format = header_match.group(1) if header_match else "unknown"
    arch = _architecture(file_format)
    header_output = _run(["llvm-objdump", "-h", str(target)], timeout=timeout)
    sections = _parse_sections(header_output)
    functions: dict[str, ParsedFunction] = {}
    labels: dict[str, int] = {}
    current_name = None
    current_entry = 0
    current_instructions: list[dict[str, Any]] = []
    for line in output.splitlines():
        label = re.match(r"^([0-9a-f]+)\s+<([^>]+)>:$", line.strip())
        if label:
            if current_name is not None:
                functions[current_name] = ParsedFunction(current_name, current_entry, current_instructions, int(current_instructions[-1]["va"], 16) if current_instructions else current_entry)
            current_name = label.group(2)
            current_entry = int(label.group(1), 16)
            labels[current_name] = current_entry
            current_instructions = []
            continue
        if current_name is None:
            continue
        instruction = re.match(r"^\s*([0-9a-f]+):\s+(.*?)\t(.*)$", line)
        if not instruction:
            # Fallback for LLVM builds that separate fields with spaces only.
            instruction = re.match(r"^\s*([0-9a-f]+):\s+((?:(?:[0-9a-f]{2}|[0-9a-f]{4}|[0-9a-f]{8}|[0-9a-f]{16})\s+)+)(.*)$", line)
            if not instruction:
                continue
        address = int(instruction.group(1), 16)
        raw_bytes = "".join(instruction.group(2).split())
        text = instruction.group(3).strip()
        if not text or text.startswith("#"):
            continue
        mnemonic, operand_text = _mnemonic(text)
        if not mnemonic:
            continue
        raw_operands = _split_operands(operand_text) if operand_text else []
        flow = _flow(arch, mnemonic, raw_operands, labels)
        operands = []
        read_only = {
            "cmp", "test", "tst", "teq", "cmn", "push", "pop", "b", "bl", "br", "blr", "bx", "blx",
            "jmp", "call", "callq", "ret", "retq", "nop", "beq", "bne", "je", "jne", "jmpq",
            "ecall", "svc", "syscall", "int3", "ud2", "ebreak",
        }
        for index, raw in enumerate(raw_operands):
            access = "read"
            if arch in {"x86", "x86_64", "i386"}:
                # LLVM prints x86 in AT&T order: source(s), destination last.
                if mnemonic in read_only or index < len(raw_operands) - 1:
                    access = "read"
                elif mnemonic in {"inc", "dec", "neg", "not", "pop"}:
                    access = "write" if not _is_memory(raw) else "read_write"
                elif index == len(raw_operands) - 1:
                    access = "read_write" if _is_memory(raw) or mnemonic in {"add", "sub", "and", "or", "xor", "shl", "shr", "sar", "imul"} else "write"
            elif index == 0 and mnemonic not in read_only and mnemonic not in {"str", "st", "stp"}:
                access = "write" if not _is_memory(raw) else "read_write"
            operands.append(_operand(arch, raw, access))
        reads, writes, memory = _read_write(arch, mnemonic, operands, flow)
        current_instructions.append(
            {
                "instruction_id": f"ins_{address:x}",
                "va": f"0x{address:x}",
                "bytes": raw_bytes,
                "length": len(bytes.fromhex(raw_bytes)) if re.fullmatch(r"(?:[0-9a-f]{2})+", raw_bytes) else 0,
                "mnemonic": mnemonic,
                "operands": operands,
                "reads": reads,
                "writes": writes,
                "memory": memory,
                "control_flow": flow,
                "flags_read": ["nzcv"] if mnemonic in {"cmp", "tst", "teq", "cmn"} else [],
                "flags_written": ["nzcv"] if mnemonic.endswith("s") or mnemonic in {"cmp", "tst", "teq", "cmn", "adds", "subs"} else [],
                "text": text,
            }
        )
    if current_name is not None:
        functions[current_name] = ParsedFunction(current_name, current_entry, current_instructions, int(current_instructions[-1]["va"], 16) if current_instructions else current_entry)
    # Build blocks and CFG.
    instructions_by_address = {int(item["va"], 16): item for item in (ins for fn in functions.values() for ins in fn.instructions)}
    function_records = []
    call_edges = []
    for name, fn in functions.items():
        if len(fn.instructions) < 2:
            continue
        addresses = [int(item["va"], 16) for item in fn.instructions]
        address_set = set(addresses)
        leaders = {addresses[0]}
        for item in fn.instructions:
            flow = item["control_flow"]
            if flow["kind"] != "fallthrough":
                leaders.add(int(item["va"], 16) + max(1, item["length"]))
            for branch_target in flow.get("targets", []):
                if branch_target.get("va") in address_set:
                    leaders.add(int(branch_target["va"]))
        leaders = {address for address in leaders if address in address_set}
        blocks = []
        current = []
        for address in addresses:
            if current and address in leaders:
                blocks.append(current)
                current = []
            current.append(address)
        if current:
            blocks.append(current)
        block_records = []
        for block_index, block in enumerate(blocks):
            start = block[0]
            end = block[-1] + max(1, instructions_by_address[block[-1]]["length"])
            block_id = f"bb_{_sha256(target)[:12]}_{fn.entry:x}_{start:x}"
            block_records.append(
                {
                    "block_id": block_id,
                    "start": start,
                    "end": end,
                    "instruction_ids": [f"ins_{address:x}" for address in block],
                    "successors": [],
                    "predecessors": [],
                    "confidence": 0.7,
                    "evidence": [f"ev_{_sha256(target)[:12]}_block_{start:x}"],
                }
            )
        block_by_start = {block["start"]: block for block in block_records}
        for block_index, block in enumerate(block_records):
            last = instructions_by_address[int(block["instruction_ids"][-1][4:], 16)]
            flow = last["control_flow"]
            if flow["kind"] == "conditional_branch":
                for branch_target in flow.get("targets", []):
                    target_block = block_by_start.get(branch_target.get("va"))
                    if target_block:
                        block["successors"].append(target_block["block_id"])
                        target_block["predecessors"].append(block["block_id"])
                if block_index + 1 < len(block_records):
                    block["successors"].append(block_records[block_index + 1]["block_id"])
                    block_records[block_index + 1]["predecessors"].append(block["block_id"])
            elif flow["kind"] == "unconditional_branch":
                for branch_target in flow.get("targets", []):
                    target_block = block_by_start.get(branch_target.get("va"))
                    if target_block:
                        block["successors"].append(target_block["block_id"])
                        target_block["predecessors"].append(block["block_id"])
            elif flow["kind"] in {"fallthrough", "call"} and block_index + 1 < len(block_records):
                block["successors"].append(block_records[block_index + 1]["block_id"])
                block_records[block_index + 1]["predecessors"].append(block["block_id"])
            if flow["kind"] in {"call", "indirect_call"}:
                for target_item in flow.get("targets", []):
                    if target_item.get("va") is not None:
                        call_edges.append({"source": f"fn_{_sha256(target)[:12]}_{fn.entry:x}", "target_va": target_item["va"], "site": int(last["va"], 16)})
        function_records.append(
            {
                "function_id": f"fn_{_sha256(target)[:12]}_{fn.entry:x}",
                "name": name,
                "entry_va": f"0x{fn.entry:x}",
                "ranges": [{"start_va": f"0x{fn.entry:x}", "end_va": f"0x{fn.end:x}", "size": fn.end - fn.entry}],
                "size": fn.end - fn.entry,
                "source": "llvm_objdump_label",
                "confidence": 0.75,
                "instruction_count": len(fn.instructions),
                "block_count": len(block_records),
                "blocks": block_records,
                "callees": [],
                "callers": [],
                "aliases": [],
                "data_references": [],
                "stack_frame": {},
                "semantic": {},
                "signature": {"return_type": "unknown", "parameters": [], "calling_convention": arch},
                "evidence": [f"ev_{_sha256(target)[:12]}_func_{fn.entry:x}"],
                "unknowns": [],
                "purpose": None,
            }
        )
    entries = {int(item["entry_va"], 16): item["function_id"] for item in function_records}
    for function in function_records:
        for edge in call_edges:
            if edge["source"] == function["function_id"]:
                target_id = entries.get(edge["target_va"])
                if target_id:
                    function["callees"].append({"edge_id": f"edge_{edge['site']:x}", "source": function["function_id"], "target": target_id, "kind": "direct", "call_site_va": f"0x{edge['site']:x}", "confidence": 0.7, "target_name": None, "evidence": []})
                    target_record = next(item for item in function_records if item["function_id"] == target_id)
                    target_record["callers"].append({"edge_id": f"edge_{edge['site']:x}", "source": function["function_id"], "target": target_id, "kind": "direct", "call_site_va": f"0x{edge['site']:x}", "confidence": 0.7, "target_name": None, "evidence": []})
    all_instructions = [item for fn in functions.values() for item in fn.instructions]
    ends = [int(item["va"], 16) for item in all_instructions]
    size = target.stat().st_size
    return {
        "schema_version": "nexus.program-record/v2",
        "program": {
            "sha256": _sha256(target),
            "size": size,
            "path": str(target),
            "format": "ELF",
            "class": "ELF64" if "64" in file_format else "ELF32",
            "endianness": "big" if "big" in file_format else "little",
            "machine": arch,
            "entry": hex(min(ends) if ends else 0),
            "stripped": False,
            "pie": False,
            "static": True,
            "shared_object": False,
            "file_offset_to_va": [],
            "compiler_hints": [],
        },
        "functions": function_records,
        "instructions": all_instructions,
        "cfg": {"blocks": [], "edges": [], "limitations": ["CFG recovered from a linear object disassembly; indirect branches are not resolved."]},
        "callgraph": {"nodes": [], "edges": [], "limitations": ["Direct call targets inside the object only."]},
        "sections": sections,
        "symbols": [{"name": name, "value": f"0x{entry:x}", "size": 0, "symbol_type": "FUNC", "binding": "global", "visibility": "default", "shndx": 0, "evidence": []} for name, entry in sorted(labels.items())],
        "imports": [],
        "exports": [],
        "dependencies": [],
        "strings": [],
        "relocations": [],
        "notes": [],
        "metadata": {"architecture": arch, "file_format": file_format, "function_count": len(function_records), "instruction_count": len(all_instructions), "llvm_objdump": True},
        "analysis": {"backend": "llvm-objdump", "limitations": ["Generic multi-architecture static parse.", "Operand effects are heuristic.", "No execution."], "warnings": []},
        "evidence": [],
        "unknowns": [],
        "security_findings": [],
        "jni": {"is_jni": False, "entries": [], "limitations": []},
    }
