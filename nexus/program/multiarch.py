"""LLVM-backed multi-architecture analysis records.

The existing bundle model was originally x86-64/objdump specific.  This module
keeps that data model but converts the deterministic ``llvm-objdump`` record
shape into the same ``Disassembly``/``FunctionCandidate``/``CallGraph`` types so
that ELF metadata, JNI scanning, security review, signatures, the CLI and the
API all work for additional architectures.

The converter does not claim semantic equivalence.  It preserves the decoder's
instruction text, bytes and addresses, and exposes explicitly uncertain reads,
writes, memory effects and control-flow edges.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from nexus.disassembly.llvm_objdump import _mnemonic, _split_operands, parse_llvm_object
from nexus.disassembly.model import Disassembly, Instruction, MemoryEffect, Target
from nexus.elf.model import ElfFile
from nexus.program.model import BasicBlock, CallEdge, CallGraph, FunctionCandidate


def _address(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, int):
        return value
    text = str(value).strip()
    if not text:
        return None
    try:
        return int(text, 0)
    except ValueError:
        match = re.search(r"0x[0-9a-fA-F]+|\b[0-9a-fA-F]+\b", text)
        return int(match.group(0), 0) if match else None


def _instruction_operands(record: dict[str, Any]) -> list[str]:
    text = str(record.get("text") or "")
    _, operand_text = _mnemonic(text)
    return _split_operands(operand_text) if operand_text else []


def _instruction(record: dict[str, Any]) -> Instruction:
    address = _address(record.get("va")) or 0
    operands = _instruction_operands(record)
    raw_bytes = str(record.get("bytes") or "")
    length = int(record.get("length") or 0)
    if not length and re.fullmatch(r"(?:[0-9a-fA-F]{2})+", raw_bytes):
        length = len(raw_bytes) // 2
    flow = record.get("control_flow") or {}
    targets = []
    for target in flow.get("targets", []) or []:
        target_va = _address(target.get("va"))
        targets.append(
            Target(
                va=target_va,
                kind=str(target.get("kind") or "direct"),
                confidence=float(target.get("confidence") or 0.5),
            )
        )
    memory = []
    for effect in record.get("memory", []) or []:
        memory.append(
            MemoryEffect(
                kind=str(effect.get("kind") or "unknown"),
                address=effect.get("base"),
                width=effect.get("width"),
                base=effect.get("base"),
                index=effect.get("index"),
                displacement=effect.get("displacement"),
            )
        )
    return Instruction(
        address=address,
        bytes_hex=raw_bytes,
        length=length or 1,
        text=str(record.get("text") or ""),
        mnemonic=str(record.get("mnemonic") or "unknown"),
        operands=operands,
        function_label=None,
        section=None,
        reads=[str(item) for item in record.get("reads", []) or []],
        writes=[str(item) for item in record.get("writes", []) or []],
        memory=memory,
        control_flow=str(flow.get("kind") or "fallthrough"),
        targets=targets,
        flags_read=[str(item) for item in record.get("flags_read", []) or []],
        flags_written=[str(item) for item in record.get("flags_written", []) or []],
        decoded=True,
        confidence=float(record.get("confidence") or 0.6),
    )


def _function_from_record(record: dict[str, Any], instruction_by_id: dict[str, Instruction]) -> FunctionCandidate:
    entry = _address(record.get("entry_va")) or 0
    function_id = str(record.get("function_id") or f"fn_multi_{entry:x}")
    blocks = []
    function_instructions: list[Instruction] = []
    for block_record in record.get("blocks", []) or []:
        instructions = [
            instruction_by_id[instruction_id]
            for instruction_id in block_record.get("instruction_ids", [])
            if instruction_id in instruction_by_id
        ]
        if not instructions:
            continue
        for instruction in instructions:
            instruction.function_label = str(record.get("name") or function_id)
        start = int(block_record.get("start") or instructions[0].address)
        end = int(block_record.get("end") or (instructions[-1].address + instructions[-1].length))
        block = BasicBlock(
            block_id=str(block_record.get("block_id") or f"bb_{entry:x}_{start:x}"),
            function_id=function_id,
            start=start,
            end=end,
            instructions=instructions,
            successors=[str(item) for item in block_record.get("successors", [])],
            predecessors=[str(item) for item in block_record.get("predecessors", [])],
            confidence=float(block_record.get("confidence") or 0.6),
            evidence=[str(item) for item in block_record.get("evidence", [])],
        )
        blocks.append(block)
        function_instructions.extend(instructions)
    if not blocks and function_instructions:
        blocks.append(
            BasicBlock(
                block_id=f"bb_{entry:x}_fallback",
                function_id=function_id,
                start=function_instructions[0].address,
                end=function_instructions[-1].address + function_instructions[-1].length,
                instructions=list(function_instructions),
                confidence=0.2,
            )
        )
    if function_instructions:
        end = max(instruction.address + instruction.length for instruction in function_instructions)
    else:
        ranges = record.get("ranges") or [{}]
        end = _address(ranges[0].get("end_va")) or entry
    ranges = record.get("ranges") or []
    start = _address(ranges[0].get("start_va")) if ranges else entry
    return FunctionCandidate(
        function_id=function_id,
        name=str(record.get("name") or f"sub_{entry:x}"),
        entry=entry,
        ranges=[(start or entry, end)],
        size=max(0, end - (start or entry)),
        source=str(record.get("source") or "llvm_objdump_label"),
        confidence=float(record.get("confidence") or 0.6),
        instructions=function_instructions,
        blocks=blocks,
        evidence=[str(item) for item in record.get("evidence", [])],
    )


def _build_callgraph(
    functions: list[FunctionCandidate],
    disassembly: Disassembly,
    elf: ElfFile,
) -> CallGraph:
    by_entry = {function.entry: function for function in functions}
    graph = CallGraph(
        limitations=[
            "Call graph is derived from direct branch/call operands in LLVM disassembly.",
            "Indirect calls and stripped local functions remain unresolved.",
        ]
    )
    for function in functions:
        graph.add_node(function.function_id, kind="function", name=function.name, entry_va=f"0x{function.entry:x}")
    for function in functions:
        for instruction in function.instructions:
            if instruction.control_flow not in {"call", "indirect_call"}:
                continue
            target = instruction.targets[0] if instruction.targets else None
            if target is None or target.va is None:
                label = None
                for operand in instruction.operands:
                    match = re.search(r"<([^>]+)>", operand)
                    if match:
                        label = match.group(1).split("@")[0]
                        break
                name = label or "indirect_call"
                target_id = f"ext_{re.sub(r'[^A-Za-z0-9_]', '_', name)}"
                graph.add_node(target_id, kind="external", name=name)
                edge = CallEdge(
                    edge_id=f"call_{elf.sha256[:12]}_{instruction.address:x}_{target_id}",
                    source=function.function_id,
                    target=target_id,
                    kind="call_indirect" if instruction.control_flow == "indirect_call" else "call_external",
                    call_site=instruction.address,
                    confidence=0.2 if instruction.control_flow == "indirect_call" else 0.45,
                    target_name=name,
                    evidence=[f"ev_{elf.sha256[:12]}_call_{instruction.address:x}"],
                )
                graph.add_edge(edge)
                function.callees.append(edge)
                continue
            if target.va in by_entry:
                target_function = by_entry[target.va]
                edge = CallEdge(
                    edge_id=f"call_{elf.sha256[:12]}_{instruction.address:x}_{target_function.function_id}",
                    source=function.function_id,
                    target=target_function.function_id,
                    kind="call_direct",
                    call_site=instruction.address,
                    confidence=0.85,
                    target_name=target_function.name,
                    evidence=[f"ev_{elf.sha256[:12]}_call_{instruction.address:x}"],
                )
                graph.add_edge(edge)
                function.callees.append(edge)
                target_function.callers.append(edge)
                continue
            label = None
            for operand in instruction.operands:
                match = re.search(r"<([^>]+)>", operand)
                if match:
                    label = match.group(1).split("@")[0]
                    break
            name = label or f"sub_{target.va:x}"
            target_id = f"ext_{re.sub(r'[^A-Za-z0-9_]', '_', name)}"
            graph.add_node(target_id, kind="external", name=name)
            edge = CallEdge(
                edge_id=f"call_{elf.sha256[:12]}_{instruction.address:x}_{target_id}",
                source=function.function_id,
                target=target_id,
                kind="call_external",
                call_site=instruction.address,
                confidence=0.55,
                target_name=name,
                evidence=[f"ev_{elf.sha256[:12]}_call_{instruction.address:x}"],
            )
            graph.add_edge(edge)
            function.callees.append(edge)
    for function in functions:
        for block in function.blocks:
            if not block.successors:
                continue
            by_id = {candidate.block_id: candidate for candidate in function.blocks}
            for successor in list(block.successors):
                if successor in by_id and function.function_id not in by_id[successor].predecessors:
                    by_id[successor].predecessors.append(block.block_id)
    return graph


def analyze_with_llvm(path: str | Path, elf: ElfFile) -> tuple[Disassembly, list[FunctionCandidate], CallGraph, list[str], list[str]]:
    parsed = parse_llvm_object(path)
    instruction_by_id: dict[str, Instruction] = {}
    instructions: list[Instruction] = []
    for record in parsed.get("instructions", []) or []:
        instruction = _instruction(record)
        instruction_by_id[str(record.get("instruction_id") or f"ins_{instruction.address:x}")] = instruction
        instructions.append(instruction)
    functions = []
    seen_entries: set[int] = set()
    for record in parsed.get("functions", []) or []:
        function = _function_from_record(record, instruction_by_id)
        if function.instructions:
            functions.append(function)
            seen_entries.add(function.entry)
    # Exported symbols may not appear as labels in every LLVM version.  Add a
    # size-bounded function only when the symbol maps to decoded instructions.
    for symbol in list(elf.dynamic_symbols) + list(elf.symbols):
        if symbol.symbol_type != "FUNC" or symbol.shndx == 0 or not symbol.name or not symbol.size:
            continue
        if symbol.value in seen_entries:
            continue
        selected = [
            instruction
            for instruction in instructions
            if symbol.value <= instruction.address < symbol.value + symbol.size
        ]
        if not selected:
            continue
        function_id = f"fn_{elf.sha256[:12]}_{symbol.value:x}"
        blocks = [
            BasicBlock(
                block_id=f"bb_{elf.sha256[:12]}_{symbol.value:x}_symbol",
                function_id=function_id,
                start=selected[0].address,
                end=selected[-1].address + selected[-1].length,
                instructions=selected,
                confidence=0.7,
            )
        ]
        functions.append(
            FunctionCandidate(
                function_id=function_id,
                name=symbol.name,
                entry=symbol.value,
                ranges=[(symbol.value, symbol.value + symbol.size)],
                size=symbol.size,
                source="symbol",
                confidence=0.85,
                instructions=selected,
                blocks=blocks,
                evidence=[f"ev_{elf.sha256[:12]}_sym_{symbol.value:x}"],
            )
        )
        seen_entries.add(symbol.value)
    functions.sort(key=lambda item: item.entry)
    disassembly = Disassembly(
        backend="llvm-objdump",
        version=None,
        instructions=instructions,
        functions={function.name: function.entry for function in functions},
        sections=[str(section.get("name")) for section in parsed.get("sections", [])],
        limitations=[
            "Generic LLVM disassembly; operand semantics are decoder-derived and may be conservative.",
            "Stripped local functions without symbols are not exhaustively discovered.",
            "Indirect control flow is not resolved.",
        ],
    )
    graph = _build_callgraph(functions, disassembly, elf)
    warnings = [
        f"Using generic LLVM backend for {elf.header.machine_name}; semantic and reconstruction coverage is lower than validated x86-64.",
    ]
    limitations = list(disassembly.limitations)
    return disassembly, functions, graph, warnings, limitations
