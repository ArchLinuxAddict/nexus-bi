import re
from collections import deque

from nexus.disassembly.model import Disassembly, Instruction
from nexus.elf.model import ElfFile
from nexus.program.model import BasicBlock, CallEdge, CallGraph, FunctionCandidate


def _sanitize_name(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_]", "_", name)
    if not cleaned:
        return "unnamed"
    if cleaned[0].isdigit():
        return f"func_{cleaned}"
    return cleaned


def _is_exec_section(elf: ElfFile, section_index: int) -> bool:
    section = elf.section_by_index(section_index)
    return bool(section and (section.flags & 4) and section.sh_type != 8)


def _in_exec_segment(elf: ElfFile, address: int) -> bool:
    for segment in elf.program_headers:
        if segment.p_type == 1 and segment.flags & 1 and segment.vaddr <= address < segment.vaddr + segment.memsz:
            return True
    return False


def _function_id(elf: ElfFile, entry: int) -> str:
    return f"fn_{elf.sha256[:12]}_{entry:x}"


def _evidence_id(elf: ElfFile, kind: str, address: int) -> str:
    return f"ev_{elf.sha256[:12]}_{kind}_{address:x}"


def _collect_symbols(elf: ElfFile) -> dict[int, dict]:
    candidates: dict[int, dict] = {}
    for symbol in list(elf.symbols) + list(elf.dynamic_symbols):
        if symbol.symbol_type != "FUNC" or symbol.shndx == 0:
            continue
        if not _in_exec_segment(elf, symbol.value):
            continue
        if not symbol.name:
            continue
        existing = candidates.get(symbol.value)
        if existing:
            if symbol.name not in existing["aliases"]:
                existing["aliases"].append(symbol.name)
            if symbol.size:
                existing["size"] = max(existing["size"], symbol.size)
            continue
        candidates[symbol.value] = {
            "name": symbol.name,
            "source": "symbol",
            "confidence": 0.95 if symbol.size else 0.8,
            "size": symbol.size,
            "aliases": [],
            "evidence": [_evidence_id(elf, "sym", symbol.value)],
        }
    return candidates


def _collect_call_targets(disassembly: Disassembly, elf: ElfFile, candidates: dict[int, dict]) -> None:
    for instruction in disassembly.instructions:
        if instruction.control_flow not in {"call", "indirect_call"}:
            continue
        for target in instruction.targets:
            if target.va is None or not _in_exec_segment(elf, target.va):
                continue
            if target.va in candidates:
                continue
            candidates[target.va] = {
                "name": None,
                "source": "call_target",
                "confidence": 0.65,
                "size": 0,
                "aliases": [],
                "evidence": [_evidence_id(elf, "call", instruction.address)],
            }


def _collect_prologues(disassembly: Disassembly, elf: ElfFile, candidates: dict[int, dict]) -> None:
    instructions = disassembly.instructions
    by_address = {instruction.address: instruction for instruction in instructions}
    known_ranges = []
    for entry, data in candidates.items():
        if data["size"]:
            known_ranges.append((entry, entry + data["size"]))
    for index, instruction in enumerate(instructions):
        address = instruction.address
        if not _in_exec_segment(elf, address) or address in candidates:
            continue
        if any(start <= address < end for start, end in known_ranges):
            continue
        prologue = False
        if instruction.mnemonic == "push" and instruction.operands[:1] == ["rbp"]:
            following = by_address.get(instruction.end)
            prologue = bool(following and following.mnemonic == "mov" and len(following.operands) == 2 and following.operands[0] == "rbp" and following.operands[1] == "rsp")
        elif instruction.mnemonic in {"endbr64", "endbr32"}:
            following = by_address.get(instruction.end)
            prologue = bool(following and following.mnemonic == "push" and following.operands[:1] == ["rbp"])
        if prologue:
            candidates[address] = {
                "name": None,
                "source": "prologue",
                "confidence": 0.35,
                "size": 0,
                "aliases": [],
                "evidence": [_evidence_id(elf, "prologue", address)],
            }


def _reachable(entry: int, limit: int, by_address: dict[int, Instruction], stop_entries: set[int] | None = None) -> list[Instruction]:
    selected: dict[int, Instruction] = {}
    queue = deque([entry])
    visited = set()
    while queue:
        address = queue.popleft()
        if address in visited or address < entry or address >= limit:
            continue
        if stop_entries and address != entry and address in stop_entries:
            continue
        visited.add(address)
        instruction = by_address.get(address)
        if instruction is None:
            continue
        selected[address] = instruction
        flow = instruction.control_flow
        if flow in {"return", "trap"}:
            continue
        if flow in {"indirect_branch"}:
            continue
        if flow in {"unconditional_branch"}:
            for target in instruction.targets:
                if target.va is not None:
                    queue.append(target.va)
            continue
        if flow in {"conditional_branch"}:
            for target in instruction.targets:
                if target.va is not None:
                    queue.append(target.va)
            queue.append(instruction.end)
            continue
        queue.append(instruction.end)
    return [selected[address] for address in sorted(selected)]


def _infer_function(elf: ElfFile, entry: int, limit: int, by_address: dict[int, Instruction], size_hint: int) -> list[Instruction]:
    if size_hint:
        limit = min(limit, entry + size_hint) if limit > entry else entry + size_hint
        selected = [instruction for address, instruction in by_address.items() if entry <= address < limit]
        return sorted(selected, key=lambda item: item.address)
    return _reachable(entry, limit, by_address)


def _build_cfg(elf: ElfFile, function: FunctionCandidate) -> None:
    instructions = sorted(function.instructions, key=lambda item: item.address)
    if not instructions:
        return
    addresses = {instruction.address for instruction in instructions}
    leaders = {instructions[0].address}
    for instruction in instructions:
        if instruction.control_flow in {"conditional_branch", "unconditional_branch", "indirect_branch", "return", "trap"}:
            leaders.add(instruction.end)
        for target in instruction.targets:
            if target.va is not None and target.va in addresses:
                leaders.add(target.va)
    leaders = {address for address in leaders if address in addresses}
    blocks: list[BasicBlock] = []
    current: list[Instruction] = []
    for instruction in instructions:
        if current and instruction.address in leaders:
            blocks.append(BasicBlock(
                block_id=f"bb_{elf.sha256[:12]}_{function.entry:x}_{current[0].address:x}",
                function_id=function.function_id,
                start=current[0].address,
                end=current[-1].end,
                instructions=current,
                confidence=0.85,
                evidence=[_evidence_id(elf, "block", current[0].address)],
            ))
            current = []
        current.append(instruction)
    if current:
        blocks.append(BasicBlock(
            block_id=f"bb_{elf.sha256[:12]}_{function.entry:x}_{current[0].address:x}",
            function_id=function.function_id,
            start=current[0].address,
            end=current[-1].end,
            instructions=current,
            confidence=0.85,
            evidence=[_evidence_id(elf, "block", current[0].address)],
        ))
    block_by_start = {block.start: block for block in blocks}
    for index, block in enumerate(blocks):
        last = block.instructions[-1]
        possible: list[tuple[str, str, float]] = []
        if last.control_flow == "conditional_branch":
            for target in last.targets:
                if target.va in block_by_start:
                    possible.append((block_by_start[target.va].block_id, "cfg_conditional", target.confidence))
            next_block = blocks[index + 1] if index + 1 < len(blocks) else None
            if next_block is not None and last.end == next_block.start:
                possible.append((next_block.block_id, "cfg_fallthrough", 0.9))
        elif last.control_flow == "unconditional_branch":
            for target in last.targets:
                if target.va in block_by_start:
                    possible.append((block_by_start[target.va].block_id, "cfg_unconditional", target.confidence))
        elif last.control_flow in {"indirect_branch", "indirect_call"}:
            possible.append((f"unknown_{function.function_id}", "cfg_indirect", 0.2))
        elif last.control_flow in {"return", "trap"}:
            pass
        else:
            next_block = blocks[index + 1] if index + 1 < len(blocks) else None
            if next_block is not None and last.end == next_block.start:
                possible.append((next_block.block_id, "cfg_fallthrough", 0.9))
        for successor, kind, confidence in possible:
            block.successors.append(successor)
            function.unknowns = function.unknowns
            if successor in block_by_start:
                block_by_start[successor].predecessors.append(block.block_id)
        if last.control_flow in {"indirect_branch", "indirect_call"}:
            function.unknowns.append({
                "unknown_id": f"unk_{elf.sha256[:12]}_{last.address:x}_indirect",
                "question": f"Unresolved indirect control flow at 0x{last.address:x}",
                "severity": "medium",
                "status": "open",
                "required_evidence": ["indirect target resolution"],
                "next_probe": "analyze jump table or pointer provenance",
            })
    function.blocks = blocks


def _build_callgraph(elf: ElfFile, functions: list[FunctionCandidate], disassembly: Disassembly) -> CallGraph:
    by_entry = {function.entry: function for function in functions}
    graph = CallGraph(limitations=[
        "Indirect calls are not resolved; they are represented as unknown edges.",
        "PLT/external call names are derived from objdump labels and may be incomplete.",
    ])
    for function in functions:
        graph.add_node(function.function_id, kind="function", name=function.name, entry_va=f"0x{function.entry:x}")
    for function in functions:
        for instruction in function.instructions:
            if instruction.control_flow not in {"call", "indirect_call"}:
                continue
            if instruction.control_flow == "indirect_call" or not instruction.targets or instruction.targets[0].va is None:
                unknown_id = f"call_unknown_{function.function_id}"
                graph.add_node(unknown_id, kind="unknown", name="indirect_call")
                edge = CallEdge(
                    edge_id=f"call_{elf.sha256[:12]}_{instruction.address:x}_unknown",
                    source=function.function_id,
                    target=unknown_id,
                    kind="call_indirect",
                    call_site=instruction.address,
                    confidence=0.2,
                    target_name=None,
                    evidence=[_evidence_id(elf, "call", instruction.address)],
                )
                graph.add_edge(edge)
                function.callees.append(edge)
                continue
            target = instruction.targets[0]
            if target.va in by_entry:
                target_function = by_entry[target.va]
                edge = CallEdge(
                    edge_id=f"call_{elf.sha256[:12]}_{instruction.address:x}_{target_function.function_id}",
                    source=function.function_id,
                    target=target_function.function_id,
                    kind="call_direct",
                    call_site=instruction.address,
                    confidence=0.9,
                    target_name=target_function.name,
                    evidence=[_evidence_id(elf, "call", instruction.address)],
                )
                graph.add_edge(edge)
                function.callees.append(edge)
                target_function.callers.append(edge)
                continue
            label = None
            for operand in instruction.operands:
                match = re.search(r"<([^>]+)>", operand)
                if match:
                    label = match.group(1)
                    break
            target_name = (label or f"sub_{target.va:x}").split("@")[0]
            external_id = f"ext_{_sanitize_name(target_name)}"
            graph.add_node(external_id, kind="external", name=target_name)
            edge = CallEdge(
                edge_id=f"call_{elf.sha256[:12]}_{instruction.address:x}_{external_id}",
                source=function.function_id,
                target=external_id,
                kind="call_external",
                call_site=instruction.address,
                confidence=0.7,
                target_name=target_name,
                evidence=[_evidence_id(elf, "call", instruction.address)],
            )
            graph.add_edge(edge)
            function.callees.append(edge)
    return graph


def recover_functions(elf: ElfFile, disassembly: Disassembly) -> tuple[list[FunctionCandidate], CallGraph]:
    candidates = _collect_symbols(elf)
    if _in_exec_segment(elf, elf.header.entry):
        candidates.setdefault(elf.header.entry, {
            "name": "_start",
            "source": "entry",
            "confidence": 0.9,
            "size": 0,
            "aliases": [],
            "evidence": [_evidence_id(elf, "entry", elf.header.entry)],
        })
    for name, address in disassembly.functions.items():
        if _in_exec_segment(elf, address) and address not in candidates:
            candidates[address] = {
                "name": name,
                "source": "disassembly_label",
                "confidence": 0.75,
                "size": 0,
                "aliases": [],
                "evidence": [_evidence_id(elf, "label", address)],
            }
    _collect_call_targets(disassembly, elf, candidates)
    _collect_prologues(disassembly, elf, candidates)
    by_address_all = {instruction.address: instruction for instruction in disassembly.instructions}
    text_end = max((instruction.end for instruction in disassembly.instructions), default=0)
    strong_entries = {address for address, data in candidates.items() if data.get("source") != "prologue"}
    stop_entries = set(strong_entries)
    for strong_entry in sorted(strong_entries):
        reachable = _reachable(strong_entry, text_end, by_address_all, stop_entries=stop_entries - {strong_entry})
        reachable_addresses = {instruction.address for instruction in reachable}
        for address in list(candidates):
            if address != strong_entry and candidates[address].get("source") == "prologue" and address in reachable_addresses:
                del candidates[address]
    entries = sorted(candidates)
    by_address = {instruction.address: instruction for instruction in disassembly.instructions}
    functions: list[FunctionCandidate] = []
    for index, entry in enumerate(entries):
        data = candidates[entry]
        limit = entries[index + 1] if index + 1 < len(entries) else None
        if limit is None:
            limit = max((instruction.end for instruction in disassembly.instructions), default=entry + 1)
        instructions = _infer_function(elf, entry, limit, by_address, data["size"])
        if not instructions:
            instructions = [instruction for instruction in disassembly.instructions if entry <= instruction.address < limit]
        if not instructions:
            continue
        end = max(max(instruction.end for instruction in instructions), entry + max(data["size"], 1))
        if data["size"] and end < entry + data["size"]:
            end = entry + data["size"]
        name = data["name"] or f"sub_{entry:x}"
        function = FunctionCandidate(
            function_id=_function_id(elf, entry),
            name=_sanitize_name(name),
            entry=entry,
            ranges=[(entry, end)],
            size=end - entry,
            source=data["source"],
            confidence=min(0.99, data["confidence"] + (0.05 if instructions else -0.1)),
            instructions=instructions,
            aliases=sorted(set(data.get("aliases", []))),
            evidence=list(data.get("evidence", [])) + [_evidence_id(elf, "func", entry)],
        )
        if not data["name"]:
            function.unknowns.append({
                "unknown_id": f"unk_{elf.sha256[:12]}_{entry:x}_name",
                "question": f"Original name of function at 0x{entry:x}",
                "severity": "low",
                "status": "open",
                "required_evidence": ["symbol table", "debug information", "call-site context"],
                "next_probe": "inspect callers and strings",
            })
        _build_cfg(elf, function)
        functions.append(function)
    functions.sort(key=lambda item: item.entry)
    callgraph = _build_callgraph(elf, functions, disassembly)
    return functions, callgraph
