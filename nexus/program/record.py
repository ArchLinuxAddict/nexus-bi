import re
from typing import Any

from nexus.disassembly.model import Instruction
from nexus.disassembly.objdump import operand_register_width
from nexus.elf.model import ElfFile
from nexus.program.bundle import AnalysisBundle
from nexus.program.model import BasicBlock, FunctionCandidate

ANALYSIS_VERSION = "0.2.0"
SCHEMA_VERSION = "0.1.0"
PIPELINE = "nexus-bi/0.2.0"


def _address(value: int | None) -> str | None:
    return f"0x{value:x}" if value is not None else None


def _range(start: int, end: int, elf: ElfFile) -> dict[str, Any]:
    offset = elf.va_to_offset(start)
    return {
        "start_va": f"0x{start:x}",
        "end_va": f"0x{end:x}",
        "file_offset": offset,
        "size": max(0, end - start),
    }


def _type_ref(candidate: dict[str, Any] | None) -> dict[str, Any]:
    if not candidate:
        return {"name": "unknown", "kind": "unknown", "width": None, "signed": None, "confidence": 0.1, "evidence": []}
    width = candidate.get("size")
    return {
        "name": candidate.get("name", "unknown"),
        "kind": "integer" if candidate.get("kind") == "integer" else candidate.get("kind", "unknown"),
        "width": width * 8 if width else None,
        "signed": candidate.get("signed", False),
        "type_id": candidate.get("type_id"),
        "confidence": candidate.get("confidence", 0.3),
        "evidence": candidate.get("evidence", []),
    }


def _memory_from_operand(instruction: Instruction, operand: str) -> dict[str, Any] | None:
    for effect in instruction.memory:
        if effect.address == operand or (effect.base and effect.base in operand):
            return {
                "base": effect.base,
                "index": effect.index,
                "scale": 1 if effect.index else None,
                "displacement": effect.displacement,
                "segment": effect.segment,
                "width": effect.width,
            }
    return None


def _canonical_for_arch(architecture: str, raw: str) -> str | None:
    from nexus.architectures import get_adapter
    from nexus.disassembly.llvm_objdump import _canonical_reg as llvm_canonical
    from nexus.disassembly.objdump import canonical_register as x86_canonical

    try:
        adapter = get_adapter(architecture)
    except Exception:
        adapter = None
    if adapter is not None and adapter.name in {"x86_64", "x86"}:
        return x86_canonical(raw) or llvm_canonical(adapter.name, raw) or None
    if adapter is not None:
        return llvm_canonical("arm" if adapter.name == "armv7" else adapter.name, raw) or None
    return x86_canonical(raw)


def _operand_record(instruction: Instruction, operand: str, architecture: str) -> dict[str, Any]:
    raw = operand.strip()
    if "[" in raw or "PTR" in raw.upper():
        memory = _memory_from_operand(instruction, raw)
        if memory is None:
            for effect in instruction.memory:
                if effect.base and effect.base in raw:
                    memory = {
                        "base": effect.base,
                        "index": effect.index,
                        "scale": 1 if effect.index else None,
                        "displacement": effect.displacement,
                        "segment": effect.segment,
                        "width": effect.width,
                    }
                    break
        memory = memory or {}
        access = "unknown"
        for effect in instruction.memory:
            if effect.address == raw or (effect.base and effect.base in raw):
                access = effect.kind
                break
        return {
            "kind": "memory",
            "value": raw,
            "width": memory.get("width"),
            "access": access,
            "memory": memory,
        }
    register = _canonical_for_arch(architecture, raw)
    if register:
        access = "unknown"
        if register in instruction.reads and register in instruction.writes:
            access = "read_write"
        elif register in instruction.writes:
            access = "write"
        elif register in instruction.reads:
            access = "read"
        return {
            "kind": "register",
            "value": raw,
            "register": register,
            "width": operand_register_width(raw) if architecture in {"x86_64", "x86"} else None,
            "access": access,
        }
    if re.match(r"^\$?-?(?:0x[0-9a-fA-F]+|\d+)$", raw):
        return {"kind": "immediate", "value": raw, "width": None, "access": "read"}
    if "<" in raw or re.match(r"^[0-9a-fA-F]+$", raw):
        return {"kind": "address", "value": raw, "width": 64, "access": "read"}
    return {"kind": "unknown", "value": raw, "width": None, "access": "unknown"}


def _instruction_record(instruction: Instruction, elf: ElfFile, architecture: str) -> dict[str, Any]:
    file_offset = elf.va_to_offset(instruction.address)
    return {
        "instruction_id": f"ins_{instruction.address:x}",
        "va": f"0x{instruction.address:x}",
        "file_offset": file_offset if file_offset is not None else 0,
        "bytes": instruction.bytes_hex,
        "length": instruction.length,
        "arch": architecture,
        "mode": "long" if architecture == "x86_64" else "unknown",
        "decoded": instruction.decoded,
        "mnemonic": instruction.mnemonic,
        "operands": [_operand_record(instruction, operand, architecture) for operand in instruction.operands],
        "reads": instruction.reads,
        "writes": instruction.writes,
        "memory": [
            {
                "kind": effect.kind,
                "address": effect.address,
                "width": effect.width,
                "base": effect.base,
                "index": effect.index,
                "displacement": effect.displacement,
            }
            for effect in instruction.memory
        ],
        "control_flow": {
            "kind": instruction.control_flow,
            "targets": [target.to_dict() for target in instruction.targets],
        },
        "flags_read": instruction.flags_read,
        "flags_written": instruction.flags_written,
        "decoder": {"name": "gnu-objdump", "version": "2.47"},
        "confidence": instruction.confidence,
        "evidence": [f"ev_{elf.sha256[:12]}_ins_{instruction.address:x}"],
        "limitations": [] if instruction.decoded else ["decoder reported invalid instruction"],
    }


def _block_record(block: BasicBlock) -> dict[str, Any]:
    return {
        "block_id": block.block_id,
        "range": _range(block.start, block.end, _BLOCK_ELF[0]),
        "instruction_ids": block.instruction_ids,
        "successors": block.successors,
        "predecessors": block.predecessors,
        "loop_header": False,
        "confidence": block.confidence,
        "evidence": block.evidence,
    }


_BLOCK_ELF: list[ElfFile] = []


def _abi_type(name: str, width: int | None, evidence: list[str], confidence: float) -> dict[str, Any]:
    return {
        "name": name,
        "kind": "void" if name == "void" else "integer",
        "width": width,
        "signed": None,
        "type_id": None,
        "confidence": confidence,
        "evidence": evidence,
    }


def _signature_record(
    abi_signature: dict[str, Any],
    parameters: list[dict[str, Any]],
    return_type: dict[str, Any] | None,
    function: FunctionCandidate,
    elf: ElfFile,
) -> dict[str, Any]:
    if abi_signature:
        abi_parameters = abi_signature.get("parameters") or []
        rendered = []
        for index, parameter in enumerate(abi_parameters):
            rendered.append(
                {
                    "index": index,
                    "name": parameter.get("name") or f"arg{index}",
                    "type": _abi_type(
                        str(parameter.get("type") or "uintptr_t"),
                        parameter.get("width"),
                        list(parameter.get("evidence", [])),
                        float(parameter.get("confidence", 0.4)),
                    ),
                    "location": parameter.get("register"),
                    "confidence": float(parameter.get("confidence", 0.4)),
                    "evidence": list(parameter.get("evidence", [])),
                    "provenance": parameter.get("provenance", "statically_inferred_abi_registers"),
                }
            )
        return {
            "return_type": _abi_type(
                str(abi_signature.get("return_type") or "void"),
                abi_signature.get("return_width"),
                list(abi_signature.get("evidence", [])),
                float(abi_signature.get("confidence", 0.4)),
            ),
            "parameters": rendered,
            "calling_convention": abi_signature.get("calling_convention"),
            "variadic": abi_signature.get("variadic"),
            "noreturn": abi_signature.get("noreturn"),
            "confidence": abi_signature.get("confidence", 0.4),
            "register_model": abi_signature.get("register_model"),
            "limitations": abi_signature.get("limitations", []),
        }
    return {
        "return_type": _type_ref(return_type),
        "parameters": [
            {
                "index": parameter.get("parameter_index", 0),
                "name": f"arg{parameter.get('parameter_index', 0)}",
                "type": _type_ref(parameter),
                "location": parameter.get("register"),
                "confidence": parameter.get("confidence", 0.3),
                "evidence": parameter.get("evidence", []),
            }
            for parameter in parameters
        ],
        "calling_convention": "SysV-AMD64" if elf.header.machine_name == "x86_64" else None,
        "variadic": None,
        "noreturn": None,
    }


def _function_record(function: FunctionCandidate, bundle: AnalysisBundle) -> dict[str, Any]:
    elf = bundle.elf
    dataflow = bundle.dataflows.get(function.function_id, {})
    types = bundle.type_candidates.get(function.function_id, [])
    parameters = sorted(
        [item for item in types if item.get("role") == "parameter"],
        key=lambda item: item.get("parameter_index", 0),
    )
    return_type = next((item for item in types if item.get("role") == "return"), None)
    purpose = bundle.purposes.get(function.function_id, {})
    abi_signature = (bundle.signatures or {}).get(function.function_id, {})
    names = [{
        "value": function.name,
        "source": "symbol" if function.source in {"symbol", "disassembly_label"} else "inferred",
        "confidence": function.confidence,
        "evidence": function.evidence,
    }]
    for alias in function.aliases:
        if alias != function.name:
            names.append({
                "value": alias,
                "source": "symbol",
                "confidence": 0.9,
                "evidence": function.evidence,
            })
    callees = []
    for edge in function.callees:
        callees.append({
            "function_id": edge.target if edge.kind == "call_direct" else None,
            "symbol": {"name": edge.target_name or edge.target, "symbol_id": None, "library": None, "version": None, "confidence": edge.confidence, "evidence": edge.evidence},
            "call_site_va": f"0x{edge.call_site:x}",
            "kind": "direct" if edge.kind == "call_direct" else "indirect" if edge.kind == "call_indirect" else "external",
            "confidence": edge.confidence,
            "evidence": edge.evidence,
        })
    blocks = []
    for block in function.blocks:
        blocks.append({
            "block_id": block.block_id,
            "range": _range(block.start, block.end, elf),
            "instruction_ids": block.instruction_ids,
            "successors": block.successors,
            "predecessors": block.predecessors,
            "loop_header": False,
            "confidence": block.confidence,
            "evidence": block.evidence,
        })
    if not blocks:
        blocks.append({
            "block_id": f"bb_{elf.sha256[:12]}_{function.entry:x}_fallback",
            "range": _range(function.start, function.end, elf),
            "instruction_ids": [f"ins_{instruction.address:x}" for instruction in function.instructions],
            "successors": [],
            "predecessors": [],
            "loop_header": False,
            "confidence": 0.2,
            "evidence": function.evidence,
        })
    data_references = []
    for memory in dataflow.get("globals", []):
        data_references.append({
            "va": memory["address"],
            "width": memory.get("width"),
            "access": "read_write" if memory.get("reads") and memory.get("writes") else "read" if memory.get("reads") else "write",
            "target_kind": "global_candidate",
            "evidence": [f"ev_{elf.sha256[:12]}_dataflow_{function.entry:x}"],
        })
    return {
        "function_id": function.function_id,
        "entry_va": f"0x{function.entry:x}",
        "ranges": [{"start_va": f"0x{start:x}", "end_va": f"0x{end:x}", "file_offset": elf.va_to_offset(start), "size": end - start} for start, end in function.ranges],
        "names": names,
        "signature": _signature_record(abi_signature, parameters, return_type, function, elf),
        "blocks": blocks,
        "callees": callees,
        "callers": [edge.source for edge in function.callers],
        "data_references": data_references,
        "stack_frame": {
            "size": None,
            "alignment": 16 if elf.header.machine_name == "x86_64" else None,
            "locals": [
                {"name": f"stack_{slot.get('offset')}", "offset": slot.get("offset"), "width": slot.get("width"), "confidence": slot.get("confidence")}
                for slot in dataflow.get("stack_slots", [])
            ],
        },
        "semantic": {
            "purpose": purpose.get("purpose"),
            "side_effects": [],
            "algorithm": None,
            "confidence": purpose.get("confidence", 0.1),
            "evidence": purpose.get("evidence", []),
        },
        "confidence": function.confidence,
        "evidence": function.evidence,
        "unknowns": function.unknowns,
    }


def _section_records(elf: ElfFile) -> tuple[list[dict], list[dict], list[dict]]:
    segments = []
    for segment in elf.program_headers:
        if segment.p_type not in {1, 2, 3, 4, 6, 7, 0x6474E550, 0x6474E551, 0x6474E552, 0x6474E553}:
            continue
        segments.append({
            "segment_id": f"seg_{elf.sha256[:12]}_{segment.index}",
            "type": segment.type_name,
            "flags": segment.permissions,
            "range": {"start_va": f"0x{segment.vaddr:x}", "end_va": f"0x{segment.vaddr + segment.memsz:x}", "file_offset": segment.offset, "size": segment.memsz},
            "alignment": segment.align,
            "evidence": [f"ev_{elf.sha256[:12]}_seg_{segment.index}"],
        })
    sections = []
    regions = []
    for section in elf.sections:
        if not section.name:
            continue
        sections.append({
            "section_id": f"sec_{elf.sha256[:12]}_{section.index}",
            "name": section.name,
            "type": section.type_name,
            "flags": section.flag_names,
            "range": {"start_va": f"0x{section.addr:x}", "end_va": f"0x{section.addr + section.size:x}", "file_offset": section.offset, "size": section.size},
            "alignment": section.addralign if section.addralign else None,
            "entry_size": section.entsize,
            "link": section.link if section.link < len(elf.sections) else None,
            "info": section.info,
            "evidence": [f"ev_{elf.sha256[:12]}_sec_{section.index}"],
        })
        if section.flags & 4:
            kind = "code"
        elif section.sh_type == 8:
            kind = "bss"
        elif section.flags & 1:
            kind = "data"
        elif section.flags & 2:
            kind = "rodata"
        else:
            kind = "unknown"
        regions.append({
            "region_id": f"reg_{elf.sha256[:12]}_{section.index}",
            "kind": kind,
            "range": {"start_va": f"0x{section.addr:x}", "end_va": f"0x{section.addr + section.size:x}", "file_offset": section.offset, "size": section.size},
            "confidence": 0.9,
            "evidence": [f"ev_{elf.sha256[:12]}_sec_{section.index}"],
        })
    return segments, sections, regions


def _symbol_records(elf: ElfFile) -> tuple[list[dict], list[dict], list[dict]]:
    symbols = []
    imports = []
    exports = []
    seen_imports = set()
    seen_exports = set()
    for symbol in list(elf.symbols) + list(elf.dynamic_symbols):
        if not symbol.name:
            continue
        source = "static" if symbol.table == "symtab" else "dynamic"
        entry = {
            "symbol_id": f"sym_{elf.sha256[:12]}_{source}_{symbol.index}",
            "name": symbol.name,
            "demangled_name": None,
            "value": f"0x{symbol.value:x}",
            "size": symbol.size,
            "binding": symbol.binding,
            "type": symbol.symbol_type,
            "visibility": symbol.visibility,
            "section_id": f"sec_{elf.sha256[:12]}_{symbol.shndx}" if symbol.shndx < len(elf.sections) else None,
            "version": symbol.version,
            "source": source,
            "confidence": 0.95,
            "evidence": [f"ev_{elf.sha256[:12]}_sym_{symbol.value:x}_{symbol.index}"],
        }
        symbols.append(entry)
        if symbol.is_import and symbol.name not in seen_imports:
            seen_imports.add(symbol.name)
            imports.append({"name": symbol.name, "symbol_id": entry["symbol_id"], "library": None, "version": symbol.version, "confidence": 0.85, "evidence": entry["evidence"]})
        if source == "dynamic" and symbol.shndx != 0 and symbol.binding in {"GLOBAL", "WEAK"} and symbol.visibility in {"DEFAULT", "PROTECTED"} and symbol.name not in seen_exports:
            seen_exports.add(symbol.name)
            exports.append({"name": symbol.name, "symbol_id": entry["symbol_id"], "library": None, "version": symbol.version, "confidence": 0.9, "evidence": entry["evidence"]})
    return symbols, imports, exports


def _relocation_records(elf: ElfFile) -> list[dict]:
    records = []
    for relocation in elf.relocations:
        records.append({
            "relocation_id": f"rel_{elf.sha256[:12]}_{relocation.table}_{relocation.index}",
            "offset": f"0x{relocation.offset:x}",
            "type": relocation.type_name,
            "addend": relocation.addend,
            "target": {
                "name": relocation.symbol_name or "<unknown>",
                "symbol_id": None,
                "library": None,
                "version": None,
                "confidence": 0.8 if relocation.symbol_name else 0.2,
                "evidence": [f"ev_{elf.sha256[:12]}_rel_{relocation.index}"],
            },
            "resolved_value": None,
            "confidence": 0.8 if relocation.symbol_name else 0.2,
            "evidence": [f"ev_{elf.sha256[:12]}_rel_{relocation.index}"],
        })
    return records


def _dependency_records(elf: ElfFile) -> list[dict]:
    records = []
    for index, dependency in enumerate(elf.dependencies):
        name = dependency.get("name", "")
        if not name:
            continue
        kind = dependency.get("kind", "unknown")
        if kind not in {"needed", "soname", "rpath", "runpath", "weak", "unknown"}:
            kind = "unknown"
        records.append({
            "name": name,
            "kind": kind,
            "version": None,
            "resolved_path": None,
            "evidence": [f"ev_{elf.sha256[:12]}_dep_{index}"],
        })
    return records


def _security_evidence(bundle: AnalysisBundle, registry: dict[str, dict]) -> None:
    for finding in bundle.security_findings:
        for evidence_id in finding.get("evidence", []):
            registry.setdefault(evidence_id, {
                "evidence_id": evidence_id,
                "kind": "inferred",
                "source": {"type": "tool", "path": bundle.path, "va": finding.get("address"), "file_offset": None, "tool": "nexus-security", "version": ANALYSIS_VERSION},
                "claim": finding.get("description", "Security finding"),
                "payload_sha256": None,
                "confidence": finding.get("confidence", 0.3),
                "analysis_version": ANALYSIS_VERSION,
                "timestamp": bundle.finished_at,
                "untrusted": False,
            })


def _ensure_evidence(record: dict[str, Any], registry: dict[str, dict], bundle: AnalysisBundle) -> None:
    referenced: set[str] = set()

    def walk(value: Any) -> None:
        if isinstance(value, dict):
            for item in value.values():
                walk(item)
        elif isinstance(value, list):
            for item in value:
                walk(item)
        elif isinstance(value, str) and value.startswith("ev_"):
            referenced.add(value)

    walk(record)
    for evidence_id in sorted(referenced):
        if evidence_id in registry:
            continue
        registry[evidence_id] = {
            "evidence_id": evidence_id,
            "kind": "derived",
            "source": {"type": "tool", "path": bundle.path, "va": None, "file_offset": None, "tool": "nexus", "version": ANALYSIS_VERSION},
            "claim": "Derived analysis fact referenced by the ProgramRecord.",
            "payload_sha256": None,
            "confidence": 0.5,
            "analysis_version": ANALYSIS_VERSION,
            "timestamp": bundle.finished_at,
            "untrusted": False,
        }


def build_program_record(bundle: AnalysisBundle) -> dict[str, Any]:
    elf = bundle.elf
    _BLOCK_ELF[:] = [elf]
    segments, sections, regions = _section_records(elf)
    symbols, imports, exports = _symbol_records(elf)
    relocations = _relocation_records(elf)
    dependencies = _dependency_records(elf)
    functions = [_function_record(function, bundle) for function in bundle.functions]
    instructions = []
    if bundle.disassembly:
        seen = set()
        for instruction in bundle.disassembly.instructions:
            if instruction.address in seen:
                continue
            seen.add(instruction.address)
            instructions.append(_instruction_record(instruction, elf, elf.header.machine_name))
    cfg_nodes = []
    cfg_edges = []
    for function in bundle.functions:
        for block in function.blocks:
            cfg_nodes.append({"node_id": block.block_id, "kind": "basic_block", "function_id": function.function_id, "start": f"0x{block.start:x}", "end": f"0x{block.end:x}"})
            last = block.instructions[-1] if block.instructions else None
            for successor in block.successors:
                if last and last.control_flow == "conditional_branch":
                    kind = "cfg_conditional" if successor != (block.successors[-1] if block.successors else None) else "cfg_fallthrough"
                elif last and last.control_flow == "unconditional_branch":
                    kind = "cfg_unconditional"
                elif last and last.control_flow in {"indirect_branch", "indirect_call"}:
                    kind = "cfg_indirect"
                else:
                    kind = "cfg_fallthrough"
                cfg_edges.append({
                    "edge_id": f"cfgedge_{elf.sha256[:12]}_{block.start:x}_{successor}",
                    "source": block.block_id,
                    "target": successor,
                    "kind": kind,
                    "confidence": 0.8 if kind != "cfg_indirect" else 0.2,
                    "evidence": block.evidence,
                })
                if successor.startswith("unknown_"):
                    cfg_nodes.append({"node_id": successor, "kind": "unknown"})
    cfg = {"nodes": cfg_nodes, "edges": cfg_edges, "confidence": 0.65, "limitations": ["Indirect CFG edges are not resolved."]}
    call_nodes = list(bundle.callgraph.nodes.values())
    call_edges = []
    for edge in bundle.callgraph.edges:
        call_edges.append({
            "edge_id": edge.edge_id,
            "source": edge.source,
            "target": edge.target,
            "kind": "call_direct" if edge.kind in {"call_direct", "call_external"} else "call_indirect",
            "attributes": {"external": edge.kind == "call_external", "target_name": edge.target_name},
            "confidence": edge.confidence,
            "evidence": edge.evidence,
        })
    callgraph = {"nodes": call_nodes, "edges": call_edges, "confidence": 0.6, "limitations": bundle.callgraph.limitations}
    type_records = []
    for function_id, candidates in bundle.type_candidates.items():
        for candidate in candidates:
            type_records.append({
                "type_id": candidate["type_id"],
                "name": candidate.get("name"),
                "kind": "integer",
                "size": candidate.get("size"),
                "signed": candidate.get("signed", False),
                "fields": [],
                "confidence": candidate.get("confidence", 0.3),
                "evidence": candidate.get("evidence", []),
            })
    evidence_registry: dict[str, dict] = {}
    binary_evidence = f"ev_{elf.sha256[:12]}_binary"
    evidence_registry[binary_evidence] = {
        "evidence_id": binary_evidence,
        "kind": "observed",
        "source": {"type": "elf", "path": bundle.path, "va": None, "file_offset": None, "tool": "nexus-elf", "version": ANALYSIS_VERSION},
        "claim": "Binary identity and ELF metadata were read from the selected file.",
        "payload_sha256": elf.sha256,
        "confidence": 1.0,
        "analysis_version": ANALYSIS_VERSION,
        "timestamp": bundle.finished_at,
        "untrusted": False,
    }
    for instruction in instructions:
        evidence_id = instruction["evidence"][0]
        evidence_registry[evidence_id] = {
            "evidence_id": evidence_id,
            "kind": "observed",
            "source": {"type": "disassembly", "path": bundle.path, "va": instruction["va"], "file_offset": instruction["file_offset"], "tool": "gnu-objdump", "version": "2.47"},
            "claim": f"Decoded instruction at {instruction['va']}.",
            "payload_sha256": None,
            "confidence": instruction["confidence"],
            "analysis_version": ANALYSIS_VERSION,
            "timestamp": bundle.finished_at,
            "untrusted": False,
        }
    for function in bundle.functions:
        evidence_id = f"ev_{elf.sha256[:12]}_func_{function.entry:x}"
        evidence_registry[evidence_id] = {
            "evidence_id": evidence_id,
            "kind": "derived",
            "source": {"type": "tool", "path": bundle.path, "va": f"0x{function.entry:x}", "file_offset": elf.va_to_offset(function.entry), "tool": "nexus-program", "version": ANALYSIS_VERSION},
            "claim": f"Function candidate {function.name} at 0x{function.entry:x}.",
            "payload_sha256": None,
            "confidence": function.confidence,
            "analysis_version": ANALYSIS_VERSION,
            "timestamp": bundle.finished_at,
            "untrusted": False,
        }
    for string in bundle.strings:
        for evidence_id in string["evidence"]:
            evidence_registry[evidence_id] = {
                "evidence_id": evidence_id,
                "kind": "observed",
                "source": {"type": "tool", "path": bundle.path, "va": string["va"], "file_offset": string.get("file_offset"), "tool": "nexus-strings", "version": ANALYSIS_VERSION},
                "claim": f"Printable string at {string['va']}.",
                "payload_sha256": None,
                "confidence": string["confidence"],
                "analysis_version": ANALYSIS_VERSION,
                "timestamp": bundle.finished_at,
                "untrusted": True,
            }
    _security_evidence(bundle, evidence_registry)
    record = {
        "schema_version": SCHEMA_VERSION,
        "program": {
            "sha256": elf.sha256,
            "size": elf.size,
            "format": "ELF",
            "class": elf.header.elf_class,
            "endianness": elf.header.endianness,
            "machine": elf.header.machine_name,
            "entry_va": f"0x{elf.header.entry:x}",
            "build_id": elf.build_id,
            "os_abi": {0: "System V", 3: "GNU", 9: "FreeBSD"}.get(elf.header.osabi, f"OSABI_{elf.header.osabi}"),
            "abi_version": elf.header.abi_version,
            "stripped": elf.is_stripped,
            "pie": elf.is_pie,
            "static": elf.is_static,
            "shared_object": elf.is_shared_object,
            "compiler_hints": sorted({item["value"] for item in bundle.strings if "GCC:" in item["value"] or "clang version" in item["value"]})[:10],
            "file_offset_to_va": [
                {"file_offset": segment.offset, "va": f"0x{segment.vaddr:x}", "size": segment.filesz}
                for segment in elf.program_headers if segment.p_type == 1 and segment.filesz
            ],
        },
        "segments": segments,
        "sections": sections,
        "symbols": symbols,
        "relocations": relocations,
        "imports": imports,
        "exports": exports,
        "dependencies": dependencies,
        "regions": regions,
        "functions": functions,
        "instructions": instructions,
        "cfg": cfg,
        "callgraph": callgraph,
        "dataflow": bundle.dataflows,
        "types": type_records,
        "strings": bundle.strings,
        "metadata": {
            "format": "ELF",
            "header": elf.header.to_dict(),
            "function_count": len(bundle.functions),
            "instruction_count": len(instructions),
            "security_findings": bundle.security_findings,
            "jni": bundle.jni_report,
            "binary_sha256": elf.sha256,
        },
        "security_findings": bundle.security_findings,
        "jni": bundle.jni_report,
        "analysis": {
            "analysis_version": ANALYSIS_VERSION,
            "pipeline": PIPELINE,
            "started_at": bundle.started_at,
            "finished_at": bundle.finished_at,
            "tool_versions": {"objdump": "2.47", "nexus-elf": ANALYSIS_VERSION, "nexus-program": ANALYSIS_VERSION},
            "limitations": bundle.limitations,
            "unknowns": [],
        },
        "evidence": [],
        "unknowns": [],
    }
    unknown_ids = set()
    for function in bundle.functions:
        for unknown in function.unknowns:
            if unknown.get("unknown_id") and unknown["unknown_id"] not in unknown_ids:
                record["unknowns"].append(unknown)
                unknown_ids.add(unknown["unknown_id"])
    _ensure_evidence(record, evidence_registry, bundle)
    record["evidence"] = [evidence_registry[key] for key in sorted(evidence_registry)]
    return record
