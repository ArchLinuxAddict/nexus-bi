import bisect
import hashlib
import os
from collections import Counter
from pathlib import Path
from typing import Any

from elftools.elf.elffile import ELFFile

from nexus.core.hashing import sha256_file
from nexus.disassembly.model import Instruction
from nexus.disassembly.objdump import disassemble_elf
from nexus.elf.model import ElfFile
from nexus.elf.parser import parse_elf
from nexus.program.analysis import recover_functions

ALIGNMENT_SCHEMA_VERSION = "0.2.0"
ALIGNMENT_CLASSIFICATIONS = ("EXACT", "DEBUG_DERIVED", "SYMBOL_DERIVED", "HEURISTIC", "UNVERIFIED", "UNAVAILABLE")
MAX_CUS = 500
MAX_FUNCTIONS = 5000
MAX_LINE_ENTRIES = 500_000
MAX_VARIABLES = 10_000
MAX_INSTRUCTIONS = 300_000


def _text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _attribute(die: Any, name: str) -> Any:
    return die.attributes.get(name)


def _attr_value(die: Any, name: str, default: Any = None) -> Any:
    attribute = die.attributes.get(name)
    return default if attribute is None else attribute.value


def _int_attr(die: Any, name: str) -> int | None:
    attribute = die.attributes.get(name)
    if attribute is None:
        return None
    try:
        return int(attribute.value)
    except (TypeError, ValueError):
        return None


def _file_name(line_program: Any, index: int | None) -> tuple[str, str]:
    if index is None:
        return "", ""
    header = getattr(line_program, "header", None)
    entries = getattr(header, "file_entry", None) or []
    if not entries:
        return "", ""
    entry = None
    for candidate_index in (index, index - 1):
        try:
            candidate = entries[candidate_index]
        except (IndexError, TypeError):
            continue
        if candidate is not None and _text(getattr(candidate, "name", "")):
            entry = candidate
            break
    if entry is None:
        return "", ""
    name = _text(getattr(entry, "name", ""))
    directory = ""
    directories = getattr(header, "include_directory", None) or []
    directory_index = getattr(entry, "dir_index", None)
    if directory_index is not None:
        try:
            directory = _text(directories[directory_index])
        except (IndexError, TypeError):
            directory = ""
    if directory and not os.path.isabs(name):
        name = os.path.join(directory, name)
    return name, directory


def _normalize_source(path: str, comp_dir: str, source_root: Path, inventory: dict[str, str]) -> tuple[str, str | None, str]:
    value = (path or "").replace("\\", "/")
    if not value:
        return "", None, "unresolved"
    comp = (comp_dir or "").replace("\\", "/")
    if comp and value.startswith(comp):
        value = value[len(comp) :].lstrip("/")
    if value.startswith("./"):
        value = value[2:]
    value = os.path.normpath(value).replace("\\", "/")
    if os.path.isabs(value):
        try:
            value = str(Path(value).relative_to(source_root)).replace("\\", "/")
        except ValueError:
            value = Path(value).name
    if value in inventory:
        return value, inventory[value], "exact"
    matches = [candidate for candidate in inventory if candidate.endswith("/" + value) or candidate == value]
    if len(matches) == 1:
        return matches[0], inventory[matches[0]], "suffix"
    basename = Path(value).name
    basename_matches = [candidate for candidate in inventory if Path(candidate).name == basename]
    if len(basename_matches) == 1:
        return basename_matches[0], inventory[basename_matches[0]], "basename"
    return value, None, "unresolved"


def _source_inventory(source_root: Path) -> dict[str, str]:
    inventory = {}
    if not source_root.exists():
        return inventory
    for path in source_root.rglob("*"):
        if not path.is_file() or path.is_symlink():
            continue
        relative = str(path.relative_to(source_root)).replace("\\", "/")
        if path.suffix.lower() in {".c", ".cc", ".cpp", ".cxx", ".h", ".hpp", ".inc"}:
            inventory[relative] = sha256_file(path)
    return inventory


def _type_name(die: Any, dies_by_offset: dict[int, Any], depth: int = 0) -> str:
    if depth > 6:
        return "unknown"
    attribute = die.attributes.get("DW_AT_type")
    if attribute is None:
        return "void"
    target = dies_by_offset.get(int(attribute.value))
    if target is None:
        return "unknown"
    name = target.attributes.get("DW_AT_name")
    if name is not None and _text(name.value):
        return _text(name.value)
    if target.tag in {"DW_TAG_pointer_type", "DW_TAG_reference_type", "DW_TAG_rvalue_reference_type"}:
        suffix = "*" if target.tag == "DW_TAG_pointer_type" else "&"
        return _type_name(target, dies_by_offset, depth + 1) + suffix
    if target.tag in {"DW_TAG_const_type", "DW_TAG_volatile_type", "DW_TAG_restrict_type"}:
        prefix = {"DW_TAG_const_type": "const ", "DW_TAG_volatile_type": "volatile ", "DW_TAG_restrict_type": "restrict "}[target.tag]
        return prefix + _type_name(target, dies_by_offset, depth + 1)
    if target.tag == "DW_TAG_array_type":
        return _type_name(target, dies_by_offset, depth + 1) + "[]"
    if target.tag in {"DW_TAG_structure_type", "DW_TAG_union_type", "DW_TAG_enumeration_type", "DW_TAG_class_type"}:
        kind = target.tag.replace("DW_TAG_", "").replace("_type", "")
        return f"{kind} {{anonymous}}"
    if target.tag == "DW_TAG_typedef":
        return _text(name.value) if name is not None else "typedef"
    if target.tag == "DW_TAG_base_type":
        return _text(name.value) if name is not None else "base"
    if target.tag == "DW_TAG_subroutine_type":
        return "function"
    return target.tag.replace("DW_TAG_", "").lower()


def _ranges_from_die(die: Any, cu: Any, dwarf: Any) -> list[tuple[int, int]]:
    ranges: list[tuple[int, int]] = []
    low = _int_attr(die, "DW_AT_low_pc")
    high_attr = die.attributes.get("DW_AT_high_pc")
    if low is not None and high_attr is not None:
        try:
            high = int(high_attr.value)
            end = high if high_attr.form == "DW_FORM_addr" else low + high
            if end > low:
                ranges.append((low, end))
        except (TypeError, ValueError):
            pass
    if not ranges:
        ranges_attr = die.attributes.get("DW_AT_ranges")
        if ranges_attr is not None:
            try:
                range_list = dwarf.range_lists().get_range_list_at_offset(int(ranges_attr.value), cu=cu)
                base = _int_attr(cu.get_top_DIE(), "DW_AT_low_pc") or 0
                for item in range_list:
                    start = int(item.begin_offset)
                    end = int(item.end_offset)
                    if base and start < base:
                        start += base
                        end += base
                    if end > start:
                        ranges.append((start, end))
            except Exception:
                pass
    ranges.sort()
    merged: list[tuple[int, int]] = []
    for start, end in ranges:
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def _parse_location(value: Any, address_size: int = 8, dwarf: Any = None, cu: Any = None) -> dict[str, Any]:
    if value is None:
        return {"kind": "unavailable", "expression_hex": None, "address": None, "register": None, "offset": None}
    if isinstance(value, int):
        return {"kind": "location_list", "expression_hex": None, "address": None, "register": None, "offset": None}
    if isinstance(value, str):
        value = value.encode("latin-1", errors="ignore")
    if not isinstance(value, (bytes, bytearray, list, tuple)):
        return {"kind": "unparsed", "expression_hex": None, "address": None, "register": None, "offset": None}
    data = bytes(value)
    expression = data.hex()
    if not data:
        return {"kind": "unavailable", "expression_hex": expression, "address": None, "register": None, "offset": None}
    opcode = data[0]
    if opcode == 0xA1 and len(data) >= 2 and dwarf is not None and cu is not None:
        try:
            index = _uleb128(data[1:])
            address = int(dwarf.get_addr(cu, index))
            return {"kind": "indexed_address", "expression_hex": expression, "address": f"0x{address:x}", "register": None, "offset": None}
        except Exception:
            return {"kind": "indexed_address_unresolved", "expression_hex": expression, "address": None, "register": None, "offset": None}
    if opcode == 0x03 and len(data) >= 1 + address_size:
        address = int.from_bytes(data[1 : 1 + address_size], "little")
        return {"kind": "absolute_address", "expression_hex": expression, "address": f"0x{address:x}", "register": None, "offset": None}
    if opcode == 0x91 and len(data) >= 2:
        offset = _sleb128(data[1:])
        return {"kind": "frame_base", "expression_hex": expression, "address": None, "register": None, "offset": offset}
    if 0x50 <= opcode <= 0x6F:
        return {"kind": "register", "expression_hex": expression, "address": None, "register": opcode - 0x50, "offset": None}
    if opcode == 0x9C:
        return {"kind": "call_frame_cfa", "expression_hex": expression, "address": None, "register": None, "offset": None}
    if opcode == 0x9F:
        return {"kind": "stack_value", "expression_hex": expression, "address": None, "register": None, "offset": None}
    if opcode == 0x23 and len(data) >= 2:
        return {"kind": "plus_uconst", "expression_hex": expression, "address": None, "register": None, "offset": _uleb128(data[1:])}
    return {"kind": "unparsed", "expression_hex": expression, "address": None, "register": None, "offset": None}


def _sleb128(data: bytes) -> int:
    result = 0
    shift = 0
    for byte in data:
        result |= (byte & 0x7F) << shift
        shift += 7
        if not (byte & 0x80):
            if byte & 0x40:
                result -= 1 << shift
            break
    return result


def _uleb128(data: bytes) -> int:
    result = 0
    shift = 0
    for byte in data:
        result |= (byte & 0x7F) << shift
        shift += 7
        if not (byte & 0x80):
            break
    return result


def _name_from_die(die: Any, dies_by_offset: dict[int, Any]) -> tuple[str, str, Any]:
    seen = set()
    current = die
    while current is not None and current.offset not in seen:
        seen.add(current.offset)
        name = _attr_value(current, "DW_AT_name")
        linkage = _attr_value(current, "DW_AT_linkage_name")
        if name or linkage:
            return _text(name), _text(linkage), current
        reference = current.attributes.get("DW_AT_abstract_origin") or current.attributes.get("DW_AT_specification")
        if reference is None:
            break
        current = dies_by_offset.get(int(reference.value))
    return "", "", die


def _symbol_tables(elf: ElfFile) -> tuple[dict[int, list[dict[str, Any]]], dict[int, list[dict[str, Any]]], list[dict[str, Any]]]:
    functions: dict[int, list[dict[str, Any]]] = {}
    objects: dict[int, list[dict[str, Any]]] = {}
    records = []
    seen = set()
    for symbol in list(elf.symbols) + list(elf.dynamic_symbols):
        if symbol.shndx == 0 or not symbol.name:
            continue
        if symbol.symbol_type not in {"FUNC", "OBJECT"}:
            continue
        key = (symbol.name, symbol.value, symbol.size, symbol.symbol_type, symbol.table)
        if key in seen:
            continue
        seen.add(key)
        record = {
            "name": symbol.name,
            "value": symbol.value,
            "size": symbol.size,
            "type": symbol.symbol_type,
            "binding": symbol.binding,
            "visibility": symbol.visibility,
            "table": symbol.table,
            "section_index": symbol.shndx,
            "range": [symbol.value, symbol.value + symbol.size] if symbol.size else None,
        }
        records.append(record)
        if symbol.symbol_type == "FUNC":
            functions.setdefault(symbol.value, []).append(record)
        elif symbol.symbol_type == "OBJECT":
            objects.setdefault(symbol.value, []).append(record)
    records.sort(key=lambda item: (item["value"], item["name"]))
    return functions, objects, records


def _best_symbol(candidates: list[dict[str, Any]] | None) -> dict[str, Any] | None:
    if not candidates:
        return None
    return sorted(candidates, key=lambda item: (item["size"] > 0, item["binding"] == "GLOBAL", item["name"]))[-1]


def _build_blocks(function_id: str, instructions: list[Instruction]) -> list[dict[str, Any]]:
    if not instructions:
        return []
    by_address = {instruction.address: instruction for instruction in instructions}
    addresses = set(by_address)
    address_list = [instruction.address for instruction in instructions]
    leaders = {instructions[0].address}
    for instruction in instructions:
        if instruction.control_flow in {"conditional_branch", "unconditional_branch", "return", "trap"}:
            if instruction.end in addresses:
                leaders.add(instruction.end)
        for target in instruction.targets:
            if target.va in addresses:
                leaders.add(target.va)
    ordered = sorted(leaders)
    blocks = []
    for index, start in enumerate(ordered):
        end = ordered[index + 1] if index + 1 < len(ordered) else None
        lower = bisect.bisect_left(address_list, start)
        upper = bisect.bisect_left(address_list, end) if end is not None else len(instructions)
        members = instructions[lower:upper]
        if not members:
            continue
        blocks.append(
            {
                "block_id": f"bb_{function_id.split('_')[-1]}_{start:x}",
                "function_id": function_id,
                "start_va": f"0x{start:x}",
                "end_va": f"0x{members[-1].end:x}",
                "instruction_ids": [f"ins_{instruction.address:x}" for instruction in members],
                "successors": [],
                "classification": "HEURISTIC",
                "confidence": 0.65,
                "_last_address": members[-1].address,
            }
        )
    by_start = {int(block["start_va"], 16): block for block in blocks}
    for index, block in enumerate(blocks):
        last = by_address.get(block.pop("_last_address", None))
        if last is None:
            continue
        if last.control_flow == "conditional_branch":
            for target in last.targets:
                if target.va in by_start:
                    block["successors"].append(by_start[target.va]["block_id"])
            if index + 1 < len(blocks) and last.end == int(blocks[index + 1]["start_va"], 16):
                block["successors"].append(blocks[index + 1]["block_id"])
        elif last.control_flow == "unconditional_branch":
            for target in last.targets:
                if target.va in by_start:
                    block["successors"].append(by_start[target.va]["block_id"])
        elif last.control_flow in {"return", "trap"}:
            continue
        elif index + 1 < len(blocks) and last.end == int(blocks[index + 1]["start_va"], 16):
            block["successors"].append(blocks[index + 1]["block_id"])
        block["successors"] = sorted(set(block["successors"]))
    return blocks


def _line_sequences(line_program: Any) -> list[tuple[list[Any], int | None]]:
    sequences = []
    current = []
    end_address = None
    for entry in line_program.get_entries():
        state = entry.state
        if state is None:
            continue
        if state.end_sequence:
            if current:
                sequences.append((current, int(state.address) if state.address else end_address))
            current = []
            end_address = None
            continue
        if not state.address or not state.line:
            continue
        current.append(state)
        end_address = int(state.address)
    if current:
        sequences.append((current, end_address))
    return sequences


def _collect_dwarf(binary: Path, source_root: Path, inventory: dict[str, str], record: dict[str, Any] | None) -> dict[str, Any]:
    compilation_units = []
    functions = []
    line_mappings = []
    variables = []
    inlined = []
    limitations = []
    try:
        with open(binary, "rb") as handle:
            elf = ELFFile(handle)
            if not elf.has_dwarf_info():
                return {"status": "unavailable", "reason": "no DWARF information", "compilation_units": [], "functions": [], "line_mappings": [], "variables": [], "inlined": [], "limitations": ["No .debug_info section."]}
            dwarf = elf.get_dwarf_info()
            line_index = 0
            for cu_index, cu in enumerate(dwarf.iter_CUs()):
                if cu_index >= MAX_CUS:
                    limitations.append(f"compilation-unit cap reached ({MAX_CUS})")
                    break
                top = cu.get_top_DIE()
                cu_name_raw = _text(_attr_value(top, "DW_AT_name"))
                comp_dir = _text(_attr_value(top, "DW_AT_comp_dir"))
                cu_name, cu_sha, cu_match = _normalize_source(cu_name_raw, comp_dir, source_root, inventory)
                cu_id = f"cu_{binary.stem}_{cu.cu_offset:x}"
                line_program = dwarf.line_program_for_CU(cu)
                dies_by_offset = {die.offset: die for die in cu.iter_DIEs()}
                cu_record = {
                    "cu_id": cu_id,
                    "offset": cu.cu_offset,
                    "name": cu_name,
                    "raw_name": cu_name_raw,
                    "comp_dir": comp_dir,
                    "producer": _text(_attr_value(top, "DW_AT_producer")),
                    "language": _text(_attr_value(top, "DW_AT_language")),
                    "source_sha256": cu_sha,
                    "path_match": cu_match,
                    "function_ids": [],
                    "line_ranges": 0,
                    "classification": "DEBUG_DERIVED",
                    "confidence": 0.95 if cu_sha else 0.7,
                    "method": "dwarf_compilation_unit",
                }
                function_by_die: dict[int, dict[str, Any]] = {}
                for die in dies_by_offset.values():
                    if die.tag not in {"DW_TAG_subprogram", "DW_TAG_inlined_subroutine"}:
                        continue
                    if len(functions) + len(inlined) >= MAX_FUNCTIONS * 2:
                        limitations.append("function cap reached")
                        break
                    ranges = _ranges_from_die(die, cu, dwarf)
                    if not ranges:
                        continue
                    name, linkage, name_die = _name_from_die(die, dies_by_offset)
                    if not name and not linkage:
                        continue
                    decl_file = _int_attr(name_die, "DW_AT_decl_file")
                    decl_line = _int_attr(name_die, "DW_AT_decl_line")
                    decl_name, _ = _file_name(line_program, decl_file) if decl_file is not None else ("", "")
                    source_file, source_sha, source_match = _normalize_source(decl_name, comp_dir, source_root, inventory) if decl_name else (cu_name, cu_sha, cu_match)
                    entry = ranges[0][0]
                    function_id = f"fn_{binary.stem}_{entry:x}"
                    if any(item["function_id"] == function_id for item in functions):
                        function_id = f"fn_{binary.stem}_{entry:x}_{len(functions):x}"
                    record_item = {
                        "function_id": function_id,
                        "name": name or linkage,
                        "linkage_name": linkage,
                        "entry_va": f"0x{entry:x}",
                        "ranges": [{"start_va": f"0x{start:x}", "end_va": f"0x{end:x}", "size": end - start} for start, end in ranges],
                        "size": sum(end - start for start, end in ranges),
                        "source_file": source_file,
                        "source_sha256": source_sha,
                        "source_line": decl_line,
                        "source_path_match": source_match,
                        "compilation_unit": cu_id,
                        "die_offset": die.offset,
                        "inline": die.tag == "DW_TAG_inlined_subroutine",
                        "classification": "DEBUG_DERIVED",
                        "confidence": 0.95 if source_sha else 0.75,
                        "methods": ["dwarf"],
                        "variables": [],
                        "instruction_ids": [],
                        "block_ids": [],
                    }
                    if record_item["inline"]:
                        inlined.append(record_item)
                    else:
                        functions.append(record_item)
                        function_by_die[die.offset] = record_item
                        cu_record["function_ids"].append(function_id)
                if line_program is not None:
                    for sequence, sequence_end in _line_sequences(line_program):
                        sequence.sort(key=lambda state: state.address)
                        for index, state in enumerate(sequence):
                            start = int(state.address)
                            end = int(sequence[index + 1].address) if index + 1 < len(sequence) else (sequence_end if sequence_end and sequence_end > start else start + 1)
                            if end <= start:
                                continue
                            raw_name, _ = _file_name(line_program, state.file)
                            source_file, source_sha, source_match = _normalize_source(raw_name, comp_dir, source_root, inventory)
                            line_id = f"line_{binary.stem}_{line_index:x}"
                            line_index += 1
                            line_mappings.append(
                                {
                                    "line_id": line_id,
                                    "start_va": f"0x{start:x}",
                                    "end_va": f"0x{end:x}",
                                    "source_file": source_file,
                                    "source_sha256": source_sha,
                                    "source_line": int(state.line),
                                    "column": int(state.column) if state.column else None,
                                    "is_stmt": bool(state.is_stmt),
                                    "discriminator": int(state.discriminator) if getattr(state, "discriminator", None) else None,
                                    "compilation_unit": cu_id,
                                    "path_match": source_match,
                                    "classification": "DEBUG_DERIVED",
                                    "confidence": 0.95 if state.is_stmt else 0.8,
                                    "method": "dwarf_line_table",
                                    "instruction_ids": [],
                                    "function_id": None,
                                }
                            )
                            cu_record["line_ranges"] += 1
                            if len(line_mappings) >= MAX_LINE_ENTRIES:
                                limitations.append("line-entry cap reached")
                                break
                for die in dies_by_offset.values():
                    if die.tag not in {"DW_TAG_variable", "DW_TAG_formal_parameter"} or len(variables) >= MAX_VARIABLES:
                        continue
                    name = _text(_attr_value(die, "DW_AT_name"))
                    if not name:
                        continue
                    parent = die.get_parent()
                    function = None
                    seen_parents = set()
                    while parent is not None and parent.offset not in seen_parents:
                        seen_parents.add(parent.offset)
                        function = function_by_die.get(parent.offset)
                        if function is not None:
                            break
                        parent = parent.get_parent()
                    decl_file = _int_attr(die, "DW_AT_decl_file")
                    decl_line = _int_attr(die, "DW_AT_decl_line")
                    if decl_file is not None:
                        decl_name, _ = _file_name(line_program, decl_file)
                        source_file, source_sha, source_match = _normalize_source(decl_name, comp_dir, source_root, inventory)
                    else:
                        source_file, source_sha, source_match = cu_name, cu_sha, cu_match
                    location = _parse_location(_attr_value(die, "DW_AT_location"), dwarf=dwarf, cu=cu)
                    var_id = f"var_{binary.stem}_{die.offset:x}"
                    item = {
                        "variable_id": var_id,
                        "name": name,
                        "type": _type_name(die, dies_by_offset),
                        "role": "parameter" if die.tag == "DW_TAG_formal_parameter" else "variable",
                        "source_file": source_file,
                        "source_sha256": source_sha,
                        "source_line": decl_line,
                        "path_match": source_match,
                        "compilation_unit": cu_id,
                        "function_id": function["function_id"] if function else None,
                        "location": location,
                        "classification": "DEBUG_DERIVED" if location["kind"] not in {"unavailable", "unparsed", "location_list", "indexed_address_unresolved"} else "UNAVAILABLE",
                        "confidence": 0.9 if location["kind"] in {"absolute_address", "indexed_address"} else 0.7 if location["kind"] not in {"unavailable", "unparsed", "location_list", "indexed_address_unresolved"} else 0.2,
                        "method": "dwarf_location_expression",
                        "die_offset": die.offset,
                    }
                    variables.append(item)
                    if function:
                        function["variables"].append(var_id)
                compilation_units.append(cu_record)
            return {
                "status": "available",
                "compilation_units": compilation_units,
                "functions": functions,
                "line_mappings": line_mappings,
                "variables": variables,
                "inlined": inlined,
                "limitations": limitations,
            }
    except Exception as exc:
        return {
            "status": "unavailable",
            "reason": f"{type(exc).__name__}: {exc}",
            "compilation_units": compilation_units,
            "functions": functions,
            "line_mappings": line_mappings,
            "variables": variables,
            "inlined": inlined,
            "limitations": limitations + [f"DWARF extraction failed: {exc}"],
        }


def extract_alignment(
    binary: str | Path,
    source_root: str | Path,
    *,
    project_id: str | None = None,
    build_variant: dict[str, Any] | None = None,
    record: dict[str, Any] | None = None,
) -> dict[str, Any]:
    binary_path = Path(binary).resolve()
    source_path = Path(source_root).resolve()
    elf = parse_elf(binary_path)
    inventory = _source_inventory(source_path)
    symbol_functions, symbol_objects, symbol_records = _symbol_tables(elf)
    dwarf = _collect_dwarf(binary_path, source_path, inventory, record)
    limitations = list(dwarf.get("limitations", []))
    limitations.extend(
        [
            "Instruction alignment is DWARF line-table address coverage, not token-level or AST-level source mapping.",
            "EXACT is used only when DWARF and symbol-table function boundaries agree; line-table mappings are DEBUG_DERIVED.",
            "Basic-block source ranges are intersections of CFG blocks and line-table ranges, classified HEURISTIC.",
            "Variable addresses are recovered only from simple location expressions; location lists and complex expressions remain UNAVAILABLE.",
            "Optimized builds may merge, split or inline functions; absence of a mapping is not evidence that source is absent.",
        ]
    )
    disassembly = None
    try:
        disassembly = disassemble_elf(elf)
    except Exception as exc:
        limitations.append(f"Disassembly unavailable: {type(exc).__name__}: {exc}")
    instructions = sorted(disassembly.instructions, key=lambda item: item.address) if disassembly is not None else []
    if len(instructions) > MAX_INSTRUCTIONS:
        instructions = instructions[:MAX_INSTRUCTIONS]
        limitations.append(f"instruction cap reached ({MAX_INSTRUCTIONS})")
    instruction_addresses = [instruction.address for instruction in instructions]

    def select_instructions(start: int, end: int) -> list[Instruction]:
        lower = bisect.bisect_left(instruction_addresses, start)
        upper = bisect.bisect_left(instruction_addresses, end)
        return instructions[lower:upper]

    functions = dwarf.get("functions", [])
    by_entry: dict[int, dict[str, Any]] = {}
    for function in functions:
        entry = int(function["entry_va"], 16)
        symbol = _best_symbol(symbol_functions.get(entry))
        function["symbol"] = None
        if symbol is not None:
            symbol_start, symbol_end = symbol["range"] if symbol["range"] else (symbol["value"], symbol["value"])
            dwarf_start = entry
            dwarf_end = max(int(item["end_va"], 16) for item in function["ranges"])
            names = {function.get("name"), function.get("linkage_name")}
            name_match = symbol["name"] in names
            function["symbol"] = {key: symbol[key] for key in ("name", "value", "size", "binding", "table")}
            if symbol["range"] and symbol_start == dwarf_start and symbol_end == dwarf_end and name_match:
                function["classification"] = "EXACT"
                function["confidence"] = 0.99
                function["methods"] = ["dwarf", "symbol"]
            elif symbol_start == dwarf_start:
                function["classification"] = "DEBUG_DERIVED"
                function["confidence"] = 0.9
                function["methods"] = ["dwarf", "symbol"]
                if not name_match:
                    function["name_mismatch"] = True
            else:
                function["methods"] = ["dwarf"]
        by_entry[entry] = function
    matched_entries = set(by_entry)
    for entry, candidates in sorted(symbol_functions.items()):
        if entry in matched_entries:
            continue
        symbol = _best_symbol(candidates)
        if symbol is None:
            continue
        end = symbol["range"][1] if symbol["range"] else entry
        function_id = f"fn_{binary_path.stem}_{entry:x}"
        item = {
            "function_id": function_id,
            "name": symbol["name"],
            "linkage_name": symbol["name"],
            "entry_va": f"0x{entry:x}",
            "ranges": [{"start_va": f"0x{entry:x}", "end_va": f"0x{end:x}", "size": max(0, end - entry)}] if symbol["range"] else [],
            "size": symbol["size"],
            "source_file": None,
            "source_sha256": None,
            "source_line": None,
            "source_path_match": "unavailable",
            "compilation_unit": None,
            "inline": False,
            "classification": "SYMBOL_DERIVED" if symbol["size"] > 0 else "UNVERIFIED",
            "confidence": 0.85 if symbol["size"] > 0 else 0.4,
            "methods": ["symbol"],
            "symbol": {key: symbol[key] for key in ("name", "value", "size", "binding", "table")},
            "variables": [],
            "instruction_ids": [],
            "block_ids": [],
        }
        functions.append(item)
        by_entry[entry] = item
        matched_entries.add(entry)
    if not functions and disassembly is not None:
        recovered, _ = recover_functions(elf, disassembly)
        for candidate in recovered[:MAX_FUNCTIONS]:
            item = {
                "function_id": candidate.function_id,
                "name": candidate.name,
                "linkage_name": None,
                "entry_va": f"0x{candidate.entry:x}",
                "ranges": [{"start_va": f"0x{start:x}", "end_va": f"0x{end:x}", "size": end - start} for start, end in candidate.ranges],
                "size": candidate.size,
                "source_file": None,
                "source_sha256": None,
                "source_line": None,
                "source_path_match": "unavailable",
                "compilation_unit": None,
                "inline": False,
                "classification": "HEURISTIC",
                "confidence": 0.35,
                "methods": ["recovery"],
                "symbol": None,
                "variables": [],
                "instruction_ids": [f"ins_{instruction.address:x}" for instruction in candidate.instructions],
                "block_ids": [],
            }
            functions.append(item)
            by_entry[candidate.entry] = item
        limitations.append("No DWARF or symbol functions; boundaries come from heuristic recovery and have no source mapping.")
    line_mappings = dwarf.get("line_mappings", [])
    function_ranges = []
    for function in functions:
        for item in function.get("ranges", []):
            function_ranges.append((int(item["start_va"], 16), int(item["end_va"], 16), function))
    function_ranges.sort(key=lambda item: (item[0], item[1]))
    starts = [item[0] for item in function_ranges]
    for mapping in line_mappings:
        start = int(mapping["start_va"], 16)
        end = int(mapping["end_va"], 16)
        index = bisect.bisect_right(starts, start) - 1
        owner = None
        if index >= 0:
            candidate_start, candidate_end, candidate = function_ranges[index]
            if candidate_start <= start < candidate_end:
                owner = candidate
        if owner is not None:
            mapping["function_id"] = owner["function_id"]
            if end > candidate_end:
                mapping["crosses_function_end"] = True
        mapping["instruction_ids"] = [f"ins_{instruction.address:x}" for instruction in select_instructions(start, end)]
        mapping["instruction_count"] = len(mapping["instruction_ids"])
    instruction_alignments = []
    mapped_addresses = set()
    for mapping in line_mappings:
        for instruction_id in mapping["instruction_ids"]:
            if instruction_id in mapped_addresses:
                continue
            mapped_addresses.add(instruction_id)
            address = int(instruction_id.split("_")[-1], 16)
            instruction_alignments.append(
                {
                    "instruction_id": instruction_id,
                    "address": f"0x{address:x}",
                    "function_id": mapping.get("function_id"),
                    "line_id": mapping["line_id"],
                    "source_file": mapping.get("source_file"),
                    "source_sha256": mapping.get("source_sha256"),
                    "source_line": mapping.get("source_line"),
                    "column": mapping.get("column"),
                    "is_stmt": mapping.get("is_stmt"),
                    "classification": "DEBUG_DERIVED",
                    "confidence": mapping.get("confidence", 0.8),
                    "method": "dwarf_line_table_address_coverage",
                }
            )
    for function in functions:
        selected = []
        for item in function.get("ranges", []):
            start = int(item["start_va"], 16)
            end = int(item["end_va"], 16)
            selected.extend(select_instructions(start, end))
        selected = sorted(selected, key=lambda item: item.address)
        function["instruction_ids"] = [f"ins_{instruction.address:x}" for instruction in selected]
        function["instruction_count"] = len(selected)
        function["mnemonic_histogram"] = dict(sorted(Counter(instruction.mnemonic for instruction in selected).items()))
        code = bytearray()
        for item in function.get("ranges", []):
            start = int(item["start_va"], 16)
            size = int(item["end_va"], 16) - start
            payload = elf.read_va(start, size)
            if payload:
                code.extend(payload)
        function["code_sha256"] = hashlib.sha256(bytes(code)).hexdigest() if code else None
        blocks = _build_blocks(function["function_id"], selected)
        function["block_ids"] = [block["block_id"] for block in blocks]
        function["blocks"] = blocks
    block_alignments = []
    for function in functions:
        line_by_instruction = {item["instruction_id"]: item for item in instruction_alignments}
        for block in function.get("blocks", []):
            lines = [line_by_instruction[instruction_id] for instruction_id in block["instruction_ids"] if instruction_id in line_by_instruction]
            if not lines:
                block_alignments.append(
                    {
                        **block,
                        "source_file": None,
                        "start_line": None,
                        "end_line": None,
                        "classification": "UNAVAILABLE",
                        "confidence": 0.0,
                        "method": "cfg_line_table_intersection",
                        "line_count": 0,
                    }
                )
                continue
            files = {item.get("source_file") for item in lines if item.get("source_file")}
            block_alignments.append(
                {
                    **block,
                    "source_file": lines[0].get("source_file"),
                    "start_line": min(item.get("source_line") or 0 for item in lines) or None,
                    "end_line": max(item.get("source_line") or 0 for item in lines) or None,
                    "line_count": len({(item.get("source_file"), item.get("source_line")) for item in lines}),
                    "multiple_source_files": len(files) > 1,
                    "classification": "HEURISTIC",
                    "confidence": 0.55 if len(files) == 1 else 0.35,
                    "method": "cfg_line_table_intersection",
                }
            )
    call_graph_nodes = []
    call_graph_edges = []
    function_entries = {int(function["entry_va"], 16): function for function in functions if function.get("entry_va")}
    ranges = [(int(item["start_va"], 16), int(item["end_va"], 16), function) for function in functions for item in function.get("ranges", [])]
    ranges.sort(key=lambda item: item[0])
    range_starts = [item[0] for item in ranges]
    for function in functions:
        call_graph_nodes.append({"node_id": function["function_id"], "name": function.get("name"), "entry_va": function.get("entry_va"), "classification": function.get("classification")})
    for instruction in instructions:
        if instruction.control_flow not in {"call", "indirect_call"}:
            continue
        source_function = None
        index = bisect.bisect_right(range_starts, instruction.address) - 1
        if index >= 0:
            start, end, candidate = ranges[index]
            if start <= instruction.address < end:
                source_function = candidate
        if source_function is None:
            continue
        for target in instruction.targets:
            target_function = function_entries.get(target.va) if target.va is not None else None
            if target_function is None and target.va is not None:
                target_index = bisect.bisect_right(range_starts, target.va) - 1
                if target_index >= 0:
                    start, end, candidate = ranges[target_index]
                    if start <= target.va < end:
                        target_function = candidate
            if target_function is not None:
                call_graph_edges.append(
                    {
                        "edge_id": f"call_{instruction.address:x}",
                        "source": source_function["function_id"],
                        "target": target_function["function_id"],
                        "call_site_va": f"0x{instruction.address:x}",
                        "kind": "direct" if instruction.control_flow == "call" else "indirect_resolved",
                        "classification": "SYMBOL_DERIVED" if "symbol" in source_function.get("methods", []) and "symbol" in target_function.get("methods", []) else "DEBUG_DERIVED",
                        "confidence": 0.9 if instruction.control_flow == "call" else 0.6,
                    }
                )
            elif instruction.control_flow == "indirect_call":
                call_graph_edges.append(
                    {
                        "edge_id": f"call_{instruction.address:x}",
                        "source": source_function["function_id"],
                        "target": None,
                        "call_site_va": f"0x{instruction.address:x}",
                        "kind": "indirect_unresolved",
                        "classification": "UNVERIFIED",
                        "confidence": 0.2,
                    }
                )
    call_graph = {"nodes": call_graph_nodes, "edges": call_graph_edges, "classification": "HEURISTIC", "confidence": 0.6}
    variables = dwarf.get("variables", [])
    data_objects = []
    for variable in variables:
        address = variable.get("location", {}).get("address")
        if not address:
            continue
        address_int = int(address, 16)
        symbol = _best_symbol(symbol_objects.get(address_int))
        record_item = {
            "data_object_id": f"data_{binary_path.stem}_{address_int:x}",
            "name": variable["name"],
            "address": address,
            "type": variable["type"],
            "source_file": variable.get("source_file"),
            "source_line": variable.get("source_line"),
            "variable_id": variable["variable_id"],
            "symbol": {key: symbol[key] for key in ("name", "value", "size", "binding", "table")} if symbol else None,
            "classification": "EXACT" if symbol and symbol["name"] == variable["name"] and symbol["size"] > 0 else "DEBUG_DERIVED",
            "confidence": 0.99 if symbol and symbol["name"] == variable["name"] else 0.9,
            "method": "dwarf_location_expression",
            "methods": ["dwarf", "symbol"] if symbol else ["dwarf"],
        }
        data_objects.append(record_item)
    for variable, data in zip([item for item in variables if item.get("location", {}).get("address")], data_objects):
        variable["data_object_id"] = data["data_object_id"]
    source_level = []
    for relative, digest in sorted(inventory.items()):
        source_functions = [function for function in functions if function.get("source_file") == relative]
        source_lines = [mapping for mapping in line_mappings if mapping.get("source_file") == relative]
        cus = sorted({function.get("compilation_unit") for function in source_functions if function.get("compilation_unit")})
        if not source_functions and not source_lines:
            continue
        source_level.append(
            {
                "source_file": relative,
                "sha256": digest,
                "compilation_units": cus,
                "function_count": len(source_functions),
                "line_range_count": len(source_lines),
                "classification": "DEBUG_DERIVED",
                "confidence": 0.95 if source_functions else 0.8,
                "method": "dwarf_cu_and_line_table",
            }
        )
    classifications = {name: 0 for name in ALIGNMENT_CLASSIFICATIONS}
    for collection in (functions, line_mappings, instruction_alignments, block_alignments, variables, data_objects, dwarf.get("inlined", [])):
        for item in collection:
            classification = item.get("classification") or "UNVERIFIED"
            classifications[classification] = classifications.get(classification, 0) + 1
    mapped_instructions = len({item["instruction_id"] for item in instruction_alignments})
    blocks_mapped = sum(1 for block in block_alignments if block.get("classification") != "UNAVAILABLE")
    functions_with_source = sum(1 for function in functions if function.get("source_file") and function.get("source_sha256"))
    functions_with_instructions = sum(1 for function in functions if function.get("instruction_ids") and any(item["function_id"] == function["function_id"] for item in instruction_alignments))
    variables_with_location = sum(1 for variable in variables if variable.get("classification") == "DEBUG_DERIVED")
    variables_with_address = sum(1 for variable in variables if variable.get("location", {}).get("address"))
    coverage = {
        "compilation_units": len(dwarf.get("compilation_units", [])),
        "functions": len(functions),
        "functions_with_source": functions_with_source,
        "functions_exact": sum(1 for function in functions if function.get("classification") == "EXACT"),
        "functions_debug_derived": sum(1 for function in functions if function.get("classification") == "DEBUG_DERIVED"),
        "functions_symbol_derived": sum(1 for function in functions if function.get("classification") == "SYMBOL_DERIVED"),
        "functions_heuristic": sum(1 for function in functions if function.get("classification") == "HEURISTIC"),
        "functions_unverified": sum(1 for function in functions if function.get("classification") == "UNVERIFIED"),
        "functions_with_instruction_mapping": functions_with_instructions,
        "line_ranges": len(line_mappings),
        "instructions": len(instructions),
        "instructions_mapped": mapped_instructions,
        "instruction_coverage": round(mapped_instructions / max(1, len(instructions)), 4),
        "basic_blocks": len(block_alignments),
        "basic_blocks_mapped": blocks_mapped,
        "basic_block_coverage": round(blocks_mapped / max(1, len(block_alignments)), 4),
        "variables": len(variables),
        "variables_with_location": variables_with_location,
        "variables_with_address": variables_with_address,
        "data_objects": len(data_objects),
        "data_objects_with_address": len(data_objects),
        "source_files_inventory": len(inventory),
        "source_files_mapped": len(source_level),
        "inlined_functions": len(dwarf.get("inlined", [])),
    }
    status = "available" if functions or line_mappings else "unavailable"
    return {
        "schema_version": ALIGNMENT_SCHEMA_VERSION,
        "status": status,
        "binary": {
            "path": str(binary_path),
            "sha256": elf.sha256,
            "size": elf.size,
            "machine": elf.header.machine_name,
            "type": elf.header.type_name,
            "stripped": elf.is_stripped,
            "has_debug_info": dwarf.get("status") == "available",
        },
        "project_id": project_id,
        "build_variant": build_variant,
        "source_root": str(source_path),
        "dwarf_status": dwarf.get("status"),
        "dwarf_reason": dwarf.get("reason"),
        "project": {
            "project_id": project_id,
            "source_root": str(source_path),
            "source_file_count": len(inventory),
            "source_archive_sha256": (record or {}).get("source_archive_sha256"),
            "commit": (record or {}).get("commit"),
            "license": (record or {}).get("license"),
            "classification": "DEBUG_DERIVED" if dwarf.get("status") == "available" else "SYMBOL_DERIVED" if symbol_functions else "UNAVAILABLE",
            "confidence": 0.95 if dwarf.get("status") == "available" else 0.6 if symbol_functions else 0.0,
        },
        "compilation_units": dwarf.get("compilation_units", []),
        "source_files": source_level,
        "functions": functions,
        "inlined_functions": dwarf.get("inlined", []),
        "line_mappings": line_mappings,
        "instruction_alignments": instruction_alignments,
        "basic_block_alignments": block_alignments,
        "call_graph": call_graph,
        "variables": variables,
        "data_objects": data_objects,
        "symbols": symbol_records[:20000],
        "coverage": coverage,
        "classifications": classifications,
        "methods": {
            "dwarf": sum(1 for function in functions if "dwarf" in function.get("methods", [])),
            "symbol": sum(1 for function in functions if "symbol" in function.get("methods", [])),
            "recovery": sum(1 for function in functions if "recovery" in function.get("methods", [])),
        },
        "limitations": limitations,
    }


def _text_sha(binary: str | Path) -> str | None:
    try:
        elf = parse_elf(binary)
    except Exception:
        return None
    section = elf.section_by_name(".text")
    if section is None or section.offset + section.size > len(elf.raw):
        return None
    return hashlib.sha256(elf.raw[section.offset : section.offset + section.size]).hexdigest()


def transfer_alignment(debug_index: dict[str, Any], stripped_binary: str | Path, debug_binary: str | Path) -> dict[str, Any]:
    debug_text = _text_sha(debug_binary)
    stripped_text = _text_sha(stripped_binary)
    if not debug_text or not stripped_text:
        return {"status": "unavailable", "reason": "unable to hash .text"}
    if debug_text != stripped_text:
        return {"status": "unavailable", "reason": "text section differs; no transfer", "debug_text_sha256": debug_text, "target_text_sha256": stripped_text}
    try:
        target = parse_elf(stripped_binary)
    except Exception as exc:
        return {"status": "unavailable", "reason": f"unable to parse target: {exc}"}
    transferred = {
        "schema_version": ALIGNMENT_SCHEMA_VERSION,
        "status": "available",
        "binary": {
            "path": str(Path(stripped_binary).resolve()),
            "sha256": target.sha256,
            "size": target.size,
            "machine": target.header.machine_name,
            "type": target.header.type_name,
            "stripped": target.is_stripped,
            "has_debug_info": False,
        },
        "project_id": debug_index.get("project_id"),
        "build_variant": debug_index.get("build_variant"),
        "source_root": debug_index.get("source_root"),
        "dwarf_status": "transferred",
        "project": debug_index.get("project"),
        "compilation_units": debug_index.get("compilation_units", []),
        "source_files": debug_index.get("source_files", []),
        "inlined_functions": debug_index.get("inlined_functions", []),
        "call_graph": debug_index.get("call_graph", {}),
        "symbols": debug_index.get("symbols", []),
        "coverage": debug_index.get("coverage", {}),
        "classifications": debug_index.get("classifications", {}),
        "methods": {**debug_index.get("methods", {}), "text_identity_transfer": 1},
        "transfer": {
            "method": "text-identity",
            "debug_binary_sha256": debug_index.get("binary", {}).get("sha256"),
            "target_binary_sha256": target.sha256,
            "text_sha256": debug_text,
            "verified": True,
            "confidence": 0.95,
        },
        "limitations": [
            "Source mappings were transferred from a debug build only because the .text section is byte-identical.",
            "Transfer does not prove semantic equivalence under other build flags; it proves identical code bytes for the compared builds.",
            "Instruction mappings remain DWARF-derived line-table coverage, not token-level AST mapping.",
        ],
    }
    function_map = {}
    for function in debug_index.get("functions", []):
        entry = function.get("entry_va")
        function_id = f"fn_{target.sha256[:12]}_{int(entry, 16):x}" if entry else function.get("function_id")
        function_map[function.get("function_id")] = function_id
        item = {**function, "function_id": function_id, "source_alignment_transferred": True, "methods": [*function.get("methods", []), "text_identity_transfer"]}
        transferred.setdefault("functions", []).append(item)
    line_map = {}
    for index, mapping in enumerate(debug_index.get("line_mappings", [])):
        line_id = f"line_{target.sha256[:12]}_{index:x}"
        line_map[mapping.get("line_id")] = line_id
        item = {**mapping, "line_id": line_id, "function_id": function_map.get(mapping.get("function_id"), mapping.get("function_id")), "method": f"{mapping.get('method', 'dwarf_line_table')}+text_identity_transfer"}
        transferred.setdefault("line_mappings", []).append(item)
    for mapping in transferred.get("line_mappings", []):
        mapping["instruction_ids"] = list(mapping.get("instruction_ids", []))
    for item in debug_index.get("instruction_alignments", []):
        transferred.setdefault("instruction_alignments", []).append(
            {
                **item,
                "function_id": function_map.get(item.get("function_id"), item.get("function_id")),
                "line_id": line_map.get(item.get("line_id"), item.get("line_id")),
                "source_alignment_transferred": True,
                "method": f"{item.get('method', 'dwarf_line_table_address_coverage')}+text_identity_transfer",
            }
        )
    for item in debug_index.get("basic_block_alignments", []):
        transferred.setdefault("basic_block_alignments", []).append(
            {
                **item,
                "function_id": function_map.get(item.get("function_id"), item.get("function_id")),
                "source_alignment_transferred": True,
                "method": f"{item.get('method', 'cfg_line_table_intersection')}+text_identity_transfer",
            }
        )
    for item in debug_index.get("variables", []):
        transferred.setdefault("variables", []).append({**item, "function_id": function_map.get(item.get("function_id"), item.get("function_id")), "source_alignment_transferred": True, "method": f"{item.get('method', 'dwarf_location_expression')}+text_identity_transfer"})
    for item in debug_index.get("data_objects", []):
        transferred.setdefault("data_objects", []).append({**item, "source_alignment_transferred": True, "method": f"{item.get('method', 'dwarf_location_expression')}+text_identity_transfer"})
    if not transferred.get("functions"):
        transferred["status"] = "unavailable"
        transferred["status_reason"] = "debug index contains no functions"
    return transferred


def write_alignment(index: dict[str, Any], path: str | Path) -> None:
    import json

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")
    temporary.replace(target)
