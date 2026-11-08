import os
from pathlib import Path
from typing import Any

from elftools.elf.elffile import ELFFile


def _text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _safe_source_path(path: str) -> str:
    value = path.replace("\\", "/")
    marker = "/sources/"
    if marker in value:
        return value.split(marker, 1)[1]
    if value.startswith("./"):
        return value[2:]
    if os.path.isabs(value):
        return os.path.basename(value)
    return value


def _file_name(line_program: Any, index: int | None) -> str:
    if index is None:
        return ""
    entries = line_program.get("file_entry", [])
    try:
        entry = entries[index]
    except (IndexError, TypeError):
        try:
            entry = entries[index - 1]
        except (IndexError, TypeError):
            return ""
    name = _text(entry.get("name"))
    directory_index = entry.get("dir_index")
    directories = line_program.get("include_directory", [])
    if directory_index is not None:
        try:
            directory = _text(directories[directory_index])
        except (IndexError, TypeError):
            directory = ""
        if directory and not os.path.isabs(name):
            name = os.path.join(directory, name)
    return _safe_source_path(name)


def _type_name(die: Any, dies_by_offset: dict[int, Any], depth: int = 0) -> str:
    if depth > 4:
        return "unknown"
    attribute = die.attributes.get("DW_AT_type")
    if attribute is None:
        return "void"
    target = dies_by_offset.get(attribute.value)
    if target is None:
        return "unknown"
    name = target.attributes.get("DW_AT_name")
    if name is not None:
        return _text(name.value)
    if target.tag in {"DW_TAG_pointer_type", "DW_TAG_reference_type"}:
        return _type_name(target, dies_by_offset, depth + 1) + "*"
    if target.tag == "DW_TAG_const_type":
        return "const " + _type_name(target, dies_by_offset, depth + 1)
    if target.tag == "DW_TAG_array_type":
        return _type_name(target, dies_by_offset, depth + 1) + "[]"
    return target.tag.replace("DW_TAG_", "").lower()


def _ranges_from_die(die: Any) -> list[tuple[int, int]]:
    low = die.attributes.get("DW_AT_low_pc")
    high = die.attributes.get("DW_AT_high_pc")
    if low is None or high is None:
        return []
    start = int(low.value)
    if high.form == "DW_FORM_addr":
        end = int(high.value)
    else:
        end = start + int(high.value)
    return [(start, end)] if end > start else []


def extract_dwarf_alignment(binary: str | Path, record: dict[str, Any] | None = None, max_functions: int = 5000, max_line_entries: int = 200000, max_variables: int = 5000) -> dict[str, Any]:
    path = Path(binary)
    functions: list[dict[str, Any]] = []
    line_mappings: list[dict[str, Any]] = []
    compilation_units: list[dict[str, Any]] = []
    variables: list[dict[str, Any]] = []
    limitations: list[str] = []
    classifications: dict[str, int] = {"EXACT": 0, "DEBUG_DERIVED": 0, "SYMBOL_DERIVED": 0, "HEURISTIC": 0, "UNVERIFIED": 0, "UNAVAILABLE": 0}
    try:
        with open(path, "rb") as handle:
            elf = ELFFile(handle)
            dwarf = elf.get_dwarf_info()
            cu_count = 0
            for cu in dwarf.iter_CUs():
                if cu_count >= 500:
                    limitations.append("compilation-unit cap reached")
                    break
                cu_count += 1
                top = cu.get_top_DIE()
                cu_name = _safe_source_path(_text(top.attributes.get("DW_AT_name").value if top.attributes.get("DW_AT_name") else ""))
                producer = _text(top.attributes.get("DW_AT_producer").value if top.attributes.get("DW_AT_producer") else "")
                language = top.attributes.get("DW_AT_language")
                compilation_units.append({"offset": cu.cu_offset, "name": cu_name, "producer": producer, "language": language.value if language else None})
                line_program = dwarf.line_program_for_CU(cu)
                dies_by_offset = {die.offset: die for die in cu.iter_DIEs()}
                for die in dies_by_offset.values():
                    if die.tag == "DW_TAG_subprogram" and len(functions) < max_functions:
                        name_attr = die.attributes.get("DW_AT_name")
                        linkage_attr = die.attributes.get("DW_AT_linkage_name")
                        decl_file_attr = die.attributes.get("DW_AT_decl_file")
                        decl_line_attr = die.attributes.get("DW_AT_decl_line")
                        ranges = _ranges_from_die(die)
                        if not ranges:
                            classifications["UNAVAILABLE"] += 1
                            continue
                        source_file = _file_name(line_program, decl_file_attr.value if decl_file_attr else None)
                        source_line = int(decl_line_attr.value) if decl_line_attr else None
                        functions.append({
                            "name": _text(name_attr.value if name_attr else ""),
                            "linkage_name": _text(linkage_attr.value if linkage_attr else ""),
                            "ranges": [{"start_va": f"0x{start:x}", "end_va": f"0x{end:x}"} for start, end in ranges],
                            "source_file": source_file or cu_name,
                            "source_line": source_line,
                            "compilation_unit": cu_name,
                            "classification": "DEBUG_DERIVED",
                            "confidence": 0.95,
                        })
                        classifications["DEBUG_DERIVED"] += 1
                    elif die.tag in {"DW_TAG_variable", "DW_TAG_formal_parameter"} and len(variables) < max_variables:
                        name_attr = die.attributes.get("DW_AT_name")
                        decl_file_attr = die.attributes.get("DW_AT_decl_file")
                        decl_line_attr = die.attributes.get("DW_AT_decl_line")
                        location_attr = die.attributes.get("DW_AT_location")
                        variables.append({
                            "name": _text(name_attr.value if name_attr else ""),
                            "source_file": _file_name(line_program, decl_file_attr.value if decl_file_attr else None) or cu_name,
                            "source_line": int(decl_line_attr.value) if decl_line_attr else None,
                            "type": _type_name(die, dies_by_offset),
                            "role": "parameter" if die.tag == "DW_TAG_formal_parameter" else "variable",
                            "location_available": location_attr is not None,
                            "location_classification": "DEBUG_DERIVED" if location_attr is not None else "UNAVAILABLE",
                            "address_mapping": "UNAVAILABLE",
                            "classification": "DEBUG_DERIVED",
                            "confidence": 0.7,
                        })
                if line_program is not None:
                    entries = []
                    for entry in line_program.get_entries():
                        state = entry.state
                        if state is None or state.end_sequence or not state.address or not state.line:
                            continue
                        entries.append((int(state.address), _file_name(line_program, state.file), int(state.line)))
                        if len(entries) >= max_line_entries:
                            limitations.append("line-entry cap reached")
                            break
                    entries.sort(key=lambda item: item[0])
                    for index, (address, source_file, source_line) in enumerate(entries):
                        end = entries[index + 1][0] if index + 1 < len(entries) else address + 1
                        if end <= address:
                            continue
                        line_mappings.append({
                            "start_va": f"0x{address:x}",
                            "end_va": f"0x{end:x}",
                            "source_file": source_file,
                            "source_line": source_line,
                            "classification": "DEBUG_DERIVED",
                            "confidence": 0.9,
                        })
                        classifications["DEBUG_DERIVED"] += 1
    except Exception as exc:
        return {
            "status": "UNAVAILABLE",
            "binary_path": str(path),
            "binary_sha256": record.get("program", {}).get("sha256") if record else None,
            "reason": f"{type(exc).__name__}: {exc}",
            "functions": [],
            "line_mappings": [],
            "variables": [],
            "compilation_units": [],
            "instruction_alignment": [],
            "basic_block_alignment": [],
            "classifications": classifications,
            "coverage": {},
            "limitations": ["No usable DWARF debug information; stripped or non-debug binary."],
        }
    instruction_alignment = []
    basic_block_alignment = []
    if record:
        instructions = record.get("instructions", [])
        instruction_by_id = {item.get("instruction_id"): item for item in instructions}
        for mapping in line_mappings:
            start = int(mapping["start_va"], 16)
            end = int(mapping["end_va"], 16)
            ids = [item["instruction_id"] for item in instructions if start <= int(item["va"], 16) < end]
            if ids:
                mapping["instruction_ids"] = ids
                instruction_alignment.extend(ids)
        for function in record.get("functions", []):
            ranges = [(int(item["start_va"], 16), int(item["end_va"], 16)) for item in function.get("ranges", [])]
            lines = [mapping for mapping in line_mappings if any(start <= int(mapping["start_va"], 16) < end for start, end in ranges)]
            if lines:
                for block in function.get("blocks", []):
                    block_start = int(block.get("range", {}).get("start_va", "0x0"), 16)
                    block_end = int(block.get("range", {}).get("end_va", "0x0"), 16)
                    block_lines = [mapping for mapping in lines if block_start <= int(mapping["start_va"], 16) < block_end]
                    if block_lines:
                        basic_block_alignment.append({
                            "block_id": block.get("block_id"),
                            "function_id": function.get("function_id"),
                            "source_file": block_lines[0]["source_file"],
                            "start_line": block_lines[0]["source_line"],
                            "end_line": block_lines[-1]["source_line"],
                            "classification": "HEURISTIC",
                            "confidence": 0.5,
                        })
                        classifications["HEURISTIC"] += 1
        source_instructions = sorted(set(instruction_alignment))
        instruction_records = [
            {"instruction_id": instruction_id, "classification": "DEBUG_DERIVED", "confidence": 0.8}
            for instruction_id in source_instructions
        ]
    else:
        instruction_records = []
        source_instructions = []
    function_match = []
    if record:
        record_functions = record.get("functions", [])
        by_entry = {int(item.get("entry_va", "0x0"), 16): item for item in record_functions}
        for function in functions:
            start = int(function["ranges"][0]["start_va"], 16)
            end = max(int(item["end_va"], 16) for item in function["ranges"])
            matched = by_entry.get(start)
            if matched is None:
                matched = next((item for item in record_functions if any(int(r["start_va"], 16) <= start < int(r["end_va"], 16) for r in item.get("ranges", []))), None)
            function["function_id"] = matched.get("function_id") if matched else None
            function["binary_name"] = matched.get("names", [{}])[0].get("value") if matched else None
            function["name_match"] = bool(matched and function.get("name") in {name.get("value") for name in matched.get("names", [])})
            if matched:
                function["classification"] = "DEBUG_DERIVED"
                function["confidence"] = 0.95 if int(matched.get("entry_va", "0x0"), 16) == start else 0.8
                function["instruction_ids"] = [item["instruction_id"] for item in instruction_by_id.values() if any(r["start_va"] and start <= int(item["va"], 16) < end for r in matched.get("ranges", []))]
            else:
                function["classification"] = "UNVERIFIED"
                function["confidence"] = 0.4
            function_match.append(function)
    else:
        function_match = functions
    coverage = {
        "compilation_units": len(compilation_units),
        "functions": len(function_match),
        "functions_with_source": sum(1 for item in function_match if item.get("source_file")),
        "record_functions": len(record.get("functions", [])) if record else 0,
        "record_functions_aligned": sum(1 for item in function_match if item.get("function_id")),
        "line_ranges": len(line_mappings),
        "instructions": len(record.get("instructions", [])) if record else 0,
        "instructions_with_source": len(source_instructions),
        "instruction_coverage": round(len(source_instructions) / max(1, len(record.get("instructions", []))) if record else 0, 4),
        "variables": len(variables),
        "variables_with_location": sum(1 for item in variables if item.get("location_available")),
        "source_files": len({item.get("source_file") for item in line_mappings if item.get("source_file")}),
        "basic_blocks_aligned": len(basic_block_alignment),
    }
    limitations.append("Instruction alignment is line-table address coverage, not token-level AST mapping.")
    limitations.append("Variable addresses are not recovered from DWARF locations.")
    limitations.append("Basic-block source ranges are heuristic intersections of line entries and CFG blocks.")
    status = "available" if compilation_units and line_mappings else "unavailable"
    return {
        "schema_version": "0.1.0",
        "status": status,
        "binary_path": str(path),
        "binary_sha256": record.get("program", {}).get("sha256") if record else None,
        "compilation_units": compilation_units,
        "functions": function_match,
        "line_mappings": line_mappings,
        "instruction_alignment": instruction_records,
        "basic_block_alignment": basic_block_alignment,
        "variables": variables,
        "coverage": coverage,
        "classifications": classifications,
        "limitations": limitations,
    }
