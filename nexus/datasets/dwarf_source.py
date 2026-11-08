"""Extract source-backed function labels from DWARF dumps and source trees."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any

_SUBPROGRAM = re.compile(r"^(0x[0-9a-f]+):\s+DW_TAG_subprogram\s*$", re.MULTILINE)


def _attr(block: str, name: str) -> str | None:
    match = re.search(rf"DW_AT_{re.escape(name)}\s+\((.*?)\)", block)
    if not match:
        return None
    value = match.group(1).strip()
    if value.startswith('"') and value.endswith('"'):
        return value[1:-1]
    return value


def parse_dwarf_functions(text: str) -> list[dict[str, Any]]:
    starts = [match.start() for match in _SUBPROGRAM.finditer(text)]
    functions = []
    for index, start in enumerate(starts):
        end = starts[index + 1] if index + 1 < len(starts) else len(text)
        block = text[start:end]
        name = _attr(block, "name")
        file_name = _attr(block, "decl_file")
        line = _attr(block, "decl_line")
        low = _attr(block, "low_pc")
        high = _attr(block, "high_pc")
        if not name or not file_name or not line or not low or not high:
            continue
        try:
            low_pc = int(low, 16)
            high_pc = int(high, 16)
        except ValueError:
            continue
        functions.append({
            "name": name,
            "file": file_name,
            "line": int(line),
            "low_pc": low_pc,
            "high_pc": high_pc,
        })
    return functions


def _strip_c_comments_and_strings(text: str) -> str:
    output = []
    index = 0
    state = "code"
    while index < len(text):
        char = text[index]
        nxt = text[index + 1] if index + 1 < len(text) else ""
        if state == "code":
            if char == "/" and nxt == "*":
                state = "block"
                output.extend("  ")
                index += 2
                continue
            if char == "/" and nxt == "/":
                state = "line"
                output.extend("  ")
                index += 2
                continue
            if char == '"':
                state = "string"
                output.append(" ")
                index += 1
                continue
            output.append(char)
            index += 1
        elif state == "block":
            if char == "*" and nxt == "/":
                state = "code"
                output.extend("  ")
                index += 2
                continue
            output.append("\n" if char == "\n" else " ")
            index += 1
        elif state == "line":
            if char == "\n":
                state = "code"
                output.append("\n")
            else:
                output.append(" ")
            index += 1
        else:
            if char == "\\" and nxt:
                output.extend("  ")
                index += 2
                continue
            if char == '"':
                state = "code"
            output.append("\n" if char == "\n" else " ")
            index += 1
    return "".join(output)


def extract_function_body(source: str, start_line: int) -> str | None:
    lines = source.splitlines()
    if start_line < 1 or start_line > len(lines):
        return None
    offset = sum(len(line) + 1 for line in lines[: start_line - 1])
    cleaned = _strip_c_comments_and_strings(source[offset:])
    open_index = cleaned.find("{")
    if open_index < 0:
        return None
    depth = 0
    end_index = None
    for index in range(open_index, len(cleaned)):
        char = cleaned[index]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                end_index = index + 1
                break
    if end_index is None:
        return None
    snippet = source[offset : offset + end_index].strip()
    return snippet or None


def extract_source_labels(record: dict[str, Any], source_root: Path, dwarf_text: str) -> dict[str, dict[str, Any]]:
    dwarf_functions = parse_dwarf_functions(dwarf_text)
    by_pc = {item["low_pc"]: item for item in dwarf_functions}
    by_name = {item["name"]: item for item in dwarf_functions}
    source_cache: dict[str, str] = {}
    labels: dict[str, dict[str, Any]] = {}
    for function in record.get("functions", []):
        entry = function.get("entry_va")
        try:
            entry_pc = int(str(entry), 16)
        except (TypeError, ValueError):
            entry_pc = None
        item = by_pc.get(entry_pc) if entry_pc is not None else None
        if item is None:
            item = by_name.get(str(function.get("name") or ""))
        if item is None:
            continue
        relative = item["file"].replace("././", "").lstrip("./")
        path = (source_root / relative).resolve()
        try:
            path.relative_to(source_root.resolve())
        except ValueError:
            continue
        if not path.exists():
            continue
        if relative not in source_cache:
            source_cache[relative] = path.read_text(encoding="utf-8", errors="replace")
        body = extract_function_body(source_cache[relative], item["line"])
        if not body:
            continue
        labels[str(function.get("function_id"))] = {
            "text": body,
            "source_sha256": hashlib.sha256(body.encode("utf-8")).hexdigest(),
            "source_path": relative,
            "source_line": item["line"],
            "alignment": "dwarf",
            "confidence": 1.0,
        }
    return labels
