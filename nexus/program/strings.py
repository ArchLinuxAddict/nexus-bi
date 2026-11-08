import re
from typing import Any

from nexus.elf.model import ElfFile

PRINTABLE = re.compile(rb"[\x20-\x7e]{%d,}")


def extract_strings(elf: ElfFile, min_length: int = 4, limit: int = 5000) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    seen: set[tuple[int | None, str]] = set()
    for section in elf.sections:
        if section.sh_type == 8 or section.size == 0 or section.addr == 0:
            continue
        if not (section.flags & 2):
            continue
        data = elf.raw[section.offset:section.offset + section.size]
        if not data:
            continue
        for match in PRINTABLE.finditer(data):
            if len(results) >= limit:
                return results
            value_bytes = match.group(0)[:512]
            value = value_bytes.decode("ascii", errors="replace")
            if len(value) < min_length:
                continue
            va = section.addr + match.start()
            key = (va, value)
            if key in seen:
                continue
            seen.add(key)
            results.append({
                "string_id": f"str_{elf.sha256[:12]}_{va:x}",
                "va": f"0x{va:x}",
                "file_offset": section.offset + match.start(),
                "encoding": "ascii",
                "value": value,
                "terminated": match.end() < len(data) and data[match.end():match.end() + 1] == b"\x00",
                "confidence": 0.9,
                "evidence": [f"ev_{elf.sha256[:12]}_str_{va:x}"],
                "untrusted": True,
            })
    return results


def strings_by_address(strings: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    return {int(item["va"], 16): item for item in strings}
