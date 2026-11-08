"""Minimal read-only PE32/PE32+ structural parser."""

from __future__ import annotations

import struct
from pathlib import Path
from typing import Any


def _u16(data: bytes, offset: int) -> int:
    return struct.unpack_from("<H", data, offset)[0]


def _u32(data: bytes, offset: int) -> int:
    return struct.unpack_from("<I", data, offset)[0]


def _u64(data: bytes, offset: int) -> int:
    return struct.unpack_from("<Q", data, offset)[0]


def _rva_to_offset(rva: int, sections: list[dict[str, Any]]) -> int | None:
    for section in sections:
        start = section["virtual_address"]
        size = max(section["virtual_size"], section["raw_size"])
        if start <= rva < start + size:
            delta = rva - start
            if delta < section["raw_size"]:
                return section["raw_pointer"] + delta
    return None


def _cstring(data: bytes, offset: int, limit: int = 512) -> str:
    end = data.find(b"\x00", offset, offset + limit)
    if end < 0:
        end = min(len(data), offset + limit)
    return data[offset:end].decode("ascii", errors="replace")


def parse_pe(path: str | Path) -> dict[str, Any]:
    target = Path(path)
    data = target.read_bytes()
    result: dict[str, Any] = {
        "path": str(target),
        "format": "PE",
        "size": len(data),
        "header": {},
        "sections": [],
        "imports": [],
        "exports": [],
        "dependencies": [],
        "limitations": [],
    }
    if not data.startswith(b"MZ"):
        raise ValueError("not a DOS MZ image")
    e_lfanew = _u32(data, 0x3C)
    if data[e_lfanew : e_lfanew + 4] != b"PE\x00\x00":
        raise ValueError("missing PE signature")
    coff = e_lfanew + 4
    machine = _u16(data, coff)
    number_of_sections = _u16(data, coff + 2)
    timestamp = _u32(data, coff + 4)
    pointer_to_symbols = _u32(data, coff + 8)
    number_of_symbols = _u32(data, coff + 12)
    optional_size = _u16(data, coff + 16)
    characteristics = _u16(data, coff + 18)
    optional = coff + 20
    magic = _u16(data, optional)
    is_64 = magic == 0x20B
    result["class"] = "PE32+" if is_64 else "PE32"
    architecture = {
        0x14C: "i386",
        0x8664: "x86_64",
        0x1C0: "arm",
        0xAA64: "aarch64",
        0x5064: "riscv64",
        0x166: "mips",
    }.get(machine, f"machine_{machine}")
    result["architecture"] = architecture
    result["header"] = {
        "machine": machine,
        "architecture": architecture,
        "number_of_sections": number_of_sections,
        "timestamp": timestamp,
        "characteristics": characteristics,
        "optional_magic": hex(magic),
        "pointer_to_symbols": pointer_to_symbols,
        "number_of_symbols": number_of_symbols,
        "image_base": _u64(data, optional + 24) if is_64 else _u32(data, optional + 28),
        "entry_point_rva": _u32(data, optional + 16),
        "section_alignment": _u32(data, optional + 32),
        "file_alignment": _u32(data, optional + 36),
        "size_of_image": _u32(data, optional + 56),
        "size_of_headers": _u32(data, optional + 60),
        "subsystem": _u16(data, optional + 68),
        "dll_characteristics": _u16(data, optional + 70),
        "number_of_rva_and_sizes": _u32(data, optional + (108 if is_64 else 92)),
    }
    directory_count_offset = optional + (108 if is_64 else 92)
    directory_offset = optional + (112 if is_64 else 96)
    directory_count = min(16, _u32(data, directory_count_offset))
    directories = []
    for index in range(directory_count):
        offset = directory_offset + index * 8
        directories.append({"index": index, "virtual_address": _u32(data, offset), "size": _u32(data, offset + 4)})
    result["data_directories"] = directories
    section_offset = optional + optional_size
    for index in range(min(number_of_sections, 512)):
        offset = section_offset + index * 40
        if offset + 40 > len(data):
            break
        name = data[offset : offset + 8].rstrip(b"\x00").decode("ascii", errors="replace")
        result["sections"].append(
            {
                "index": index,
                "name": name,
                "virtual_size": _u32(data, offset + 8),
                "virtual_address": _u32(data, offset + 12),
                "raw_size": _u32(data, offset + 16),
                "raw_pointer": _u32(data, offset + 20),
                "characteristics": _u32(data, offset + 36),
            }
        )
    # Imports.
    import_dir = directories[1] if len(directories) > 1 else {"virtual_address": 0, "size": 0}
    if import_dir["virtual_address"] and import_dir["size"]:
        offset = _rva_to_offset(import_dir["virtual_address"], result["sections"])
        if offset is not None:
            cursor = offset
            for _ in range(512):
                if cursor + 20 > len(data):
                    break
                original_first_thunk = _u32(data, cursor)
                name_rva = _u32(data, cursor + 12)
                first_thunk = _u32(data, cursor + 16)
                if not name_rva:
                    break
                name_offset = _rva_to_offset(name_rva, result["sections"])
                dll = _cstring(data, name_offset) if name_offset is not None else ""
                functions = []
                thunk_rva = original_first_thunk or first_thunk
                thunk_offset = _rva_to_offset(thunk_rva, result["sections"]) if thunk_rva else None
                if thunk_offset is not None:
                    step = 8 if is_64 else 4
                    for slot in range(4096):
                        entry_offset = thunk_offset + slot * step
                        if entry_offset + step > len(data):
                            break
                        value = _u64(data, entry_offset) if is_64 else _u32(data, entry_offset)
                        if value == 0:
                            break
                        if value & (1 << (63 if is_64 else 31)):
                            functions.append({"ordinal": value & 0xFFFF})
                        else:
                            hint_name = _rva_to_offset(value & 0x7FFFFFFF, result["sections"])
                            functions.append({"name": _cstring(data, hint_name + 2) if hint_name is not None else None})
                result["imports"].append({"dll": dll, "functions": functions[:2048]})
                if dll:
                    result["dependencies"].append(dll)
                cursor += 20
    # Exports.
    export_dir = directories[0] if directories else {"virtual_address": 0, "size": 0}
    if export_dir["virtual_address"] and export_dir["size"]:
        offset = _rva_to_offset(export_dir["virtual_address"], result["sections"])
        if offset is not None and offset + 40 <= len(data):
            number_of_names = _u32(data, offset + 24)
            address_of_functions = _u32(data, offset + 28)
            address_of_names = _u32(data, offset + 32)
            address_of_ordinals = _u32(data, offset + 36)
            names_offset = _rva_to_offset(address_of_names, result["sections"])
            ordinals_offset = _rva_to_offset(address_of_ordinals, result["sections"])
            functions_offset = _rva_to_offset(address_of_functions, result["sections"])
            if names_offset is not None and ordinals_offset is not None and functions_offset is not None:
                for index in range(min(number_of_names, 8192)):
                    name_rva = _u32(data, names_offset + index * 4)
                    ordinal_index = _u16(data, ordinals_offset + index * 2)
                    name_offset = _rva_to_offset(name_rva, result["sections"])
                    function_rva = _u32(data, functions_offset + ordinal_index * 4) if functions_offset + ordinal_index * 4 + 4 <= len(data) else None
                    result["exports"].append(
                        {
                            "name": _cstring(data, name_offset) if name_offset is not None else None,
                            "ordinal": ordinal_index,
                            "rva": function_rva,
                        }
                    )
    result["limitations"] = [
        "Static PE parsing only; no execution, relocation application, Authenticode verification or runtime loading.",
        "Import/export names are read from the file image; bound imports and forwarders are not resolved.",
    ]
    return result
