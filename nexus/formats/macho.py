"""Minimal read-only Mach-O structural parser."""

from __future__ import annotations

import struct
from pathlib import Path
from typing import Any


def parse_macho(path: str | Path) -> dict[str, Any]:
    target = Path(path)
    data = target.read_bytes()
    result: dict[str, Any] = {
        "path": str(target),
        "format": "Mach-O",
        "size": len(data),
        "architecture": None,
        "class": None,
        "endianness": None,
        "segments": [],
        "sections": [],
        "dependencies": [],
        "symbols": [],
        "rpaths": [],
        "limitations": [
            "Static Mach-O header/section parsing only; no code signing, dyld resolution, Objective-C/Swift runtime metadata or execution.",
        ],
    }
    if len(data) < 28:
        raise ValueError("truncated Mach-O header")
    magic = struct.unpack_from(">I", data, 0)[0]
    little = magic in {0xCEFAEDFE, 0xCFFAEDFE}
    fmt = "<" if little else ">"
    if magic in {0xFEEDFACE, 0xCEFAEDFE}:
        is_64 = False
        header_size = 28
    elif magic in {0xFEEDFACF, 0xCFFAEDFE}:
        is_64 = True
        header_size = 32
    else:
        raise ValueError("not a Mach-O image")
    cputype, cpusubtype, filetype, ncmds, sizeofcmds, flags = struct.unpack_from(fmt + "iiIIII", data, 4)
    result.update(
        {
            "class": "Mach-O 64" if is_64 else "Mach-O 32",
            "endianness": "little" if little else "big",
            "architecture": {
                7: "i386",
                0x01000007: "x86_64",
                12: "arm",
                0x0100000C: "arm64",
                0x0100000D: "arm64e",
                18: "powerpc",
                0x01000012: "powerpc64",
            }.get(cputype, f"cputype_{cputype}"),
            "header": {
                "cputype": cputype,
                "cpusubtype": cpusubtype,
                "filetype": filetype,
                "ncmds": ncmds,
                "sizeofcmds": sizeofcmds,
                "flags": flags,
            },
        }
    )
    cursor = header_size
    lc_segment = 0x1
    lc_segment_64 = 0x19
    lc_load_dylib = 0xC
    lc_id_dylib = 0xD
    lc_load_weak_dylib = 0x80000018
    lc_rpath = 0x8000001C
    lc_symtab = 0x2
    for _ in range(min(ncmds, 4096)):
        if cursor + 8 > len(data):
            break
        cmd, cmdsize = struct.unpack_from(fmt + "II", data, cursor)
        if cmdsize < 8 or cursor + cmdsize > len(data):
            result["limitations"].append("truncated or invalid load command")
            break
        base_cmd = cmd & 0x7FFFFFFF
        if base_cmd in {lc_segment, lc_segment_64}:
            if base_cmd == lc_segment_64:
                if cmdsize >= 72:
                    segname = data[cursor + 8 : cursor + 24].rstrip(b"\x00").decode("ascii", errors="replace")
                    vmaddr, vmsize, fileoff, filesize = struct.unpack_from(fmt + "QQQQ", data, cursor + 24)
                    nsects = struct.unpack_from(fmt + "I", data, cursor + 64)[0]
                    section_offset = cursor + 72
                    section_size = 80
                else:
                    nsects = 0
                    section_offset = cursor + cmdsize
                    section_size = 0
                    segname = ""
            else:
                if cmdsize >= 56:
                    segname = data[cursor + 8 : cursor + 24].rstrip(b"\x00").decode("ascii", errors="replace")
                    vmaddr, vmsize, fileoff, filesize = struct.unpack_from(fmt + "IIII", data, cursor + 24)
                    nsects = struct.unpack_from(fmt + "I", data, cursor + 48)[0]
                    section_offset = cursor + 56
                    section_size = 68
                else:
                    nsects = 0
                    section_offset = cursor + cmdsize
                    section_size = 0
                    segname = ""
            result["segments"].append(
                {
                    "name": segname,
                    "vmaddr": vmaddr,
                    "vmsize": vmsize,
                    "fileoff": fileoff,
                    "filesize": filesize,
                    "nsects": nsects,
                }
            )
            for index in range(min(nsects, 1024)):
                offset = section_offset + index * section_size
                if section_size == 0 or offset + section_size > len(data):
                    break
                sectname = data[offset : offset + 16].rstrip(b"\x00").decode("ascii", errors="replace")
                seg = data[offset + 16 : offset + 32].rstrip(b"\x00").decode("ascii", errors="replace")
                if section_size == 80:
                    addr, size = struct.unpack_from(fmt + "QQ", data, offset + 32)
                    sec_offset, align, reloff, nreloc, flags = struct.unpack_from(fmt + "IIIII", data, offset + 48)
                else:
                    addr, size = struct.unpack_from(fmt + "II", data, offset + 32)
                    sec_offset, align, reloff, nreloc, flags = struct.unpack_from(fmt + "IIIII", data, offset + 40)
                result["sections"].append(
                    {
                        "segment": seg,
                        "name": sectname,
                        "address": addr,
                        "size": size,
                        "file_offset": sec_offset,
                        "align": align,
                        "flags": flags,
                    }
                )
        elif cmd in {lc_load_dylib, lc_id_dylib, lc_load_weak_dylib} and cmdsize >= 24:
            name_offset = struct.unpack_from(fmt + "I", data, cursor + 8)[0]
            start = cursor + name_offset
            end = data.find(b"\x00", start, cursor + cmdsize)
            if end < 0:
                end = cursor + cmdsize
            name = data[start:end].decode("utf-8", errors="replace")
            if cmd != lc_id_dylib:
                result["dependencies"].append(name)
            else:
                result["install_name"] = name
        elif cmd == lc_rpath and cmdsize >= 12:
            name_offset = struct.unpack_from(fmt + "I", data, cursor + 8)[0]
            start = cursor + name_offset
            end = data.find(b"\x00", start, cursor + cmdsize)
            if end < 0:
                end = cursor + cmdsize
            result["rpaths"].append(data[start:end].decode("utf-8", errors="replace"))
        elif cmd == lc_symtab and cmdsize >= 24:
            symoff, nsyms, stroff, strsize = struct.unpack_from(fmt + "IIII", data, cursor + 8)
            step = 16 if is_64 else 12
            for index in range(min(nsyms, 20000)):
                offset = symoff + index * step
                if offset + step > len(data):
                    break
                if is_64:
                    strx, n_type, n_sect, n_desc, value = struct.unpack_from(fmt + "IBBHQ", data, offset)
                else:
                    strx, n_type, n_sect, n_desc, value = struct.unpack_from(fmt + "IBBHI", data, offset)
                name_offset = stroff + strx
                if name_offset >= len(data):
                    continue
                end = data.find(b"\x00", name_offset, min(len(data), name_offset + 512))
                if end < 0:
                    end = min(len(data), name_offset + 512)
                name = data[name_offset:end].decode("utf-8", errors="replace")
                if name:
                    result["symbols"].append(
                        {"name": name, "type": n_type, "section": n_sect, "description": n_desc, "value": value}
                    )
        cursor += cmdsize
    return result
