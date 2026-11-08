"""Magic-based format identification with explicit unknown outcomes."""

from __future__ import annotations

import struct
from pathlib import Path
from typing import Any


def _read(path: str | Path, length: int = 4096) -> bytes:
    with Path(path).open("rb") as handle:
        return handle.read(length)


def identify_format(path: str | Path) -> dict[str, Any]:
    target = Path(path)
    header = _read(target, 64)
    result: dict[str, Any] = {
        "path": str(target),
        "exists": target.exists(),
        "size": target.stat().st_size if target.exists() else None,
        "format": "unknown",
        "architecture": None,
        "class": None,
        "endianness": None,
        "confidence": 0.0,
        "evidence": [],
        "limitations": [],
    }
    if not header:
        result["limitations"].append("empty or unreadable file")
        return result
    if header.startswith(b"\x7fELF"):
        result.update({"format": "ELF", "confidence": 0.99, "evidence": ["ELF magic"]})
        if len(header) >= 20:
            result["class"] = {1: "ELF32", 2: "ELF64"}.get(header[4], "unknown")
            result["endianness"] = {1: "little", 2: "big"}.get(header[5], "unknown")
            machine = struct.unpack_from("<H" if header[5] == 1 else ">H", header, 18)[0]
            from nexus.architectures import get_adapter
            try:
                adapter = get_adapter(machine, elf_class=result.get("class"))
                result["architecture"] = adapter.name
                result["abi"] = adapter.abi
                result["pointer_width"] = adapter.pointer_width
            except Exception:
                result["architecture"] = {
                    2: "sparc",
                    21: "powerpc64",
                }.get(machine, f"machine_{machine}")
        return result
    if header.startswith(b"MZ"):
        result.update({"format": "PE", "confidence": 0.95, "evidence": ["DOS MZ magic"]})
        try:
            e_lfanew = struct.unpack_from("<I", header, 0x3C)[0]
            with target.open("rb") as handle:
                handle.seek(e_lfanew)
                signature = handle.read(6)
            if signature.startswith(b"PE\x00\x00"):
                result["evidence"].append("PE signature")
                result["class"] = "PE32+" if signature[4:6] == b"\x0b\x02" else "PE32"
            else:
                result["limitations"].append("MZ header without a PE signature")
        except (OSError, struct.error) as exc:
            result["limitations"].append(f"PE header parse failed: {exc}")
        return result
    magic = struct.unpack_from(">I", header, 0)[0]
    if magic in {0xFEEDFACE, 0xFEEDFACF, 0xCEFAEDFE, 0xCFFAEDFE}:
        result.update({"format": "Mach-O", "confidence": 0.98, "evidence": ["Mach-O magic"]})
        little = magic in {0xCEFAEDFE, 0xCFFAEDFE}
        result["endianness"] = "little" if little else "big"
        result["class"] = "Mach-O 64" if magic in {0xFEEDFACF, 0xCFFAEDFE} else "Mach-O 32"
        if len(header) >= 12:
            cputype = struct.unpack_from("<i" if little else ">i", header, 4)[0]
            result["architecture"] = {
                7: "i386",
                0x01000007: "x86_64",
                12: "arm",
                0x0100000C: "arm64",
                18: "powerpc",
                0x01000012: "powerpc64",
            }.get(cputype, f"cputype_{cputype}")
        return result
    if header.startswith(b"PK\x03\x04"):
        result.update({"format": "ZIP/APK", "confidence": 0.9, "evidence": ["ZIP local file header"]})
        if target.suffix.lower() == ".apk":
            result["format"] = "APK"
        return result
    if header.startswith(b"dex\n"):
        result.update({"format": "DEX", "confidence": 0.99, "evidence": ["DEX magic"]})
        version = header[4:7].decode("ascii", errors="replace")
        result["class"] = f"dex {version}"
        return result
    if header.startswith(b"\xca\xfe\xba\xbe"):
        result.update({"format": "Java class", "confidence": 0.99, "evidence": ["Java class magic"]})
        return result
    if header.startswith(b"ANDROID!"):
        result.update({"format": "Android boot image", "confidence": 0.8, "evidence": ["ANDROID! magic"]})
        return result
    if header[:4] == b"\x27\x05\x19\x56":
        result.update({"format": "U-Boot image", "confidence": 0.8, "evidence": ["U-Boot magic"]})
        return result
    if header.startswith(b"hsqs") or header.startswith(b"sqsh"):
        result.update({"format": "SquashFS", "confidence": 0.8, "evidence": ["SquashFS magic"]})
        return result
    result["limitations"].append("unrecognized magic; treat as unknown bytes")
    return result
