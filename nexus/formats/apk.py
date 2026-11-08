"""Read-only APK inventory with manifest, DEX and native-library correlation."""

from __future__ import annotations

import struct
import tempfile
import zipfile
from pathlib import Path
from typing import Any

from nexus.formats.android import manifest_summary, parse_axml
from nexus.formats.dex import parse_dex

MAX_ENTRIES = 200000
MAX_DEX_BYTES = 128 * 1024 * 1024


def _elf_identity(data: bytes) -> dict[str, Any]:
    if len(data) < 20 or not data.startswith(b"\x7fELF"):
        return {}
    elf_class = {1: "ELF32", 2: "ELF64"}.get(data[4], "unknown")
    endian = "little" if data[5] == 1 else "big"
    machine = struct.unpack_from("<H" if endian == "little" else ">H", data, 18)[0]
    from nexus.architectures import get_adapter
    try:
        adapter = get_adapter(machine, elf_class=elf_class)
        name, abi, pointer_width = adapter.name, adapter.abi, adapter.pointer_width
    except Exception:
        name, abi, pointer_width = f"machine_{machine}", None, None
    return {"class": elf_class, "endianness": endian, "machine": name, "abi": abi, "pointer_width": pointer_width}


def _native_methods(dex_reports: list[dict[str, Any]]) -> list[dict[str, Any]]:
    methods = []
    for report in dex_reports:
        for class_item in report.get("classes", []):
            for method in class_item.get("methods", []):
                if method.get("native"):
                    methods.append(
                        {
                            "class_name": class_item.get("name"),
                            "package": class_item.get("package"),
                            "method": method.get("name"),
                            "descriptor": method.get("descriptor"),
                            "return_type": method.get("return_type"),
                            "parameters": [item.get("type") for item in method.get("parameters") or []],
                            "access": method.get("access"),
                            "dex": report.get("path"),
                        }
                    )
    return methods


def parse_apk(path: str | Path) -> dict[str, Any]:
    target = Path(path)
    result: dict[str, Any] = {
        "path": str(target),
        "format": "APK",
        "size": target.stat().st_size,
        "entries": [],
        "dex": [],
        "native_libraries": [],
        "signatures": [],
        "assets": [],
        "limitations": [
            "APK inventory only; APK signature verification is not claimed.",
            "DEX parsing is static table/bytecode inventory; no Dalvik/ART execution or exact Java source recovery.",
            "Native libraries are listed and identified, not executed or fully re-parsed here.",
        ],
    }
    try:
        archive = zipfile.ZipFile(target, "r")
    except (OSError, zipfile.BadZipFile) as exc:
        raise ValueError(f"not a readable APK/ZIP archive: {exc}") from exc
    dex_reports: list[dict[str, Any]] = []
    manifest_data = None
    with archive:
        infos = archive.infolist()[:MAX_ENTRIES]
        result["entry_count"] = len(infos)
        for info in infos:
            entry = {
                "name": info.filename,
                "size": info.file_size,
                "compressed_size": info.compress_size,
                "crc32": f"{info.CRC:08x}",
            }
            result["entries"].append(entry)
            name = info.filename
            if name == "AndroidManifest.xml":
                try:
                    manifest_data = archive.read(info)
                except (OSError, zipfile.BadZipFile) as exc:
                    result["manifest_error"] = str(exc)
            elif name.startswith("classes") and name.endswith(".dex"):
                result["dex"].append({"name": name, "size": info.file_size})
                if info.file_size <= MAX_DEX_BYTES:
                    try:
                        with tempfile.NamedTemporaryFile(suffix=".dex") as handle:
                            payload = archive.read(info)
                            handle.write(payload)
                            handle.flush()
                            parsed = parse_dex(handle.name)
                        parsed["path"] = name
                        dex_reports.append(parsed)
                        result["dex"][-1]["classes"] = len(parsed.get("classes", []))
                        result["dex"][-1]["methods"] = parsed.get("method_count")
                        result["dex"][-1]["fields"] = parsed.get("field_count")
                        result["dex"][-1]["class_names"] = [item["name"] for item in parsed.get("classes", [])[:500]]
                    except (ValueError, OSError, zipfile.BadZipFile, struct.error) as exc:
                        result["dex"][-1]["error"] = str(exc)
                else:
                    result["dex"][-1]["error"] = "DEX exceeds parser budget"
            elif name.startswith("lib/") and name.endswith(".so"):
                abi = name.split("/")[1] if "/" in name else None
                native = {"name": name, "size": info.file_size, "abi_directory": abi}
                try:
                    with archive.open(info) as handle:
                        header = handle.read(8192)
                    native.update(_elf_identity(header))
                except (OSError, zipfile.BadZipFile) as exc:
                    native["error"] = str(exc)
                result["native_libraries"].append(native)
            elif name.startswith("META-INF/") and name.endswith((".RSA", ".DSA", ".EC", ".SF")):
                result["signatures"].append({"name": name, "size": info.file_size})
            elif name.startswith(("assets/", "res/", "resources.arsc")):
                result["assets"].append({"name": name, "size": info.file_size})
    if manifest_data is not None:
        try:
            axml = parse_axml(manifest_data)
            result["manifest"] = manifest_summary(axml)
            result["manifest"]["binary_xml"] = True
        except (ValueError, IndexError, struct.error) as exc:
            result["manifest_error"] = str(exc)
    result["java_classes"] = sum(len(report.get("classes", [])) for report in dex_reports)
    result["java_methods"] = sum(report.get("method_count", 0) for report in dex_reports)
    result["native_methods"] = _native_methods(dex_reports)
    result["dex_reports"] = dex_reports
    result["android_manifest_present"] = any(item["name"] == "AndroidManifest.xml" for item in result["entries"])
    result["classes_dex_count"] = len(result["dex"])
    result["native_library_count"] = len(result["native_libraries"])
    result["limitations"].append("Only DEX classes and methods in the archive are inventoried; split APK/DEX semantics are not reconstructed.")
    return result
