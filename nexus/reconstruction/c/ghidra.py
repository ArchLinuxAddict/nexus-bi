"""Optional Ghidra headless decompiler backend for multi-architecture C reconstruction."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
CACHE_ROOT = ROOT / "artifacts" / "reconstruction" / "ghidra_cache"
SCRIPT_ROOT = ROOT / "nexus" / "reconstruction" / "ghidra"
SCRIPT_NAME = "DecompileAll.java"


def find_ghidra_headless() -> Path | None:
    configured = os.environ.get("NEXUS_GHIDRA_HEADLESS")
    if configured and Path(configured).exists():
        return Path(configured)
    found = shutil.which("analyzeHeadless")
    if found:
        return Path(found)
    candidates = sorted(Path.home().glob(".local/share/ghidra/*/support/analyzeHeadless"), reverse=True)
    candidates += sorted(Path("/opt").glob("ghidra*/support/analyzeHeadless"), reverse=True)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_FIELD_HIGH = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*)\._4_4_\b")


def support_header(functions: list[dict[str, Any]]) -> list[str]:
    text = "\n".join(item.get("source") or "" for item in functions)
    data_symbols = sorted(set(re.findall(r"\b(?:DAT|PTR_DAT|PTR_LOOP|PTR|UNK|SUB)_[0-9A-Fa-f]+\b", text)))
    lines = [
        "#ifndef NEXUS_GHIDRA_SUPPORT_H",
        "#define NEXUS_GHIDRA_SUPPORT_H",
        "",
        "#include <stdint.h>",
        "#include <stddef.h>",
        "#include <stdlib.h>",
        "#include <stdbool.h>",
        "#include <string.h>",
        "",
        "typedef unsigned char undefined1;",
        "typedef unsigned short undefined2;",
        "typedef unsigned int undefined4;",
        "typedef unsigned long undefined8;",
        "typedef void undefined;",
        "typedef uintptr_t (*code)();",
        "typedef unsigned char byte;",
        "typedef unsigned int uint;",
        "typedef unsigned short ushort;",
        "typedef unsigned char uchar;",
        "typedef unsigned long ulong;",
        "",
        "extern uintptr_t tpidr_el0;",
        "extern uintptr_t __stack_chk_guard;",
        "extern void __stack_chk_fail();",
        "extern void __cxa_finalize();",
        "",
        "#define CONCAT44(hi, lo) (((uint64_t)(hi) << 32) | (uint32_t)(lo))",
        "#define CONCAT22(hi, lo) (((uint32_t)(hi) << 16) | (uint16_t)(lo))",
        "#define SBORROW4(a, b) ((int64_t)((a) - (b)) < 0)",
        "#define SBORROW8(a, b) ((int64_t)((a) - (b)) < 0)",
        "#define SUB84(a, b) ((uint64_t)(a) - (uint64_t)(b))",
        "",
    ]
    for symbol in data_symbols:
        if symbol.startswith(("PTR_DAT", "PTR_LOOP")):
            lines.append(f"extern void *{symbol};")
        else:
            lines.append(f"extern unsigned char {symbol}[];")
    lines.extend(["", "#endif", ""])
    return lines


def file_prefix(current_name: str, functions: list[dict[str, Any]], header_name: str, source: str) -> list[str]:
    referenced = set(re.findall(r"\b[A-Za-z_][A-Za-z0-9_]*\b", source))
    lines = [f'#include "{header_name}"', ""]
    for item in functions:
        name = str(item.get("name") or "")
        if not name or name == current_name or name not in referenced or not _IDENTIFIER.match(name):
            continue
        lines.append(f"extern uintptr_t {name}();")
    lines.append("")
    return lines


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _cache_path(binary_sha256: str) -> Path:
    return CACHE_ROOT / f"{binary_sha256}.json"


def _parse_export(text: str) -> tuple[list[dict[str, Any]], int]:
    image_base = 0
    base_match = re.search(r"/\* IMAGE_BASE (?:0x)?([0-9A-Fa-f]+) \*/", text)
    if base_match:
        image_base = int(base_match.group(1), 16)
    marker = re.compile(r"/\* FUNCTION (.+?) @ ([0-9A-Fa-f]+) \*/")
    matches = list(marker.finditer(text))
    functions = []
    for index, match in enumerate(matches):
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        source = text[start:end].strip()
        if not source:
            continue
        source = _FIELD_HIGH.sub(r"((uint32_t)((\1) >> 32))", source)
        raw_entry = int(match.group(2), 16)
        entry = raw_entry - image_base if image_base and raw_entry >= image_base else raw_entry
        functions.append({
            "name": match.group(1).strip(),
            "entry": entry,
            "entry_va": f"0x{entry:x}",
            "ghidra_entry": f"0x{raw_entry:x}",
            "source": source,
        })
    return functions, image_base


def decompile_binary(binary_path: str | Path, timeout: int = 600) -> dict[str, Any]:
    path = Path(binary_path).resolve()
    ghidra = find_ghidra_headless()
    if ghidra is None:
        return {"status": "unavailable", "reason": "Ghidra analyzeHeadless was not found; set NEXUS_GHIDRA_HEADLESS.", "functions": []}
    if not SCRIPT_ROOT.joinpath(SCRIPT_NAME).exists():
        return {"status": "unavailable", "reason": "Ghidra export script is missing.", "functions": []}
    binary_sha = _file_hash(path)
    cache = _cache_path(binary_sha)
    if cache.exists():
        try:
            cached = json.loads(cache.read_text(encoding="utf-8"))
            if cached.get("status") == "completed" and cached.get("functions") and cached.get("cache_version") == 3:
                cached["cached"] = True
                return cached
        except json.JSONDecodeError:
            pass
    started = time.time()
    with tempfile.TemporaryDirectory(prefix="nexus_ghidra_") as directory:
        project_dir = Path(directory) / "project"
        project_dir.mkdir(parents=True, exist_ok=True)
        export_path = Path(directory) / "export.c"
        command = [
            str(ghidra),
            str(project_dir),
            "nexus_reconstruction",
            "-import", str(path),
            "-scriptPath", str(SCRIPT_ROOT),
            "-postScript", SCRIPT_NAME,
            "-deleteProject",
            "-analysisTimeoutPerFile", str(max(60, timeout // 2)),
        ]
        environment = dict(os.environ)
        environment["GHIDRA_EXPORT_PATH"] = str(export_path)
        try:
            process = subprocess.run(command, capture_output=True, text=True, timeout=timeout, env=environment)
        except subprocess.TimeoutExpired as exc:
            return {"status": "timeout", "reason": f"Ghidra decompilation exceeded {timeout}s", "functions": [], "elapsed_seconds": round(time.time() - started, 3), "stderr": (exc.stderr or "")[-2000:] if isinstance(exc.stderr, str) else ""}
        except OSError as exc:
            return {"status": "failed", "reason": str(exc), "functions": [], "elapsed_seconds": round(time.time() - started, 3)}
        if process.returncode != 0 or not export_path.exists():
            return {
                "status": "failed",
                "reason": f"Ghidra exited with {process.returncode}",
                "functions": [],
                "elapsed_seconds": round(time.time() - started, 3),
                "stderr": (process.stderr or process.stdout or "")[-3000:],
            }
        text = export_path.read_text(encoding="utf-8", errors="replace")
    functions, image_base = _parse_export(text)
    result = {
        "status": "completed" if functions else "no_functions",
        "cache_version": 3,
        "backend": "ghidra",
        "ghidra": str(ghidra),
        "image_base": image_base,
        "binary_sha256": binary_sha,
        "functions": functions,
        "elapsed_seconds": round(time.time() - started, 3),
        "cached": False,
        "limitations": [
            "Ghidra output is a static decompiler candidate, not original source.",
            "Decompiled names and types are inferred by Ghidra.",
            "Compilation of Ghidra output is attempted separately and is not behavioral verification.",
        ],
    }
    if functions:
        CACHE_ROOT.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result
