"""Discovery and capability matrix for external analysis tools."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from nexus.toolchain.models import ToolIdentity


@dataclass(frozen=True)
class ToolSpec:
    name: str
    capabilities: tuple[str, ...]
    env: str | None = None
    candidates: tuple[str, ...] = ()
    version_args: tuple[str, ...] = ("--version",)
    license: str | None = None
    required: bool = False


TOOL_SPECS: tuple[ToolSpec, ...] = (
    ToolSpec(
        "ghidra_headless",
        ("native_decompile", "function_discovery", "control_flow", "types", "xrefs", "pcode"),
        env="NEXUS_GHIDRA_HEADLESS",
        candidates=("analyzeHeadless",),
        version_args=(),
        license="Apache-2.0",
        required=True,
    ),
    ToolSpec(
        "llvm_objdump",
        ("multiarch_disassembly", "object_headers", "relocations"),
        candidates=("llvm-objdump",),
        version_args=("--version",),
        license="Apache-2.0 WITH LLVM-exception",
    ),
    ToolSpec(
        "objdump",
        ("disassembly", "object_headers"),
        candidates=("objdump",),
        version_args=("--version",),
        license="GPL-3.0-or-later",
    ),
    ToolSpec(
        "readelf",
        ("elf_metadata", "sections", "symbols", "relocations", "notes"),
        candidates=("readelf",),
        version_args=("--version",),
        license="GPL-3.0-or-later",
    ),
    ToolSpec(
        "nm",
        ("symbols",),
        candidates=("nm",),
        version_args=("--version",),
        license="GPL-3.0-or-later",
    ),
    ToolSpec(
        "clang",
        ("compile", "syntax_check", "cross_target_compile"),
        candidates=("clang",),
        version_args=("--version",),
        license="Apache-2.0 WITH LLVM-exception",
    ),
    ToolSpec(
        "gcc",
        ("compile", "syntax_check"),
        candidates=("gcc",),
        version_args=("--version",),
        license="GPL-3.0-or-later",
    ),
    ToolSpec(
        "java",
        ("java_runtime",),
        candidates=("java",),
        version_args=("-version",),
        license="GPL-2.0-with-classpath-exception",
    ),
    ToolSpec(
        "javac",
        ("compile", "java_syntax_check"),
        candidates=("javac",),
        version_args=("-version",),
        license="GPL-2.0-with-classpath-exception",
    ),
    ToolSpec(
        "javap",
        ("bytecode_disassembly", "java_metadata"),
        candidates=("javap",),
        version_args=("-version",),
        license="GPL-2.0-with-classpath-exception",
    ),
    ToolSpec(
        "jadx",
        ("java_decompile", "dex_decompile", "apk_decompile"),
        env="NEXUS_JADX",
        candidates=("jadx", "jadx-cli"),
        version_args=("--version",),
        license="Apache-2.0",
        required=True,
    ),
    ToolSpec(
        "jadx_headless_mcp",
        ("java_decompile", "dex_decompile", "apk_decompile"),
        env="NEXUS_JADX_HEADLESS_JAR",
        candidates=(),
        version_args=(),
        license="Apache-2.0",
        required=False,
    ),
    ToolSpec(
        "cfr",
        ("java_decompile", "independent_java_check"),
        env="NEXUS_CFR_JAR",
        candidates=("cfr",),
        version_args=("--version",),
        license="MIT",
        required=True,
    ),
    ToolSpec(
        "z3",
        ("constraint_solving", "equivalence_probe"),
        candidates=("z3",),
        version_args=("--version",),
        license="MIT",
        required=False,
    ),
)


def _resolve(spec: ToolSpec) -> Path | None:
    if spec.env and os.environ.get(spec.env):
        candidate = Path(os.environ[spec.env])
        if candidate.exists():
            return candidate
    for name in spec.candidates:
        found = shutil.which(name)
        if found:
            return Path(found)
    if spec.name == "ghidra_headless":
        default = sorted(Path.home().glob(".local/share/ghidra/*/support/analyzeHeadless"), reverse=True)
        if default:
            return default[0]
    if spec.name == "jadx_headless_mcp":
        default = Path.home() / ".local" / "share" / "rea" / "android" / "jadx-headless-mcp-0.7.1-all.jar"
        if default.exists():
            return default
    tools_root = Path(__file__).resolve().parents[2] / "artifacts" / "toolchain" / "tools"
    if spec.name == "jadx":
        matches = sorted(tools_root.glob("jadx-*/bin/jadx"), reverse=True)
        if matches:
            return matches[0]
    if spec.name == "cfr":
        matches = sorted(tools_root.glob("cfr-*.jar"), reverse=True)
        if matches:
            return matches[0]
    return None


def _version(spec: ToolSpec, path: Path | None) -> str:
    if path is None:
        return "missing"
    if spec.name == "ghidra_headless":
        match = re.search(r"ghidra_([0-9][0-9.]*)", str(path))
        return match.group(1) if match else "unknown"
    if spec.name == "jadx_headless_mcp":
        match = re.search(r"jadx-headless-mcp-([0-9][0-9.]*)", path.name)
        return match.group(1) if match else "unknown"
    if spec.name == "cfr" and path.suffix == ".jar":
        try:
            result = subprocess.run(["java", "-jar", str(path), "--version"], capture_output=True, text=True, timeout=15)
            text = (result.stdout or result.stderr or "").strip().splitlines()
            return text[0].strip() if text else "unknown"
        except (OSError, subprocess.TimeoutExpired):
            return "unknown"
    if not spec.version_args:
        return "unknown"
    try:
        result = subprocess.run([str(path), *spec.version_args], capture_output=True, text=True, timeout=15)
        text = (result.stdout or result.stderr or "").strip().splitlines()
        if spec.name == "llvm_objdump":
            for line in text:
                if "version" in line.lower():
                    return line.strip()
        return text[0].strip() if text else "unknown"
    except (OSError, subprocess.TimeoutExpired):
        return "unknown"


class ToolRegistry:
    def __init__(self, root: str | Path | None = None) -> None:
        self.root = Path(root) if root else Path(__file__).resolve().parents[2] / "artifacts" / "toolchain"

    def discover(self) -> dict[str, ToolIdentity]:
        identities: dict[str, ToolIdentity] = {}
        for spec in TOOL_SPECS:
            path = _resolve(spec)
            status = "missing"
            if path is not None:
                status = "available"
                if spec.name == "jadx_headless_mcp":
                    status = "present_unconfigured"
            identities[spec.name] = ToolIdentity(
                name=spec.name,
                version=_version(spec, path),
                path=str(path) if path else None,
                status=status,
                capabilities=spec.capabilities,
                license=spec.license,
                notes=(() if status == "available" else ("not found or not configured",)),
            )
        return identities

    def matrix(self) -> dict[str, Any]:
        identities = self.discover()
        required = {spec.name for spec in TOOL_SPECS if spec.required}
        missing_required = [name for name in sorted(required) if identities[name].status == "missing"]
        unconfigured_required = [name for name in sorted(required) if identities[name].status == "present_unconfigured"]
        return {
            "schema_version": "nexus.tool-matrix/v1",
            "created_at": datetime.now(UTC).isoformat(),
            "host": {
                "cpu_count": os.cpu_count(),
                "python": "3.13",
            },
            "tools": {name: identity.to_dict() for name, identity in sorted(identities.items())},
            "required": sorted(required),
            "missing_required": missing_required,
            "unconfigured_required": unconfigured_required,
            "status": "ready" if not missing_required and not unconfigured_required else "incomplete",
        }

    def write_matrix(self, path: str | Path | None = None) -> Path:
        target = Path(path) if path else self.root / "tool_matrix.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.matrix(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return target


def discover_tools(root: str | Path | None = None) -> dict[str, Any]:
    return ToolRegistry(root).matrix()
