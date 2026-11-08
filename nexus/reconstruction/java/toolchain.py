"""Tool-first Java/DEX/APK reconstruction using JADX and CFR."""

from __future__ import annotations

import hashlib
import json
import shutil
import zipfile
from pathlib import Path
from typing import Any

from nexus.toolchain.adapters import CfrAdapter, JadxAdapter

TOOL_OK_STATUSES = {"completed", "completed_with_errors"}

FRAMEWORK_PREFIXES = (
    "android/",
    "androidx/",
    "com/google/android/",
    "dalvik/",
    "java/",
    "javax/",
    "kotlin/",
    "kotlinx/",
    "org/intellij/",
    "org/jetbrains/",
)

RESOURCE_CLASS_NAMES = {"R.java", "BuildConfig.java"}

MAX_SOURCE_FILE_BYTES = 1_500_000
MAX_SOURCE_TOTAL_BYTES = 24 * 1024 * 1024


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _android_package(target: Path) -> str | None:
    if target.suffix.lower() != ".apk":
        return None
    try:
        with zipfile.ZipFile(target) as archive:
            manifest = archive.read("AndroidManifest.xml")
        from nexus.formats.android import manifest_summary, parse_axml

        package = manifest_summary(parse_axml(manifest)).get("package")
        return str(package) if package else None
    except Exception:
        return None


def _project_files(project_name: str) -> dict[str, str]:
    return {
        "pom.xml": (
            '<?xml version="1.0" encoding="UTF-8"?>\n'
            '<project xmlns="http://maven.apache.org/POM/4.0.0">\n'
            "  <modelVersion>4.0.0</modelVersion>\n"
            "  <groupId>local.nexus</groupId>\n"
            f"  <artifactId>{project_name}</artifactId>\n"
            "  <version>0.1.0</version>\n"
            "  <properties><maven.compiler.release>17</maven.compiler.release></properties>\n"
            "</project>\n"
        ),
        "build.gradle": (
            "plugins { id 'java' }\n"
            "group = 'local.nexus'\nversion = '0.1.0'\n"
            "java { toolchain { languageVersion = JavaLanguageVersion.of(17) } }\n"
        ),
        "README.md": (
            f"# {project_name}\n\n"
            "Java reconstruction produced by external decompilers. JADX and CFR output are "
            "reconstruction candidates, not original source. Compilation is not behavioral verification.\n"
        ),
    }


def _source_priority(relative: Path, package_hint: str | None) -> int:
    name = relative.name
    if name in RESOURCE_CLASS_NAMES or name.startswith("R$") or name == "BuildConfig.java":
        return 3
    text = relative.as_posix()
    if package_hint:
        prefix = package_hint.replace(".", "/").rstrip("/") + "/"
        if text.startswith(prefix):
            return 0
    if any(text.startswith(prefix) for prefix in FRAMEWORK_PREFIXES):
        return 2
    return 1


def _collect_sources(
    root: Path,
    max_classes: int,
    *,
    package_hint: str | None = None,
) -> tuple[dict[str, str], dict[str, Any]]:
    candidates: list[tuple[int, int, str, Path]] = []
    for path in root.rglob("*.java"):
        relative = path.relative_to(root)
        parts = relative.parts
        if parts and parts[0] in {"sources", "src"}:
            relative = Path(*parts[1:])
        try:
            size = path.stat().st_size
        except OSError:
            continue
        candidates.append((_source_priority(relative, package_hint), size, relative.as_posix(), path))
    candidates.sort(key=lambda item: (item[0], item[2]))
    sources: dict[str, str] = {}
    total_bytes = 0
    skipped_large = 0
    skipped_budget = 0
    for _, size, relative, path in candidates:
        if len(sources) >= max_classes:
            skipped_budget += 1
            continue
        if size > MAX_SOURCE_FILE_BYTES:
            skipped_large += 1
            continue
        if total_bytes + size > MAX_SOURCE_TOTAL_BYTES:
            skipped_budget += 1
            continue
        key = f"src/main/java/{relative}"
        sources[key] = path.read_text(encoding="utf-8", errors="replace")
        total_bytes += size
    stats = {
        "source_files_total": len(candidates),
        "source_files_returned": len(sources),
        "source_files_skipped_large": skipped_large,
        "source_files_skipped_budget": skipped_budget,
        "source_bytes_returned": total_bytes,
        "package_hint": package_hint,
    }
    return sources, stats


def tool_first_java_reconstruction(
    path: str | Path,
    *,
    out_dir: str | Path | None = None,
    project_name: str = "nexus_java_reconstruction",
    max_classes: int = 2000,
) -> dict[str, Any] | None:
    target = Path(path)
    if target.suffix.lower() not in {".apk", ".dex", ".jar", ".zip", ".class"}:
        return None
    root = Path(out_dir) if out_dir is not None else Path("artifacts") / "java_reconstructions" / project_name
    root.mkdir(parents=True, exist_ok=True)
    tools_root = root / "tools"
    package_hint = _android_package(target)
    evidence: dict[str, Any] = {
        "binary_sha256": _sha256(target),
        "project": project_name,
        "source_format": target.suffix.lower().lstrip("."),
        "package_hint": package_hint,
        "tools": {},
        "claims": [],
    }
    primary = JadxAdapter().decompile(target, tools_root / "jadx")
    evidence["tools"]["jadx"] = primary.to_dict()
    evidence["jadx_status"] = primary.status
    sources: dict[str, str] = {}
    stats: dict[str, Any] = {}
    classification = "jadx_decompiler_reconstruction"
    backend = "jadx"
    if primary.status in TOOL_OK_STATUSES:
        sources, stats = _collect_sources(tools_root / "jadx", max_classes, package_hint=package_hint)
    if not sources and target.suffix.lower() in {".jar", ".zip", ".class"}:
        fallback = CfrAdapter().decompile(target, tools_root / "cfr")
        evidence["tools"]["cfr"] = fallback.to_dict()
        evidence["cfr_status"] = fallback.status
        if fallback.status in TOOL_OK_STATUSES:
            sources, stats = _collect_sources(tools_root / "cfr", max_classes, package_hint=package_hint)
            classification = "cfr_decompiler_reconstruction"
            backend = "cfr"
    elif target.suffix.lower() in {".jar", ".zip", ".class"}:
        cfr = CfrAdapter().decompile(target, tools_root / "cfr")
        evidence["tools"]["cfr"] = cfr.to_dict()
        evidence["cfr_status"] = cfr.status
    if not sources:
        return None
    files = dict(sources)
    files.update(_project_files(project_name))
    evidence["backend"] = backend
    evidence["source_selection"] = stats
    evidence["java_files"] = sorted(sources)
    evidence["limitations"] = [
        "JADX/CFR output is reconstructed Java, not original source.",
        "Decompiler names, types and control flow may be wrong or incomplete.",
        f"Only {stats.get('source_files_returned', len(sources))} of {stats.get('source_files_total', len(sources))} decompiled sources are returned in this payload; the full tool output remains under tools/.",
        "Compilation is not behavioral verification.",
    ]
    if primary.status == "completed_with_errors":
        evidence["limitations"].insert(0, "JADX reported errors on some classes; returned sources are partial.")
    files["evidence_map.json"] = json.dumps(evidence, indent=2, sort_keys=True) + "\n"
    sources_root = root / "src"
    if sources_root.exists():
        shutil.rmtree(sources_root)
    for relative, content in files.items():
        destination = root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(content, encoding="utf-8")
    return {
        "language": "java",
        "source_format": target.suffix.lower().lstrip(".") or "archive",
        "project_name": project_name,
        "classification": classification,
        "files": files,
        "evidence_map": evidence,
        "limitations": evidence["limitations"],
    }
