"""Unified Java/JNI artifact analysis and reconstruction service."""

from __future__ import annotations

import json
import tempfile
import zipfile
from pathlib import Path
from typing import Any

from nexus.formats.apk import parse_apk
from nexus.formats.dex import parse_dex
from nexus.formats.java_class import parse_java_class
from nexus.reconstruction.java.classfile import (
    reconstruct_java_class_file,
    reconstruct_java_class_source,
)
from nexus.reconstruction.java.dex import reconstruct_java_dex
from nexus.reconstruction.java.toolchain import tool_first_java_reconstruction


def _class_summary(parsed: dict[str, Any]) -> dict[str, Any]:
    methods = parsed.get("methods") or []
    return {
        "name": parsed.get("name"),
        "package": parsed.get("package"),
        "superclass": parsed.get("superclass"),
        "interfaces": parsed.get("interfaces"),
        "source_file": parsed.get("source_file"),
        "fields": len(parsed.get("fields") or []),
        "methods": len(methods),
        "native_methods": [method.get("name") for method in methods if method.get("native")],
        "constructors": [method.get("name") for method in methods if method.get("constructor")],
        "access": parsed.get("access"),
        "version": parsed.get("version"),
        "limitations": parsed.get("limitations"),
    }


def analyze_java_artifact(path: str | Path) -> dict[str, Any]:
    target = Path(path)
    suffix = target.suffix.lower()
    if suffix == ".class":
        parsed = parse_java_class(target)
        return {"format": "class", "path": str(target), "summary": _class_summary(parsed), "class": parsed}
    if suffix == ".dex":
        parsed = parse_dex(target)
        return {
            "format": "dex",
            "path": str(target),
            "summary": {
                "version": parsed.get("version"),
                "classes": parsed.get("class_count"),
                "methods": parsed.get("method_count"),
                "fields": parsed.get("field_count"),
                "class_names": [item.get("name") for item in parsed.get("classes", [])[:500]],
            },
            "dex": parsed,
        }
    if suffix == ".apk":
        parsed = parse_apk(target)
        return {
            "format": "apk",
            "path": str(target),
            "summary": {
                "package": (parsed.get("manifest") or {}).get("package"),
                "manifest": parsed.get("manifest"),
                "classes": parsed.get("java_classes"),
                "methods": parsed.get("java_methods"),
                "native_methods": parsed.get("native_methods"),
                "native_libraries": parsed.get("native_libraries"),
                "dex_count": parsed.get("classes_dex_count"),
                "native_library_count": parsed.get("native_library_count"),
            },
            "apk": parsed,
        }
    if suffix in {".jar", ".zip"}:
        classes = []
        with zipfile.ZipFile(target) as archive:
            for info in archive.infolist():
                if not info.filename.endswith(".class"):
                    continue
                with archive.open(info) as handle, tempfile.NamedTemporaryFile(suffix=".class") as temporary:
                    temporary.write(handle.read())
                    temporary.flush()
                    try:
                        classes.append(parse_java_class(temporary.name))
                    except ValueError:
                        continue
        return {
            "format": "archive",
            "path": str(target),
            "summary": {
                "classes": len(classes),
                "class_names": [item.get("name") for item in classes],
                "native_methods": [
                    {"class_name": item.get("name"), "method": method.get("name"), "descriptor": method.get("descriptor")}
                    for item in classes
                    for method in item.get("methods") or []
                    if method.get("native")
                ],
            },
            "classes": classes,
        }
    if target.is_dir():
        classes = [parse_java_class(item) for item in sorted(target.rglob("*.class"))]
        return {
            "format": "class-directory",
            "path": str(target),
            "summary": {
                "classes": len(classes),
                "class_names": [item.get("name") for item in classes],
                "native_methods": [
                    {"class_name": item.get("name"), "method": method.get("name"), "descriptor": method.get("descriptor")}
                    for item in classes
                    for method in item.get("methods") or []
                    if method.get("native")
                ],
            },
            "classes": classes,
        }
    if suffix == ".so":
        from nexus.program.bundle import analyze_binary
        bundle = analyze_binary(target)
        return {"format": "native", "path": str(target), "summary": bundle.summary(), "jni": bundle.jni_report}
    raise ValueError(f"unsupported Java artifact: {target.suffix or 'no suffix'}")


def _project_files(project_name: str) -> dict[str, str]:
    return {
        "pom.xml": (
            '<?xml version="1.0" encoding="UTF-8"?>\n'
            '<project xmlns="http://maven.apache.org/POM/4.0.0" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" '
            'xsi:schemaLocation="http://maven.apache.org/POM/4.0.0 https://maven.apache.org/xsd/maven-4.0.0.xsd">\n'
            "  <modelVersion>4.0.0</modelVersion>\n  <groupId>local.nexus</groupId>\n"
            f"  <artifactId>{project_name}</artifactId>\n  <version>0.1.0</version>\n"
            "  <properties><maven.compiler.release>17</maven.compiler.release></properties>\n</project>\n"
        ),
        "build.gradle": (
            "plugins { id 'java' }\n"
            "group = 'local.nexus'\nversion = '0.1.0'\n"
            "java { toolchain { languageVersion = JavaLanguageVersion.of(17) } }\n"
        ),
        "README.md": (
            f"# {project_name}\n\n"
            "Approximate Java reconstruction. Declarations are preserved; a straight-line bytecode subset is translated. "
            "Unresolved control flow, constructors, exceptions and dynamic language features are explicit. "
            "Compilation is a syntax check, not behavioral verification.\n"
        ),
    }


def reconstruct_java_artifact(
    path: str | Path,
    *,
    out_dir: str | Path | None = None,
    project_name: str = "nexus_java_reconstruction",
    max_classes: int = 2000,
) -> dict[str, Any]:
    target = Path(path)
    suffix = target.suffix.lower()
    tool_result = tool_first_java_reconstruction(
        target,
        out_dir=out_dir,
        project_name=project_name,
        max_classes=max_classes,
    )
    if tool_result is not None:
        return tool_result
    if suffix == ".class":
        return reconstruct_java_class_file(target, out_dir=out_dir, project_name=project_name)
    if suffix == ".dex":
        return reconstruct_java_dex(parse_dex(target), out_dir=out_dir, project_name=project_name, max_classes=max_classes)
    if suffix == ".apk":
        apk = parse_apk(target)
        classes = [item for report in apk.get("dex_reports", []) for item in report.get("classes", [])]
        merged = {
            "version": ",".join(sorted({str(report.get("version")) for report in apk.get("dex_reports", [])})),
            "classes": classes,
            "path": str(target),
        }
        result = reconstruct_java_dex(merged, out_dir=out_dir, project_name=project_name, max_classes=max_classes)
        result["source_format"] = "APK/DEX"
        result["apk"] = {
            "manifest": apk.get("manifest"),
            "native_libraries": apk.get("native_libraries"),
            "native_methods": apk.get("native_methods"),
        }
        if out_dir is not None:
            evidence_path = Path(out_dir) / "evidence_map.json"
            evidence = json.loads(evidence_path.read_text(encoding="utf-8")) if evidence_path.exists() else result["evidence_map"]
            evidence["apk"] = result["apk"]
            evidence_path.write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
            result["evidence_map"] = evidence
        return result
    if suffix in {".jar", ".zip", ".class-dir"} or target.is_dir():
        classes = []
        if target.is_dir():
            for item in sorted(target.rglob("*.class")):
                classes.append(parse_java_class(item))
        else:
            with zipfile.ZipFile(target) as archive:
                for info in archive.infolist():
                    if not info.filename.endswith(".class"):
                        continue
                    with archive.open(info) as handle, tempfile.NamedTemporaryFile(suffix=".class") as temporary:
                        temporary.write(handle.read())
                        temporary.flush()
                        try:
                            classes.append(parse_java_class(temporary.name))
                        except ValueError:
                            continue
        files: dict[str, str] = {}
        evidence = []
        for parsed in classes[:max_classes]:
            source, class_evidence = reconstruct_java_class_source(parsed, project_name=project_name)
            package = parsed.get("package") or ""
            class_name = parsed.get("simple_name") or "Reconstructed"
            path_name = f"src/main/java/{package.replace('.', '/') + '/' if package else ''}{class_name}.java"
            files[path_name] = source
            evidence.append(class_evidence)
        files.update(_project_files(project_name))
        files["evidence_map.json"] = json.dumps({"classes": evidence, "limitations": ["Archive reconstruction is bounded."]}, indent=2) + "\n"
        if out_dir is not None:
            root = Path(out_dir)
            for relative, content in files.items():
                destination = root / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_text(content, encoding="utf-8")
        return {
            "language": "java",
            "source_format": "archive",
            "project_name": project_name,
            "classification": "structural_archive_reconstruction",
            "files": files,
            "evidence_map": {"classes": evidence},
            "limitations": ["Only a bounded number of classes is emitted.", "Compilation is not behavioral verification."],
        }
    if suffix == ".so":
        from nexus.program.bundle import analyze_binary
        from nexus.reconstruction.java.generator import reconstruct_java
        bundle = analyze_binary(target)
        return reconstruct_java(bundle, out_dir=out_dir, project_name=project_name)
    raise ValueError(f"unsupported Java artifact: {target.suffix or 'no suffix'}")
