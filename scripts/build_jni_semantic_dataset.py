#!/usr/bin/env python3
"""Build residual training pairs from native JNI semantic reconstruction.

Each row maps a Ghidra-decompiled native JNI function to the Java method that
the semantic reconstructor recovered from its JNI call sequence. The residual
model can learn to generalize these mappings beyond the deterministic
templates. No training is started by this script.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from nexus.program.bundle import analyze_binary
from nexus.reconstruction.java.jni_semantics import SemanticMethod, analyze_method
from nexus.toolchain.adapters import GhidraAdapter

SKIP_TEMPLATES = {"trace"}


def _identifier(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_$]", "_", value)
    if not cleaned:
        cleaned = "Reconstructed"
    if cleaned[0].isdigit():
        cleaned = f"J_{cleaned}"
    return cleaned


FIELD_TYPES = {
    "extraDataMap": "androidx.collection.SimpleArrayMap",
    "lifecycleRegistry": "androidx.lifecycle.LifecycleRegistry",
}


def _method_target(entry: dict[str, Any], semantic: SemanticMethod) -> str:
    package = entry.get("package") or ""
    class_name = entry.get("simple_class") or (entry.get("class_name") or "Native").split(".")[-1]
    simple = _identifier(class_name)
    method = entry.get("method") or "nativeMethod"
    arguments = entry.get("arguments") or []
    params = ", ".join(f"{argument} arg{index}" for index, argument in enumerate(arguments))
    body_lines = [
        line for line in semantic.body.splitlines()
        if line.strip() and not line.strip().startswith("//")
    ]
    if not body_lines:
        body_lines = ["return;" if semantic.return_type == "void" else "return null;"]
    fields = sorted(set(re.findall(r"this\.([A-Za-z_$][\w$]*)\b(?!\s*\()", semantic.body)))
    extends = " extends android.app.Activity" if semantic.template in {"key_dispatch", "super_dispatch"} else ""
    implements = " implements androidx.lifecycle.LifecycleOwner" if "lifecycleRegistry" in fields else ""
    lines = []
    if package:
        lines.append(f"package {package};")
        lines.append("")
    class_prefix = "public class" if semantic.instance or method == "<init>" else "public final class"
    lines.append(f"{class_prefix} {simple}{extends}{implements} {{")
    for field in fields:
        field_type = FIELD_TYPES.get(field, "Object")
        initializer = ""
        if field == "extraDataMap":
            initializer = f" = new {field_type}()"
        elif field == "lifecycleRegistry" and "LifecycleOwner" in implements:
            initializer = f" = new {field_type}(this)"
        lines.append(f"    private {field_type} {field}{initializer};")
    if fields:
        lines.append("")
    class_fqn = f"{package}.{simple}" if package else simple
    if f"{class_fqn}.ExtraData" in semantic.body or f"{class_fqn}.ExtraData" in semantic.return_type:
        lines.append("    public static class ExtraData {")
        lines.append("    }")
        lines.append("")
    if method == "<init>":
        lines.append(f"    public {simple}({params}) {{")
        for line in body_lines:
            if line.strip() in {"return null;", "return false;", "return;"}:
                continue
            lines.append(f"        {line.rstrip()}")
        lines.append("    }")
    else:
        modifier = "public" if semantic.instance else "public static"
        lines.append(f"    {modifier} {semantic.return_type} {_identifier(method)}({params}) {{")
        for line in body_lines:
            lines.append(f"        {line.rstrip()}")
        lines.append("    }")
    lines.append("}")
    return "\n".join(lines)


def _tool_text(entry: dict[str, Any], source: str) -> str:
    header = [
        f"JNI symbol: {entry.get('symbol')}",
        f"Java class: {entry.get('class_name')}",
        f"Java method: {entry.get('method')}",
        f"Descriptor: {entry.get('descriptor')}",
        f"Parameters: {', '.join(entry.get('arguments') or [])}",
        "",
        "Ghidra decompilation:",
    ]
    return "\n".join(header) + "\n" + source


def _binary_sources(args: argparse.Namespace) -> list[Path]:
    paths: list[Path] = []
    for value in args.path or []:
        path = Path(value)
        if path.is_dir():
            paths.extend(sorted(path.rglob("*.so")))
        elif path.exists():
            paths.append(path)
    if args.catalog:
        catalog = Path(args.catalog)
        rows = [json.loads(line) for line in catalog.read_text(encoding="utf-8").splitlines() if line.strip()]
        for row in rows[: args.limit]:
            candidate = Path(str(row.get("resolved_path") or row.get("path") or ""))
            if candidate.exists() and candidate.suffix == ".so":
                paths.append(candidate)
    seen: set[str] = set()
    unique = []
    for path in paths:
        key = str(path.resolve())
        if key in seen:
            continue
        seen.add(key)
        unique.append(path)
    return unique[: args.limit] if args.limit else unique


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", action="append", help="Native .so file or directory (repeatable)")
    parser.add_argument("--catalog", default=None, help="Optional real-world catalog.jsonl")
    parser.add_argument("--out", default="artifacts/dataset/jni_semantic_v1")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--min-confidence", type=float, default=0.4)
    parser.add_argument("--max-methods-per-binary", type=int, default=200)
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    sources = _binary_sources(args)
    if not sources:
        raise SystemExit("no native sources found")
    rows: list[dict[str, Any]] = []
    for binary in sources:
        try:
            bundle = analyze_binary(binary)
        except Exception as exc:
            print(f"skip {binary}: {exc}", flush=True)
            continue
        entries = bundle.jni_report.get("entries") or []
        if not entries:
            continue
        try:
            ghidra = GhidraAdapter().decompile(binary)
        except Exception as exc:
            print(f"ghidra failed {binary}: {exc}", flush=True)
            continue
        if ghidra.status not in {"completed", "completed_with_errors"}:
            print(f"ghidra status {ghidra.status} for {binary}: {ghidra.error}", flush=True)
            continue
        functions = {item.get("name"): item for item in (ghidra.artifacts.get("functions") or [])}
        image_base = ghidra.artifacts.get("image_base")
        count = 0
        for entry in entries:
            if count >= args.max_methods_per_binary:
                break
            source = (functions.get(entry.get("symbol")) or {}).get("source") or ""
            semantic = analyze_method(entry, source, elf=bundle.elf, image_base=image_base)
            if semantic is None or semantic.template in SKIP_TEMPLATES:
                continue
            if semantic.confidence < args.min_confidence:
                continue
            target = _method_target(entry, semantic)
            if not target:
                continue
            rows.append({
                "sample_id": f"{bundle.elf.sha256[:16]}::{entry.get('symbol')}",
                "language": "java",
                "tool": "ghidra_headless",
                "tool_status": ghidra.status,
                "compile_status": "unknown",
                "warning_count": len(ghidra.warnings),
                "cached": bool(ghidra.cached),
                "tool_text": _tool_text(entry, source)[:64000],
                "target_text": target[:64000],
                "failure_label": 0,
                "candidate_label": 1.0,
                "confidence_label": float(semantic.confidence),
                "source": "jni-semantic",
                "project_id": bundle.elf.sha256,
                "template": semantic.template,
                "jni_symbol": entry.get("symbol"),
                "class_name": entry.get("class_name"),
                "method": entry.get("method"),
                "descriptor": entry.get("descriptor"),
            })
            count += 1
        print(f"{binary}: {count} JNI semantic pairs", flush=True)
    groups = sorted({row["project_id"] for row in rows})
    split_for = {}
    for index, group in enumerate(groups):
        if index == 0:
            split_for[group] = "train"
        elif index % 10 == 1:
            split_for[group] = "validation"
        elif index % 10 == 9:
            split_for[group] = "test"
        else:
            split_for[group] = "train"
    buckets = {"train": [], "validation": [], "test": []}
    for row in rows:
        buckets[split_for[row["project_id"]]].append(row)
    for split, items in buckets.items():
        path = out / f"{split}.jsonl"
        path.write_text("\n".join(json.dumps(row, sort_keys=True) for row in items) + ("\n" if items else ""), encoding="utf-8")
    summary = {
        "schema_version": "nexus.jni-semantic/v1",
        "samples": len(rows),
        "splits": {key: len(value) for key, value in buckets.items()},
        "binaries": len(groups),
        "templates": dict(Counter(row["template"] for row in rows)),
        "root": str(out),
        "limitations": [
            "Targets are JNI semantic reconstructions, not original Java source.",
            "Control flow and unknown return types in targets are approximated.",
            "No behavioral equivalence is claimed.",
        ],
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
