#!/usr/bin/env python3
"""Build JNI facade training pairs from ELF/JNI evidence.

This is the protector-aware dataset lane: it recovers exported Java_* methods
and JNINativeMethod tables (including libraries whose JNI_OnLoad is obfuscated)
and maps them to exact Java native declarations. It also records protection
evidence such as runtime decryption markers. No training is started here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from nexus.elf.parser import parse_elf
from nexus.jni.analysis import decode_jni_descriptor, decode_jni_symbol
from nexus.jni.registry import guess_class_name, protection_profile, scan_static_tables


def _java_identifier(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_$]", "_", value or "")
    if not cleaned:
        cleaned = "Reconstructed"
    if cleaned[0].isdigit():
        cleaned = f"J_{cleaned}"
    return cleaned


def _split_class(class_name: str) -> tuple[str, str]:
    value = (class_name or "UnassignedNatives").replace("/", ".")
    package, _, simple = value.rpartition(".")
    return package, simple


def _target_class(class_name: str, methods: list[dict[str, Any]], protection: str, evidence: list[str]) -> str:
    package, simple = _split_class(class_name)
    lines = []
    if package:
        lines.append(f"package {package};")
        lines.append("")
    lines.append("// Recovered from the native JNI registration, not original Java.")
    if protection != "none":
        lines.append(f"// protection: {protection}")
        if evidence:
            lines.append(f"// evidence: {', '.join(evidence[:6])}")
    lines.append(f"public final class {_java_identifier(simple)} {{")
    lines.append(f"    private {_java_identifier(simple)}() {{")
    lines.append("    }")
    lines.append("")
    seen = set()
    for method in methods:
        name = method["name"]
        signature = method.get("signature") or "()"
        key = (name, signature)
        if key in seen:
            continue
        seen.add(key)
        descriptor = method.get("descriptor") or (signature if signature.startswith("(") and ")" in signature else None)
        arguments, return_type = decode_jni_descriptor(descriptor) if descriptor else ([], "unknown")
        if return_type == "unknown":
            return_type = "Object"
        params = ", ".join(f"{argument} arg{index}" for index, argument in enumerate(arguments))
        lines.append(f"    public static native {return_type} {_java_identifier(name)}({params});")
    lines.append("}")
    return "\n".join(lines)


def _entries(elf: Any) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for symbol in list(elf.symbols) + list(elf.dynamic_symbols):
        if not symbol.name or not symbol.name.startswith("Java_"):
            continue
        decoded = decode_jni_symbol(symbol.name)
        if decoded is None:
            continue
        entries.append({
            "name": decoded["method"],
            "signature": decoded.get("descriptor"),
            "descriptor": decoded.get("descriptor"),
            "class_name": decoded["class_name"],
            "address": symbol.value,
            "source": "exported-symbol",
        })
    guessed = guess_class_name(elf)
    for table in scan_static_tables(elf, class_name=guessed):
        class_name = (table.get("class_name") or guessed or "UnassignedNatives").replace("/", ".")
        entries.append({
            "name": table["name"],
            "signature": table["signature"],
            "descriptor": table["signature"],
            "class_name": class_name,
            "address": table["address"],
            "source": "JNINativeMethod-table",
        })
    return entries


def _sources(args: argparse.Namespace) -> list[Path]:
    paths: list[Path] = []
    for value in args.path or []:
        path = Path(value)
        if path.is_dir():
            paths.extend(sorted(path.rglob("*.so")))
        elif path.exists():
            paths.append(path)
    seen: set[str] = set()
    unique: list[Path] = []
    for path in paths:
        key = str(path.resolve())
        if key in seen:
            continue
        seen.add(key)
        unique.append(path)
    return unique


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", action="append", help="Native .so file or directory (repeatable)")
    parser.add_argument("--out", default="artifacts/dataset/jni_facade_v1")
    parser.add_argument("--validation-mod", type=int, default=10, help="Hold out one binary hash class in N")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for path in _sources(args):
        try:
            elf = parse_elf(path)
        except Exception as exc:
            print(f"skip {path}: {exc}", flush=True)
            continue
        entries = _entries(elf)
        if not entries:
            print(f"{path.name}: no JNI entries", flush=True)
            continue
        profile = protection_profile(elf)
        protection = profile["label"]
        evidence = profile["evidence"]
        capabilities = profile["capabilities"]
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for entry in entries:
            grouped[entry["class_name"]].append(entry)
        for class_name, methods in sorted(grouped.items()):
            target = _target_class(class_name, methods, protection, evidence)
            tool_lines = [
                f"ELF: {path.name}",
                f"SHA256: {elf.sha256}",
                f"Architecture: {elf.header.machine_name}",
                f"JNI registration: {', '.join(sorted({method['source'] for method in methods}))}",
                f"Java class: {class_name}",
                f"Protection: {protection}",
            ]
            if evidence:
                tool_lines.append(f"Protection evidence: {', '.join(evidence[:8])}")
            tool_lines.append("Methods:")
            for method in sorted(methods, key=lambda item: (item["name"], item.get("signature") or "")):
                tool_lines.append(f"  {method['name']}{method.get('signature') or ''}")
            rows.append({
                "sample_id": f"{elf.sha256[:16]}::{class_name}",
                "language": "java",
                "tool": "jni_table",
                "tool_status": "completed",
                "compile_status": "unknown",
                "warning_count": 0,
                "cached": True,
                "tool_text": "\n".join(tool_lines)[:64000],
                "target_text": target[:64000],
                "failure_label": 0,
                "candidate_label": 1.0,
                "confidence_label": 0.6 if protection == "none" else 0.5,
                "source": "jni-facade",
                "project_id": elf.sha256,
                "protection": protection,
                "protection_evidence": evidence,
                "capabilities": capabilities,
                "method_count": len(methods),
            })
        print(f"{path.name}: {len(entries)} methods, {len(grouped)} classes, protection={protection}", flush=True)
    if not rows:
        raise SystemExit("no JNI facade rows")
    validation_mod = max(2, args.validation_mod)
    train: list[dict[str, Any]] = []
    validation: list[dict[str, Any]] = []
    for row in rows:
        bucket = int(hashlib.sha256(row["project_id"].encode("utf-8")).hexdigest(), 16) % validation_mod
        (validation if bucket == 0 else train).append(row)
    if not train:
        train, validation = rows, []
    for split, items in (("train", train), ("validation", validation)):
        path = out / f"{split}.jsonl"
        path.write_text("\n".join(json.dumps(row, sort_keys=True) for row in items) + ("\n" if items else ""), encoding="utf-8")
    summary = {
        "schema_version": "nexus.jni-facade/v1",
        "rows": len(rows),
        "splits": {"train": len(train), "validation": len(validation)},
        "methods": sum(row["method_count"] for row in rows),
        "classes": len(rows),
        "protections": dict(Counter(row["protection"] for row in rows)),
        "capabilities": dict(Counter(item for row in rows for item in row["capabilities"])),
        "sources": dict(Counter(row["tool"] for row in rows)),
        "root": str(out),
        "limitations": [
            "Targets are JNI facades, not original Java source.",
            "Stack-built RegisterNatives tables require Ghidra evidence and are not included by this lightweight builder.",
            "Runtime-decrypted payloads are not visible statically.",
        ],
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
