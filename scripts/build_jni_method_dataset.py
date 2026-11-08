#!/usr/bin/env python3
"""Build method-level JNI training pairs and protection classification rows.

Whole-class generation is too long for the small residual decoder. This
builder emits one Java native declaration per sample and a short protection
classification target, which is the granularity needed for protector-like
libraries.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

METHOD_PATTERN = re.compile(r"^\s*public static native (.+?) ([A-Za-z_$][\w$]*)\((.*?)\);\s*$")


def _read(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _method_rows(record: dict) -> list[dict]:
    rows = []
    for line in record.get("target_text", "").splitlines():
        match = METHOD_PATTERN.match(line)
        if not match:
            continue
        declaration = line.strip()
        method_name = match.group(2)
        rows.append({
            "sample_id": f"{record['sample_id']}::{method_name}",
            "language": "java",
            "tool": "jni_table",
            "tool_status": "completed",
            "compile_status": "unknown",
            "warning_count": 0,
            "cached": True,
            "tool_text": (
                record.get("tool_text", "")
                + f"\nTASK: generate the exact native declaration for {method_name}.\n"
            )[:64000],
            "target_text": declaration,
            "failure_label": 0,
            "candidate_label": 1.0,
            "confidence_label": record.get("confidence_label", 0.6),
            "source": "jni-method",
            "project_id": record.get("project_id"),
            "protection": record.get("protection"),
            "method": method_name,
        })
    return rows


def _protection_rows(record: dict) -> list[dict]:
    label = record.get("protection") or "none"
    evidence = record.get("protection_evidence") or []
    if label == "none" and not record.get("capabilities"):
        return []
    lines = [f"// protection: {label}"]
    if evidence:
        lines.append(f"// evidence: {', '.join(evidence[:6])}")
    if record.get("capabilities"):
        lines.append(f"// capabilities: {', '.join(record['capabilities'])}")
    return [{
        "sample_id": f"{record['sample_id']}::protection",
        "language": "java",
        "tool": "jni_table",
        "tool_status": "completed",
        "compile_status": "unknown",
        "warning_count": 0,
        "cached": True,
        "tool_text": record.get("tool_text", "") + "\nTASK: classify the protection and cite evidence.\n",
        "target_text": "\n".join(lines),
        "failure_label": 0,
        "candidate_label": 1.0,
        "confidence_label": 0.6,
        "source": "jni-protection",
        "project_id": record.get("project_id"),
        "protection": label,
    }]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--facade", default="artifacts/dataset/jni_facade_v1")
    parser.add_argument("--semantic", default="artifacts/dataset/jni_semantic_v1")
    parser.add_argument("--general", default="artifacts/dataset/residual_v6")
    parser.add_argument("--out", default="artifacts/dataset/jni_method_v1")
    parser.add_argument("--repeat", type=int, default=10)
    parser.add_argument("--semantic-repeat", type=int, default=2)
    parser.add_argument("--general-train", type=int, default=500)
    parser.add_argument("--general-validation", type=int, default=120)
    parser.add_argument("--holdout-sha", default=None)
    args = parser.parse_args()
    facade = Path(args.facade)
    semantic = Path(args.semantic)
    general = Path(args.general)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    facade_train = _read(facade / "train.jsonl")
    facade_validation = _read(facade / "validation.jsonl")
    if args.holdout_sha:
        held = [row for row in facade_train if str(row.get("project_id", "")).startswith(args.holdout_sha)]
        facade_train = [row for row in facade_train if not str(row.get("project_id", "")).startswith(args.holdout_sha)]
        facade_validation.extend(held)
    method_train = [row for record in facade_train for row in _method_rows(record)]
    method_validation = [row for record in facade_validation for row in _method_rows(record)]
    protection_train = [row for record in facade_train for row in _protection_rows(record)]
    protection_validation = [row for record in facade_validation for row in _protection_rows(record)]
    semantic_train = _read(semantic / "train.jsonl")
    semantic_validation = _read(semantic / "validation.jsonl")
    train = []
    validation = []
    for repeat in range(max(1, args.repeat)):
        for row in method_train + protection_train:
            clone = dict(row)
            clone["sample_id"] = f"{row['sample_id']}::r{repeat}"
            train.append(clone)
    for repeat in range(max(1, args.semantic_repeat)):
        for row in semantic_train:
            clone = dict(row)
            clone["sample_id"] = f"{row['sample_id']}::r{repeat}"
            train.append(clone)
    validation.extend(method_validation + protection_validation + semantic_validation)
    general_train = _read(general / "train.jsonl")
    general_validation = _read(general / "validation.jsonl")
    train.extend(general_train[: args.general_train])
    validation.extend(general_validation[: args.general_validation])
    for split, rows in (("train", train), ("validation", validation)):
        path = out / f"{split}.jsonl"
        path.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + ("\n" if rows else ""), encoding="utf-8")
    summary = {
        "schema_version": "nexus.jni-method/v1",
        "train": len(train),
        "validation": len(validation),
        "method_train": len(method_train),
        "method_validation": len(method_validation),
        "protection_train": len(protection_train),
        "protection_validation": len(protection_validation),
        "semantic_train": len(semantic_train),
        "repeat": args.repeat,
        "semantic_repeat": args.semantic_repeat,
        "general_train": min(args.general_train, len(general_train)),
        "general_validation": min(args.general_validation, len(general_validation)),
        "holdout_sha": args.holdout_sha,
        "root": str(out),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
