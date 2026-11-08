#!/usr/bin/env python3
"""Build positive/negative pairs for residual candidate and failure heads."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

METHOD_PATTERN = re.compile(r"^\s*(?:public|private|protected)?\s*(?:static\s+)?native\s+(.+?)\s+([A-Za-z_$][\w$]*)\s*\((.*?)\);\s*$")
WRONG_TYPES = ["Object", "int", "long", "boolean", "byte[]", "java.lang.String", "void"]


def _read(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _declaration(record: dict) -> str | None:
    for line in record.get("target_text", "").splitlines():
        if "native" in line and "(" in line and line.strip().endswith(";"):
            return line.strip()
    return None


def _corrupt(declaration: str) -> str:
    match = METHOD_PATTERN.match(declaration)
    if not match:
        return declaration.replace("byte[]", "Object")
    return_type, method, params = match.groups()
    wrong = next((value for value in WRONG_TYPES if value != return_type), "Object")
    return f"public static native {wrong} {method}({params});"


def _with_candidate(tool_text: str, candidate: str) -> str:
    if "CANDIDATE:" in tool_text:
        head, _, tail = tool_text.partition("CANDIDATE:")
        rest = tail.split("TASK:", 1)
        suffix = ("TASK:" + rest[1]) if len(rest) == 2 else ""
        return head + "CANDIDATE:\n" + candidate + "\n" + suffix
    return tool_text.rstrip() + "\nCANDIDATE:\n" + candidate + "\nTASK: verify the Java native declaration.\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repair", default="artifacts/dataset/jni_repair_v1")
    parser.add_argument("--methods", default="artifacts/dataset/jni_method_all")
    parser.add_argument("--out", default="artifacts/dataset/jni_ranking_v1")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    train: list[dict] = []
    validation: list[dict] = []
    for split in ("train", "validation"):
        source = _read(Path(args.repair) / f"{split}.jsonl") + _read(Path(args.methods) / f"{split}.jsonl")
        bucket = train if split == "train" else validation
        for record in source:
            declaration = _declaration(record)
            if declaration is None:
                continue
            corrupted = _corrupt(declaration)
            base = dict(record)
            base["target_text"] = declaration
            base["source"] = "jni-ranking"
            positive = dict(base)
            positive["sample_id"] = f"{record['sample_id']}::positive"
            positive["tool_text"] = _with_candidate(record.get("tool_text", ""), declaration)
            positive["candidate_text"] = declaration
            positive["candidate_label"] = 1.0
            positive["failure_label"] = 0
            positive["confidence_label"] = 1.0
            negative = dict(base)
            negative["sample_id"] = f"{record['sample_id']}::negative"
            negative["tool_text"] = _with_candidate(record.get("tool_text", ""), corrupted)
            negative["candidate_text"] = corrupted
            negative["candidate_label"] = 0.0
            negative["failure_label"] = 1
            negative["confidence_label"] = 0.0
            bucket.extend([positive, negative])
    for split, rows in (("train", train), ("validation", validation)):
        path = out / f"{split}.jsonl"
        path.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + ("\n" if rows else ""), encoding="utf-8")
    summary = {
        "schema_version": "nexus.jni-ranking/v1",
        "train": len(train),
        "validation": len(validation),
        "root": str(out),
        "positive": sum(1 for row in train if row["candidate_label"] > 0.5),
        "negative": sum(1 for row in train if row["candidate_label"] <= 0.5),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
