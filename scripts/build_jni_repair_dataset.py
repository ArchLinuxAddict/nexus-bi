#!/usr/bin/env python3
"""Build corrupted-to-correct JNI declaration repair pairs.

The residual model is a repair model: deterministic tools produce the facade,
and the model fixes wrong types, names and modifiers. This dataset teaches that
task on JNI native declarations, including protector libraries.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

METHOD_PATTERN = re.compile(r"^\s*public static native (.+?) ([A-Za-z_$][\w$]*)\((.*?)\);\s*$")
WRONG_RETURNS = ["Object", "int", "long", "boolean", "byte[]", "java.lang.String", "void"]
WRONG_PARAMS = ["Object", "int", "long", "boolean", "byte[]", "java.lang.String"]


def _read(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _declarations(record: dict) -> list[tuple[str, str, str, str]]:
    values = []
    for line in record.get("target_text", "").splitlines():
        match = METHOD_PATTERN.match(line)
        if not match:
            continue
        values.append((match.group(1), match.group(2), match.group(3), line.strip()))
    return values


def _corrupt(return_type: str, params: str, variant: int) -> str:
    parts = [part.strip() for part in params.split(",") if part.strip()]
    if variant % 3 == 0:
        wrong_return = next((value for value in WRONG_RETURNS if value != return_type), "Object")
        return f"public static native {wrong_return} METHOD({params});"
    if variant % 3 == 1:
        if parts:
            wrong = next((value for value in WRONG_PARAMS if value != parts[0].split()[0]), "Object")
            parts[0] = f"{wrong} arg0"
        return f"public static native {return_type} METHOD({', '.join(parts)});"
    modifier = "public final native" if variant % 2 else "private static native"
    return f"{modifier} {return_type} METHOD({params});"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--methods", default="artifacts/dataset/jni_method_holdout")
    parser.add_argument("--general", default="artifacts/dataset/residual_v6")
    parser.add_argument("--out", default="artifacts/dataset/jni_repair_v1")
    parser.add_argument("--variants", type=int, default=3)
    parser.add_argument("--general-train", type=int, default=500)
    parser.add_argument("--general-validation", type=int, default=120)
    args = parser.parse_args()
    methods = Path(args.methods)
    general = Path(args.general)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    train = []
    validation = []
    for split in ("train", "validation"):
        source_rows = _read(methods / f"{split}.jsonl")
        for record in source_rows:
            declarations = _declarations(record)
            for return_type, method, params, declaration in declarations:
                for variant in range(max(1, args.variants)):
                    corrupted = _corrupt(return_type, params, variant).replace("METHOD", method)
                    row = dict(record)
                    row["sample_id"] = f"{record['sample_id']}::{method}::repair{variant}"
                    row["source"] = "jni-repair"
                    row["tool_text"] = (
                        record.get("tool_text", "")
                        + f"\nCANDIDATE:\n{corrupted}\nTASK: repair the Java native declaration to match the JNI descriptor.\n"
                    )[:64000]
                    row["target_text"] = declaration
                    (train if split == "train" else validation).append(row)
    general_train = _read(general / "train.jsonl")[: args.general_train]
    general_validation = _read(general / "validation.jsonl")[: args.general_validation]
    train.extend(general_train)
    validation.extend(general_validation)
    for split, rows in (("train", train), ("validation", validation)):
        path = out / f"{split}.jsonl"
        path.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + ("\n" if rows else ""), encoding="utf-8")
    summary = {
        "schema_version": "nexus.jni-repair/v1",
        "train": len(train),
        "validation": len(validation),
        "variants": args.variants,
        "general_train": len(general_train),
        "general_validation": len(general_validation),
        "root": str(out),
        "limitations": [
            "Corruption is synthetic; real tool errors may differ.",
            "Targets are recovered facades, not original Java source.",
        ],
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
