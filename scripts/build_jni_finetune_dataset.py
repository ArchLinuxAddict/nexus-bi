#!/usr/bin/env python3
"""Compose a JNI/protector-focused fine-tuning corpus.

Combines the JNI facade and JNI semantic pairs with a sample of the general
C/Java residual corpus. Supports holding out one binary (for example the
protector) so its facade is measured on unseen evidence.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path


def _read(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _sample_mixed(rows: list[dict], count: int, seed: int) -> list[dict]:
    if not rows or count <= 0:
        return []
    random.Random(seed).shuffle(rows)
    c_rows = [row for row in rows if row.get("language") == "c"]
    java_rows = [row for row in rows if row.get("language") != "c"]
    selected: list[dict] = []
    half = max(1, count // 2)
    selected.extend(c_rows[:half])
    selected.extend(java_rows[: max(0, count - len(selected))])
    if len(selected) < count:
        remaining = [row for row in rows if row not in selected]
        selected.extend(remaining[: count - len(selected)])
    return selected[:count]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--facade", default="artifacts/dataset/jni_facade_v1")
    parser.add_argument("--semantic", default="artifacts/dataset/jni_semantic_v1")
    parser.add_argument("--general", default="artifacts/dataset/residual_v6")
    parser.add_argument("--out", default="artifacts/dataset/jni_finetune_v1")
    parser.add_argument("--repeat", type=int, default=8)
    parser.add_argument("--general-train", type=int, default=600)
    parser.add_argument("--general-validation", type=int, default=120)
    parser.add_argument("--holdout-sha", default=None, help="Binary SHA prefix to hold out of JNI training")
    parser.add_argument("--seed", type=int, default=20261107)
    args = parser.parse_args()
    facade = Path(args.facade)
    semantic = Path(args.semantic)
    general = Path(args.general)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    jni_train = _read(facade / "train.jsonl") + _read(semantic / "train.jsonl")
    jni_validation = _read(facade / "validation.jsonl") + _read(semantic / "validation.jsonl")
    if args.holdout_sha:
        held = [row for row in jni_train if str(row.get("project_id", "")).startswith(args.holdout_sha)]
        jni_train = [row for row in jni_train if not str(row.get("project_id", "")).startswith(args.holdout_sha)]
        jni_validation.extend(held)
    repeated: list[dict] = []
    for repeat in range(max(1, args.repeat)):
        for row in jni_train:
            clone = dict(row)
            clone["sample_id"] = f"{row['sample_id']}::r{repeat}"
            repeated.append(clone)
    general_train = _sample_mixed(_read(general / "train.jsonl"), args.general_train, args.seed)
    general_validation = _sample_mixed(_read(general / "validation.jsonl"), args.general_validation, args.seed + 1)
    train = repeated + general_train
    validation = jni_validation + general_validation
    for split, rows in (("train", train), ("validation", validation)):
        path = out / f"{split}.jsonl"
        path.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + ("\n" if rows else ""), encoding="utf-8")
    summary = {
        "schema_version": "nexus.jni-finetune/v1",
        "train": len(train),
        "validation": len(validation),
        "jni_train_unique": len(jni_train),
        "jni_validation_unique": len(jni_validation),
        "repeat": args.repeat,
        "general_train": len(general_train),
        "general_validation": len(general_validation),
        "holdout_sha": args.holdout_sha,
        "languages": dict(Counter(row.get("language", "unknown") for row in train)),
        "tools": dict(Counter(row.get("tool", "none") for row in train)),
        "root": str(out),
        "limitations": [
            "JNI targets are facades/semantic reconstructions, not original Java source.",
            "Runtime-decrypted payloads are absent from static-only data.",
        ],
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
