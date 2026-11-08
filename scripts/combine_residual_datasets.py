#!/usr/bin/env python3
"""Combine C and Java residual datasets into one training corpus."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--c", default="artifacts/dataset/c_v6_residual")
    parser.add_argument("--java", default="artifacts/dataset/java_v6_residual")
    parser.add_argument("--jni", default="artifacts/dataset/jni_semantic_v1")
    parser.add_argument("--out", default="artifacts/dataset/residual_v6")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    summary = {"schema_version": "nexus.residual-combined/v1", "sources": {}, "splits": {}}
    for split in ("train", "validation", "test"):
        rows = []
        for name, root_name in (("c", args.c), ("java", args.java), ("jni", args.jni)):
            path = Path(root_name) / f"{split}.jsonl"
            if not path.exists():
                continue
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    row = json.loads(line)
                    row["dataset"] = name
                    rows.append(row)
        (out / f"{split}.jsonl").write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + ("\n" if rows else ""), encoding="utf-8")
        summary["splits"][split] = len(rows)
    all_rows = []
    for split in ("train", "validation", "test"):
        all_rows.extend(json.loads(line) for line in (out / f"{split}.jsonl").read_text(encoding="utf-8").splitlines() if line.strip())
    summary["samples"] = len(all_rows)
    summary["datasets"] = dict(Counter(row["dataset"] for row in all_rows))
    summary["languages"] = dict(Counter(row["language"] for row in all_rows))
    summary["tools"] = dict(Counter(row["tool"] for row in all_rows))
    (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
