#!/usr/bin/env python3
"""Add corrupted-tool-output negatives to residual training data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _corrupt(text: str) -> str:
    lines = text.splitlines()
    if len(lines) > 4:
        keep = max(1, int(len(lines) * 0.6))
        lines = lines[:keep]
    text = "\n".join(lines)
    if len(text) > 80:
        text = text[: int(len(text) * 0.7)]
    return text + "\n/* NEXUS negative: truncated or corrupted tool output */\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="artifacts/dataset/residual_v6")
    parser.add_argument("--out", default="artifacts/dataset/residual_v6_aug")
    args = parser.parse_args()
    root = Path(args.root)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    summary = {"schema_version": "nexus.residual-augmented/v1", "splits": {}}
    for split in ("train", "validation", "test"):
        source = root / f"{split}.jsonl"
        if not source.exists():
            continue
        rows = []
        negatives = 0
        for line in source.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            rows.append(row)
            if split in {"train", "validation"}:
                negative = dict(row)
                negative["sample_id"] = row["sample_id"] + "::negative"
                negative["tool_text"] = _corrupt(row.get("tool_text") or "")
                negative["failure_label"] = 1
                negative["candidate_label"] = 0.0
                negative["confidence_label"] = 0.0
                negative["augmentation"] = "truncated_tool_output"
                rows.append(negative)
                negatives += 1
        (out / f"{split}.jsonl").write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + ("\n" if rows else ""), encoding="utf-8")
        summary["splits"][split] = {"rows": len(rows), "negatives": negatives}
    (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
