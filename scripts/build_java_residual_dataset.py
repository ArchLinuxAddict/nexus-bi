#!/usr/bin/env python3
"""Build Java tool-output residual samples from Maven source/binary pairs."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _strip_sources_prefix(path: str) -> str:
    marker = "sources/"
    return path.split(marker, 1)[1] if marker in path else path


def _read(path: Path, limit: int = 65536) -> str:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    return text[:limit]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tools-root", default="artifacts/dataset/java_v6_tools")
    parser.add_argument("--out", default="artifacts/dataset/java_v6_residual")
    parser.add_argument("--max-chars", type=int, default=65536)
    args = parser.parse_args()
    tools_root = Path(args.tools_root)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    labels = [json.loads(line) for line in (tools_root / "labels.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    rows = []
    seen = set()
    for record in labels:
        source_root = tools_root / "sources" / record["sample_id"]
        source_map = {str(path.relative_to(source_root)): path for path in source_root.rglob("*.java")}
        for tool_name in ("jadx", "cfr"):
            tool_base = tools_root / "tools" / tool_name / record["sample_id"]
            tool_root = tool_base / "sources" if (tool_base / "sources").exists() else tool_base
            if not tool_root.exists():
                continue
            for tool_path in sorted(tool_root.rglob("*.java")):
                relative = str(tool_path.relative_to(tool_root))
                target_path = source_map.get(relative)
                if target_path is None:
                    continue
                target = _read(target_path, args.max_chars)
                tool_text = _read(tool_path, args.max_chars)
                if not target or not tool_text:
                    continue
                sample_id = f"{record['sample_id']}::{tool_name}::{relative}"
                if sample_id in seen:
                    continue
                seen.add(sample_id)
                rows.append({
                    "sample_id": sample_id,
                    "language": "java",
                    "tool": tool_name,
                    "tool_status": record[tool_name]["status"],
                    "compile_status": "unknown",
                    "warning_count": 0,
                    "cached": bool(record[tool_name].get("cached")),
                    "tool_text": tool_text,
                    "target_text": target,
                    "failure_label": 0,
                    "candidate_label": 1.0,
                    "confidence_label": 1.0,
                    "source": "maven-source-jar",
                    "artifact": record["artifact"],
                    "version": record["version"],
                    "license": record["license"],
                })
    groups = sorted({f"{row['artifact']}::{row['version']}" for row in rows})
    split_for = {group: ("validation" if index % 10 == 1 else "test" if index % 10 == 0 else "train") for index, group in enumerate(groups)}
    buckets = {"train": [], "validation": [], "test": []}
    for row in rows:
        buckets[split_for[f"{row['artifact']}::{row['version']}"]].append(row)
    for split, items in buckets.items():
        path = out / f"{split}.jsonl"
        path.write_text("\n".join(json.dumps(row, sort_keys=True) for row in items) + ("\n" if items else ""), encoding="utf-8")
    summary = {
        "schema_version": "nexus.java-residual/v1",
        "samples": len(rows),
        "splits": {key: len(value) for key, value in buckets.items()},
        "tools": dict(Counter(row["tool"] for row in rows)),
        "artifacts": len(groups),
        "total_chars": sum(len(row["target_text"]) + len(row["tool_text"]) for row in rows),
        "root": str(out),
        "limitations": [
            "Pairs are Maven source JAR vs decompiler output, not compiled original source.",
            "Only class files with matching source paths are included.",
            "No Java behavior or test equivalence is claimed.",
        ],
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
