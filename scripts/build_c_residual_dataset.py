#!/usr/bin/env python3
"""Build C residual samples from real source+binary builds and Ghidra output."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from nexus.foundation.config import config_for_vocabulary, profile_config
from nexus.foundation.representation import RepresentationOptions, encode_function_sample
from nexus.foundation.vocab import Vocabulary
from nexus.toolchain.adapters import GhidraAdapter


def _read(path: Path, limit: int = 65536) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")[:limit]
    except OSError:
        return ""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="artifacts/dataset/realworld_v6_pilot2")
    parser.add_argument("--out", default="artifacts/dataset/c_v6_residual")
    parser.add_argument("--max-projects", type=int, default=10)
    parser.add_argument("--max-functions", type=int, default=80)
    args = parser.parse_args()
    root = Path(args.root)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    manifests = [json.loads(line) for line in (root / "manifests.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    debug = [item for item in manifests if item.get("build", {}).get("debug_info")]
    projects = {}
    for item in debug:
        projects.setdefault(str(item.get("project_id")), []).append(item)
    vocab = Vocabulary()
    config = config_for_vocabulary(profile_config("cpu_dev"), vocab)
    rows = []
    for project_id, items in list(projects.items())[: args.max_projects]:
        item = items[0]
        record_path = next((root / artifact["path"] for artifact in item.get("artifacts", []) if artifact["path"].endswith("program_record.json")), None)
        binary_path = next((root / artifact["path"] for artifact in item.get("artifacts", []) if artifact.get("media_type") == "application/x-elf"), None)
        if record_path is None or not record_path.exists() or binary_path is None:
            continue
        record = json.loads(record_path.read_text(encoding="utf-8"))
        functions = {int(str(function.get("entry_va")), 16): function for function in record.get("functions", []) if function.get("entry_va")}
        result = GhidraAdapter().decompile(binary_path)
        if result.status != "completed":
            print(f"ghidra failed {project_id}: {result.error}", flush=True)
            continue
        count = 0
        for ghidra_function in result.artifacts.get("functions", []):
            if count >= args.max_functions:
                break
            entry = int(ghidra_function.get("entry") or -1)
            function = functions.get(entry)
            if function is None:
                continue
            target = function.get("source_label_text") or ""
            tool_text = ghidra_function.get("source") or ""
            if not target or not tool_text:
                continue
            program = encode_function_sample(
                record,
                function,
                vocab,
                config,
                identity={
                    "sample_id": f"{item['sample_id']}::{entry:x}",
                    "function_id": function.get("function_id"),
                    "project_id": project_id,
                    "dataset_id": "realworld-v6",
                    "split": item.get("split", "train"),
                    "binary_sha256": item.get("binary", {}).get("sha256"),
                    "source_family": project_id,
                    "match_group": project_id,
                    "binary_path": str(binary_path),
                    "language": "c",
                    "build": item.get("build", {}),
                },
                options=RepresentationOptions(include_source_target=False),
            )
            if program is None:
                continue
            rows.append({
                "sample_id": f"{item['sample_id']}::ghidra::{entry:x}",
                "language": "c",
                "tool": "ghidra_headless",
                "tool_status": result.status,
                "compile_status": "unknown",
                "warning_count": len(result.warnings),
                "cached": result.cached,
                "tool_text": tool_text[:args.max_functions * 8192],
                "target_text": target,
                "failure_label": 0,
                "candidate_label": 1.0,
                "confidence_label": 1.0,
                "source": "real-world-debug-build",
                "project_id": project_id,
                "architecture": item.get("build", {}).get("arch"),
                "program": program,
            })
            count += 1
        print(f"project {project_id}: {count} residual samples", flush=True)
    groups = sorted({row["project_id"] for row in rows})
    split_for = {group: ("validation" if index % 10 == 1 else "test" if index % 10 == 0 else "train") for index, group in enumerate(groups)}
    buckets = {"train": [], "validation": [], "test": []}
    for row in rows:
        buckets[split_for[row["project_id"]]].append(row)
    for split, items in buckets.items():
        (out / f"{split}.jsonl").write_text("\n".join(json.dumps(row, sort_keys=True) for row in items) + ("\n" if items else ""), encoding="utf-8")
    summary = {
        "schema_version": "nexus.c-residual/v1",
        "samples": len(rows),
        "splits": {key: len(value) for key, value in buckets.items()},
        "projects": len(groups),
        "tools": dict(Counter(row["tool"] for row in rows)),
        "total_chars": sum(len(row["target_text"]) + len(row["tool_text"]) for row in rows),
        "root": str(out),
        "limitations": [
            "Ghidra output is a decompiler candidate, not original source.",
            "Only functions with DWARF source bodies are included.",
            "Compilation is not behavioral equivalence.",
        ],
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
