#!/usr/bin/env python3
"""Attach JADX/CFR tool evidence to real Java source/binary pairs."""

from __future__ import annotations

import argparse
import json
import shutil
import zipfile
from pathlib import Path

from nexus.toolchain.adapters import CfrAdapter, JadxAdapter


def _safe_extract(archive: Path, target: Path) -> list[str]:
    target.mkdir(parents=True, exist_ok=True)
    files = []
    with zipfile.ZipFile(archive) as handle:
        for info in handle.infolist():
            if info.is_dir():
                continue
            destination = (target / info.filename).resolve()
            try:
                destination.relative_to(target.resolve())
            except ValueError:
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            with handle.open(info) as source, destination.open("wb") as output:
                shutil.copyfileobj(source, output)
            if destination.suffix == ".java":
                files.append(str(destination.relative_to(target.resolve())))
    return sorted(files)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--roots", nargs="+", default=["artifacts/dataset/java_v6_pilot", "artifacts/dataset/java_v6_pilot2"])
    parser.add_argument("--out", default="artifacts/dataset/java_v6_tools")
    parser.add_argument("--max-pairs", type=int, default=0)
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for root_name in args.roots:
        root = Path(root_name)
        pairs_path = root / "pairs.jsonl"
        if not pairs_path.exists():
            continue
        for line in pairs_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            pair = json.loads(line)
            if args.max_pairs and len(rows) >= args.max_pairs:
                break
            key = f"{pair['group']}__{pair['artifact']}__{pair['version']}"
            key = key.replace(".", "_").replace("/", "_")
            binary = root / pair["binary_jar"]
            sources = root / pair["source_jar"]
            if not binary.exists() or not sources.exists():
                continue
            source_dir = out / "sources" / key
            source_files = _safe_extract(sources, source_dir)
            jadx = JadxAdapter().decompile(binary, out / "tools" / "jadx" / key)
            cfr = CfrAdapter().decompile(binary, out / "tools" / "cfr" / key)
            rows.append({
                "sample_id": key,
                "group": pair["group"],
                "artifact": pair["artifact"],
                "version": pair["version"],
                "license": pair["license"],
                "binary_sha256": pair["binary_sha256"],
                "source_sha256": pair["source_sha256"],
                "source_files": source_files,
                "source_count": len(source_files),
                "jadx": {"status": jadx.status, "java_files": jadx.artifacts.get("java_files", []), "evidence_id": jadx.to_dict()["evidence_id"], "cached": jadx.cached},
                "cfr": {"status": cfr.status, "java_files": cfr.artifacts.get("java_files", []), "evidence_id": cfr.to_dict()["evidence_id"], "cached": cfr.cached},
                "limitations": [
                    "Maven source JAR is ground-truth source for the same coordinates/version.",
                    "JADX/CFR output is reconstructed Java, not original source.",
                    "Dependencies and Android/JNI behavior are not verified.",
                ],
            })
            print(f"labelled {key} source={len(source_files)} jadx={jadx.status} cfr={cfr.status}", flush=True)
    (out / "labels.jsonl").write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + ("\n" if rows else ""), encoding="utf-8")
    summary = {
        "schema_version": "nexus.java-tool-labels/v1",
        "pairs": len(rows),
        "source_files": sum(row["source_count"] for row in rows),
        "jadx_completed": sum(1 for row in rows if row["jadx"]["status"] == "completed"),
        "cfr_completed": sum(1 for row in rows if row["cfr"]["status"] == "completed"),
        "root": str(out),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
