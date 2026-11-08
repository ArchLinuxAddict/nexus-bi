#!/usr/bin/env python3
"""Audit the real-world v6 pilot dataset."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="artifacts/dataset/realworld_v6_pilot")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    root = Path(args.root)
    catalog_path = root / "catalog.json"
    manifests_path = root / "manifests.jsonl"
    summary_path = root / "dataset_summary.json"
    if not catalog_path.exists():
        raise SystemExit(f"catalog missing: {catalog_path}")
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    manifests = [json.loads(line) for line in manifests_path.read_text(encoding="utf-8").splitlines() if line.strip()] if manifests_path.exists() else []
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    checkes = {
        "catalog_nonempty": bool(catalog),
        "manifests_nonempty": bool(manifests),
        "unique_project_ids": len({item.get("project_id") for item in catalog}) == len(catalog),
        "unique_project_names": len({item.get("name") for item in catalog}) == len(catalog),
        "split_consistency": all(
            len({item.get("split") for item in catalog if item.get("project_id") == project}) == 1
            for project in {item.get("project_id") for item in catalog}
        ),
        "license_present": all(item.get("license") for item in catalog),
        "source_archives_present": all((root / str(item.get("source_archive_path") or "")).exists() for item in catalog) if catalog else False,
    }
    report = {
        "status": "PASS" if all(checkes.values()) else "FAIL",
        "checks": checkes,
        "root": str(root),
        "projects": len(catalog),
        "manifests": len(manifests),
        "splits": dict(Counter(item.get("split") for item in catalog)),
        "licenses": dict(Counter(item.get("license") for item in catalog)),
        "source_bytes": sum(int(item.get("source_bytes") or 0) for item in catalog),
        "built_variants": sum(int(item.get("built_variants") or 0) for item in catalog),
        "failed_variants": sum(int(item.get("failed_variants") or 0) for item in catalog),
        "summary": summary,
        "limitations": [
            "Pilot audit checks schema/project/split integrity, not source-level function alignment.",
            "A successful build is not proof that the project's original build system was reproduced.",
            "Ghidra/JADX pseudo-labels are not ground-truth source.",
        ],
    }
    target = root / "v6_pilot_audit.json"
    target.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("status", "projects", "manifests", "splits", "licenses", "built_variants", "failed_variants", "root")}, indent=2))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
