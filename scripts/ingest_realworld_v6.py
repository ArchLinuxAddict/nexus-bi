#!/usr/bin/env python3
"""Ingest real-world projects into a v6 pilot dataset."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nexus.datasets.realworld import BuildVariant, ingest_candidates  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", default="configs/realworld_candidates_new.json")
    parser.add_argument("--out", default="artifacts/dataset/realworld_v6_pilot")
    parser.add_argument("--max-projects", type=int, default=50)
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--variants", choices=["debug", "debug-stripped", "matrix"], default="debug-stripped")
    args = parser.parse_args()
    if args.variants == "debug":
        variants = [BuildVariant("gcc", "O0", True)]
    elif args.variants == "matrix":
        variants = [
            BuildVariant("gcc", "O0", True),
            BuildVariant("gcc", "O2", False),
            BuildVariant("clang", "O2", True),
        ]
    else:
        variants = [
            BuildVariant("gcc", "O0", True),
            BuildVariant("gcc", "O2", False),
        ]
    result = ingest_candidates(args.candidates, args.out, max_projects=args.max_projects, variants=variants, timeout=args.timeout)
    print(json.dumps({
        "status": result.get("status"),
        "path": result.get("path"),
        "summary": result.get("summary"),
    }, indent=2))
    return 0 if result.get("status") == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
