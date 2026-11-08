#!/usr/bin/env python3
"""Generate a large permissive real-world candidate list from GitHub search."""

from __future__ import annotations

import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "configs" / "realworld_candidates_500.json"
QUERIES = [
    "language:C license:mit stars:>30",
    "language:C license:bsd-3-clause stars:>20",
    "language:C license:apache-2.0 stars:>20",
    "language:C++ license:mit stars:>30",
    "language:C++ license:apache-2.0 stars:>20",
]


def search(query: str, page: int) -> list[dict]:
    params = urllib.parse.urlencode({"q": query, "sort": "stars", "order": "desc", "per_page": 100, "page": page})
    request = urllib.request.Request(f"https://api.github.com/search/repositories?{params}", headers={"Accept": "application/vnd.github+json", "User-Agent": "nexus-bi-candidates"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.load(response).get("items", [])


def main() -> int:
    candidates: dict[str, dict] = {}
    for query in QUERIES:
        for page in (1, 2):
            try:
                items = search(query, page)
            except Exception as exc:
                print(f"query failed: {query} page {page}: {exc}", flush=True)
                time.sleep(10)
                continue
            for item in items:
                license_info = item.get("license") or {}
                spdx = license_info.get("spdx_id")
                if spdx not in {"MIT", "BSD-2-Clause", "BSD-3-Clause", "Apache-2.0", "ISC", "Unlicense", "CC0-1.0", "Zlib", "0BSD", "WTFPL"}:
                    continue
                if item.get("size", 0) > 30000:
                    continue
                full_name = item.get("full_name")
                if not full_name or full_name in candidates:
                    continue
                branch = item.get("default_branch") or "main"
                candidates[full_name] = {
                    "full_name": full_name,
                    "description": item.get("description") or "",
                    "stars": item.get("stargazers_count", 0),
                    "size_kb": item.get("size", 0),
                    "default_branch": branch,
                    "license": spdx,
                    "topics": item.get("topics") or [],
                    "html_url": item.get("html_url"),
                    "pushed_at": item.get("pushed_at"),
                    "commit": f"refs/heads/{branch}",
                    "language": item.get("language"),
                }
            print(f"{query} page {page}: total={len(candidates)}", flush=True)
            time.sleep(7)
    rows = sorted(candidates.values(), key=lambda item: item["stars"], reverse=True)
    OUT.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(OUT), "candidates": len(rows), "licenses": sorted({item["license"] for item in rows})}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
