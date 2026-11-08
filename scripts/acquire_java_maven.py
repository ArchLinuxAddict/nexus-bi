#!/usr/bin/env python3
"""Acquire real-world Java source/binary pairs from Maven Central."""

from __future__ import annotations

import argparse
import hashlib
import json
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GROUPS = [
    "org.apache.commons",
    "org.apache.httpcomponents",
    "org.apache.httpcomponents.client5",
    "org.apache.logging.log4j",
    "com.google.guava",
    "com.fasterxml.jackson.core",
    "org.slf4j",
    "io.netty",
    "org.jsoup",
    "com.squareup.okhttp3",
    "org.eclipse.jetty",
    "org.apache.poi",
]
PERMISSIVE = {"Apache-2.0", "MIT", "BSD-2-Clause", "BSD-3-Clause", "ISC", "EPL-2.0", "MPL-2.0"}


def _get(url: str, timeout: int = 60) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "nexus-bi-java-acquisition"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _licenses(pom_text: str) -> list[str]:
    names = []
    try:
        root = ET.fromstring(pom_text)
    except ET.ParseError:
        return names
    for element in root.iter():
        if element.tag.endswith("license"):
            for child in element:
                if child.tag.endswith("name") and child.text:
                    names.append(child.text.strip())
    return names


def _search(group: str, rows: int) -> list[dict]:
    query = urllib.parse.urlencode({"q": f"g:{group}", "rows": rows, "wt": "json"})
    payload = json.loads(_get(f"https://search.maven.org/solrsearch/select?{query}"))
    return payload.get("response", {}).get("docs", [])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="artifacts/dataset/java_v6_pilot")
    parser.add_argument("--max-artifacts", type=int, default=50)
    parser.add_argument("--rows", type=int, default=50)
    parser.add_argument("--per-group", type=int, default=5)
    parser.add_argument("--groups", nargs="*", default=DEFAULT_GROUPS)
    args = parser.parse_args()
    out = Path(args.out)
    jars = out / "jars"
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    seen = set()
    for group in args.groups:
        try:
            docs = _search(group, args.rows)
        except Exception as exc:
            print(f"search failed {group}: {exc}", flush=True)
            continue
        group_count = 0
        for doc in docs:
            if len(rows) >= args.max_artifacts or group_count >= args.per_group:
                break
            key = f"{doc.get('g')}:{doc.get('a')}"
            if key in seen:
                continue
            seen.add(key)
            version = doc.get("latestVersion")
            if not version:
                continue
            base = f"https://repo1.maven.org/maven2/{doc['g'].replace('.', '/')}/{doc['a']}/{version}"
            try:
                pom = _get(f"{base}/{doc['a']}-{version}.pom").decode("utf-8", errors="replace")
            except Exception:
                continue
            licenses = _licenses(pom)
            if not any(any(license_id in name for license_id in PERMISSIVE) for name in licenses):
                continue
            target = jars / doc["g"] / doc["a"] / version
            target.mkdir(parents=True, exist_ok=True)
            binary = target / f"{doc['a']}-{version}.jar"
            sources = target / f"{doc['a']}-{version}-sources.jar"
            try:
                if not binary.exists():
                    binary.write_bytes(_get(f"{base}/{doc['a']}-{version}.jar"))
                if not sources.exists():
                    sources.write_bytes(_get(f"{base}/{doc['a']}-{version}-sources.jar"))
            except Exception as exc:
                print(f"download failed {key}:{version}: {exc}", flush=True)
                continue
            group_count += 1
            rows.append({
                "group": doc["g"],
                "artifact": doc["a"],
                "version": version,
                "license": licenses,
                "binary_jar": str(binary.relative_to(out)),
                "source_jar": str(sources.relative_to(out)),
                "binary_sha256": _sha(binary),
                "source_sha256": _sha(sources),
                "binary_bytes": binary.stat().st_size,
                "source_bytes": sources.stat().st_size,
            })
            print(f"acquired {key}:{version}", flush=True)
        if len(rows) >= args.max_artifacts:
            break
    (out / "pairs.jsonl").write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + ("\n" if rows else ""), encoding="utf-8")
    summary = {
        "schema_version": "nexus.java-source-binary/v1",
        "created_at": datetime.now(UTC).isoformat(),
        "root": str(out),
        "pairs": len(rows),
        "binary_bytes": sum(row["binary_bytes"] for row in rows),
        "source_bytes": sum(row["source_bytes"] for row in rows),
        "groups": dict(sorted({row["group"]: sum(1 for item in rows if item["group"] == row["group"]) for row in rows}.items())),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
