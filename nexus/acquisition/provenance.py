import os
from pathlib import Path
from typing import Any

from nexus.core.hashing import canonical_json, sha256_bytes, sha256_file

SOURCE_SUFFIXES = {".c", ".h", ".cc", ".cpp", ".cxx", ".hpp", ".s", ".S", ".asm", ".inc", ".def", ".map", ".ld", ".mk", ".cmake", ".m4", ".am", ".ac", ".in"}
BUILD_FILES = {"makefile", "gnumakefile", "cmakelists.txt", "meson.build", "configure", "configure.ac", "config.h.in", "build", "build.ninja"}
TEXT_SUFFIXES = SOURCE_SUFFIXES | {".md", ".txt", ".rst", ".json", ".yaml", ".yml", ".toml", ".py", ".sh", ".pl"}


def _classify(path: Path) -> str:
    name = path.name.lower()
    if name in BUILD_FILES:
        return "build"
    if path.suffix.lower() in {".c", ".cc", ".cpp", ".cxx"}:
        return "c_source"
    if path.suffix.lower() in {".h", ".hpp", ".inc"}:
        return "header"
    if path.suffix.lower() in {".s", ".asm"}:
        return "assembly_source"
    if name in {"license", "license.txt", "licence", "licence.md", "copying", "unlicense", "unlicense.txt"} or name.startswith("license."):
        return "license"
    if path.suffix.lower() in {".md", ".txt", ".rst"}:
        return "documentation"
    return "other"


def hash_tree(root: str | Path, max_files: int = 20000, max_bytes: int = 1_000_000_000, include_file_hashes: bool = True) -> dict[str, Any]:
    base = Path(root)
    files: list[dict[str, Any]] = []
    total = 0
    skipped_links = 0
    skipped_special = 0
    classification: dict[str, int] = {}
    for path in sorted(base.rglob("*")):
        if path.is_symlink():
            skipped_links += 1
            continue
        if not path.is_file():
            if not path.is_dir():
                skipped_special += 1
            continue
        if len(files) >= max_files:
            raise ValueError(f"file inventory exceeds {max_files} files")
        size = path.stat().st_size
        if total + size > max_bytes:
            raise ValueError(f"source tree exceeds {max_bytes} bytes")
        relative = str(path.relative_to(base)).replace(os.sep, "/")
        kind = _classify(path)
        classification[kind] = classification.get(kind, 0) + 1
        item = {"path": relative, "size": size, "kind": kind}
        if include_file_hashes:
            item["sha256"] = sha256_file(path)
        files.append(item)
        total += size
    tree_payload = canonical_json([{"path": item["path"], "size": item["size"], "sha256": item.get("sha256")} for item in files])
    return {
        "file_count": len(files),
        "total_bytes": total,
        "tree_sha256": sha256_bytes(tree_payload.encode("utf-8")),
        "files": files,
        "skipped_links": skipped_links,
        "skipped_special": skipped_special,
        "classification": dict(sorted(classification.items())),
    }


def provenance_confidence(
    *,
    archive_sha256: str | None,
    commit: str | None,
    commit_verified: bool,
    license_sha256: str | None,
    inventory_sha256: str | None,
    provider: str,
    swh_status: str = "not_checked",
    warnings: list[str] | None = None,
) -> tuple[str, list[str]]:
    reasons = []
    score = 0
    if archive_sha256:
        score += 2
        reasons.append("source archive SHA-256 recorded")
    if commit and commit_verified:
        score += 2
        reasons.append("commit resolved and verified against provider ref")
    elif commit:
        score += 1
        reasons.append("commit recorded but not independently verified")
    if license_sha256:
        score += 1
        reasons.append("license-file SHA-256 recorded")
    if inventory_sha256:
        score += 1
        reasons.append("file inventory SHA-256 recorded")
    if provider in {"github", "gitlab", "debian", "softwareheritage"}:
        score += 1
        reasons.append(f"recognized provider: {provider}")
    if swh_status == "corroborated":
        score += 1
        reasons.append("Software Heritage revision corroborated")
    if warnings:
        reasons.extend(f"warning: {item}" for item in warnings)
    if score >= 7:
        confidence = "high"
    elif score >= 5:
        confidence = "medium"
    elif score >= 3:
        confidence = "low"
    else:
        confidence = "unknown"
    return confidence, reasons


def verify_swh(origin_url: str, commit: str, client) -> tuple[str, str | None]:
    import urllib.parse

    encoded = urllib.parse.quote(origin_url, safe="")
    try:
        revision = client.get_json(f"https://archive.softwareheritage.org/api/1/revision/{commit}/")
    except Exception:
        revision = None
    if isinstance(revision, dict) and revision.get("id"):
        return "corroborated", str(revision["id"])
    try:
        client.get_json(f"https://archive.softwareheritage.org/api/1/origin/{encoded}/visits/?limit=3")
    except Exception:
        return "unavailable", None
    return "not_found", None
