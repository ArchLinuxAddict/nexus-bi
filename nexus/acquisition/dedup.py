import hashlib
import json
import re
import tarfile
from pathlib import Path
from typing import Any

from nexus.core.hashing import sha256_file

SOURCE_SUFFIXES = (".c", ".h", ".cc", ".cpp", ".cxx", ".hpp")
EXCLUDED_PARTS = {"test", "tests", "testing", "examples", "example", "third_party", "third-party", "vendor", "externals", "crates", "contrib", "platforms", "docs", "doc"}


def _tokens(payload: bytes) -> list[str]:
    text = payload.decode("utf-8", errors="replace").lower()
    return re.findall(r"[a-z_][a-z0-9_]*|\d+", text)


def _iter_source_members(archive: Path):
    with tarfile.open(archive, "r:*") as handle:
        for member in handle:
            if not member.isfile() or not member.name.lower().endswith(SOURCE_SUFFIXES):
                continue
            if member.size > 2 * 1024 * 1024:
                continue
            parts = Path(member.name).parts
            if any(part.lower() in EXCLUDED_PARTS for part in parts[:-1]):
                continue
            source = handle.extractfile(member)
            if source is None:
                continue
            yield member.name, source.read()


def source_file_hashes(archive: str | Path) -> dict[str, str]:
    result = {}
    for name, payload in _iter_source_members(Path(archive)):
        result[name] = hashlib.sha256(payload).hexdigest()
    return result


def project_signature(archive: str | Path, dimensions: int = 64, max_tokens: int = 200_000) -> list[int]:
    tokens: list[str] = []
    for _, payload in _iter_source_members(Path(archive)):
        tokens.extend(_tokens(payload))
        if len(tokens) >= max_tokens:
            tokens = tokens[:max_tokens]
            break
    if len(tokens) < 5:
        return [0] * dimensions
    shingles = {" ".join(tokens[index : index + 5]) for index in range(0, len(tokens) - 4, 2)}
    return [
        min(int(hashlib.blake2b(shingle.encode("utf-8"), digest_size=8, salt=dimension.to_bytes(8, "little")).hexdigest(), 16) for shingle in shingles)
        for dimension in range(dimensions)
    ]


def signature_from_tree(root: str | Path, dimensions: int = 64, max_tokens: int = 200_000) -> list[int]:
    tokens: list[str] = []
    for path in sorted(Path(root).rglob("*")):
        if not path.is_file() or path.suffix.lower() not in SOURCE_SUFFIXES:
            continue
        if path.stat().st_size > 2 * 1024 * 1024:
            continue
        relative = path.relative_to(root)
        if any(part.lower() in EXCLUDED_PARTS for part in relative.parts[:-1]):
            continue
        tokens.extend(_tokens(path.read_bytes()))
        if len(tokens) >= max_tokens:
            tokens = tokens[:max_tokens]
            break
    if len(tokens) < 5:
        return [0] * dimensions
    shingles = {" ".join(tokens[index : index + 5]) for index in range(0, len(tokens) - 4, 2)}
    return [
        min(int(hashlib.blake2b(shingle.encode("utf-8"), digest_size=8, salt=dimension.to_bytes(8, "little")).hexdigest(), 16) for shingle in shingles)
        for dimension in range(dimensions)
    ]


def file_hashes_from_tree(root: str | Path) -> dict[str, str]:
    result = {}
    for path in sorted(Path(root).rglob("*")):
        if not path.is_file() or path.suffix.lower() not in SOURCE_SUFFIXES:
            continue
        if path.stat().st_size > 2 * 1024 * 1024:
            continue
        relative = path.relative_to(root)
        if any(part.lower() in EXCLUDED_PARTS for part in relative.parts[:-1]):
            continue
        result[str(relative).replace("\\", "/")] = sha256_file(path)
    return result


def jaccard(first: list[int], second: list[int]) -> float:
    if not first or not second:
        return 0.0
    return sum(1 for a, b in zip(first, second) if a == b) / len(first)


def file_jaccard(first: dict[str, str], second: dict[str, str]) -> float:
    a = set(first.values())
    b = set(second.values())
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def build_historical_index(dataset_root: str | Path, cache_path: str | Path | None = None, dimensions: int = 64) -> dict[str, Any]:
    root = Path(dataset_root)
    catalog = json.loads((root / "catalog.json").read_text(encoding="utf-8"))
    projects = []
    for project in catalog:
        archive = root / project["source_archive_path"]
        if not archive.exists():
            continue
        projects.append(
            {
                "project_id": project["project_id"],
                "name": project["name"],
                "category": project.get("category"),
                "archive_sha256": sha256_file(archive),
                "signature": project_signature(archive, dimensions=dimensions),
                "file_hashes": source_file_hashes(archive),
            }
        )
    index = {
        "schema_version": "0.1.0",
        "dataset_root": str(root.resolve()),
        "dimensions": dimensions,
        "projects": projects,
    }
    if cache_path:
        target = Path(cache_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(index, sort_keys=True) + "\n", encoding="utf-8")
    return index


def load_or_build_index(dataset_root: str | Path, cache_path: str | Path) -> dict[str, Any]:
    target = Path(cache_path)
    if target.exists():
        try:
            return json.loads(target.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            pass
    return build_historical_index(dataset_root, target)


def compare_against_index(
    *,
    archive_sha256: str,
    signature: list[int],
    file_hashes: dict[str, str],
    index: dict[str, Any],
    near_threshold: float = 0.80,
    watch_threshold: float = 0.60,
) -> dict[str, Any]:
    exact_archive = []
    matches = []
    for project in index.get("projects", []):
        if project.get("archive_sha256") == archive_sha256:
            exact_archive.append(project["project_id"])
        similarity = jaccard(signature, project.get("signature") or [])
        files_similarity = file_jaccard(file_hashes, project.get("file_hashes") or {})
        shared_files = sorted(set(file_hashes.values()) & set((project.get("file_hashes") or {}).values()))
        if similarity >= watch_threshold or files_similarity >= watch_threshold or shared_files:
            matches.append(
                {
                    "project_id": project["project_id"],
                    "name": project.get("name"),
                    "minhash_jaccard": round(similarity, 4),
                    "file_jaccard": round(files_similarity, 4),
                    "shared_file_hashes": len(shared_files),
                }
            )
    matches.sort(key=lambda item: max(item["minhash_jaccard"], item["file_jaccard"]), reverse=True)
    highest = max((max(item["minhash_jaccard"], item["file_jaccard"]) for item in matches), default=0.0)
    if exact_archive:
        status = "duplicate"
        reason = "exact archive hash matches a historical project"
    elif highest >= near_threshold:
        status = "near_duplicate"
        reason = f"similarity {highest:.3f} >= {near_threshold:.2f}"
    elif highest >= watch_threshold:
        status = "watch"
        reason = f"similarity {highest:.3f} in review band"
    else:
        status = "unique"
        reason = "no historical duplicate above watch threshold"
    return {
        "status": status,
        "reason": reason,
        "max_similarity": round(highest, 4),
        "exact_archive_matches": exact_archive,
        "matches": matches[:10],
    }
