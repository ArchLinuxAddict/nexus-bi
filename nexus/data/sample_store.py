"""Sharded, lazy, content-addressed sample store for NEXUS-BI.

Large function-level corpora do not fit comfortably in a single JSON list on a
32 GiB CPU host.  This store keeps one index record per sample and immutable
JSONL shards; training reads only the samples in the current microbatch.  The
store is append-only during construction and becomes read-only after
``finalize``.  Every artifact is SHA-256 fingerprinted.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

STORE_SCHEMA_VERSION = "nexus.sample-store/v1"
SHARD_SAMPLES = 512
SPLITS = ("train", "validation", "test", "extended_test", "quarantine")


def _json_hash(payload: Any) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass
class SampleIndexRecord:
    sample_id: str
    split: str
    shard: str
    offset: int
    length: int
    project_id: str = ""
    source_function_id: str | None = None
    source_file_id: str | None = None
    source_archive_id: str | None = None
    source_raw_hash: str | None = None
    source_normalized_hash: str | None = None
    instruction_hash: str | None = None
    binary_sha256: str | None = None
    group_id: str | None = None
    match_group: str | None = None
    has_source: bool = False
    source_target_truncated: bool = False
    purpose_id: int = -100
    instruction_count: int = 0
    source_token_count: int = 0
    language: str = "c"
    architecture: str = "unknown"
    binary_format: str = "unknown"
    compiler: str | None = None
    optimization: str | None = None
    dataset_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "split": self.split,
            "shard": self.shard,
            "offset": self.offset,
            "length": self.length,
            "project_id": self.project_id,
            "source_function_id": self.source_function_id,
            "source_file_id": self.source_file_id,
            "source_archive_id": self.source_archive_id,
            "source_raw_hash": self.source_raw_hash,
            "source_normalized_hash": self.source_normalized_hash,
            "instruction_hash": self.instruction_hash,
            "binary_sha256": self.binary_sha256,
            "group_id": self.group_id,
            "match_group": self.match_group,
            "has_source": self.has_source,
            "source_target_truncated": self.source_target_truncated,
            "purpose_id": self.purpose_id,
            "instruction_count": self.instruction_count,
            "source_token_count": self.source_token_count,
            "language": self.language,
            "architecture": self.architecture,
            "binary_format": self.binary_format,
            "compiler": self.compiler,
            "optimization": self.optimization,
            "dataset_id": self.dataset_id,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "SampleIndexRecord":
        known = {item.name for item in cls.__dataclass_fields__.values()}
        return cls(**{key: value for key, value in payload.items() if key in known})


class SampleStoreWriter:
    """Append samples to split-specific shards with a byte-offset index."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "shards").mkdir(exist_ok=True)
        self.index_path = self.root / "index.jsonl"
        self._index_handle = self.index_path.open("w", encoding="utf-8")
        self._shard_handles: dict[tuple[str, int], Any] = {}
        self._shard_bytes: dict[str, int] = {}
        self._counts: dict[str, int] = {split: 0 for split in SPLITS}
        self._shard_counts: dict[str, int] = {split: 0 for split in SPLITS}
        self._closed = False

    def _handle(self, split: str, shard_index: int):
        key = (split, shard_index)
        if key not in self._shard_handles:
            path = self.root / "shards" / f"{split}-{shard_index:05d}.jsonl"
            self._shard_handles[key] = path.open("a", encoding="utf-8")
        return self._shard_handles[key]

    def append(self, sample: dict[str, Any], split: str, record: SampleIndexRecord) -> None:
        if split not in SPLITS:
            raise ValueError(f"unknown split: {split}")
        if self._closed:
            raise RuntimeError("store writer already finalized")
        shard_index = self._shard_counts[split] // SHARD_SAMPLES
        shard_name = f"{split}-{shard_index:05d}.jsonl"
        handle = self._handle(split, shard_index)
        payload = json.dumps(sample, sort_keys=True, separators=(",", ":"), default=str) + "\n"
        data = payload.encode("utf-8")
        offset = self._shard_bytes.get(shard_name, 0)
        handle.write(payload)
        self._shard_bytes[shard_name] = offset + len(data)
        record.shard = shard_name
        record.offset = offset
        record.length = len(data)
        record.split = split
        self._index_handle.write(json.dumps(record.to_dict(), sort_keys=True) + "\n")
        self._counts[split] += 1
        self._shard_counts[split] += 1

    def finalize(self, metadata: dict[str, Any]) -> dict[str, Any]:
        if self._closed:
            raise RuntimeError("store writer already finalized")
        self._closed = True
        self._index_handle.close()
        for handle in self._shard_handles.values():
            handle.close()
        shard_manifest = []
        total_bytes = 0
        for path in sorted((self.root / "shards").glob("*.jsonl")):
            size = path.stat().st_size
            total_bytes += size
            shard_manifest.append({"path": path.name, "bytes": size, "sha256": _file_hash(path)})
        manifest = {
            "schema_version": STORE_SCHEMA_VERSION,
            "created_at": metadata.get("created_at"),
            "counts": dict(self._counts),
            "shard_samples": SHARD_SAMPLES,
            "shards": shard_manifest,
            "total_bytes": total_bytes,
            "index_sha256": _file_hash(self.index_path),
            "vocab_sha256": metadata.get("vocab_sha256"),
            "frozen_id": metadata.get("frozen_id"),
            "protected_test_file_sha256": metadata.get("protected_test_file_sha256"),
            "extended_test_file_sha256": metadata.get("extended_test_file_sha256"),
            "source_fingerprints": metadata.get("source_fingerprints") or [],
            "split_assignments_sha256": metadata.get("split_assignments_sha256"),
            "statistics": metadata.get("statistics") or {},
            "exclusions": metadata.get("exclusions") or {},
            "limitations": metadata.get("limitations") or [],
            "store_sha256": None,
        }
        body = dict(manifest)
        body.pop("store_sha256", None)
        manifest["store_sha256"] = _json_hash(body)
        (self.root / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return manifest


@dataclass
class SampleStore:
    root: Path
    manifest: dict[str, Any]
    index: dict[str, SampleIndexRecord] = field(default_factory=dict)
    by_split: dict[str, list[str]] = field(default_factory=dict)
    _shard_cache: dict[str, list[tuple[int, str]]] = field(default_factory=dict, repr=False)

    @classmethod
    def load(cls, root: str | Path) -> "SampleStore":
        base = Path(root).resolve()
        manifest_path = base / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"sample store manifest is missing: {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("schema_version") != STORE_SCHEMA_VERSION:
            raise ValueError(f"unsupported sample-store schema: {manifest.get('schema_version')}")
        index_path = base / "index.jsonl"
        if _file_hash(index_path) != manifest.get("index_sha256"):
            raise ValueError("sample-store index digest mismatch")
        index: dict[str, SampleIndexRecord] = {}
        by_split: dict[str, list[str]] = {split: [] for split in SPLITS}
        with index_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                record = SampleIndexRecord.from_dict(json.loads(line))
                if record.sample_id in index:
                    raise ValueError(f"duplicate sample id in store: {record.sample_id}")
                index[record.sample_id] = record
                by_split.setdefault(record.split, []).append(record.sample_id)
        counts = {split: len(values) for split, values in by_split.items()}
        if counts != manifest.get("counts"):
            raise ValueError(f"sample-store counts do not match manifest: {counts} vs {manifest.get('counts')}")
        store = cls(base, manifest, index, by_split)
        store._verify_shards()
        return store

    def _verify_shards(self) -> None:
        expected = {item["path"]: item for item in self.manifest.get("shards", [])}
        actual = {path.name for path in (self.root / "shards").glob("*.jsonl")}
        if set(expected) != actual:
            raise ValueError("sample-store shard set does not match manifest")
        for name, item in expected.items():
            path = self.root / "shards" / name
            if path.stat().st_size != item["bytes"]:
                raise ValueError(f"sample-store shard size mismatch: {name}")
            if _file_hash(path) != item["sha256"]:
                raise ValueError(f"sample-store shard digest mismatch: {name}")

    @property
    def frozen_id(self) -> str | None:
        return self.manifest.get("frozen_id")

    def sample_ids(self, split: str) -> list[str]:
        return list(self.by_split.get(split, []))

    def __len__(self) -> int:
        return sum(len(values) for values in self.by_split.values())

    def _read_record(self, record: SampleIndexRecord) -> dict[str, Any]:
        path = self.root / "shards" / record.shard
        with path.open("rb") as handle:
            handle.seek(record.offset)
            data = handle.read(record.length)
        if len(data) != record.length:
            raise ValueError(f"short read for sample {record.sample_id}")
        return json.loads(data.decode("utf-8"))

    def get(self, sample_id: str) -> dict[str, Any]:
        record = self.index.get(sample_id)
        if record is None:
            raise KeyError(sample_id)
        return self._read_record(record)

    def iter_split(self, split: str) -> Iterator[dict[str, Any]]:
        for sample_id in self.by_split.get(split, []):
            yield self.get(sample_id)

    def metadata(self, split: str) -> SampleIndexRecord | None:
        values = self.by_split.get(split) or []
        return self.index[values[0]] if values else None


class _LazySamples:
    """Sequence facade so existing trainer/evaluator code reads lazily."""

    def __init__(self, store: SampleStore, split: str, ids: list[str] | None = None):
        self.store = store
        self.split = split
        self._ids = list(ids) if ids is not None else store.by_split.get(split, [])

    def __len__(self) -> int:
        return len(self._ids)

    def __getitem__(self, index: int | slice) -> Any:
        if isinstance(index, slice):
            return [self.store.get(sample_id) for sample_id in self._ids[index]]
        return self.store.get(self._ids[index])

    def __iter__(self):
        for sample_id in self._ids:
            yield self.store.get(sample_id)
