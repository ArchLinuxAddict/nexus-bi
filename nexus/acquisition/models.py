from dataclasses import asdict, dataclass, field
from typing import Any

ACQUISITION_SCHEMA_VERSION = "0.1.0"
ALIGNMENT_CLASSIFICATIONS = ("EXACT", "DEBUG_DERIVED", "SYMBOL_DERIVED", "HEURISTIC", "UNVERIFIED", "UNAVAILABLE")
PROVENANCE_CONFIDENCE = ("high", "medium", "low", "unknown")
ACQUISITION_STATUS = ("accepted", "quarantined", "failed", "skipped")


@dataclass
class SourceCandidate:
    project_id: str
    adapter: str
    repository: str
    canonical_url: str
    category: str
    description: str = ""
    topics: list[str] = field(default_factory=list)
    ref: str = "HEAD"
    commit: str | None = None
    expected_license: str | None = None
    recipe: dict[str, Any] = field(default_factory=dict)
    provider_metadata: dict[str, Any] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ResolvedSource:
    provider: str
    repository: str
    canonical_url: str
    commit: str
    ref: str
    metadata: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ArchiveRecord:
    url: str
    path: str
    sha256: str
    size: int
    media_type: str = "application/gzip"
    resumed: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class LicenseEvidence:
    found: bool
    status: str
    spdx: str | None = None
    expression: str | None = None
    license_file: str | None = None
    sha256: str | None = None
    size: int | None = None
    method: str = "none"
    confidence: float = 0.0
    detected_from: list[str] = field(default_factory=list)
    reason: str | None = None
    secondary_files: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class FileInventory:
    file_count: int
    total_bytes: int
    tree_sha256: str
    files: list[dict[str, Any]] = field(default_factory=list)
    skipped_links: int = 0
    skipped_special: int = 0
    classification: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ProvenanceRecord:
    source_url: str
    provider: str
    repository: str
    commit: str
    retrieved_at: str
    archive_sha256: str
    archive_size: int
    license_file_sha256: str | None
    file_inventory_sha256: str
    file_count: int
    source_bytes: int
    confidence: str
    confidence_reasons: list[str] = field(default_factory=list)
    commit_verified: bool = False
    commit_verification_method: str | None = None
    swh_revision: str | None = None
    swh_status: str = "not_checked"
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AcquisitionRecord:
    project_id: str
    candidate: dict[str, Any]
    status: str
    stage: str
    resolved: dict[str, Any] | None = None
    archive: dict[str, Any] | None = None
    license: dict[str, Any] | None = None
    provenance: dict[str, Any] | None = None
    inventory: dict[str, Any] | None = None
    deduplication: dict[str, Any] | None = None
    storage_bytes: int = 0
    quarantine_reason: str | None = None
    started_at: str | None = None
    finished_at: str | None = None
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
