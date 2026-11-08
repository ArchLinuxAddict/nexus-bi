from nexus.acquisition.adapters import (
    AdapterError,
    DebianAdapter,
    GitHubAdapter,
    GitLabAdapter,
    LocalArchiveAdapter,
    SoftwareHeritageAdapter,
    adapter_for,
    discover,
)
from nexus.acquisition.dedup import (
    build_historical_index,
    compare_against_index,
    project_signature,
    source_file_hashes,
)
from nexus.acquisition.licensing import (
    ALLOWED_SPDX,
    classify_license,
    detect_license,
    license_allowed,
)
from nexus.acquisition.pipeline import (
    AcquisitionState,
    acquire_candidate,
    run_pilot,
    safe_extract_tar,
)
from nexus.acquisition.provenance import hash_tree, provenance_confidence

__all__ = [
    "ALLOWED_SPDX",
    "AcquisitionState",
    "AdapterError",
    "DebianAdapter",
    "GitHubAdapter",
    "GitLabAdapter",
    "LocalArchiveAdapter",
    "SoftwareHeritageAdapter",
    "acquire_candidate",
    "adapter_for",
    "build_historical_index",
    "classify_license",
    "compare_against_index",
    "detect_license",
    "discover",
    "hash_tree",
    "license_allowed",
    "project_signature",
    "provenance_confidence",
    "run_pilot",
    "safe_extract_tar",
    "source_file_hashes",
]
