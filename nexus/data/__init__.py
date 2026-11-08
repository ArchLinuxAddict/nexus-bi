"""Versioned data identity and split tooling for NEXUS-BI."""

from nexus.data.identity import (
    IDENTITY_SCHEMA_VERSION,
    FunctionIdentityV2,
    c_tokens,
    extract_function_by_lines,
    normalize_c_source,
    source_hashes,
)
from nexus.data.split_v2 import SplitV2Result, build_split_v2
from nexus.data.validate_v2 import validate_split_v2

__all__ = [
    "IDENTITY_SCHEMA_VERSION",
    "FunctionIdentityV2",
    "SplitV2Result",
    "build_split_v2",
    "c_tokens",
    "extract_function_by_lines",
    "normalize_c_source",
    "source_hashes",
    "validate_split_v2",
]
