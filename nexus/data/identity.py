"""Versioned function-identity primitives.

The identity layer intentionally distinguishes what is observed from what is
inferred.  A missing source mapping is represented as unavailable, never as an
empty or fabricated hash.
"""

from __future__ import annotations

import hashlib
import json
import re
import zlib
from dataclasses import dataclass, field
from typing import Any, Iterable

IDENTITY_SCHEMA_VERSION = "nexus.function-identity/v2"
PROTECTED_TEST_SCHEMA_VERSION = "nexus.protected-test/v2"

_COMMENT_OR_LITERAL = re.compile(
    r"//[^\n]*|/\*.*?\*/|\"(?:\\.|[^\"\\])*\"|'(?:\\.|[^'\\])*'",
    re.DOTALL,
)
_TOKEN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|0[xX][0-9a-fA-F]+|\d+(?:\.\d+)?|[^\s]")


def _digest(payload: Any) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _text_digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _strip_comments(text: str) -> str:
    return _COMMENT_OR_LITERAL.sub(lambda match: match.group(0) if match.group(0)[:1] in {'"', "'"} else " ", text)


def c_tokens(text: str) -> list[str]:
    """Tokenize C-like source while preserving literals and rejecting comments."""
    if not text:
        return []
    return _TOKEN.findall(_strip_comments(text))


def normalize_c_source(text: str) -> str:
    """Conservative normalization: comments/whitespace only.

    Identifiers and literals are deliberately preserved.  Replacing identifiers
    would turn independent implementations into false duplicates.
    """
    return " ".join(c_tokens(text))


def _shingle_hashes(tokens: list[str], n: int = 5, limit: int = 512) -> tuple[int, ...]:
    if not tokens:
        return ()
    if len(tokens) < n:
        grams = [" ".join(tokens)]
    else:
        grams = [" ".join(tokens[index : index + n]) for index in range(0, len(tokens) - n + 1)]
    values = sorted({zlib.crc32(gram.encode("utf-8")) & 0xFFFFFFFF for gram in grams})
    return tuple(values[:limit])


def source_hashes(text: str) -> dict[str, Any]:
    tokens = c_tokens(text)
    normalized = " ".join(tokens)
    return {
        "raw_source_hash": _text_digest(text),
        "normalized_source_hash": _text_digest(normalized),
        "normalized_function_hash": _text_digest(normalized),
        "token_count": len(tokens),
        "shingle_hashes": _shingle_hashes(tokens),
    }


def extract_function_by_lines(text: str, start_line: int, end_line: int) -> str | None:
    if start_line < 1 or end_line < start_line:
        return None
    lines = text.splitlines()
    if start_line > len(lines):
        return None
    return "\n".join(lines[start_line - 1 : min(end_line, len(lines))]).strip() or None


def extract_function_by_name(text: str, name: str) -> str | None:
    """Deterministic brace-matching extraction used only when the name is exact."""
    if not text or not name:
        return None
    pattern = re.compile(r"(?<![A-Za-z0-9_])" + re.escape(name) + r"\s*\(")
    for match in pattern.finditer(text):
        start = text.rfind("\n", 0, match.start()) + 1
        if "#" in text[start : match.start()]:
            continue
        brace = text.find("{", match.end())
        semicolon = text.find(";", match.end())
        if brace < 0 or (semicolon >= 0 and semicolon < brace):
            continue
        depth = 0
        for index in range(brace, len(text)):
            character = text[index]
            if character == "{":
                depth += 1
            elif character == "}":
                depth -= 1
                if depth == 0:
                    return text[start : index + 1].strip()
    return None


def _instruction_hash(sample: dict[str, Any]) -> str:
    return _digest(
        {
            "instr": sample.get("instr_id", []),
            "ir": sample.get("instr_ir", []),
            "blocks": sample.get("block_mask", []),
            "cfg": sample.get("cfg_target", []),
        }
    )


@dataclass(frozen=True)
class FunctionIdentityV2:
    function_uid: str
    project_id: str
    repository_id: str
    source_archive_id: str | None
    source_file_id: str | None
    source_function_id: str | None
    sample_id: str
    function_id: str
    source_path: str | None
    source_line_start: int | None
    source_line_end: int | None
    raw_source_hash: str | None
    normalized_source_hash: str | None
    normalized_function_hash: str | None
    source_shingle_hashes: tuple[int, ...]
    source_available: bool
    source_alignment: str
    source_alignment_confidence: float
    binary_sha256: str
    instruction_hash: str
    instruction_count: int
    build_id: str
    compiler_id: str
    compiler_version: str | None
    optimization_level: str | None
    architecture: str
    language: str
    binary_format: str
    debug_info: bool | None
    split: str
    protected_test: bool
    license_id: str | None
    provenance: dict[str, Any] = field(default_factory=dict)
    notes: tuple[str, ...] = ()
    identity_version: str = IDENTITY_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "identity_version": self.identity_version,
            "function_uid": self.function_uid,
            "project_id": self.project_id,
            "repository_id": self.repository_id,
            "source_archive_id": self.source_archive_id,
            "source_file_id": self.source_file_id,
            "source_function_id": self.source_function_id,
            "sample_id": self.sample_id,
            "function_id": self.function_id,
            "source_path": self.source_path,
            "source_line_start": self.source_line_start,
            "source_line_end": self.source_line_end,
            "raw_source_hash": self.raw_source_hash,
            "normalized_source_hash": self.normalized_source_hash,
            "normalized_function_hash": self.normalized_function_hash,
            "source_shingle_hashes": list(self.source_shingle_hashes),
            "source_available": self.source_available,
            "source_alignment": self.source_alignment,
            "source_alignment_confidence": self.source_alignment_confidence,
            "binary_sha256": self.binary_sha256,
            "instruction_hash": self.instruction_hash,
            "instruction_count": self.instruction_count,
            "build_id": self.build_id,
            "compiler_id": self.compiler_id,
            "compiler_version": self.compiler_version,
            "optimization_level": self.optimization_level,
            "architecture": self.architecture,
            "language": self.language,
            "binary_format": self.binary_format,
            "debug_info": self.debug_info,
            "split": self.split,
            "protected_test": self.protected_test,
            "license_id": self.license_id,
            "provenance": self.provenance,
            "notes": list(self.notes),
        }
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "FunctionIdentityV2":
        data = dict(payload)
        data["source_shingle_hashes"] = tuple(data.get("source_shingle_hashes") or ())
        data["notes"] = tuple(data.get("notes") or ())
        return cls(**data)


def identity_from_foundation_sample(sample: dict[str, Any], *, protected_test: bool = False) -> FunctionIdentityV2:
    labels = sample.get("labels") or {}
    source_text = sample.get("source_text") or ""
    source_path = sample.get("source_path")
    project_id = str(sample.get("project_id") or labels.get("project_id") or "unknown")
    source_archive_id = labels.get("source_archive_sha256")
    source_file_id = _digest([project_id, source_path]) if source_path else None
    hashes = source_hashes(source_text) if source_text else {}
    source_function_id = (
        _digest([project_id, source_path, hashes["raw_source_hash"]]) if hashes else None
    )
    compiler = str(labels.get("compiler") or "unknown")
    optimization = labels.get("optimization")
    architecture = str(labels.get("arch") or "unknown")
    debug_info = labels.get("debug_info")
    binary_sha = str(sample.get("binary_sha256") or "")
    build_id = _digest([binary_sha, compiler, optimization, architecture, debug_info])
    function_uid = _digest(
        [
            source_function_id or f"{binary_sha}:{sample.get('function_id')}",
            compiler,
            optimization,
            architecture,
            binary_sha,
            str(sample.get("function_id") or ""),
        ]
    )
    return FunctionIdentityV2(
        function_uid=function_uid,
        project_id=project_id,
        repository_id=project_id,
        source_archive_id=str(source_archive_id) if source_archive_id else None,
        source_file_id=source_file_id,
        source_function_id=source_function_id,
        sample_id=str(sample.get("sample_id") or ""),
        function_id=str(sample.get("function_id") or ""),
        source_path=source_path,
        source_line_start=None,
        source_line_end=None,
        raw_source_hash=hashes.get("raw_source_hash"),
        normalized_source_hash=hashes.get("normalized_source_hash"),
        normalized_function_hash=hashes.get("normalized_function_hash"),
        source_shingle_hashes=tuple(hashes.get("shingle_hashes") or ()),
        source_available=bool(source_text),
        source_alignment=str((labels.get("source_alignment") or {}).get("alignment") or "unavailable"),
        source_alignment_confidence=float((labels.get("source_alignment") or {}).get("confidence") or 0.0),
        binary_sha256=binary_sha,
        instruction_hash=_instruction_hash(sample),
        instruction_count=len(sample.get("instr_id") or []),
        build_id=build_id,
        compiler_id=compiler,
        compiler_version=None,
        optimization_level=str(optimization) if optimization else None,
        architecture=architecture,
        language="c",
        binary_format="ELF",
        debug_info=bool(debug_info) if debug_info is not None else None,
        split=str(sample.get("split") or labels.get("split") or "unassigned"),
        protected_test=protected_test,
        license_id=labels.get("license"),
        provenance={
            "dataset_id": sample.get("dataset_id") or labels.get("dataset_id"),
            "sample_id": sample.get("sample_id"),
            "manifest_sample_id": labels.get("sample_id"),
            "record_path": sample.get("record_path"),
            "binary_path": sample.get("binary_path"),
            "source_path": source_path,
        },
    )


def identity_from_phase2b_function(
    index: dict[str, Any],
    function: dict[str, Any],
    source_text: str,
) -> FunctionIdentityV2 | None:
    start_line = int(function.get("source_line") or 0)
    if start_line <= 0 or not source_text:
        return None
    lines = [
        int(item.get("source_line") or 0)
        for item in index.get("line_mappings", [])
        if item.get("function_id") == function.get("function_id") and item.get("source_line")
    ]
    end_line = max(lines) if lines else start_line
    text = extract_function_by_lines(source_text, start_line, end_line)
    if not text:
        return None
    hashes = source_hashes(text)
    project_id = str(index.get("project_id") or "unknown")
    project = index.get("project") or {}
    build = index.get("build_variant") or {}
    binary = index.get("binary") or {}
    source_file = function.get("source_file")
    source_sha = function.get("source_sha256")
    source_function_id = _digest([project_id, source_file, hashes["raw_source_hash"]])
    function_uid = _digest(
        [source_function_id, build.get("compiler"), build.get("optimization"), binary.get("sha256"), function.get("function_id")]
    )
    return FunctionIdentityV2(
        function_uid=function_uid,
        project_id=project_id,
        repository_id=str(project.get("project_id") or project_id),
        source_archive_id=project.get("source_archive_sha256"),
        source_file_id=_digest([project_id, source_file]) if source_file else None,
        source_function_id=source_function_id,
        sample_id=f"p2b:{project_id}:{build.get('variant_id')}:{function.get('function_id')}",
        function_id=str(function.get("function_id") or ""),
        source_path=source_file,
        source_line_start=start_line,
        source_line_end=end_line,
        raw_source_hash=hashes["raw_source_hash"],
        normalized_source_hash=hashes["normalized_source_hash"],
        normalized_function_hash=hashes["normalized_function_hash"],
        source_shingle_hashes=tuple(hashes["shingle_hashes"]),
        source_available=True,
        source_alignment=str(function.get("classification") or "dwarf_line_table"),
        source_alignment_confidence=float(function.get("confidence") or 0.0),
        binary_sha256=str(binary.get("sha256") or ""),
        instruction_hash=str(function.get("code_sha256") or _digest(function.get("instruction_ids") or [])),
        instruction_count=int(function.get("instruction_count") or len(function.get("instruction_ids") or [])),
        build_id=_digest([binary.get("sha256"), build.get("compiler"), build.get("optimization"), build.get("variant_id")]),
        compiler_id=str(build.get("compiler") or "unknown"),
        compiler_version=None,
        optimization_level=build.get("optimization"),
        architecture=str(binary.get("machine") or "unknown"),
        language="c",
        binary_format="ELF",
        debug_info=bool(binary.get("has_debug_info")),
        split="unassigned",
        protected_test=False,
        license_id=project.get("license"),
        provenance={
            "source": "phase2b_alignment",
            "source_sha256": source_sha,
            "source_file": source_file,
            "source_root": index.get("source_root"),
            "variant_id": build.get("variant_id"),
            "entry_va": function.get("entry_va"),
        },
        notes=("source text recovered from DWARF line range",),
    )


def identity_from_realworld_function(
    manifest: dict[str, Any],
    function: dict[str, Any],
    instruction_ids: Iterable[str],
) -> FunctionIdentityV2:
    build = manifest.get("build") or {}
    binary = manifest.get("binary") or {}
    project_id = str(manifest.get("project_id") or "unknown")
    binary_sha = str(binary.get("sha256") or "")
    function_id = str(function.get("function_id") or "")
    instruction_ids = sorted(set(str(item) for item in instruction_ids))
    instruction_hash = _digest(instruction_ids)
    return FunctionIdentityV2(
        function_uid=_digest([binary_sha, function_id, build.get("compiler"), build.get("optimization")]),
        project_id=project_id,
        repository_id=str((manifest.get("provenance") or {}).get("repository") or project_id),
        source_archive_id=(manifest.get("provenance") or {}).get("source_archive_sha256"),
        source_file_id=None,
        source_function_id=None,
        sample_id=f"rw:{manifest.get('sample_id')}:{function_id}",
        function_id=function_id,
        source_path=None,
        source_line_start=None,
        source_line_end=None,
        raw_source_hash=None,
        normalized_source_hash=None,
        normalized_function_hash=None,
        source_shingle_hashes=(),
        source_available=False,
        source_alignment="unverified",
        source_alignment_confidence=0.0,
        binary_sha256=binary_sha,
        instruction_hash=instruction_hash,
        instruction_count=len(instruction_ids),
        build_id=_digest([binary_sha, build.get("compiler"), build.get("optimization"), build.get("arch"), build.get("debug_info")]),
        compiler_id=str(build.get("compiler") or "unknown"),
        compiler_version=None,
        optimization_level=build.get("optimization"),
        architecture=str(build.get("arch") or "unknown"),
        language="c",
        binary_format="ELF",
        debug_info=bool(build.get("debug_info")),
        split="unassigned",
        protected_test=False,
        license_id=(manifest.get("license") or {}).get("spdx"),
        provenance={"source": "realworld_program_record", "sample_id": manifest.get("sample_id")},
        notes=("source alignment unavailable in realworld corpus",),
    )
