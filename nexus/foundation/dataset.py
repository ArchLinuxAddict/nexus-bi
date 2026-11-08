"""Registered, provenance-enforced dataset construction for NEXUS-BI.

Controlled and real-world corpora remain distinguishable.  Splits are taken
from project-level manifests; test data is never accepted for training.
Source text is attached only as a reconstruction/alignment target and never
enters binary-only input tensors.
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator

import torch

from nexus.core.hashing import sha256_file
from nexus.foundation.config import FoundationConfig
from nexus.foundation.representation import (
    PURPOSE_INDEX,
    RepresentationOptions,
    collate_samples,
    encode_function_sample,
    mask_byte_batch,
    mask_instruction_batch,
    source_label_for_function,
)
from nexus.foundation.vocab import Vocabulary
from nexus.mission.enforcement import dataset_fingerprint, validate_dataset_for_training
from nexus.mission.policy import load_registry, mission_for_dataset

PROJECT_ROOT = Path(__file__).resolve().parents[2]
APPROVED_ROOT = PROJECT_ROOT / "artifacts" / "dataset"


def _json_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _artifact_path(manifest: dict[str, Any], predicate) -> dict[str, Any] | None:
    for artifact in manifest.get("artifacts", []):
        if predicate(artifact):
            return artifact
    return None


def _verify_artifact(root: Path, artifact: dict[str, Any]) -> None:
    path = root / artifact["path"]
    if not path.exists():
        raise FileNotFoundError(f"dataset artifact is missing: {artifact['path']}")
    if sha256_file(path) != artifact["sha256"]:
        raise ValueError(f"dataset artifact hash mismatch: {artifact['path']}")


def _read_sources(manifest: dict[str, Any], root: Path) -> dict[str, str]:
    sources = {}
    for artifact in manifest.get("artifacts", []):
        path = root / artifact["path"]
        if not path.exists():
            continue
        if path.suffix in {".c", ".h"}:
            sources[artifact["path"]] = path.read_text(encoding="utf-8", errors="replace")
        elif path.name == "spec.json":
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue
            if isinstance(payload.get("source"), str):
                sources[artifact["path"]] = payload["source"]
    return sources


def _purpose_from_spec(manifest: dict[str, Any], root: Path) -> str | None:
    for artifact in manifest.get("artifacts", []):
        if not artifact["path"].endswith("spec.json"):
            continue
        path = root / artifact["path"]
        if not path.exists():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        category = str(payload.get("category", "")).lower()
        category_aliases = {
            "searching": "algorithms",
            "sorting": "algorithms",
            "bitwise": "arithmetic",
            "state_transitions": "systems",
            "arrays": "data_structures",
            "encoding": "serialization",
            "boundary": "utilities",
            "signed_unsigned": "arithmetic",
            "numeric": "arithmetic",
            "text": "strings",
        }
        if category in PURPOSE_INDEX:
            return category
        if category in category_aliases:
            return category_aliases[category]
        text = " ".join(
            [
                str(payload.get("behavior_id", "")),
                str((payload.get("purpose") or {}).get("summary", "")),
                str((payload.get("purpose") or {}).get("transformation", "")),
                str((payload.get("spec") or {}).get("summary", "")),
            ]
        ).lower()
        keywords = {
            "crypt": "cryptography",
            "hash": "cryptography",
            "compress": "compression",
            "network": "networking",
            "parse": "parsing",
            "serial": "serialization",
            "string": "strings",
            "sort": "algorithms",
            "search": "algorithms",
            "math": "arithmetic",
            "alloc": "memory",
            "buffer": "memory",
            "list": "data_structures",
            "tree": "data_structures",
            "thread": "concurrency",
            "file": "io",
            "bitwise": "arithmetic",
            "encode": "serialization",
            "decode": "serialization",
            "array": "data_structures",
            "boundary": "utilities",
        }
        for token, purpose in keywords.items():
            if token in text:
                return purpose
    return None


def _function_name(function: dict[str, Any]) -> str | None:
    names = [entry.get("value") for entry in function.get("names", []) if entry.get("value")]
    return names[0] if names else None


def _manifest_split(manifest: dict[str, Any]) -> str:
    return str(manifest.get("split") or "train")


def select_manifests(manifests: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    """Round-robin by project so compiler variants do not crowd out programs."""
    if limit <= 0:
        return []
    groups: dict[str, list[dict[str, Any]]] = {}
    for manifest in manifests:
        groups.setdefault(str(manifest.get("project_id") or manifest.get("sample_id")), []).append(manifest)
    selected = []
    index = 0
    while len(selected) < limit:
        added = False
        for group in groups.values():
            if index < len(group):
                selected.append(group[index])
                added = True
                if len(selected) >= limit:
                    break
        if not added:
            break
        index += 1
    return selected


@dataclass
class DatasetSource:
    path: str
    dataset_id: str
    category: str
    classification: str
    sha256: str
    max_manifests: int = 48
    max_functions: int = 8
    include_source: bool = True
    include_purpose: bool = True
    include_byte_modality: bool = True
    include_ir_modality: bool = True
    include_graph_modality: bool = True
    derived_records_root: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class FoundationDataset:
    samples: list[dict[str, Any]]
    config: FoundationConfig
    vocab: Vocabulary
    split: str
    sources: list[DatasetSource] = field(default_factory=list)
    provenance: dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.samples)

    def batches(
        self,
        batch_size: int = 4,
        shuffle: bool = True,
        seed: int = 20261103,
        mask_instructions: bool = True,
        mask_bytes: bool = True,
        mask_probability: float = 0.15,
        byte_mask_probability: float = 0.08,
        modality_dropout: float = 0.0,
        fixed_order: list[int] | None = None,
        grouped_fraction: float = 0.0,
        groups_per_batch: int = 2,
        members_per_group: int = 2,
    ) -> Iterator[dict[str, Any]]:
        order = list(fixed_order) if fixed_order is not None else list(range(len(self.samples)))
        if shuffle and fixed_order is None:
            random.Random(seed).shuffle(order)
        for start in range(0, len(order), batch_size):
            chosen = [self.samples[index] for index in order[start : start + batch_size]]
            batch = collate_samples(chosen, self.config)
            if mask_instructions:
                batch = mask_instruction_batch(batch, probability=mask_probability, seed=seed + start)
            if mask_bytes:
                batch = mask_byte_batch(batch, probability=byte_mask_probability, seed=seed + start)
            if modality_dropout > 0:
                batch = apply_modality_dropout(batch, modality_dropout, seed=seed + start)
            yield batch

    def evaluation_batches(self, batch_size: int = 4) -> Iterator[dict[str, Any]]:
        for start in range(0, len(self.samples), batch_size):
            yield collate_samples(self.samples[start : start + batch_size], self.config)

    def stats(self) -> dict[str, Any]:
        by_dataset: dict[str, int] = {}
        by_project: dict[str, set[str]] = {}
        source_labels = sum(1 for sample in self.samples if sample.get("source_text"))
        purpose_labels = sum(1 for sample in self.samples if sample.get("purpose_id", -100) >= 0)
        for sample in self.samples:
            dataset_id = sample.get("dataset_id", "")
            by_dataset[dataset_id] = by_dataset.get(dataset_id, 0) + 1
            by_project.setdefault(dataset_id, set()).add(sample.get("project_id", ""))
        return {
            "split": self.split,
            "samples": len(self.samples),
            "by_dataset": by_dataset,
            "unique_projects_by_dataset": {key: len(value) for key, value in by_project.items()},
            "unique_projects_total": len({sample.get("project_id") for sample in self.samples}),
            "source_labelled_samples": source_labels,
            "purpose_labelled_samples": purpose_labels,
            "source_inputs_used": False,
            "symbol_names_used_as_inputs": False,
            "datasets": [source.to_dict() for source in self.sources],
            "provenance": self.provenance,
        }


def apply_modality_dropout(batch: dict[str, Any], probability: float, seed: int = 0) -> dict[str, Any]:
    generator = torch.Generator().manual_seed(seed + 104729)
    result = dict(batch)
    for modality in ("byte_ids", "instr_ir"):
        drop = torch.rand((batch["instr_id"].size(0),), generator=generator) < probability
        if modality == "byte_ids" and "byte_mask" in batch:
            result["byte_mask"] = batch["byte_mask"] & ~drop.unsqueeze(1)
        elif modality == "instr_ir":
            ids = batch["instr_ir"].clone()
            ids[drop.unsqueeze(1).expand_as(ids)] = 0
            result["instr_ir"] = ids
    if "instruction_edge_mask" in batch:
        drop = torch.rand((batch["instr_id"].size(0),), generator=generator) < probability
        result["instruction_edge_mask"] = batch["instruction_edge_mask"] & ~drop.unsqueeze(1)
    if "block_edge_mask" in batch:
        drop = torch.rand((batch["instr_id"].size(0),), generator=generator) < probability
        result["block_edge_mask"] = batch["block_edge_mask"] & ~drop.unsqueeze(1)
    return result


def _derive_record(
    manifest: dict[str, Any],
    root: Path,
    cache_root: Path,
    binary_artifact: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Derive a ProgramRecord for a registered binary without modifying the dataset."""
    if binary_artifact is None:
        return None
    binary_path = root / binary_artifact["path"]
    if not binary_path.exists():
        return None
    import re

    from nexus.program.bundle import analyze_binary
    from nexus.program.record import build_program_record

    dataset_key = hashlib.sha256(str(root).encode()).hexdigest()[:16]
    cache_root = cache_root.resolve()
    safe_sample = re.sub(r"[^A-Za-z0-9_.-]", "_", str(manifest.get("sample_id") or "sample"))
    target = cache_root / dataset_key / f"{safe_sample}.json"
    sidecar = target.with_suffix(".provenance.json")
    expected = sha256_file(binary_path)
    if target.exists() and sidecar.exists():
        try:
            metadata = json.loads(sidecar.read_text(encoding="utf-8"))
            if metadata.get("binary_sha256") == expected:
                return {
                    "path": str(target),
                    "sha256": _json_hash(target),
                    "size": target.stat().st_size,
                    "media_type": "application/json",
                    "derived": True,
                }
        except json.JSONDecodeError:
            pass
    bundle = analyze_binary(binary_path)
    record = build_program_record(bundle)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    sidecar.write_text(
        json.dumps(
            {
                "schema_version": "0.1.0",
                "source_dataset": str(root),
                "sample_id": manifest.get("sample_id"),
                "binary_sha256": expected,
                "record_sha256": _json_hash(target),
                "analysis_pipeline": record.get("analysis", {}).get("pipeline") if isinstance(record.get("analysis"), dict) else None,
                "note": "Deterministically derived cache; original dataset untouched.",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return {
        "path": str(target),
        "sha256": _json_hash(target),
        "size": target.stat().st_size,
        "media_type": "application/json",
        "derived": True,
    }


def _validate_dataset(path: Path, split: str, objectives: list[str], verify_artifacts: bool) -> dict[str, Any]:
    if split in {"train", "validation"}:
        provenance = validate_dataset_for_training(
            path,
            split,
            objectives,
            verify_artifacts=verify_artifacts,
        )
        return provenance.to_dict()
    fingerprint = dataset_fingerprint(path)
    entry = mission_for_dataset(path)
    if entry.get("classification") in {"PROHIBITED", "IRRELEVANT"}:
        raise ValueError(f"dataset is not usable for evaluation: {entry.get('classification')}")
    return {
        "dataset_id": fingerprint["dataset_id"],
        "dataset_hash": fingerprint["manifest_sha256"],
        "dataset_category": entry["category"],
        "dataset_classification": entry["classification"],
        "objectives": objectives,
        "split": split,
        "root": str(path),
        "evaluation_only": True,
    }


def build_dataset(
    sources: Iterable[DatasetSource],
    vocab: Vocabulary,
    config: FoundationConfig,
    split: str = "train",
    include_source: bool | None = None,
    verify_artifacts: bool = True,
    objectives: list[str] | None = None,
    strict_source_input_isolation: bool = True,
    split_assignments: dict[str, str] | None = None,
) -> FoundationDataset:
    if split not in {"train", "validation", "test"}:
        raise ValueError(f"unknown split: {split}")
    objectives = objectives or ["binary_representation_masking"]
    all_samples: list[dict[str, Any]] = []
    provenance_records: list[dict[str, Any]] = []
    quarantine: list[dict[str, Any]] = []
    for source in sources:
        root = Path(source.path)
        if not root.is_absolute():
            root = PROJECT_ROOT / root
        root = root.resolve()
        try:
            root.relative_to(APPROVED_ROOT.resolve())
        except ValueError as exc:
            raise ValueError(f"dataset path is outside approved roots: {root}") from exc
        provenance = _validate_dataset(root, split, objectives, verify_artifacts=False)
        provenance["declared_dataset_id"] = source.dataset_id
        provenance_records.append(provenance)
        manifests = [
            json.loads(line)
            for line in (root / "manifests.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if split_assignments is not None:
            selected = sorted(
                [
                    manifest
                    for manifest in manifests
                    if str(manifest.get("sample_id")) in split_assignments
                    and split_assignments[str(manifest.get("sample_id"))] == split
                ],
                key=lambda manifest: str(manifest.get("sample_id")),
            )
        else:
            selected = select_manifests(
                [manifest for manifest in manifests if _manifest_split(manifest) == split],
                source.max_manifests,
            )
        if not selected:
            continue
        for manifest in selected:
            if manifest.get("quality", {}).get("validation_status") == "rejected":
                quarantine.append({"sample_id": manifest.get("sample_id"), "reason": "rejected_manifest"})
                continue
            record_artifact = _artifact_path(
                manifest,
                lambda artifact: artifact.get("role") == "program_record" or str(artifact["path"]).endswith("program_record.json"),
            )
            binary_artifact = _artifact_path(
                manifest,
                lambda artifact: artifact.get("role") == "binary" or artifact.get("sha256") == manifest.get("binary", {}).get("sha256"),
            )
            derived = False
            if record_artifact is None and source.derived_records_root:
                record_artifact = _derive_record(manifest, root, Path(source.derived_records_root), binary_artifact)
                derived = record_artifact is not None
            if record_artifact is None:
                quarantine.append({"sample_id": manifest.get("sample_id"), "reason": "missing_program_record"})
                continue
            try:
                _verify_artifact(root, record_artifact)
                if binary_artifact is not None:
                    _verify_artifact(root, binary_artifact)
            except (FileNotFoundError, ValueError) as exc:
                quarantine.append({"sample_id": manifest.get("sample_id"), "reason": str(exc)})
                continue
            record_path = root / record_artifact["path"]
            record = json.loads(record_path.read_text(encoding="utf-8"))
            use_source = source.include_source if include_source is None else include_source
            source_cache = _read_sources(manifest, root) if use_source else {}
            purpose_hint = _purpose_from_spec(manifest, root) if source.include_purpose else None
            options = RepresentationOptions(
                include_bytes=source.include_byte_modality,
                include_instructions=True,
                include_ir=source.include_ir_modality,
                include_graph=source.include_graph_modality,
                include_metadata=True,
                include_source_target=use_source,
                include_purpose_target=source.include_purpose,
            )
            functions = [function for function in record.get("functions", []) if isinstance(function, dict)]
            preferred = []
            other = []
            for function in functions:
                name = _function_name(function) or ""
                has_source = bool(source_cache) and any(name and name in text for text in source_cache.values())
                if has_source or (source.include_purpose and name == "nexus_target"):
                    preferred.append(function)
                else:
                    other.append(function)
            ordered = (preferred + other)[: source.max_functions]
            for function in ordered:
                source_label = source_label_for_function(manifest, root, function, source_cache) if use_source else None
                purpose = None
                if source.include_purpose and purpose_hint:
                    name = _function_name(function)
                    if name == "nexus_target" or len(ordered) <= 2:
                        purpose = purpose_hint
                identity = {
                    "sample_id": f"{manifest.get('sample_id')}:{function.get('function_id')}",
                    "function_id": function.get("function_id", ""),
                    "project_id": manifest.get("project_id", ""),
                    "dataset_id": provenance["dataset_id"],
                    "split": split,
                    "binary_sha256": manifest.get("binary", {}).get("sha256", ""),
                    "source_family": manifest.get("source_family", ""),
                    "match_group": manifest.get("project_id", ""),
                    "binary_path": str((root / binary_artifact["path"]).resolve()) if binary_artifact else None,
                    "record_path": str((root / record_artifact["path"]).resolve()),
                    "build": manifest.get("build", {}),
                }
                sample = encode_function_sample(
                    record,
                    function,
                    vocab,
                    config,
                    identity=identity,
                    options=options,
                    source_label=source_label,
                    purpose_label=purpose,
                    register_targets=vocab.register_targets(config.max_register_targets),
                )
                if sample is None:
                    continue
                sample["labels"].update(
                    {
                        "dataset_id": provenance["dataset_id"],
                        "sample_id": manifest.get("sample_id"),
                        "project_id": manifest.get("project_id"),
                        "split": split,
                        "license": (manifest.get("license") or {}).get("spdx"),
                        "source_archive_sha256": (manifest.get("provenance") or {}).get("source_archive_sha256"),
                        "compiler": (manifest.get("build") or {}).get("compiler"),
                        "optimization": (manifest.get("build") or {}).get("optimization"),
                        "debug_info": (manifest.get("build") or {}).get("debug_info"),
                        "arch": (manifest.get("build") or {}).get("arch"),
                        "record_sha256": record_artifact["sha256"],
                        "record_derived": derived,
                        "purpose_hint": purpose_hint,
                    }
                )
                if strict_source_input_isolation:
                    provenance_flags = sample.get("input_provenance", {})
                    if provenance_flags.get("names_used") or provenance_flags.get("source_used"):
                        raise AssertionError("source-derived information entered the binary-only input path")
                all_samples.append(sample)
    if split in {"train", "validation"}:
        projects = {sample.get("project_id") for sample in all_samples}
        test_projects: set[str] = set()
        for source in sources:
            root = Path(source.path)
            if not root.is_absolute():
                root = PROJECT_ROOT / root
            manifests_path = root.resolve() / "manifests.jsonl"
            if not manifests_path.exists():
                continue
            for line in manifests_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                manifest = json.loads(line)
                if _manifest_split(manifest) == "test":
                    test_projects.add(manifest.get("project_id"))
        leakage = sorted(project for project in projects & test_projects if project)
        if leakage:
            raise ValueError(f"project leakage between training and test splits: {leakage}")
    report = {
        "split": split,
        "samples": len(all_samples),
        "sources": [source.to_dict() for source in sources],
        "provenance": provenance_records,
        "quarantine": quarantine,
        "unique_projects": len({sample.get("project_id") for sample in all_samples}),
        "source_labelled": sum(1 for sample in all_samples if sample.get("source_text")),
        "purpose_labelled": sum(1 for sample in all_samples if sample.get("purpose_id", -100) >= 0),
        "test_projects_in_training": [],
    }
    return FoundationDataset(
        samples=all_samples,
        config=config,
        vocab=vocab,
        split=split,
        sources=list(sources),
        provenance=report,
    )


def _registry_matches(dataset_id: str) -> list[dict[str, Any]]:
    registry = load_registry()
    matches = []
    for entry in registry["datasets"]:
        aliases = set(entry.get("aliases") or [])
        if entry.get("dataset_id") == dataset_id or dataset_id in aliases:
            matches.append(entry)
    return matches


def resolve_dataset_identity(dataset_id: str) -> tuple[dict[str, Any], str]:
    matches = _registry_matches(dataset_id)
    if len(matches) != 1:
        raise ValueError(
            f"registry must contain exactly one entry or alias for {dataset_id!r}; found {len(matches)}"
        )
    entry = matches[0]
    canonical = str(entry.get("dataset_id") or "").strip()
    if not canonical:
        raise ValueError(f"registered dataset entry for {dataset_id!r} has no dataset_id")
    return entry, canonical


def dataset_source_from_registry(dataset_id: str, **overrides: Any) -> DatasetSource:
    entry, canonical = resolve_dataset_identity(dataset_id)
    path = Path(entry["path"])
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    if not path.exists():
        raise FileNotFoundError(f"registered dataset path does not exist: {path}")
    summary_path = path / "dataset_summary.json"
    manifest_path = path / "manifests.jsonl"
    if not summary_path.exists():
        raise ValueError(f"registered dataset has no dataset_summary.json: {path}")
    if not manifest_path.exists():
        raise ValueError(f"registered dataset has no manifests.jsonl: {path}")
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"dataset summary is not valid JSON: {summary_path}") from exc
    summary_id = summary.get("canonical_id") or summary.get("dataset_id")
    if not summary_id:
        raise ValueError(f"dataset summary is missing dataset_id/canonical_id: {summary_path}")
    if summary_id != canonical:
        raise ValueError(
            f"dataset summary identity {summary_id!r} does not match registry canonical ID {canonical!r}"
        )
    manifest_hash = _json_hash(manifest_path)
    declared_manifest = summary.get("manifest_sha256")
    if declared_manifest is not None and declared_manifest != manifest_hash:
        raise ValueError(
            f"dataset summary manifest fingerprint does not match manifests.jsonl for {canonical}"
        )
    aliases = set(summary.get("aliases") or []) | set(entry.get("aliases") or [])
    if dataset_id != canonical and dataset_id not in aliases and dataset_id != manifest_hash:
        raise ValueError(f"dataset alias {dataset_id!r} does not resolve to canonical ID {canonical!r}")
    return DatasetSource(
        path=str(path),
        dataset_id=canonical,
        category=entry["category"],
        classification=entry["classification"],
        sha256=manifest_hash,
        **overrides,
    )


def describe_dataset(sources: Iterable[DatasetSource]) -> dict[str, Any]:
    summary = []
    for source in sources:
        path = Path(source.path)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        manifests = [
            json.loads(line)
            for line in (path / "manifests.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        projects = {manifest.get("project_id") for manifest in manifests}
        summary.append(
            {
                "path": source.path,
                "dataset_id": source.dataset_id,
                "category": source.category,
                "classification": source.classification,
                "manifests": len(manifests),
                "unique_projects": len(projects),
                "variants_per_project": round(len(manifests) / max(1, len(projects)), 3),
                "by_split": {
                    split: sum(1 for manifest in manifests if _manifest_split(manifest) == split)
                    for split in ("train", "validation", "test")
                },
                "by_compiler": {
                    compiler: sum(1 for manifest in manifests if (manifest.get("build") or {}).get("compiler") == compiler)
                    for compiler in sorted({(manifest.get("build") or {}).get("compiler") for manifest in manifests})
                },
                "by_optimization": {
                    optimization: sum(
                        1 for manifest in manifests if (manifest.get("build") or {}).get("optimization") == optimization
                    )
                    for optimization in sorted({(manifest.get("build") or {}).get("optimization") for manifest in manifests})
                },
                "debug": sum(1 for manifest in manifests if (manifest.get("build") or {}).get("debug_info")),
                "stripped": sum(1 for manifest in manifests if not (manifest.get("build") or {}).get("debug_info")),
                "licenses": sorted({(manifest.get("license") or {}).get("spdx") for manifest in manifests}),
                "license_coverage": round(
                    sum(1 for manifest in manifests if (manifest.get("license") or {}).get("spdx")) / max(1, len(manifests)),
                    4,
                ),
            }
        )
    return {
        "sources": summary,
        "total_manifests": sum(item["manifests"] for item in summary),
        "total_unique_projects": sum(item["unique_projects"] for item in summary),
        "note": "Manifest counts include compiler/optimization variants; unique program counts are reported separately.",
    }
