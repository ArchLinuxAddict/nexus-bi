"""Frozen-dataset gate and loader for the standalone trainer."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from nexus.data.sample_store import SampleStore
from nexus.data.validate_v2 import validate_split_v2
from nexus.data.validate_v3 import validate_corpus_v3
from nexus.foundation.config import FoundationConfig, config_for_vocabulary
from nexus.foundation.dataset import DatasetSource, FoundationDataset, build_dataset, dataset_source_from_registry
from nexus.foundation.store_dataset import SampleStoreDataset
from nexus.foundation.vocab import Vocabulary
from nexus_train.dataset_freeze import (
    DEFAULT_FREEZE_ROOT,
    SPLITS,
    load_frozen_vocab,
    validate_frozen_dataset,
)

ROOT = Path(__file__).resolve().parents[1]
DERIVED_RECORDS = ROOT / "artifacts" / "training_runtime" / "derived_records"


@dataclass
class FrozenDatasetGate:
    root: Path
    manifest: dict[str, Any]
    validation: dict[str, Any]
    assignments: dict[str, str]
    manifest_assignments: dict[str, str]
    splits: dict[str, list[dict[str, Any]]]
    vocab: Vocabulary
    store: SampleStore | None = None

    @property
    def frozen_id(self) -> str:
        return str(self.manifest.get("frozen_id"))

    def identity(self) -> dict[str, Any]:
        if self.manifest.get("schema_version") == "nexus.corpus-v3/v1":
            return {
                "schema_version": "nexus.corpus-v3/v1",
                "frozen_id": self.frozen_id,
                "manifest_sha256": hashlib.sha256((self.root / "manifest.json").read_bytes()).hexdigest(),
                "counts": self.manifest.get("counts"),
                "vocab_sha256": self.manifest.get("vocab_sha256"),
                "protected_test_file_sha256": self.manifest.get("protected_test_file_sha256"),
                "extended_test_file_sha256": self.manifest.get("extended_test_file_sha256"),
                "store_sha256": self.manifest.get("store", {}).get("store_sha256"),
                "index_sha256": self.manifest.get("store", {}).get("index_sha256"),
                "store_manifest_sha256": self.manifest.get("store", {}).get("manifest_sha256"),
                "source_fingerprints": (self.store.manifest.get("source_fingerprints") if self.store is not None else None),
                "rule": self.manifest.get("split_rule"),
            }
        return {
            "schema_version": self.manifest.get("schema_version", "nexus.frozen-split/v1"),
            "frozen_id": self.frozen_id,
            "manifest_sha256": self.validation.get("manifest_sha256"),
            "counts": self.manifest.get("counts"),
            "vocab_sha256": self.manifest.get("vocab_sha256"),
            "protected_test_file_sha256": self.manifest.get("protected_test_file_sha256"),
            "source_manifests": self.manifest.get("source_manifests") or self.manifest.get("source_datasets"),
            "rule": self.manifest.get("rule"),
        }

    def require_clean(self) -> None:
        if self.validation.get("status") != "PASS":
            raise ValueError(f"frozen dataset gate failed: {self.validation.get('problems')}")
        counts = self.manifest.get("counts") or {}
        if not set(SPLITS).issubset(set(counts)):
            raise ValueError("frozen split set is incomplete")
        if self.manifest.get("schema_version") == "nexus.corpus-v3/v1" and "extended_test" not in counts:
            raise ValueError("corpus v3 extended test split is missing")
        if int(counts.get("test", 0)) <= 0:
            raise ValueError("frozen test split is empty")
        if self.manifest.get("schema_version") == "nexus.corpus-v3/v1":
            if self.validation.get("status") != "PASS":
                raise ValueError(f"corpus v3 validation failed: {self.validation.get('problems')}")
            if self.manifest.get("protected_test_file_sha256") != self.validation.get("protected_test_file_sha256"):
                raise ValueError("protected test manifest hash changed")
            if self.store is None:
                raise ValueError("corpus v3 store is not loaded")
            if self.store.manifest.get("vocab_sha256") != self.vocab.digest():
                raise ValueError("corpus v3 store vocabulary digest mismatch")
            if self.store.manifest.get("store_sha256") != self.manifest.get("store", {}).get("store_sha256"):
                raise ValueError("corpus v3 store digest mismatch")
            return
        if self.manifest.get("schema_version") == "nexus.split-v2/v1":
            if self.manifest.get("protected_test_file_sha256") != self.validation.get("protected_test_file_sha256"):
                raise ValueError("protected test manifest hash changed")

    def build_split(
        self,
        split: str,
        config: FoundationConfig,
        max_functions: int = 8,
        allowed_datasets: set[str] | None = None,
    ) -> FoundationDataset:
        if split not in {"train", "validation", "test", "extended_test"}:
            raise ValueError(f"invalid split: {split}")
        self.require_clean()
        if split == "test" and not self.manifest.get("counts", {}).get("test"):
            raise ValueError("test split is empty")
        if self.manifest.get("schema_version") == "nexus.corpus-v3/v1":
            if self.store is None:
                raise ValueError("corpus v3 store is not loaded")
            source = DatasetSource(
                path=str(self.root),
                dataset_id=str(self.frozen_id),
                category="real_world_aligned_source",
                classification="CORE",
                sha256=str(self.manifest.get("store", {}).get("store_sha256") or ""),
                max_manifests=1,
                max_functions=1,
                include_source=True,
            )
            sample_ids = None
            if allowed_datasets is not None:
                sample_ids = [
                    sample_id
                    for sample_id in self.store.by_split.get(split, [])
                    if self.store.index[sample_id].dataset_id in allowed_datasets
                ]
                if not sample_ids:
                    raise ValueError(f"dataset filter produced an empty {split} split for {sorted(allowed_datasets)}")
            return SampleStoreDataset(
                self.store,
                split,
                config_for_vocabulary(config, self.vocab),
                sources=[source],
                provenance={"corpus": self.identity(), "split": split, "dataset_filter": sorted(allowed_datasets) if allowed_datasets else None},
                vocab=self.vocab,
                sample_ids=sample_ids,
            )
        roots = self.manifest.get("dataset_roots") or {}
        if not roots:
            roots = {dataset_id: dataset_id for dataset_id in self.manifest.get("source_datasets", [])}
        sources = []
        for dataset_id in sorted(roots):
            source = dataset_source_from_registry(
                dataset_id,
                max_manifests=100000,
                max_functions=max_functions,
                derived_records_root=str(DERIVED_RECORDS),
            )
            sources.append(source)
        dataset = build_dataset(
            sources,
            self.vocab,
            config,
            split=split,
            include_source=True,
            verify_artifacts=True,
            objectives=["binary_representation_masking", "c_reconstruction", "function_purpose_inference", "basic_block_correspondence", "instruction_semantics", "type_dataflow_inference"],
            strict_source_input_isolation=True,
            split_assignments=self.manifest_assignments if self.manifest.get("schema_version") == "nexus.split-v2/v1" else self.assignments,
        )
        if self.manifest.get("schema_version") == "nexus.split-v2/v1":
            dataset.samples = [sample for sample in dataset.samples if self.assignments.get(str(sample.get("sample_id"))) == split]
        dataset.provenance["frozen_dataset"] = self.identity()
        dataset.provenance["frozen_split"] = split
        return dataset

    def build_all(self, config: FoundationConfig, max_functions: int = 8) -> dict[str, FoundationDataset]:
        return {split: self.build_split(split, config, max_functions=max_functions) for split in ("train", "validation", "test")}


def load_gate(root: str | Path = DEFAULT_FREEZE_ROOT, *, revalidate: bool = True) -> FrozenDatasetGate:
    base = Path(root).resolve()
    if not base.exists():
        raise FileNotFoundError(f"frozen dataset is missing: {base}; run 'python -m nexus_train freeze-data'")
    manifest = json.loads((base / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version") == "nexus.corpus-v3/v1":
        validation = validate_corpus_v3(base) if revalidate else {"status": "UNVALIDATED", "manifest_sha256": None, "problems": []}
    elif manifest.get("schema_version") == "nexus.split-v2/v1":
        validation = validate_split_v2(base) if revalidate else {"status": "UNVALIDATED", "manifest_sha256": None, "problems": []}
    else:
        validation = validate_frozen_dataset(base) if revalidate else {"status": "UNVALIDATED", "manifest_sha256": None}
    if validation["status"] != "PASS":
        raise ValueError(f"frozen dataset validation failed: {validation.get('problems')}")
    if manifest.get("schema_version") == "nexus.corpus-v3/v1":
        store = SampleStore.load(base / "samples")
        vocab = Vocabulary.load(base / "vocab.json")
        assignments = {sample_id: split for split in SPLITS for sample_id in store.by_split.get(split, [])}
        splits = {split: [store.index[sample_id].to_dict() for sample_id in store.by_split.get(split, [])] for split in SPLITS}
        return FrozenDatasetGate(base, manifest, validation, assignments, {}, splits, vocab, store)
    splits = {}
    assignments = {}
    manifest_assignments: dict[str, str] = {}
    for split in SPLITS:
        path = base / f"{split}.jsonl"
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        splits[split] = rows
        for row in rows:
            sample_id = str(row.get("sample_id"))
            if sample_id in assignments:
                raise ValueError(f"duplicate frozen sample id: {sample_id}")
            assignments[sample_id] = split
            manifest_id = str((row.get("provenance") or {}).get("manifest_sample_id") or "")
            if manifest_id and split != "quarantine":
                existing = manifest_assignments.get(manifest_id)
                if existing is not None and existing != split:
                    raise ValueError(f"manifest {manifest_id} spans frozen splits: {existing} vs {split}")
                manifest_assignments[manifest_id] = split
    if manifest.get("schema_version") == "nexus.split-v2/v1":
        vocab = Vocabulary.load(base / "vocab.json")
    else:
        vocab = load_frozen_vocab(base)
    return FrozenDatasetGate(base, manifest, validation, assignments, manifest_assignments, splits, vocab)


def gate_status(root: str | Path = DEFAULT_FREEZE_ROOT) -> dict[str, Any]:
    try:
        gate = load_gate(root)
        return {"status": "PASS", "frozen_dir": str(gate.root), "identity": gate.identity(), "validation": gate.validation}
    except Exception as exc:
        return {"status": "FAIL", "frozen_dir": str(Path(root).resolve()), "error": str(exc)}
