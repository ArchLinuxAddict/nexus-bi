"""FoundationDataset-compatible view over a frozen sharded sample store."""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Any, Iterator

from nexus.data.sample_store import SampleStore, _LazySamples
from nexus.foundation.config import FoundationConfig
from nexus.foundation.dataset import DatasetSource, apply_modality_dropout
from nexus.foundation.representation import (
    collate_samples,
    mask_byte_batch,
    mask_instruction_batch,
)


class SampleStoreDataset:
    """Same batch contract as ``FoundationDataset`` but lazy per microbatch."""

    def __init__(
        self,
        store: SampleStore,
        split: str,
        config: FoundationConfig,
        *,
        sources: list[DatasetSource] | None = None,
        provenance: dict[str, Any] | None = None,
        vocab: Any | None = None,
        sample_ids: list[str] | None = None,
    ):
        if split not in {"train", "validation", "test", "extended_test"}:
            raise ValueError(f"invalid split: {split}")
        self.store = store
        self.split = split
        self.config = config
        self.sources = sources or []
        self.provenance = provenance or {}
        self.vocab = vocab
        self.sample_ids = list(sample_ids) if sample_ids is not None else list(store.by_split.get(split, []))
        self._samples = _LazySamples(store, split, self.sample_ids)
        self._group_members: dict[str, list[str]] = {}
        for sample_id in self.sample_ids:
            group = str(store.index[sample_id].match_group or "")
            if group:
                self._group_members.setdefault(group, []).append(sample_id)
        self._equivalence_groups = sorted(
            group for group, members in self._group_members.items() if len(members) >= 2
        )

    @property
    def samples(self) -> _LazySamples:
        return self._samples

    def __len__(self) -> int:
        return len(self._samples)

    def _collate(self, sample_ids: list[str]) -> dict[str, Any]:
        return collate_samples([self.store.get(sample_id) for sample_id in sample_ids], self.config)

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
        by_language: dict[str, list[str]] = defaultdict(list)
        for sample_id in self.sample_ids:
            by_language[self.store.index[sample_id].language or "c"].append(sample_id)
        ids: list[str] = []
        for language in sorted(by_language):
            values = list(by_language[language])
            if shuffle:
                random.Random(seed + (0 if language == "c" else 7919)).shuffle(values)
            ids.extend(values)
        if grouped_fraction > 0.0 and self._equivalence_groups:
            batches = (len(ids) + batch_size - 1) // batch_size
            for batch_index in range(batches):
                rng = random.Random(seed + 1_000_003 * batch_index)
                chosen: list[str] = []
                if rng.random() < grouped_fraction:
                    for _ in range(max(1, groups_per_batch)):
                        group = rng.choice(self._equivalence_groups)
                        members = self._group_members[group]
                        language = self.store.index[members[0]].language or "c" if members else "c"
                        pool = by_language.get(language, [])
                        count = min(max(1, members_per_group), len(members))
                        chosen.extend(rng.sample(members, count))
                        while len(chosen) < batch_size and pool:
                            chosen.append(rng.choice(pool))
                if not chosen:
                    language = sorted(by_language)[batch_index % max(1, len(by_language))]
                    chosen = list(by_language.get(language, []))
                while len(chosen) < batch_size:
                    chosen.append(rng.choice(ids))
                chosen = chosen[:batch_size]
                languages = {self.store.index[sample_id].language or "c" for sample_id in chosen}
                if len(languages) > 1:
                    language = self.store.index[chosen[0]].language or "c"
                    chosen = [sample_id for sample_id in chosen if (self.store.index[sample_id].language or "c") == language]
                    while len(chosen) < batch_size:
                        pool = by_language.get(language, [])
                        if not pool:
                            break
                        chosen.append(rng.choice(pool))
                    chosen = chosen[:batch_size]
                if not chosen:
                    continue
                batch = self._collate(chosen)
                if mask_instructions:
                    batch = mask_instruction_batch(batch, probability=mask_probability, seed=seed + batch_index)
                if mask_bytes:
                    batch = mask_byte_batch(batch, probability=byte_mask_probability, seed=seed + batch_index)
                if modality_dropout > 0:
                    batch = apply_modality_dropout(batch, modality_dropout, seed=seed + batch_index)
                yield batch
            return
        order = list(fixed_order) if fixed_order is not None else list(range(len(ids)))
        if shuffle and fixed_order is None:
            pass
        for start in range(0, len(order), batch_size):
            chosen = [ids[index] for index in order[start : start + batch_size] if 0 <= index < len(ids)]
            if not chosen:
                continue
            batch = self._collate(chosen)
            if mask_instructions:
                batch = mask_instruction_batch(batch, probability=mask_probability, seed=seed + start)
            if mask_bytes:
                batch = mask_byte_batch(batch, probability=byte_mask_probability, seed=seed + start)
            if modality_dropout > 0:
                batch = apply_modality_dropout(batch, modality_dropout, seed=seed + start)
            yield batch

    def evaluation_batches(self, batch_size: int = 4) -> Iterator[dict[str, Any]]:
        by_language: dict[str, list[str]] = defaultdict(list)
        for sample_id in self.sample_ids:
            by_language[self.store.index[sample_id].language or "c"].append(sample_id)
        for language in sorted(by_language):
            ids = by_language[language]
            for start in range(0, len(ids), batch_size):
                yield self._collate(ids[start : start + batch_size])

    def purpose_counts(self) -> list[int]:
        counts: dict[int, int] = {}
        for sample_id in self.sample_ids:
            purpose = int(self.store.index[sample_id].purpose_id)
            if purpose >= 0:
                counts[purpose] = counts.get(purpose, 0) + 1
        if not counts:
            return []
        values = [0] * (max(counts) + 1)
        for index, count in counts.items():
            values[index] = count
        return values

    def stats(self) -> dict[str, Any]:
        records = [self.store.index[sample_id] for sample_id in self.sample_ids]
        by_dataset: dict[str, int] = {}
        by_project: dict[str, set[str]] = {}
        by_arch: dict[str, int] = {}
        by_format: dict[str, int] = {}
        by_compiler: dict[str, int] = {}
        by_optimization: dict[str, int] = {}
        for record in records:
            by_dataset[record.dataset_id] = by_dataset.get(record.dataset_id, 0) + 1
            by_project.setdefault(record.dataset_id, set()).add(record.project_id)
            by_arch[record.architecture] = by_arch.get(record.architecture, 0) + 1
            by_format[record.binary_format] = by_format.get(record.binary_format, 0) + 1
            by_compiler[str(record.compiler)] = by_compiler.get(str(record.compiler), 0) + 1
            by_optimization[str(record.optimization)] = by_optimization.get(str(record.optimization), 0) + 1
        return {
            "split": self.split,
            "samples": len(records),
            "by_dataset": by_dataset,
            "unique_projects_by_dataset": {key: len(value) for key, value in by_project.items()},
            "unique_projects_total": len({record.project_id for record in records}),
            "unique_source_functions": len({record.source_function_id for record in records if record.source_function_id}),
            "unique_binaries": len({record.binary_sha256 for record in records if record.binary_sha256}),
            "source_labelled_samples": sum(1 for record in records if record.has_source),
            "truncated_source_targets": sum(1 for record in records if record.source_target_truncated),
            "instruction_tokens": sum(record.instruction_count for record in records),
            "source_tokens": sum(record.source_token_count for record in records),
            "architectures": by_arch,
            "formats": by_format,
            "compilers": by_compiler,
            "optimizations": by_optimization,
            "source_inputs_used": False,
            "symbol_names_used_as_inputs": False,
            "datasets": [source.to_dict() for source in self.sources],
            "provenance": self.provenance,
            "store": {
                "root": str(self.store.root),
                "frozen_id": self.store.frozen_id,
                "vocab_sha256": self.store.manifest.get("vocab_sha256"),
                "index_sha256": self.store.manifest.get("index_sha256"),
                "store_sha256": self.store.manifest.get("store_sha256"),
            },
        }
