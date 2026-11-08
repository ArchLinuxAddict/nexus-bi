"""Deterministic, auditable objective-coverage scheduler.

The scheduler yields batches selected to satisfy per-objective coverage quotas
over a fixed window.  It never trains and never invents samples.  For every
selected objective it records the attempt, the actual active labels reported by
the trainer, skipped batches and skip reasons.  A quota that cannot be met is
preserved as an explicit coverage obligation.
"""

from __future__ import annotations

import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterator

from nexus.foundation.dataset import apply_modality_dropout
from nexus.foundation.representation import collate_samples, mask_byte_batch, mask_instruction_batch

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_POLICY = ROOT / "artifacts" / "objectives" / "objective_sampling_policy.json"
DEFAULT_POOL_CACHE = ROOT / "artifacts" / "objectives" / "sample_pools.json"
def _json_hash(payload: Any) -> str:
    import hashlib

    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


class ScheduledDataset:
    """FoundationDataset-compatible wrapper that yields scheduler batches."""

    def __init__(self, base: Any, scheduler: "ObjectiveScheduler"):
        self.base = base
        self.scheduler = scheduler

    def __getattr__(self, name: str) -> Any:
        return getattr(self.base, name)

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int | slice) -> Any:
        return self.base[index]

    def batches(self, batch_size: int = 4, **kwargs: Any) -> Iterator[dict[str, Any]]:
        yield from self.scheduler.iter_batches(batch_size=batch_size)

    def evaluation_batches(self, *args: Any, **kwargs: Any) -> Iterator[dict[str, Any]]:
        return self.base.evaluation_batches(*args, **kwargs)

    def stats(self) -> dict[str, Any]:
        return self.base.stats()

    def purpose_counts(self) -> list[int]:
        return self.base.purpose_counts()


class ObjectiveScheduler:
    def __init__(
        self,
        dataset: Any,
        config: Any,
        policy: dict[str, Any] | None = None,
        *,
        enabled_objectives: list[str] | None = None,
        batch_size: int = 4,
        seed: int = 20261103,
        window_steps: int | None = None,
        pool_cache: Path | None = None,
    ):
        self.dataset = dataset
        self.config = config
        self.store = dataset.store
        self.policy = policy or json.loads(DEFAULT_POLICY.read_text(encoding="utf-8"))
        self.policy_hash = _json_hash(self.policy)
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.window_steps = int(window_steps or self.policy.get("coverage_window_steps", 200))
        self.train_ids = list(self.store.by_split.get("train", []))
        self.index = self.store.index
        self.enabled = list(enabled_objectives or self.policy.get("enabled_objectives", []))
        self.quotas: dict[str, int] = {}
        self.target_fraction: dict[str, float] = {}
        self.selectors: dict[str, str] = {}
        for objective_id in self.enabled:
            entry = (self.policy.get("objectives") or {}).get(objective_id)
            if entry is None:
                raise ValueError(f"enabled objective has no sampling policy: {objective_id}")
            if not entry.get("enabled", True):
                raise ValueError(f"enabled objective is disabled in the sampling policy: {objective_id}")
            target = float(entry.get("target_fraction", 0.0))
            if target <= 0:
                raise ValueError(f"enabled objective has no positive sampling quota: {objective_id}")
            self.target_fraction[objective_id] = target
            self.quotas[objective_id] = int(entry.get("minimum_sampling_quota") or math.ceil(target * self.window_steps))
            self.selectors[objective_id] = str(entry.get("selector") or "any")
        self.pool_cache = Path(pool_cache or DEFAULT_POOL_CACHE)
        self.pools: dict[str, list[str]] = {}
        self.group_members: dict[str, list[str]] = {}
        self.groups: list[str] = []
        self._load_or_build_pools()
        self.active_steps: dict[str, list[int]] = defaultdict(list)
        self.selected: Counter[str] = Counter()
        self.scheduled_success: Counter[str] = Counter()
        self.scheduled_failure: Counter[str] = Counter()
        self.skipped: Counter[str] = Counter()
        self.skip_reasons: dict[str, Counter[str]] = defaultdict(Counter)
        self.valid_examples: Counter[str] = Counter()
        self.token_counts: Counter[str] = Counter()
        self.loss_sum: dict[str, float] = defaultdict(float)
        self.last_success_step: dict[str, int | None] = {}
        self.last_skipped_step: dict[str, int | None] = {}
        self.last_observed_step = -1
        self.scheduled_by_step: dict[int, str | None] = {}
        self.skips_by_step: dict[int, list[dict[str, str]]] = {}
        self.last_batch_meta: dict[str, Any] = {}
        self.last_observed_scheduled: str | None = None
        self.last_observed_skips: list[dict[str, str]] = []

    def _load_or_build_pools(self) -> None:
        groups: dict[str, list[str]] = defaultdict(list)
        for sample_id in self.train_ids:
            record = self.index[sample_id]
            if record.match_group:
                groups[record.match_group].append(sample_id)
        self.group_members = dict(groups)
        self.groups = sorted(group for group, members in groups.items() if len(members) >= 2)
        source = [sample_id for sample_id in self.train_ids if self.index[sample_id].has_source]
        source_nontruncated = [
            sample_id
            for sample_id in source
            if not self.index[sample_id].source_target_truncated
        ]
        self.pools = {
            "any": list(self.train_ids),
            "source": source,
            "source_nontruncated": source_nontruncated,
            "equivalence": list(self.groups),
        }
        cache_valid = False
        if self.pool_cache.exists():
            try:
                cached = json.loads(self.pool_cache.read_text(encoding="utf-8"))
                if cached.get("store_sha256") == self.store.manifest.get("store_sha256"):
                    self.pools.update({key: list(value) for key, value in (cached.get("pools") or {}).items()})
                    cache_valid = True
            except (OSError, json.JSONDecodeError):
                cache_valid = False
        if not cache_valid:
            register: list[str] = []
            memory: list[str] = []
            cfg: list[str] = []
            ir: list[str] = []
            boundary: list[str] = []
            for sample_id in self.train_ids:
                sample = self.store.get(sample_id)
                read_targets = sample.get("read_targets") or []
                write_targets = sample.get("write_targets") or []
                if any(any(value for value in row) for row in read_targets) or any(any(value for value in row) for row in write_targets):
                    register.append(sample_id)
                if any(sample.get("instr_mem_read") or []) or any(sample.get("instr_mem_write") or []):
                    memory.append(sample_id)
                block_mask = sample.get("block_mask") or []
                cfg_target = sample.get("cfg_target") or []
                if sum(1 for value in block_mask if value) >= 2 and any(any(value > 0 for value in row) for row in cfg_target):
                    cfg.append(sample_id)
                if any(sample.get("instr_ir") or []):
                    ir.append(sample_id)
                if any(value > 0 for value in (sample.get("function_start_target") or [])):
                    boundary.append(sample_id)
            self.pools.update({"register_heavy": register, "memory_heavy": memory, "cfg_heavy": cfg, "ir_available": ir, "byte_boundary": boundary})
            self.pool_cache.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "schema_version": "nexus.sample-pools/v1",
                "store_sha256": self.store.manifest.get("store_sha256"),
                "counts": {key: len(value) for key, value in self.pools.items()},
                "pools": {key: value for key, value in self.pools.items() if key not in {"any", "equivalence"}},
            }
            temporary = self.pool_cache.with_suffix(".tmp")
            temporary.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
            temporary.replace(self.pool_cache)

    def _due(self, step: int) -> list[tuple[float, str]]:
        window_start = (step // self.window_steps) * self.window_steps
        window_end = window_start + self.window_steps
        deficits: dict[str, int] = {}
        for objective_id in self.enabled:
            required = self.quotas[objective_id]
            actual = sum(1 for observed in self.active_steps.get(objective_id, []) if window_start <= observed < window_end)
            if required > actual:
                deficits[objective_id] = required - actual
        priority = [objective_id for objective_id in self.policy.get("priority", []) if objective_id in deficits]
        rest = sorted(objective_id for objective_id in deficits if objective_id not in priority)
        return [(float(deficits[objective_id]), objective_id) for objective_id in priority + rest]

    def _make_batch(self, selector: str, step: int, batch_size: int) -> dict[str, Any]:
        rng = random.Random(self.seed + step * 1000003)
        chosen: list[str] = []
        if selector == "equivalence" and self.groups:
            group = rng.choice(self.groups)
            members = self.group_members[group]
            take = min(max(2, min(len(members), batch_size)), len(members))
            chosen.extend(rng.sample(members, take))
        else:
            pool = self.pools.get(selector) or self.pools["any"]
            if not pool:
                raise ValueError(f"empty sample pool for selector: {selector}")
            take = min(batch_size, len(pool))
            chosen.extend(rng.sample(pool, take))
        seen = set(chosen)
        attempts = 0
        while len(chosen) < batch_size and attempts < batch_size * 8:
            candidate = rng.choice(self.train_ids)
            attempts += 1
            if candidate not in seen:
                seen.add(candidate)
                chosen.append(candidate)
        samples = [self.store.get(sample_id) for sample_id in chosen[:batch_size]]
        batch = collate_samples(samples, self.config)
        batch = mask_instruction_batch(batch, probability=0.15, seed=self.seed + step)
        batch = mask_byte_batch(batch, probability=0.08, seed=self.seed + step)
        batch = apply_modality_dropout(batch, 0.02, seed=self.seed + step)
        instr_tokens = sum(len(sample.get("instr_id") or []) for sample in samples)
        byte_tokens = sum(len(sample.get("byte_ids") or []) for sample in samples)
        source_tokens = sum(len(sample.get("c_ids") or []) for sample in samples)
        boundary = sum(1 for sample in samples for value in (sample.get("function_start_target") or []) if value > 0)
        groups = {str(sample.get("match_group") or "") for sample in samples if sample.get("match_group")}
        self.last_batch_meta = {
            "selector": selector,
            "sample_ids": chosen[:batch_size],
            "instr_tokens": instr_tokens,
            "byte_tokens": byte_tokens,
            "source_tokens": source_tokens,
            "function_start_positives": boundary,
            "groups": len(groups),
        }
        return batch

    def _token_for(self, objective_id: str) -> int:
        if objective_id in {"byte_masked_modeling", "function_start_prediction"}:
            return int(self.last_batch_meta.get("byte_tokens", 0))
        if objective_id in {"c_reconstruction", "semantic_equivalence", "binary_source_alignment"}:
            return int(self.last_batch_meta.get("source_tokens", 0)) or int(self.last_batch_meta.get("instr_tokens", 0))
        return int(self.last_batch_meta.get("instr_tokens", 0))

    def iter_batches(self, batch_size: int | None = None) -> Iterator[dict[str, Any]]:
        step = 0
        batch_size = int(batch_size or self.batch_size)
        while True:
            due = self._due(step)
            selected: str | None = None
            skipped: list[dict[str, str]] = []
            batch = None
            for _deficit, objective_id in due:
                selector = self.selectors[objective_id]
                if selector != "equivalence" and not self.pools.get(selector):
                    skipped.append({"objective_id": objective_id, "reason": f"empty selector pool: {selector}"})
                    continue
                if selector == "equivalence" and not self.groups:
                    skipped.append({"objective_id": objective_id, "reason": "no same-source variant groups"})
                    continue
                batch = self._make_batch(selector, step, batch_size)
                selected = objective_id
                break
            if batch is None:
                selector = "equivalence" if self.groups and random.Random(self.seed + step).random() < 0.5 else "any"
                batch = self._make_batch(selector, step, batch_size)
                selected = None
            batch = {**batch, "run_instruction_autoregressive": selected == "instruction_autoregressive"}
            self.scheduled_by_step[step] = selected
            self.skips_by_step[step] = skipped
            yield batch
            step += 1

    def observe(self, record: dict[str, Any]) -> None:
        step = int(record.get("step", self.last_observed_step + 1))
        self.last_observed_step = max(self.last_observed_step, step)
        scheduled = self.scheduled_by_step.pop(step, None)
        skipped = self.skips_by_step.pop(step, [])
        self.last_observed_scheduled = scheduled
        self.last_observed_skips = list(skipped)
        entries = {str(item.get("objective_id")): item for item in (record.get("objectives") or [])}
        for objective_id in self.enabled:
            entry = entries.get(objective_id)
            if entry is None:
                self.skipped[objective_id] += 1
                self.skip_reasons[objective_id]["not_logged"] += 1
                self.last_skipped_step[objective_id] = step
                continue
            if entry.get("active") and int(entry.get("count", 0)) > 0:
                self.active_steps[objective_id].append(step)
                self.valid_examples[objective_id] += int(entry.get("count", 0))
                self.token_counts[objective_id] += self._token_for(objective_id)
                self.loss_sum[objective_id] += float(entry.get("raw", 0.0))
                self.last_success_step[objective_id] = step
            else:
                self.skipped[objective_id] += 1
                reason = str(entry.get("note") or "inactive/no valid labels")
                self.skip_reasons[objective_id][reason] += 1
                self.last_skipped_step[objective_id] = step
        if scheduled is not None:
            self.selected[scheduled] += 1
            entry = entries.get(scheduled)
            if entry is not None and entry.get("active") and int(entry.get("count", 0)) > 0:
                self.scheduled_success[scheduled] += 1
            else:
                self.scheduled_failure[scheduled] += 1
                self.skip_reasons[scheduled]["scheduled_batch_inactive"] += 1
        for item in skipped:
            objective_id = str(item.get("objective_id"))
            self.skipped[objective_id] += 1
            self.skip_reasons[objective_id][str(item.get("reason") or "unavailable")] += 1
            self.last_skipped_step[objective_id] = step
        # Bound historical step lists to four windows.
        cutoff = max(0, step - self.window_steps * 4)
        for objective_id, values in self.active_steps.items():
            if values and values[0] < cutoff:
                self.active_steps[objective_id] = [value for value in values if value >= cutoff]

    def coverage_report(self, step: int | None = None) -> dict[str, Any]:
        observed = int(step if step is not None else self.last_observed_step)
        window_start = (max(0, observed) // self.window_steps) * self.window_steps
        window_end = window_start + self.window_steps
        objectives: dict[str, Any] = {}
        blocking: list[dict[str, Any]] = []
        for objective_id in self.enabled:
            active = sum(1 for value in self.active_steps.get(objective_id, []) if window_start <= value < window_end)
            required = self.quotas[objective_id]
            passed = active >= required
            entry = {
                "status": "PASS" if passed else "DEFICIT",
                "target_fraction": self.target_fraction[objective_id],
                "minimum_sampling_quota": required,
                "active_batches_in_window": active,
                "coverage_of_quota": round(active / max(1, required), 4),
                "deficit": max(0, required - active),
                "selected_batches": self.selected[objective_id],
                "scheduled_success": self.scheduled_success[objective_id],
                "scheduled_failure": self.scheduled_failure[objective_id],
                "skipped_batches": self.skipped[objective_id],
                "skip_reasons": dict(self.skip_reasons[objective_id]),
                "valid_examples": self.valid_examples[objective_id],
                "tokens": self.token_counts[objective_id],
                "mean_loss": round(self.loss_sum[objective_id] / max(1, active), 6) if active else None,
                "last_successful_training_step": self.last_success_step.get(objective_id),
                "last_skipped_step": self.last_skipped_step.get(objective_id),
                "last_successful_validation_step": None,
                "selector": self.selectors[objective_id],
            }
            objectives[objective_id] = entry
            if not passed:
                blocking.append({"objective_id": objective_id, "deficit": entry["deficit"], "required": required, "active": active, "reasons": entry["skip_reasons"]})
        return {
            "schema_version": "nexus.objective-coverage/v1",
            "observed_step": observed,
            "window_start": window_start,
            "window_end": window_end,
            "window_steps": self.window_steps,
            "complete": not blocking,
            "blocking_deficits": blocking,
            "objectives": objectives,
            "skipped_total": sum(self.skipped.values()),
            "note": "Inactive/skipped objectives are explicit; no objective was silently removed or replaced.",
        }

    def enrich_record(self, record: dict[str, Any]) -> dict[str, Any]:
        enriched = dict(record)
        step = int(record.get("step", self.last_observed_step))
        enriched["scheduled_objective"] = self.last_observed_scheduled
        enriched["scheduled_skips"] = list(self.last_observed_skips)
        enriched["batch_token_counts"] = dict(self.last_batch_meta)
        enriched["objective_coverage"] = self.coverage_report(step)
        enriched["gradient_contribution"] = record.get("gradient_norms") or {}
        enriched["effective_loss_weights"] = {
            str(item.get("objective_id")): item.get("weight") for item in (record.get("objectives") or [])
        }
        return enriched

    def state_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "nexus.scheduler-state/v1",
            "policy_hash": self.policy_hash,
            "store_sha256": self.store.manifest.get("store_sha256"),
            "window_steps": self.window_steps,
            "last_observed_step": self.last_observed_step,
            "active_steps": {key: list(value) for key, value in self.active_steps.items()},
            "selected": dict(self.selected),
            "scheduled_success": dict(self.scheduled_success),
            "scheduled_failure": dict(self.scheduled_failure),
            "skipped": dict(self.skipped),
            "skip_reasons": {key: dict(value) for key, value in self.skip_reasons.items()},
            "valid_examples": dict(self.valid_examples),
            "token_counts": dict(self.token_counts),
            "loss_sum": dict(self.loss_sum),
            "last_success_step": self.last_success_step,
            "last_skipped_step": self.last_skipped_step,
        }

    def load_state_dict(self, payload: dict[str, Any]) -> None:
        if payload.get("store_sha256") != self.store.manifest.get("store_sha256"):
            raise ValueError("scheduler state store fingerprint mismatch")
        if payload.get("policy_hash") != self.policy_hash:
            raise ValueError("scheduler state policy fingerprint mismatch")
        self.last_observed_step = int(payload.get("last_observed_step", -1))
        self.active_steps = defaultdict(list, {key: list(value) for key, value in (payload.get("active_steps") or {}).items()})
        self.selected = Counter(payload.get("selected") or {})
        self.scheduled_success = Counter(payload.get("scheduled_success") or {})
        self.scheduled_failure = Counter(payload.get("scheduled_failure") or {})
        self.skipped = Counter(payload.get("skipped") or {})
        self.skip_reasons = defaultdict(Counter, {key: Counter(value) for key, value in (payload.get("skip_reasons") or {}).items()})
        self.valid_examples = Counter(payload.get("valid_examples") or {})
        self.token_counts = Counter(payload.get("token_counts") or {})
        self.loss_sum = defaultdict(float, {key: float(value) for key, value in (payload.get("loss_sum") or {}).items()})
        self.last_success_step = {key: (int(value) if value is not None else None) for key, value in (payload.get("last_success_step") or {}).items()}
        self.last_skipped_step = {key: (int(value) if value is not None else None) for key, value in (payload.get("last_skipped_step") or {}).items()}

    def save_state(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.state_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")

    def pool_summary(self) -> dict[str, Any]:
        return {
            "counts": {key: len(value) for key, value in self.pools.items()},
            "groups": len(self.groups),
            "enabled_objectives": self.enabled,
            "quotas": self.quotas,
            "selectors": self.selectors,
            "window_steps": self.window_steps,
            "pool_cache": str(self.pool_cache),
        }
