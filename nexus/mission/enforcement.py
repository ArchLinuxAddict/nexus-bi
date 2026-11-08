import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from nexus.core.errors import MissionPolicyError
from nexus.core.hashing import sha256_file
from nexus.mission.policy import load_policy, load_registry

APPROVED_DATASET_ROOT = Path(__file__).resolve().parents[2] / "artifacts" / "dataset"


def _json_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dataset_fingerprint(root: str | Path) -> dict[str, str]:
    path = Path(root).resolve()
    manifest_path = path / "manifests.jsonl"
    summary_path = path / "dataset_summary.json"
    if not manifest_path.exists() or not summary_path.exists():
        raise MissionPolicyError(f"dataset is missing manifests or summary: {path}")
    manifest_hash = _json_hash(manifest_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    canonical = summary.get("canonical_id")
    declared = summary.get("dataset_id")
    declared_manifest_hash = summary.get("manifest_sha256")
    if canonical:
        if declared not in {canonical, manifest_hash}:
            raise MissionPolicyError("dataset summary dataset_id does not match its canonical_id")
        if declared_manifest_hash != manifest_hash:
            raise MissionPolicyError("dataset summary manifest_sha256 does not match manifests.jsonl")
    else:
        if declared != manifest_hash:
            raise MissionPolicyError("dataset summary ID does not match the manifest hash")
    return {
        "dataset_id": canonical or declared,
        "manifest_sha256": manifest_hash,
        "summary_sha256": _json_hash(summary_path),
        "root": str(path),
    }


@dataclass(frozen=True)
class DatasetProvenance:
    dataset_id: str
    dataset_hash: str
    dataset_category: str
    dataset_classification: str
    objectives: tuple[str, ...]
    split: str
    sample_ids: tuple[str, ...]
    project_ids: tuple[str, ...]
    manifest_count: int
    root: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _registry_entry(dataset_id: str, root: Path) -> dict[str, Any]:
    registry = load_registry()
    matches = [entry for entry in registry["datasets"] if entry.get("dataset_id") == dataset_id]
    if len(matches) != 1:
        raise MissionPolicyError(f"registry must contain exactly one entry for dataset {dataset_id}")
    entry = matches[0]
    registered_path = Path(entry.get("path", ""))
    registered_absolute = registered_path if registered_path.is_absolute() else Path(__file__).resolve().parents[2] / registered_path
    if registered_absolute.resolve() != root:
        raise MissionPolicyError(f"registry path does not match dataset root for {dataset_id}")
    policy = load_policy()
    category = policy["dataset_categories"].get(entry.get("category"))
    if category is None:
        raise MissionPolicyError(f"unclassified dataset category: {entry.get('category')}")
    if entry.get("classification") != category.get("classification"):
        raise MissionPolicyError(f"registry classification conflicts with policy for {dataset_id}")
    if entry.get("classification") in {"PROHIBITED", "IRRELEVANT"}:
        raise MissionPolicyError(f"registry dataset is {entry.get('classification')}")
    if entry.get("category") == "synthetic_infrastructure_smoke":
        return entry
    try:
        root.relative_to(APPROVED_DATASET_ROOT.resolve())
    except ValueError as exc:
        raise MissionPolicyError(f"dataset path is outside approved roots: {root}") from exc
    return entry


def _validate_manifest(manifest: dict[str, Any], root: Path, registry_entry: dict[str, Any], verify_artifacts: bool) -> None:
    sample_id = manifest.get("sample_id", "<unknown>")
    if not manifest.get("provenance", {}).get("source_archive_sha256"):
        raise MissionPolicyError(f"sample {sample_id} has no source archive provenance")
    if not manifest.get("license", {}).get("spdx"):
        raise MissionPolicyError(f"sample {sample_id} has no license identifier")
    mission = manifest.get("mission_relevance")
    category = registry_entry["category"]
    classification = registry_entry["classification"]
    if mission:
        if mission.get("category") != category:
            raise MissionPolicyError(f"sample {sample_id} mission category conflicts with registry")
        if mission.get("classification") != classification:
            raise MissionPolicyError(f"sample {sample_id} mission classification conflicts with registry")
        if mission.get("classification") in {"PROHIBITED", "IRRELEVANT"}:
            raise MissionPolicyError(f"sample {sample_id} is {mission.get('classification')}")
        if not mission.get("allowed_for_training", False):
            raise MissionPolicyError(f"sample {sample_id} is not allowed for training")
    if verify_artifacts:
        for artifact in manifest.get("artifacts", []):
            target = root / artifact["path"]
            if not target.exists():
                raise MissionPolicyError(f"sample {sample_id} artifact is missing: {artifact['path']}")
            if sha256_file(target) != artifact["sha256"]:
                raise MissionPolicyError(f"sample {sample_id} artifact hash mismatch: {artifact['path']}")
    quality = manifest.get("quality", {})
    if quality.get("validation_status") == "rejected":
        raise MissionPolicyError(f"sample {sample_id} was rejected during validation")


def validate_dataset_for_training(
    root: str | Path,
    split: str,
    objectives: list[str] | tuple[str, ...],
    *,
    verify_artifacts: bool = True,
    max_samples: int | None = None,
) -> DatasetProvenance:
    path = Path(root).resolve()
    if split not in {"train", "validation"}:
        raise MissionPolicyError(f"split {split!r} is evaluation-only and cannot enter training")
    fingerprint = dataset_fingerprint(path)
    entry = _registry_entry(fingerprint["dataset_id"], path)
    policy = load_policy()
    category = policy["dataset_categories"].get(entry.get("category"))
    if category is None:
        raise MissionPolicyError(f"unclassified dataset category: {entry.get('category')}")
    if category.get("classification") in {"PROHIBITED", "IRRELEVANT"} or not category.get("allowed_for_training"):
        raise MissionPolicyError(f"dataset category is not trainable: {entry.get('category')}")
    objectives = list(objectives)
    unknown = [objective for objective in objectives if objective not in policy["objectives"]]
    if unknown:
        raise MissionPolicyError(f"unclassified objectives: {unknown}")
    manifests = [json.loads(line) for line in (path / "manifests.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    selected = [manifest for manifest in manifests if manifest.get("split") == split]
    if max_samples is not None:
        selected = selected[:max_samples]
    if not selected:
        raise MissionPolicyError(f"dataset has no samples in split {split}")
    for manifest in selected:
        _validate_manifest(manifest, path, entry, verify_artifacts)
    sample_ids = tuple(manifest["sample_id"] for manifest in selected)
    project_ids = tuple(sorted({manifest["project_id"] for manifest in selected}))
    return DatasetProvenance(
        dataset_id=fingerprint["dataset_id"],
        dataset_hash=fingerprint["manifest_sha256"],
        dataset_category=entry["category"],
        dataset_classification=entry["classification"],
        objectives=tuple(objectives),
        split=split,
        sample_ids=sample_ids,
        project_ids=project_ids,
        manifest_count=len(selected),
        root=str(path),
    )


def validate_batch_provenance(provenance: dict[str, Any], mission) -> DatasetProvenance:
    if not provenance:
        raise MissionPolicyError("training batch has no mission provenance")
    dataset_id = provenance.get("dataset_id")
    dataset_hash = provenance.get("dataset_hash")
    split = provenance.get("split")
    if not dataset_id or not dataset_hash:
        raise MissionPolicyError("training batch provenance is incomplete")
    if split != "train":
        raise MissionPolicyError(f"batch split {split!r} is not a training split")
    root = provenance.get("root")
    if not root:
        raise MissionPolicyError("batch provenance has no dataset root")
    if dataset_id == "synthetic-smoke":
        if provenance.get("dataset_category") != "synthetic_infrastructure_smoke":
            raise MissionPolicyError("synthetic batch has an invalid category")
        if mission.run_mode != "smoke":
            raise MissionPolicyError("synthetic infrastructure data cannot enter a non-smoke run")
        return DatasetProvenance(
            dataset_id=dataset_id,
            dataset_hash=dataset_hash,
            dataset_category="synthetic_infrastructure_smoke",
            dataset_classification="EXPERIMENTAL",
            objectives=tuple(provenance.get("objectives", [])),
            split="train",
            sample_ids=tuple(provenance.get("sample_ids", ["synthetic-batch"])),
            project_ids=("synthetic",),
            manifest_count=1,
            root="synthetic",
        )
    fingerprint = dataset_fingerprint(root)
    if fingerprint["dataset_id"] != dataset_id or fingerprint["manifest_sha256"] != dataset_hash:
        raise MissionPolicyError("training batch provenance is stale: dataset changed after validation")
    entry = _registry_entry(dataset_id, Path(root).resolve())
    if provenance.get("dataset_category") != entry.get("category"):
        raise MissionPolicyError("batch category does not match registry")
    if entry.get("classification") in {"PROHIBITED", "IRRELEVANT"}:
        raise MissionPolicyError("batch dataset is prohibited")
    if mission.dataset_id and mission.dataset_id != dataset_id:
        raise MissionPolicyError("batch dataset does not match training mission")
    if mission.dataset_category != provenance.get("dataset_category"):
        raise MissionPolicyError("batch category does not match training mission")
    if not provenance.get("sample_ids"):
        raise MissionPolicyError("training batch has no sample identities")
    allowed_samples = {
        manifest["sample_id"]
        for manifest in (json.loads(line) for line in (Path(root) / "manifests.jsonl").read_text(encoding="utf-8").splitlines() if line.strip())
        if manifest.get("split") == "train"
    }
    unknown_samples = [sample_id for sample_id in provenance.get("sample_ids", []) if sample_id not in allowed_samples]
    if unknown_samples:
        raise MissionPolicyError(f"training batch contains samples outside the train split: {unknown_samples[:5]}")
    obj = list(provenance.get("objectives", []))
    if not set(obj) & set(mission.objectives):
        raise MissionPolicyError("batch objectives do not match training mission")
    return DatasetProvenance(
        dataset_id=dataset_id,
        dataset_hash=dataset_hash,
        dataset_category=provenance["dataset_category"],
        dataset_classification=provenance.get("dataset_classification", entry.get("classification")),
        objectives=tuple(obj),
        split=split,
        sample_ids=tuple(provenance.get("sample_ids", [])),
        project_ids=tuple(provenance.get("project_ids", [])),
        manifest_count=len(provenance.get("sample_ids", [])),
        root=root,
    )


def synthetic_provenance() -> dict[str, Any]:
    return {
        "dataset_id": "synthetic-smoke",
        "dataset_hash": "synthetic",
        "dataset_category": "synthetic_infrastructure_smoke",
        "dataset_classification": "EXPERIMENTAL",
        "objectives": ["infrastructure_smoke"],
        "split": "train",
        "sample_ids": ["synthetic-batch"],
        "project_ids": ["synthetic"],
        "manifest_count": 1,
        "root": "synthetic",
    }


def batch_provenance_dict(provenance: DatasetProvenance, sample_ids: list[str] | None = None) -> dict[str, Any]:
    return {
        "dataset_id": provenance.dataset_id,
        "dataset_hash": provenance.dataset_hash,
        "dataset_category": provenance.dataset_category,
        "dataset_classification": provenance.dataset_classification,
        "objectives": list(provenance.objectives),
        "split": provenance.split,
        "sample_ids": sample_ids or list(provenance.sample_ids),
        "project_ids": list(provenance.project_ids),
        "manifest_count": provenance.manifest_count,
        "root": provenance.root,
    }
