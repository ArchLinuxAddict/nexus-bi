import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from nexus.core.errors import MissionPolicyError

ROOT = Path(__file__).resolve().parents[2]
POLICY_PATH = ROOT / "configs" / "mission_policy.json"
REGISTRY_PATH = ROOT / "configs" / "dataset_registry.json"
ALLOWED_CLASSIFICATIONS = {"CORE", "SUPPORTING", "EXPERIMENTAL", "IRRELEVANT", "PROHIBITED"}


def load_policy(path: str | Path | None = None) -> dict[str, Any]:
    target = Path(path) if path else POLICY_PATH
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise MissionPolicyError(f"unable to load mission policy: {exc}") from exc
    for name in ("capabilities", "objectives", "dataset_categories", "training_rules"):
        if name not in payload:
            raise MissionPolicyError(f"mission policy is missing {name}")
    return payload


def load_registry(path: str | Path | None = None) -> dict[str, Any]:
    target = Path(path) if path else REGISTRY_PATH
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise MissionPolicyError(f"unable to load dataset registry: {exc}") from exc
    if "datasets" not in payload:
        raise MissionPolicyError("dataset registry is missing datasets")
    return payload


@dataclass(frozen=True)
class MissionContext:
    dataset_id: str | None
    dataset_category: str
    dataset_classification: str
    objectives: tuple[str, ...]
    capabilities: tuple[str, ...]
    run_mode: str
    rationale: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _category(policy: dict[str, Any], name: str) -> dict[str, Any]:
    category = policy["dataset_categories"].get(name)
    if not category:
        raise MissionPolicyError(f"unclassified dataset category: {name}")
    classification = category.get("classification")
    if classification not in ALLOWED_CLASSIFICATIONS:
        raise MissionPolicyError(f"invalid classification for dataset category {name}: {classification}")
    return category


def _objective(policy: dict[str, Any], name: str) -> dict[str, Any]:
    objective = policy["objectives"].get(name)
    if not objective:
        raise MissionPolicyError(f"unclassified training objective: {name}")
    return objective


def mission_for_dataset(path: str | Path | None = None, dataset_id: str | None = None) -> dict[str, Any]:
    registry = load_registry()
    resolved = Path(path).resolve() if path else None
    matches = []
    for entry in registry["datasets"]:
        aliases = set(entry.get("aliases") or [])
        if dataset_id and (entry.get("dataset_id") == dataset_id or dataset_id in aliases) or resolved and entry.get("path") and (ROOT / entry["path"]).resolve() == resolved:
            matches.append(entry)
    if len(matches) > 1:
        raise MissionPolicyError(f"conflicting mission registry entries for {path or dataset_id}")
    if not matches:
        raise MissionPolicyError(f"dataset is not registered in the mission registry: {path or dataset_id}")
    return matches[0]


def validate_training_run(
    objectives: list[str] | tuple[str, ...],
    dataset_categories: list[str] | tuple[str, ...],
    dataset_ids: list[str] | tuple[str, ...] | None = None,
    run_mode: str = "pilot",
    experimental_justification: str | None = None,
) -> MissionContext:
    policy = load_policy()
    rules = policy["training_rules"]
    if rules.get("require_objectives") and not objectives:
        raise MissionPolicyError("training run must declare mission objectives")
    if rules.get("require_dataset_categories") and not dataset_categories:
        raise MissionPolicyError("training run must declare dataset categories")
    objective_records = [_objective(policy, name) for name in objectives]
    category_records = [_category(policy, name) for name in dataset_categories]
    for name, category in zip(dataset_categories, category_records):
        if category.get("classification") in set(rules.get("prohibited_classifications", [])):
            raise MissionPolicyError(f"dataset category {name} is prohibited for training")
        if not category.get("allowed_for_training"):
            raise MissionPolicyError(f"dataset category {name} is not allowed for training")
    for name, objective in zip(objectives, objective_records):
        if objective.get("status") == "infrastructure_only" and run_mode not in {"smoke"}:
            raise MissionPolicyError(f"objective {name} is infrastructure-only and cannot run in {run_mode} mode")
    if run_mode == "main":
        core_capabilities = {
            capability
            for category in policy["dataset_categories"].values()
            if category.get("classification") == "CORE"
            for capability in category.get("capabilities", [])
        }
        objective_capabilities = {capability for objective in objective_records for capability in objective.get("capabilities", [])}
        if rules.get("require_core_objective") and not (objective_capabilities & core_capabilities):
            raise MissionPolicyError("main training requires at least one CORE binary-intelligence objective")
        if rules.get("require_core_dataset_for_main") and not any(category.get("classification") == "CORE" for category in category_records):
            raise MissionPolicyError("main training requires at least one CORE dataset category")
        if rules.get("experimental_requires_justification_for_main"):
            if any(category.get("classification") == "EXPERIMENTAL" for category in category_records) and not experimental_justification:
                raise MissionPolicyError("experimental dataset categories require an explicit justification for main training")
    if dataset_ids:
        registry = load_registry()
        registered = {entry.get("dataset_id"): entry for entry in registry["datasets"]}
        for dataset_id in dataset_ids:
            entry = registered.get(dataset_id)
            if entry is None:
                raise MissionPolicyError(f"dataset is not registered: {dataset_id}")
            category = _category(policy, entry["category"])
            if category.get("classification") in set(rules.get("prohibited_classifications", [])):
                raise MissionPolicyError(f"registered dataset {dataset_id} is prohibited")
            if rules.get("reject_unregistered_datasets") and entry.get("category") not in dataset_categories:
                raise MissionPolicyError(f"dataset {dataset_id} category is not declared in the training run")
    capabilities = sorted({capability for objective in objective_records for capability in objective.get("capabilities", [])})
    classications = [category.get("classification") for category in category_records]
    dataset_id = dataset_ids[0] if dataset_ids else None
    return MissionContext(
        dataset_id=dataset_id,
        dataset_category=dataset_categories[0],
        dataset_classification=classications[0],
        objectives=tuple(objectives),
        capabilities=tuple(capabilities),
        run_mode=run_mode,
        rationale=experimental_justification or "declared mission objective and dataset category",
    )


def audit_registry() -> dict[str, Any]:
    policy = load_policy()
    registry = load_registry()
    issues = []
    seen_ids = set()
    seen_paths = set()
    approved_root = (ROOT / "artifacts" / "dataset").resolve()
    for entry in registry["datasets"]:
        dataset_id = entry.get("dataset_id")
        if dataset_id in seen_ids:
            issues.append(f"duplicate dataset ID: {dataset_id}")
        seen_ids.add(dataset_id)
        category_name = entry.get("category")
        if category_name not in policy["dataset_categories"]:
            issues.append(f"unknown category: {category_name}")
            continue
        category = policy["dataset_categories"][category_name]
        if entry.get("classification") != category.get("classification"):
            issues.append(f"classification mismatch for {dataset_id}")
        if category.get("license_required") and not entry.get("license_verified"):
            issues.append(f"license not verified for {dataset_id}")
        stored_path = entry.get("path")
        if stored_path and stored_path != "synthetic":
            actual = (ROOT / stored_path).resolve()
            if stored_path in seen_paths:
                issues.append(f"duplicate dataset path: {stored_path}")
            seen_paths.add(stored_path)
            if not actual.exists():
                issues.append(f"dataset path missing: {stored_path}")
            try:
                actual.relative_to(approved_root)
            except ValueError:
                issues.append(f"dataset path outside approved root: {stored_path}")
            summary_path = actual / "dataset_summary.json"
            manifest_path = actual / "manifests.jsonl"
            if summary_path.exists() and manifest_path.exists():
                import hashlib
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
                manifest_hash = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
                canonical = summary.get("canonical_id")
                summary_id = canonical or summary.get("dataset_id")
                if canonical:
                    if summary.get("manifest_sha256") != manifest_hash:
                        issues.append(f"summary/manifest hash mismatch for {dataset_id}")
                    if summary.get("dataset_id") not in {canonical, manifest_hash}:
                        issues.append(f"summary dataset_id does not match canonical_id for {dataset_id}")
                else:
                    if summary.get("dataset_id") != manifest_hash:
                        issues.append(f"summary/manifest hash mismatch for {dataset_id}")
                if dataset_id != summary_id:
                    issues.append(f"registry dataset ID does not match summary for {dataset_id}")
        for objective in entry.get("objectives", []):
            if objective not in policy["objectives"]:
                issues.append(f"unknown objective in registry: {objective}")
    return {
        "status": "passed" if not issues else "failed",
        "datasets": len(registry["datasets"]),
        "categories": len(policy["dataset_categories"]),
        "objectives": len(policy["objectives"]),
        "capabilities": len(policy["capabilities"]),
        "issues": issues,
    }


def manifest_mission(category_name: str, objectives: list[str], rationale: str) -> dict[str, Any]:
    policy = load_policy()
    category = _category(policy, category_name)
    for objective in objectives:
        _objective(policy, objective)
    return {
        "schema_version": "0.1.0",
        "category": category_name,
        "classification": category.get("classification"),
        "capabilities": category.get("capabilities", []),
        "objectives": objectives,
        "rationale": rationale,
        "license_required": category.get("license_required", True),
        "allowed_for_training": category.get("allowed_for_training", False),
    }


def register_dataset(
    dataset_id: str,
    path: str | Path,
    category_name: str,
    objectives: list[str],
    rationale: str,
    license_verified: bool,
    capabilities: list[str] | None = None,
    registry_path: str | Path | None = None,
    replace_path: bool = False,
) -> dict[str, Any]:
    policy = load_policy()
    category = _category(policy, category_name)
    for objective in objectives:
        _objective(policy, objective)
    registry_target = Path(registry_path) if registry_path else REGISTRY_PATH
    try:
        registry = json.loads(registry_target.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        registry = {"schema_version": "0.1.0", "datasets": []}
    candidate = Path(path).resolve()
    try:
        stored_path = str(candidate.relative_to(ROOT))
    except ValueError:
        stored_path = str(candidate)
    entry = {
        "dataset_id": dataset_id,
        "path": stored_path,
        "category": category_name,
        "classification": category.get("classification"),
        "capabilities": capabilities or category.get("capabilities", []),
        "objectives": objectives,
        "license_verified": license_verified,
        "rationale": rationale,
    }
    for existing in registry.get("datasets", []):
        if existing.get("dataset_id") == dataset_id and existing.get("path") != stored_path:
            raise MissionPolicyError(f"dataset ID {dataset_id} is already registered with a different path")
        if existing.get("path") == stored_path and existing.get("dataset_id") != dataset_id and not replace_path:
            raise MissionPolicyError(f"dataset path {stored_path} is already registered with a different ID")
    datasets = [
        item
        for item in registry.get("datasets", [])
        if item.get("dataset_id") != dataset_id and not (replace_path and item.get("path") == stored_path)
    ]
    datasets.append(entry)
    registry["datasets"] = datasets
    registry_target.parent.mkdir(parents=True, exist_ok=True)
    temporary = registry_target.with_suffix(".tmp")
    temporary.write_text(json.dumps(registry, indent=2) + "\n", encoding="utf-8")
    temporary.replace(registry_target)
    return entry
