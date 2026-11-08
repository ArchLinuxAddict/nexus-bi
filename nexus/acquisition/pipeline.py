import json
import shutil
import tarfile
import tempfile
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from nexus.acquisition.adapters import AdapterError, BaseAdapter, HttpClient, RateLimited, adapter_for
from nexus.acquisition.dedup import compare_against_index, signature_from_tree, source_file_hashes
from nexus.acquisition.licensing import detect_license, license_report
from nexus.acquisition.models import ACQUISITION_SCHEMA_VERSION, AcquisitionRecord, SourceCandidate
from nexus.acquisition.provenance import hash_tree, provenance_confidence, verify_swh
from nexus.core.hashing import sha256_file

DEFAULT_MAX_FILES = 20000
DEFAULT_MAX_EXTRACT_BYTES = 1_000_000_000
DEFAULT_MAX_ARCHIVE_BYTES = 80 * 1024 * 1024
MARKER_NAME = ".nexus_acquisition.json"


class AcquisitionError(Exception):
    pass


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(item, sort_keys=True) + "\n" for item in records), encoding="utf-8")


def _tree_bytes(root: Path) -> int:
    total = 0
    for path in root.rglob("*"):
        if path.is_file() and not path.is_symlink():
            total += path.stat().st_size
    return total


def safe_extract_tar(
    archive: str | Path,
    destination: str | Path,
    max_files: int = DEFAULT_MAX_FILES,
    max_bytes: int = DEFAULT_MAX_EXTRACT_BYTES,
) -> dict[str, Any]:
    source = Path(archive)
    target = Path(destination)
    target.mkdir(parents=True, exist_ok=True)
    files = 0
    total = 0
    skipped_links = 0
    skipped_special = 0
    with tarfile.open(source, "r:*") as handle:
        for member in handle:
            name = member.name
            if name.startswith("/") or ".." in Path(name).parts:
                raise AcquisitionError(f"unsafe archive member: {name}")
            if member.issym() or member.islnk():
                skipped_links += 1
                continue
            if member.isdev() or member.isfifo():
                skipped_special += 1
                continue
            if not member.isfile() and not member.isdir():
                skipped_special += 1
                continue
            if files >= max_files:
                raise AcquisitionError(f"archive exceeds {max_files} file entries")
            if total + member.size > max_bytes:
                raise AcquisitionError(f"archive exceeds {max_bytes} extracted bytes")
            destination_path = target / name
            try:
                destination_path.resolve().relative_to(target.resolve())
            except ValueError as exc:
                raise AcquisitionError(f"unsafe archive member: {name}") from exc
            if member.isdir():
                destination_path.mkdir(parents=True, exist_ok=True)
                continue
            destination_path.parent.mkdir(parents=True, exist_ok=True)
            source_file = handle.extractfile(member)
            if source_file is None:
                continue
            with open(destination_path, "wb") as output:
                shutil.copyfileobj(source_file, output)
            files += 1
            total += member.size
    return {"files": files, "bytes": total, "skipped_links": skipped_links, "skipped_special": skipped_special}


def _source_root(extracted: Path) -> Path:
    children = [item for item in extracted.iterdir() if item.is_dir()]
    files = [item for item in extracted.iterdir() if item.is_file()]
    if len(children) == 1 and not files:
        return children[0]
    return extracted


@dataclass
class AcquisitionState:
    path: Path
    payload: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.path.exists():
            try:
                self.payload = json.loads(self.path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                self.payload = {}
        self.payload.setdefault("schema_version", ACQUISITION_SCHEMA_VERSION)
        self.payload.setdefault("projects", {})
        self.payload.setdefault("created_at", _now())
        self.payload.setdefault("updated_at", _now())
        self.payload.setdefault("attempts", 0)

    def get(self, project_id: str) -> dict[str, Any]:
        return self.payload["projects"].setdefault(project_id, {"stages": {}, "status": "pending"})

    def set_stage(self, project_id: str, stage: str, status: str, **details: Any) -> None:
        record = self.get(project_id)
        record["stages"][stage] = {"status": status, "at": _now(), **details}
        record["status"] = status if status in {"quarantined", "failed", "accepted", "skipped"} else record.get("status", "pending")
        self.save()

    def save(self) -> None:
        self.payload["updated_at"] = _now()
        self.payload["attempts"] = int(self.payload.get("attempts", 0)) + 1
        _atomic_json(self.path, self.payload)


def _prepare_workspace(out_root: Path, pilot_id: str) -> None:
    marker = out_root / MARKER_NAME
    if out_root.exists() and any(out_root.iterdir()):
        if not marker.exists():
            raise AcquisitionError(f"refusing to write into non-empty directory without acquisition marker: {out_root}")
        try:
            existing = json.loads(marker.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise AcquisitionError(f"unreadable acquisition marker: {marker}") from exc
        if existing.get("pilot_id") != pilot_id:
            raise AcquisitionError(f"workspace belongs to pilot {existing.get('pilot_id')}, not {pilot_id}")
    out_root.mkdir(parents=True, exist_ok=True)
    _atomic_json(marker, {"schema_version": ACQUISITION_SCHEMA_VERSION, "pilot_id": pilot_id, "created_at": _now()})


def _candidate_from_dict(item: dict[str, Any]) -> SourceCandidate:
    known = {field for field in SourceCandidate.__dataclass_fields__}
    payload = {key: value for key, value in item.items() if key in known}
    payload.setdefault("project_id", str(item.get("repository") or item.get("package") or "unknown").replace("/", "__").lower())
    payload.setdefault("adapter", "github")
    payload.setdefault("repository", item.get("package") or payload["project_id"])
    payload.setdefault("canonical_url", "")
    payload.setdefault("category", "other")
    if "build_recipe" in item:
        payload["recipe"] = item["build_recipe"]
    if "package" in item or "suite" in item:
        payload.setdefault("extra", {})
        for key in ("package", "suite", "version"):
            if key in item:
                payload["extra"][key] = item[key]
    for key in ("archive_path", "license_ignore"):
        if key in item:
            payload.setdefault("extra", {})[key] = item[key]
    return SourceCandidate(**payload)


def _resolve_and_fetch(candidate: SourceCandidate, adapter: BaseAdapter, project_dir: Path, state: AcquisitionState, timeout: int) -> tuple[Any, dict[str, Any]]:
    resolved = adapter.resolve(candidate)
    state.set_stage(candidate.project_id, "resolve", "ok", commit=resolved.commit, warnings=resolved.warnings)
    archive = project_dir / "source.tar.gz"
    expected = state.get(candidate.project_id).get("stages", {}).get("fetch", {}).get("sha256")
    if archive.exists() and expected and sha256_file(archive) == expected:
        record = {"url": state.get(candidate.project_id)["stages"]["fetch"].get("url"), "path": str(archive), "sha256": expected, "size": archive.stat().st_size, "resumed": True}
    elif archive.exists() and not expected:
        digest = sha256_file(archive)
        record = {"url": None, "path": str(archive), "sha256": digest, "size": archive.stat().st_size, "resumed": True}
    else:
        fetched = adapter.fetch(resolved, archive)
        record = fetched.to_dict()
    state.set_stage(candidate.project_id, "fetch", "ok", sha256=record["sha256"], size=record["size"], url=record.get("url"))
    return resolved, record


def acquire_candidate(
    candidate: SourceCandidate,
    *,
    out_root: str | Path,
    state: AcquisitionState,
    client: HttpClient | None = None,
    historical_index: dict[str, Any] | None = None,
    max_archive_bytes: int = DEFAULT_MAX_ARCHIVE_BYTES,
    max_files: int = DEFAULT_MAX_FILES,
    max_extract_bytes: int = DEFAULT_MAX_EXTRACT_BYTES,
    swh_check: bool = True,
    timeout: int = 90,
) -> AcquisitionRecord:
    root = Path(out_root)
    project_dir = root / "corpus" / candidate.project_id
    project_dir.mkdir(parents=True, exist_ok=True)
    started = _now()
    record = AcquisitionRecord(project_id=candidate.project_id, candidate=candidate.to_dict(), status="pending", stage="start", started_at=started)
    state.set_stage(candidate.project_id, "start", "pending")
    try:
        adapter = adapter_for(candidate.adapter, client=client, cache_dir=root / ".cache")
        if not adapter.fetch_supported:
            raise AdapterError(f"adapter {candidate.adapter} does not support archive fetch")
        resolved, archive = _resolve_and_fetch(candidate, adapter, project_dir, state, timeout)
        record.resolved = resolved.to_dict()
        record.archive = archive
        if archive["size"] > max_archive_bytes:
            raise AcquisitionError(f"archive exceeds {max_archive_bytes} bytes")
        record.stage = "fetched"
        extracted = project_dir / "source"
        inventory_path = project_dir / "inventory.json"
        expected_inventory = state.get(candidate.project_id).get("stages", {}).get("extract", {}).get("tree_sha256")
        if extracted.exists() and expected_inventory and inventory_path.exists():
            inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
            if inventory.get("tree_sha256") != expected_inventory:
                shutil.rmtree(extracted)
        if not extracted.exists() or not (inventory_path.exists() and json.loads(inventory_path.read_text(encoding="utf-8")).get("tree_sha256")):
            temporary = Path(tempfile.mkdtemp(prefix=f"{candidate.project_id}-", dir=root / "corpus"))
            try:
                safe_extract_tar(project_dir / "source.tar.gz", temporary, max_files=max_files, max_bytes=max_extract_bytes)
                source_root = _source_root(temporary)
                if extracted.exists():
                    shutil.rmtree(extracted)
                shutil.move(str(source_root), str(extracted))
            finally:
                shutil.rmtree(temporary, ignore_errors=True)
            inventory = hash_tree(extracted)
            _atomic_json(inventory_path, inventory)
        else:
            inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
        record.inventory = {key: value for key, value in inventory.items() if key != "files"}
        record.stage = "extracted"
        state.set_stage(candidate.project_id, "extract", "ok", tree_sha256=inventory["tree_sha256"], files=inventory["file_count"], bytes=inventory["total_bytes"])
        license_evidence = detect_license(extracted, expected=candidate.expected_license, ignore_paths=candidate.extra.get("license_ignore") or [])
        record.license = license_evidence.to_dict()
        state.set_stage(candidate.project_id, "license", license_evidence.status, spdx=license_evidence.spdx, expression=license_evidence.expression, reason=license_evidence.reason)
        record.stage = "licensed"
        file_hashes = source_file_hashes(project_dir / "source.tar.gz")
        signature = signature_from_tree(extracted)
        if historical_index is not None:
            dedup = compare_against_index(
                archive_sha256=archive["sha256"],
                signature=signature,
                file_hashes=file_hashes,
                index=historical_index,
            )
        else:
            dedup = {"status": "not_checked", "reason": "no historical index", "max_similarity": 0.0, "matches": []}
        record.deduplication = dedup
        state.set_stage(candidate.project_id, "deduplicate", dedup["status"], max_similarity=dedup.get("max_similarity"), reason=dedup.get("reason"))
        record.stage = "deduplicated"
        swh_status, swh_revision = ("not_checked", None)
        if swh_check and candidate.adapter in {"github", "gitlab"}:
            swh_status, swh_revision = verify_swh(resolved.canonical_url, resolved.commit, client or adapter.client)
        confidence, reasons = provenance_confidence(
            archive_sha256=archive["sha256"],
            commit=resolved.commit,
            commit_verified=not any("unavailable" in item for item in resolved.warnings) and not any("not declared" in item for item in resolved.warnings),
            license_sha256=license_evidence.sha256,
            inventory_sha256=inventory["tree_sha256"],
            provider=resolved.provider,
            swh_status=swh_status,
            warnings=resolved.warnings,
        )
        record.provenance = {
            "source_url": resolved.canonical_url,
            "provider": resolved.provider,
            "repository": resolved.repository,
            "commit": resolved.commit,
            "retrieved_at": _now(),
            "archive_sha256": archive["sha256"],
            "archive_size": archive["size"],
            "license_file_sha256": license_evidence.sha256,
            "file_inventory_sha256": inventory["tree_sha256"],
            "file_count": inventory["file_count"],
            "source_bytes": inventory["total_bytes"],
            "confidence": confidence,
            "confidence_reasons": reasons,
            "commit_verified": not any("unavailable" in item for item in resolved.warnings),
            "commit_verification_method": "git-ls-remote" if not any("unavailable" in item for item in resolved.warnings) else "declared-only",
            "swh_revision": swh_revision,
            "swh_status": swh_status,
            "warnings": resolved.warnings,
        }
        state.set_stage(candidate.project_id, "provenance", "ok", confidence=confidence, swh_status=swh_status)
        record.stage = "provenanced"
        reasons_block = []
        if license_evidence.status != "accepted":
            reasons_block.append(f"license: {license_evidence.reason}")
        if dedup.get("status") in {"duplicate", "near_duplicate"}:
            reasons_block.append(f"deduplication: {dedup.get('reason')}")
        if confidence == "unknown":
            reasons_block.append("provenance confidence is unknown")
        if reasons_block:
            record.status = "quarantined"
            record.quarantine_reason = "; ".join(reasons_block)
            state.set_stage(candidate.project_id, "complete", "quarantined", reason=record.quarantine_reason)
        else:
            record.status = "accepted"
            state.set_stage(candidate.project_id, "complete", "accepted", license=license_evidence.spdx or license_evidence.expression, confidence=confidence)
        record.storage_bytes = (project_dir / "source.tar.gz").stat().st_size + _tree_bytes(project_dir / "source")
        record.finished_at = _now()
        return record
    except RateLimited as exc:
        record.status = "skipped"
        record.stage = "rate_limited"
        record.errors.append(str(exc))
        state.set_stage(candidate.project_id, "complete", "skipped", reason=str(exc))
    except (AdapterError, AcquisitionError, OSError, ValueError, tarfile.TarError) as exc:
        record.status = "failed"
        record.stage = "failed"
        record.errors.append(f"{type(exc).__name__}: {exc}")
        state.set_stage(candidate.project_id, "complete", "failed", reason=str(exc))
    record.finished_at = _now()
    return record


def run_pilot(
    candidates_path: str | Path,
    out_root: str | Path,
    reports_root: str | Path,
    *,
    max_projects: int = 30,
    max_download_bytes: int = 1_500_000_000,
    max_storage_bytes: int = 4_000_000_000,
    resume: bool = True,
    historical_dataset: str | Path | None = "artifacts/dataset/realworld_v6_pilot2",
    historical_index_path: str | Path | None = "artifacts/acquisition/historical_corpus_index.json",
    swh_check: bool = True,
    limit: int | None = None,
) -> dict[str, Any]:
    root = Path(out_root)
    reports = Path(reports_root)
    payload = json.loads(Path(candidates_path).read_text(encoding="utf-8"))
    pilot_id = payload.get("pilot_id") or "acquisition_projects_v1"
    _prepare_workspace(root, pilot_id)
    state = AcquisitionState(root / ".state" / "acquisition_state.json")
    historical_index = None
    if historical_dataset and Path(historical_dataset).exists() and historical_index_path:
        from nexus.acquisition.dedup import load_or_build_index

        historical_index = load_or_build_index(historical_dataset, historical_index_path)
    protocol = payload.get("protocol", {})
    projects = payload.get("projects", [])
    if limit is not None:
        projects = projects[:limit]
    records = []
    total_download = 0
    total_storage = 0
    for item in projects:
        candidate = _candidate_from_dict(item)
        existing = state.get(candidate.project_id)
        previous_status = existing.get("status")
        if resume and previous_status in {"accepted", "quarantined"}:
            record_path = root / "corpus" / candidate.project_id / "acquisition_record.json"
            if record_path.exists():
                record = json.loads(record_path.read_text(encoding="utf-8"))
                records.append(record)
                total_download += int((record.get("archive") or {}).get("size") or 0)
                total_storage += int(record.get("storage_bytes") or 0)
                continue
        if sum(1 for record in records if record.get("status") == "accepted") >= max_projects:
            break
        if total_download >= max_download_bytes or total_storage >= max_storage_bytes:
            break
        record = acquire_candidate(
            candidate,
            out_root=root,
            state=state,
            historical_index=historical_index,
            swh_check=swh_check,
            max_archive_bytes=min(DEFAULT_MAX_ARCHIVE_BYTES, max(1, max_download_bytes - total_download)),
        )
        records.append(record.to_dict())
        total_download += int((record.archive or {}).get("size") or 0)
        total_storage += int(record.storage_bytes or 0)
        _atomic_json(root / "corpus" / candidate.project_id / "acquisition_record.json", record.to_dict())
    accepted = [record for record in records if record.get("status") == "accepted"]
    quarantined = [record for record in records if record.get("status") == "quarantined"]
    failed = [record for record in records if record.get("status") == "failed"]
    skipped = [record for record in records if record.get("status") == "skipped"]
    by_category: dict[str, int] = {}
    by_provider: dict[str, int] = {}
    for record in accepted:
        by_category[record["candidate"]["category"]] = by_category.get(record["candidate"]["category"], 0) + 1
        by_provider[record["candidate"]["adapter"]] = by_provider.get(record["candidate"]["adapter"], 0) + 1
    storage_used = _tree_bytes(root)
    catalog = []
    for record in records:
        candidate = record["candidate"]
        catalog.append(
            {
                "project_id": record["project_id"],
                "name": candidate["repository"],
                "repository": (record.get("resolved") or {}).get("canonical_url"),
                "category": candidate["category"],
                "status": record["status"],
                "commit": (record.get("resolved") or {}).get("commit"),
                "license": (record.get("license") or {}).get("spdx") or (record.get("license") or {}).get("expression"),
                "license_file_sha256": (record.get("license") or {}).get("sha256"),
                "source_archive_sha256": (record.get("archive") or {}).get("sha256"),
                "source_archive_path": f"corpus/{record['project_id']}/source.tar.gz",
                "source_root": f"corpus/{record['project_id']}/source",
                "source_files": (record.get("inventory") or {}).get("classification", {}).get("c_source", 0),
                "source_bytes": (record.get("inventory") or {}).get("total_bytes", 0),
                "inventory_tree_sha256": (record.get("inventory") or {}).get("tree_sha256"),
                "provenance_confidence": (record.get("provenance") or {}).get("confidence"),
                "deduplication": (record.get("deduplication") or {}).get("status"),
                "storage_bytes": record.get("storage_bytes", 0),
                "build_recipe": candidate.get("recipe", {}),
            }
        )
    reports.mkdir(parents=True, exist_ok=True)
    _atomic_json(reports / "acquisition_manifest.json", {
        "schema_version": ACQUISITION_SCHEMA_VERSION,
        "pilot_id": pilot_id,
        "created_at": _now(),
        "policy": protocol,
        "counts": {"candidates": len(projects), "processed": len(records), "accepted": len(accepted), "quarantined": len(quarantined), "failed": len(failed), "skipped": len(skipped)},
        "by_category": dict(sorted(by_category.items())),
        "by_provider": dict(sorted(by_provider.items())),
        "download_bytes": total_download,
        "storage_bytes": storage_used,
        "records": records,
    })
    _atomic_json(root / "catalog.json", catalog)
    _jsonl(root / "quarantine.jsonl", [record for record in records if record.get("status") == "quarantined"])
    _atomic_json(reports / "license_report.json", {
        "schema_version": ACQUISITION_SCHEMA_VERSION,
        "created_at": _now(),
        **license_report(records),
        "records": [
            {
                "project_id": record["project_id"],
                "repository": record["candidate"]["repository"],
                "category": record["candidate"]["category"],
                "license": record.get("license"),
            }
            for record in records
        ],
    })
    _atomic_json(reports / "provenance_report.json", {
        "schema_version": ACQUISITION_SCHEMA_VERSION,
        "created_at": _now(),
        "records": [
            {
                "project_id": record["project_id"],
                "repository": record["candidate"]["repository"],
                "status": record["status"],
                "provenance": record.get("provenance"),
                "archive": record.get("archive"),
                "inventory": record.get("inventory"),
            }
            for record in records
        ],
    })
    _atomic_json(reports / "deduplication_report.json", {
        "schema_version": ACQUISITION_SCHEMA_VERSION,
        "created_at": _now(),
        "historical_dataset": str(historical_dataset) if historical_dataset else None,
        "near_threshold": 0.80,
        "watch_threshold": 0.60,
        "records": [
            {
                "project_id": record["project_id"],
                "repository": record["candidate"]["repository"],
                "status": record.get("deduplication", {}).get("status"),
                "max_similarity": record.get("deduplication", {}).get("max_similarity"),
                "matches": record.get("deduplication", {}).get("matches", []),
            }
            for record in records
        ],
    })
    _atomic_json(reports / "source_discovery_report.json", {
        "schema_version": ACQUISITION_SCHEMA_VERSION,
        "created_at": _now(),
        "pilot_id": pilot_id,
        "adapters": sorted({record["candidate"]["adapter"] for record in records}),
        "adapter_inventory": {
            "github": {"status": "used", "resolution": "git ls-remote", "fetch": "codeload tarball"},
            "gitlab": {"status": "implemented", "resolution": "git ls-remote", "fetch": "GitLab archive endpoint"},
            "softwareheritage": {"status": "used_for_corroboration", "fetch": "vault not enabled"},
            "debian": {"status": "implemented_not_exercised_in_pilot", "resolution": "stable Sources.xz with SHA-256"},
            "local": {"status": "tested", "purpose": "controlled ingestion and resumability tests"},
        },
        "candidates": len(projects),
        "resolved": sum(1 for record in records if record.get("resolved")),
        "fetched": sum(1 for record in records if record.get("archive")),
        "accepted": len(accepted),
        "quarantined": len(quarantined),
        "failed": len(failed),
        "skipped": len(skipped),
        "by_category": dict(sorted(by_category.items())),
        "by_provider": dict(sorted(by_provider.items())),
        "rate_limit_state": {
            "rest_api_calls": 0,
            "provider_requests": "git ls-remote for refs and codeload for archives",
            "note": "GitHub REST search rate limits were not consumed during the pinned pilot; discovery adapters honor returned rate-limit headers.",
        },
        "softwareheritage_corroboration": {
            "corroborated": sum(1 for record in records if (record.get("provenance") or {}).get("swh_status") == "corroborated"),
            "not_found": sum(1 for record in records if (record.get("provenance") or {}).get("swh_status") == "not_found"),
            "unavailable": sum(1 for record in records if (record.get("provenance") or {}).get("swh_status") == "unavailable"),
        },
        "access_notes": [
            "GitHub resolution uses git ls-remote, not the REST API; archive download uses codeload.",
            "GitHub/GitLab discovery adapters use provider search APIs and honor returned rate-limit headers.",
            "Software Heritage is used for provenance corroboration; vault downloads are not enabled.",
            "Debian source acquisition uses the stable Sources.xz index and SHA-256 checksums.",
            "Publicly accessible does not mean unrestricted training permission; every project is license-gated.",
        ],
        "records": [
            {
                "project_id": record["project_id"],
                "repository": record["candidate"]["repository"],
                "adapter": record["candidate"]["adapter"],
                "category": record["candidate"]["category"],
                "status": record["status"],
                "commit": (record.get("resolved") or {}).get("commit"),
            }
            for record in records
        ],
    })
    summary = {
        "schema_version": ACQUISITION_SCHEMA_VERSION,
        "pilot_id": pilot_id,
        "created_at": _now(),
        "candidates": len(projects),
        "processed": len(records),
        "accepted": len(accepted),
        "quarantined": len(quarantined),
        "failed": len(failed),
        "skipped": len(skipped),
        "by_category": dict(sorted(by_category.items())),
        "download_bytes": total_download,
        "storage_bytes": storage_used,
        "state_path": str(state.path),
        "resumable": True,
    }
    _atomic_json(root / "acquisition_summary.json", summary)
    return summary
