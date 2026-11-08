import json
import shutil
import sqlite3
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from nexus.core.config import NexusPaths, default_paths
from nexus.core.errors import NotFoundError
from nexus.core.hashing import sha256_file


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _detect_format(path: Path, original_name: str | None = None) -> str:
    suffix = Path(original_name).suffix.lower() if original_name else path.suffix.lower()
    try:
        magic = path.read_bytes()[:8]
    except OSError:
        magic = b""
    if magic.startswith(b"\x7fELF"):
        return "ELF"
    if magic.startswith(b"dex\n"):
        return "DEX"
    if magic.startswith(b"\xca\xfe\xba\xbe"):
        return "CLASS"
    if magic.startswith(b"PK\x03\x04"):
        if suffix == ".apk":
            return "APK"
        if suffix in {".jar", ".zip"}:
            return "JAR"
        return "ZIP"
    if suffix in {".apk", ".jar", ".dex", ".class"}:
        return suffix.lstrip(".").upper()
    return "UNKNOWN"


class Storage:
    def __init__(self, paths: NexusPaths | None = None):
        self.paths = (paths or default_paths()).ensure()
        self.db_path = self.paths.database
        self.cas = self.paths.cas
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
        return connection

    def _init_schema(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS binaries (
                    id TEXT PRIMARY KEY,
                    sha256 TEXT UNIQUE NOT NULL,
                    path TEXT NOT NULL,
                    size INTEGER NOT NULL,
                    format TEXT,
                    machine TEXT,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS analyses (
                    id TEXT PRIMARY KEY,
                    binary_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    summary_json TEXT NOT NULL,
                    result_path TEXT,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS reconstructions (
                    id TEXT PRIMARY KEY,
                    analysis_id TEXT NOT NULL,
                    language TEXT NOT NULL,
                    status TEXT NOT NULL,
                    output_dir TEXT,
                    summary_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS verifications (
                    id TEXT PRIMARY KEY,
                    reconstruction_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    result_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS jobs (
                    id TEXT PRIMARY KEY,
                    kind TEXT NOT NULL,
                    status TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    result_json TEXT,
                    error TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS datasets (
                    id TEXT PRIMARY KEY,
                    path TEXT NOT NULL,
                    summary_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS models (
                    id TEXT PRIMARY KEY,
                    path TEXT NOT NULL,
                    sha256 TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS experiments (
                    id TEXT PRIMARY KEY,
                    path TEXT,
                    summary_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS findings (
                    id TEXT PRIMARY KEY,
                    binary_id TEXT NOT NULL,
                    finding_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS investigations (
                    id TEXT PRIMARY KEY,
                    binary_id TEXT NOT NULL,
                    objective TEXT NOT NULL,
                    status TEXT NOT NULL,
                    summary_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                """
            )

    def put_cas(self, path: str | Path) -> tuple[str, Path]:
        source = Path(path)
        digest = sha256_file(source)
        target = self.cas / digest[:2] / digest
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            shutil.copy2(source, target)
        return digest, target

    def add_binary(self, path: str | Path, original_name: str | None = None) -> dict[str, Any]:
        source = Path(path)
        digest, target = self.put_cas(source)
        binary_id = f"bin_{digest[:16]}"
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM binaries WHERE sha256 = ?", (digest,)).fetchone()
            detected = _detect_format(source, original_name)
            if row is not None:
                if detected != row["format"]:
                    connection.execute("UPDATE binaries SET format = ? WHERE id = ?", (detected, row["id"]))
                return dict(row) | {"stored_path": row["path"], "format": detected}
            connection.execute(
                "INSERT INTO binaries (id, sha256, path, size, format, machine, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (binary_id, digest, str(target), source.stat().st_size, _detect_format(source, original_name), None, _now()),
            )
        return {"id": binary_id, "sha256": digest, "path": str(target), "size": source.stat().st_size, "format": _detect_format(source, original_name)}

    def update_binary_machine(self, binary_id: str, machine: str) -> None:
        with self._connect() as connection:
            connection.execute("UPDATE binaries SET machine = ? WHERE id = ?", (machine, binary_id))

    def get_binary(self, binary_id: str) -> dict[str, Any]:
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM binaries WHERE id = ?", (binary_id,)).fetchone()
        if row is None:
            raise NotFoundError(f"binary not found: {binary_id}")
        return dict(row)

    def list_binaries(self, limit: int = 100) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute("SELECT * FROM binaries ORDER BY created_at DESC LIMIT ?", (limit,)).fetchall()
        return [dict(row) for row in rows]

    def create_analysis(self, binary_id: str, summary: dict[str, Any], result_path: str | Path | None, status: str = "completed") -> str:
        analysis_id = f"analysis_{uuid.uuid4().hex[:16]}"
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO analyses (id, binary_id, status, summary_json, result_path, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                (analysis_id, binary_id, status, json.dumps(summary, sort_keys=True), str(result_path) if result_path else None, _now()),
            )
        return analysis_id

    def get_analysis(self, analysis_id: str) -> dict[str, Any]:
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM analyses WHERE id = ?", (analysis_id,)).fetchone()
        if row is None:
            raise NotFoundError(f"analysis not found: {analysis_id}")
        result = dict(row)
        result["analysis_id"] = result.get("id")
        result["summary"] = json.loads(result.pop("summary_json"))
        return result

    def list_analyses(self, binary_id: str | None = None, limit: int = 100) -> list[dict[str, Any]]:
        with self._connect() as connection:
            if binary_id:
                rows = connection.execute("SELECT * FROM analyses WHERE binary_id = ? ORDER BY created_at DESC LIMIT ?", (binary_id, limit)).fetchall()
            else:
                rows = connection.execute("SELECT * FROM analyses ORDER BY created_at DESC LIMIT ?", (limit,)).fetchall()
        results = []
        for row in rows:
            item = dict(row)
            item["analysis_id"] = item.get("id")
            item["summary"] = json.loads(item.pop("summary_json"))
            results.append(item)
        return results

    def create_reconstruction(self, analysis_id: str, language: str, summary: dict[str, Any], output_dir: str | Path | None, status: str = "completed") -> str:
        reconstruction_id = f"rec_{uuid.uuid4().hex[:16]}"
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO reconstructions (id, analysis_id, language, status, output_dir, summary_json, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (reconstruction_id, analysis_id, language, status, str(output_dir) if output_dir else None, json.dumps(summary, sort_keys=True), _now()),
            )
        return reconstruction_id

    def get_reconstruction(self, reconstruction_id: str) -> dict[str, Any]:
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM reconstructions WHERE id = ?", (reconstruction_id,)).fetchone()
        if row is None:
            raise NotFoundError(f"reconstruction not found: {reconstruction_id}")
        result = dict(row)
        result["reconstruction_id"] = result.get("id")
        result["summary"] = json.loads(result.pop("summary_json"))
        return result

    def create_verification(self, reconstruction_id: str, result: dict[str, Any]) -> str:
        verification_id = f"verify_{uuid.uuid4().hex[:16]}"
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO verifications (id, reconstruction_id, status, result_json, created_at) VALUES (?, ?, ?, ?, ?)",
                (verification_id, reconstruction_id, result.get("status", "unknown"), json.dumps(result, sort_keys=True), _now()),
            )
        return verification_id

    def get_verification(self, verification_id: str) -> dict[str, Any]:
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM verifications WHERE id = ?", (verification_id,)).fetchone()
        if row is None:
            raise NotFoundError(f"verification not found: {verification_id}")
        result = dict(row)
        result["result"] = json.loads(result.pop("result_json"))
        return result

    def create_job(self, kind: str, payload: dict[str, Any]) -> str:
        job_id = f"job_{uuid.uuid4().hex[:16]}"
        now = _now()
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO jobs (id, kind, status, payload_json, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?)",
                (job_id, kind, "queued", json.dumps(payload, sort_keys=True), now, now),
            )
        return job_id

    def update_job(self, job_id: str, status: str, result: dict[str, Any] | None = None, error: str | None = None) -> None:
        with self._connect() as connection:
            connection.execute(
                "UPDATE jobs SET status = ?, result_json = ?, error = ?, updated_at = ? WHERE id = ?",
                (status, json.dumps(result, sort_keys=True) if result is not None else None, error, _now(), job_id),
            )

    def list_jobs(self, limit: int = 100) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute("SELECT id, kind, status, created_at, updated_at, error FROM jobs ORDER BY created_at DESC LIMIT ?", (limit,)).fetchall()
        return [dict(row) for row in rows]

    def get_job(self, job_id: str) -> dict[str, Any]:
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        if row is None:
            raise NotFoundError(f"job not found: {job_id}")
        result = dict(row)
        result["payload"] = json.loads(result.pop("payload_json"))
        result["result"] = json.loads(result["result_json"]) if result.get("result_json") else None
        result.pop("result_json", None)
        return result

    def add_dataset(self, path: str | Path, summary: dict[str, Any]) -> str:
        dataset_id = f"dataset_{uuid.uuid4().hex[:16]}"
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO datasets (id, path, summary_json, created_at) VALUES (?, ?, ?, ?)",
                (dataset_id, str(path), json.dumps(summary, sort_keys=True), _now()),
            )
        return dataset_id

    def list_datasets(self, limit: int = 20) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute("SELECT * FROM datasets ORDER BY created_at DESC LIMIT ?", (limit,)).fetchall()
        results = []
        for row in rows:
            item = dict(row)
            item["summary"] = json.loads(item.pop("summary_json"))
            results.append(item)
        return results

    def add_model(self, path: str | Path, metadata: dict[str, Any]) -> str:
        model_id = f"model_{uuid.uuid4().hex[:16]}"
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO models (id, path, sha256, metadata_json, created_at) VALUES (?, ?, ?, ?, ?)",
                (model_id, str(path), sha256_file(path), json.dumps(metadata, sort_keys=True), _now()),
            )
        return model_id

    def list_models(self, limit: int = 20) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute("SELECT * FROM models ORDER BY created_at DESC LIMIT ?", (limit,)).fetchall()
        results = []
        for row in rows:
            item = dict(row)
            item["metadata"] = json.loads(item.pop("metadata_json"))
            results.append(item)
        return results

    def add_experiment(self, summary: dict[str, Any], path: str | None = None) -> str:
        experiment_id = f"experiment_{uuid.uuid4().hex[:16]}"
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO experiments (id, path, summary_json, created_at) VALUES (?, ?, ?, ?)",
                (experiment_id, path, json.dumps(summary, sort_keys=True), _now()),
            )
        return experiment_id

    def list_experiments(self, limit: int = 20) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute("SELECT * FROM experiments ORDER BY created_at DESC LIMIT ?", (limit,)).fetchall()
        results = []
        for row in rows:
            item = dict(row)
            item["summary"] = json.loads(item.pop("summary_json"))
            results.append(item)
        return results

    def add_findings(self, binary_id: str, findings: list[dict[str, Any]]) -> None:
        with self._connect() as connection:
            for finding in findings:
                connection.execute(
                    "INSERT OR REPLACE INTO findings (id, binary_id, finding_json, created_at) VALUES (?, ?, ?, ?)",
                    (finding.get("finding_id"), binary_id, json.dumps(finding, sort_keys=True), _now()),
                )

    def add_investigation(self, binary_id: str, report: dict[str, Any]) -> str:
        investigation_id = report.get("investigation_id") or f"inv_{uuid.uuid4().hex[:16]}"
        with self._connect() as connection:
            connection.execute(
                "INSERT OR REPLACE INTO investigations (id, binary_id, objective, status, summary_json, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                (investigation_id, binary_id, report.get("objective", ""), report.get("status", "unknown"), json.dumps(report, sort_keys=True), _now()),
            )
        return investigation_id

    def get_investigation(self, investigation_id: str) -> dict[str, Any]:
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM investigations WHERE id = ?", (investigation_id,)).fetchone()
        if row is None:
            raise NotFoundError(f"investigation not found: {investigation_id}")
        result = dict(row)
        result["report"] = json.loads(result.pop("summary_json"))
        return result

    def list_investigations(self, binary_id: str | None = None, limit: int = 50) -> list[dict[str, Any]]:
        with self._connect() as connection:
            if binary_id:
                rows = connection.execute("SELECT * FROM investigations WHERE binary_id = ? ORDER BY created_at DESC LIMIT ?", (binary_id, limit)).fetchall()
            else:
                rows = connection.execute("SELECT * FROM investigations ORDER BY created_at DESC LIMIT ?", (limit,)).fetchall()
        results = []
        for row in rows:
            item = dict(row)
            item["report"] = json.loads(item.pop("summary_json"))
            results.append(item)
        return results

    def list_findings(self, binary_id: str | None = None, limit: int = 200) -> list[dict[str, Any]]:
        with self._connect() as connection:
            if binary_id:
                rows = connection.execute("SELECT * FROM findings WHERE binary_id = ? ORDER BY created_at DESC LIMIT ?", (binary_id, limit)).fetchall()
            else:
                rows = connection.execute("SELECT * FROM findings ORDER BY created_at DESC LIMIT ?", (limit,)).fetchall()
        return [json.loads(row["finding_json"]) for row in rows]
