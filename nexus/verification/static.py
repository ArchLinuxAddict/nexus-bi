import json
from pathlib import Path
from typing import Any

from nexus.core.hashing import sha256_file


def _schema_path() -> Path:
    return Path(__file__).resolve().parents[2] / "schemas" / "program_record.schema.json"


def verify_program_record(record: dict[str, Any]) -> dict[str, Any]:
    errors = []
    warnings = []
    schema_path = _schema_path()
    if schema_path.exists():
        try:
            import jsonschema
            schema = json.loads(schema_path.read_text(encoding="utf-8"))
            for error in sorted(jsonschema.Draft202012Validator(schema).iter_errors(record), key=lambda item: list(item.path)):
                errors.append({"path": "/".join(map(str, error.path)), "message": error.message})
        except Exception as exc:
            warnings.append(f"schema validation unavailable: {exc}")
    else:
        warnings.append("schema file not found")
    evidence_ids = {item.get("evidence_id") for item in record.get("evidence", []) if isinstance(item, dict)}
    referenced = set()

    def walk(value: Any) -> None:
        if isinstance(value, dict):
            for item in value.values():
                walk(item)
        elif isinstance(value, list):
            for item in value:
                walk(item)
        elif isinstance(value, str) and value.startswith("ev_"):
            referenced.add(value)

    walk(record)
    missing = sorted(referenced - evidence_ids)
    if missing:
        errors.append({"path": "evidence", "message": f"missing evidence records: {missing[:20]}"})
    for function in record.get("functions", []):
        if not function.get("blocks"):
            errors.append({"path": f"functions/{function.get('function_id')}", "message": "function has no basic blocks"})
        if not function.get("evidence"):
            errors.append({"path": f"functions/{function.get('function_id')}", "message": "function has no evidence"})
    for instruction in record.get("instructions", []):
        if len(instruction.get("bytes", "")) // 2 != instruction.get("length"):
            errors.append({"path": f"instructions/{instruction.get('instruction_id')}", "message": "byte length mismatch"})
    return {
        "status": "passed" if not errors else "failed",
        "errors": errors,
        "warnings": warnings,
        "counts": {
            "functions": len(record.get("functions", [])),
            "instructions": len(record.get("instructions", [])),
            "evidence": len(record.get("evidence", [])),
        },
    }


def verify_binary_identity(record: dict[str, Any], path: str | Path) -> dict[str, Any]:
    expected = record.get("program", {}).get("sha256")
    actual = sha256_file(path)
    return {
        "status": "passed" if expected == actual else "failed",
        "expected_sha256": expected,
        "actual_sha256": actual,
    }


def verify_reconstruction_structure(reconstruction: dict[str, Any], project_dir: str | Path | None = None) -> dict[str, Any]:
    errors = []
    warnings = []
    if reconstruction.get("classification") == "exact_source_recovery":
        errors.append("exact source recovery requires provenance and was not established")
    files = reconstruction.get("files", {})
    if not files:
        errors.append("reconstruction contains no files")
    if project_dir is not None:
        root = Path(project_dir)
        for relative in files:
            if not (root / relative).exists():
                errors.append(f"missing generated file: {relative}")
    unresolved = sum(1 for item in reconstruction.get("functions", []) if item.get("classification") == "unresolved")
    if unresolved:
        warnings.append(f"{unresolved} functions are unresolved stubs")
    return {
        "status": "passed" if not errors else "failed",
        "errors": errors,
        "warnings": warnings,
        "unresolved_functions": unresolved,
    }
