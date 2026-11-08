import json
from pathlib import Path

from jsonschema import Draft202012Validator

from nexus.datasets.factory import generate_dataset, inspect_dataset


def test_dataset_summary_and_manifests(dataset_root):
    summary = json.loads((dataset_root / "dataset_summary.json").read_text(encoding="utf-8"))
    assert summary["samples"] > 0
    assert summary["projects"] > 0
    assert summary["quarantined"] == 0
    assert summary["leakage"] == []
    manifests = [json.loads(line) for line in (dataset_root / "manifests.jsonl").read_text(encoding="utf-8").splitlines() if line]
    assert len(manifests) == summary["samples"]
    schema = json.loads((Path(__file__).resolve().parents[2] / "schemas" / "dataset_manifest.schema.json").read_text(encoding="utf-8"))
    validator = Draft202012Validator(schema)
    for manifest in manifests:
        errors = list(validator.iter_errors(manifest))
        assert not errors, errors[:3]
        assert manifest["binary"]["sha256"]
        assert manifest["build"]["compiler"] == "gcc"
        assert manifest["quality"]["validation_status"] in {"schema_valid", "compile_reproduced"}


def test_dataset_inspect(dataset_root):
    result = inspect_dataset(dataset_root)
    assert result["status"] == "ok"
    assert result["samples"] > 0
    assert result["leakage"] == []
    assert result["stripped"] > 0
    assert result["unstripped"] > 0


def test_dataset_is_project_split(dataset_root):
    manifests = [json.loads(line) for line in (dataset_root / "manifests.jsonl").read_text(encoding="utf-8").splitlines() if line]
    project_splits = {}
    for manifest in manifests:
        project_splits.setdefault(manifest["project_id"], set()).add(manifest["split"])
    assert all(len(splits) == 1 for splits in project_splits.values())


def test_generate_dataset_small(tmp_path):
    result = generate_dataset(tmp_path / "small", fixtures=["arithmetic"], compilers=["gcc"], optimizations=["O0"], debug_modes=[False], pie_modes=[True])
    assert result["summary"]["samples"] == 1
    assert result["summary"]["quarantined"] == 0
