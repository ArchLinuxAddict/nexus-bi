import json

from nexus.core.config import NexusPaths
from nexus.core.engine import Engine
from nexus.storage.store import Storage


def test_end_to_end_c_pipeline(tmp_path, simple_binary):
    paths = NexusPaths(root=tmp_path / "e2e-home").ensure()
    engine = Engine(paths, Storage(paths))
    analysis = engine.analyze_path(simple_binary)
    summary = analysis["summary"]
    assert summary["validation"]["status"] == "passed"
    assert summary["function_count"] > 0
    record = engine.load_record(summary["analysis_id"])
    assert record["instructions"]
    functions = engine.functions(summary["analysis_id"])
    add = next(function for function in functions if function["names"][0]["value"] == "add")
    reconstruction = engine.reconstruct(summary["analysis_id"], "c", [add["function_id"]], out_dir=tmp_path / "e2e-c")
    assert reconstruction["classification"] == "partial_behavioral_reconstruction"
    verification = engine.verify_reconstruction(reconstruction["reconstruction_id"])
    assert verification["status"] == "passed"
    assert verification["compile_rate"] == 1.0
    import pytest

    from nexus.core.errors import NotFoundError
    with pytest.raises(NotFoundError):
        engine.read_source_file(reconstruction["reconstruction_id"], "../../etc/passwd")
    report = {
        "binary_sha256": summary["sha256"],
        "analysis_id": summary["analysis_id"],
        "function_id": add["function_id"],
        "reconstruction_id": reconstruction["reconstruction_id"],
        "classification": reconstruction["classification"],
        "verification_status": verification["status"],
        "dynamic_verification": verification["dynamic"]["status"],
        "security_findings": len(summary.get("security_findings_count", []) if isinstance(summary.get("security_findings_count"), list) else []),
    }
    (tmp_path / "e2e-report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    assert report["dynamic_verification"] == "capability_unavailable"
    assert report["verification_status"] == "passed"


def test_end_to_end_jni_pipeline(tmp_path, jni_binary):
    paths = NexusPaths(root=tmp_path / "jni-home").ensure()
    engine = Engine(paths, Storage(paths))
    analysis = engine.analyze_path(jni_binary)
    summary = analysis["summary"]
    jni = engine.jni(summary["analysis_id"])
    assert jni["is_jni"] is True
    assert len(jni["entries"]) == 2
    reconstruction = engine.reconstruct(summary["analysis_id"], "java", out_dir=tmp_path / "jni-java")
    assert "com/example/JniDemo.java" in " ".join(reconstruction["files"].keys())
    verification = engine.verify_reconstruction(reconstruction["reconstruction_id"])
    assert verification["status"] == "passed"
    assert verification["compile_rate"] == 1.0
