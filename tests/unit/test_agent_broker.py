import pytest

from nexus.agent import InvestigationReport, StaticAnalysisBroker
from nexus.agent.evidence import EvidenceRecord


def test_broker_is_closed_and_rejects_invalid_arguments(simple_binary, tmp_path):
    broker = StaticAnalysisBroker(allowed_roots=[simple_binary.parent, tmp_path])
    names = set(broker.registry.names())
    assert {"analyze_binary", "build_program_record", "identify_format", "parse_pe", "parse_macho", "parse_apk", "parse_dex", "parse_java_class", "sandbox_status"} <= names
    assert "shell" not in names
    assert broker.call("shell", {"command": "id"}).status == "error"
    assert broker.call("analyze_binary", {}).status == "rejected"
    assert broker.call("analyze_binary", {"path": str(simple_binary), "extra": "x"}).status == "rejected"
    outside = tmp_path.parent / "outside.bin"
    outside.write_bytes(b"x")
    assert broker.call("analyze_binary", {"path": str(outside)}).status == "error"


def test_broker_emits_hashed_evidence(simple_binary):
    broker = StaticAnalysisBroker(allowed_roots=[simple_binary.parent])
    result = broker.call("analyze_binary", {"path": str(simple_binary)})
    assert result.status == "ok"
    assert result.evidence is not None
    payload = result.evidence.to_dict()
    assert payload["tier"] == "FACT"
    assert payload["subject"]["artifact_sha256"]
    assert payload["provenance"] if "provenance" in payload else payload["reproducibility"]["command"]
    record = broker.call("build_program_record", {"path": str(simple_binary)})
    assert record.status == "ok"
    assert record.output["functions"]


def test_investigation_report_distinguishes_claim_tiers_and_unknowns():
    report = InvestigationReport("inv_1", {"sha256": "a"}, "identify behavior")
    evidence = EvidenceRecord("test", "1", {"artifact_sha256": "a"}, {"kind": "fact"})
    report.add_fact("f1", "function:add", "entry", "0x1000", [evidence.evidence_id])
    report.add_claim("c1", "likely arithmetic", "WEAK_HYPOTHESIS", 0.4, [evidence.evidence_id], missing=["trace"])
    report.add_hypothesis("h1", "adds two integers", "under_investigation", [evidence.evidence_id], tests=["differential_test"])
    report.add_unknown("u1", "is it signed?", False, "symbolic_execute", "sandbox unavailable")
    payload = report.to_dict()
    assert payload["facts"][0]["tier"] == "FACT"
    assert payload["claims"][0]["tier"] == "WEAK_HYPOTHESIS"
    assert payload["hypotheses"][0]["status"] == "under_investigation"
    assert payload["unknowns"][0]["reason"] == "sandbox unavailable"
    with pytest.raises(ValueError):
        report.add_claim("c2", "bad", "CERTAIN", 1.0, [])
