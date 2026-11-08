from nexus.program.bundle import analyze_binary
from nexus.sandbox.gate import differential_test, status


def test_unsafe_calls_produce_evidence(unsafe_binary):
    bundle = analyze_binary(unsafe_binary)
    titles = {finding["title"] for finding in bundle.security_findings}
    assert any("strcpy" in title for title in titles)
    assert any("system" in title for title in titles)
    for finding in bundle.security_findings:
        assert finding["status"] == "needs_review"
        assert finding["evidence"]
        assert finding["confidence"] < 1.0


def test_safe_binary_has_no_high_findings(simple_binary):
    bundle = analyze_binary(simple_binary)
    assert all(finding["severity"] != "high" for finding in bundle.security_findings)


def test_dynamic_execution_is_disabled():
    sandbox = status()
    assert sandbox.available is False
    result = differential_test()
    assert result["status"] == "capability_unavailable"
    assert result["executed"] is False
