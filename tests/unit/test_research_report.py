import json
from pathlib import Path

from nexus.reporting.comprehensive import build_comprehensive_report
from nexus.research.knowledge import research_topic


def test_curated_research_returns_citations():
    result = research_topic("AAPCS64 argument registers", include_headers=False)
    assert result["knowledge"]
    assert result["knowledge"][0]["id"] == "abi.aapcs64"
    assert result["knowledge"][0]["sources"]


def test_research_system_headers_is_bounded():
    result = research_topic("stdio printf", include_headers=True, limit=3)
    headers = result["system_headers"]
    assert headers is not None
    assert headers["searched_files"] <= headers["available_files"]
    for match in headers["matches"]:
        assert Path(match["source"]).is_absolute()


def test_comprehensive_report_contains_required_sections(simple_binary, tmp_path):
    result = build_comprehensive_report(simple_binary, out_dir=tmp_path / "report", max_functions=4)
    assert result["status"] == "completed"
    report = Path(result["report_markdown"])
    text = report.read_text(encoding="utf-8")
    for heading in (
        "## Library purpose",
        "## Exported APIs",
        "## Imported APIs and dependencies",
        "## ABI and calling conventions",
        "## Functions",
        "## Security review candidates",
        "## Performance proxies",
        "## Reconstructed source",
        "## Evidence, confidence and unresolved questions",
        "## Limitations",
    ):
        assert heading in text
    payload = json.loads(Path(result["report_json"]).read_text(encoding="utf-8"))
    assert payload["facts"]["binary"]["sha256"]
    assert payload["inferences"]["library_purpose"]
    assert payload["confidence_model"]["facts"]
