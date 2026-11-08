import json

from nexus.toolchain.cache import ToolCache
from nexus.toolchain.models import ToolResult, cache_key
from nexus.toolchain.registry import ToolRegistry


def test_tool_matrix_contract():
    matrix = ToolRegistry().matrix()
    assert matrix["schema_version"] == "nexus.tool-matrix/v1"
    assert "ghidra_headless" in matrix["tools"]
    assert "jadx" in matrix["tools"]
    assert "cfr" in matrix["tools"]
    assert matrix["tools"]["ghidra_headless"]["capabilities"]
    assert matrix["status"] in {"ready", "incomplete"}


def test_cache_key_is_content_and_version_addressed():
    first = cache_key("a" * 64, "ghidra_headless", "12.1.4", {"max_functions": 8})
    second = cache_key("a" * 64, "ghidra_headless", "12.1.4", {"max_functions": 8})
    changed = cache_key("a" * 64, "ghidra_headless", "12.2.0", {"max_functions": 8})
    assert first == second
    assert first != changed


def test_tool_cache_roundtrip(tmp_path):
    cache = ToolCache(tmp_path)
    result = ToolResult(
        tool="ghidra_headless",
        version="12.1.4",
        input_sha256="b" * 64,
        arguments={"language": "c"},
        command=["analyzeHeadless"],
        status="completed",
        artifacts={"functions": 3},
    )
    key = cache_key(result.input_sha256, result.tool, result.version, result.arguments)
    cache.put_result(key, result)
    loaded = cache.get_result(key)
    assert loaded is not None
    assert loaded.cached is True
    assert loaded.to_dict()["artifacts"] == {"functions": 3}
    assert json.loads((tmp_path / key[:2] / f"{key}.json").read_text())["tool"] == "ghidra_headless"


def test_tool_result_evidence_id_is_stable():
    first = ToolResult(tool="jadx", version="1.5.6", input_sha256="c" * 64, status="completed", started_at="2026-01-01T00:00:00Z")
    second = ToolResult(tool="jadx", version="1.5.6", input_sha256="c" * 64, status="completed", started_at="2026-01-02T00:00:00Z", cached=True)
    assert first.to_dict()["evidence_id"] == second.to_dict()["evidence_id"]
    assert first.to_dict()["evidence_id"].startswith("tool_")
