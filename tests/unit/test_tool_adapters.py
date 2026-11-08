import zipfile
from pathlib import Path

import pytest

from nexus.toolchain.adapters import CfrAdapter, JadxAdapter
from nexus.toolchain.registry import ToolRegistry

CLASS_ROOT = Path(__file__).resolve().parents[2] / "artifacts/dataset/java_pilot/java_classes"


def _make_jar(target: Path) -> Path:
    classes = sorted(CLASS_ROOT.rglob("*.class"))
    if not classes:
        pytest.skip("Java class fixture is unavailable")
    with zipfile.ZipFile(target, "w") as archive:
        for path in classes:
            archive.write(path, path.relative_to(CLASS_ROOT))
    return target


def test_jadx_decompiles_java_jar(tmp_path):
    tools = ToolRegistry().discover()
    if tools["jadx"].status != "available":
        pytest.skip("jadx is not available")
    jar = _make_jar(tmp_path / "sample.jar")
    result = JadxAdapter().decompile(jar, tmp_path / "jadx")
    assert result.status == "completed"
    assert result.artifacts["java_files"]
    assert any(path.endswith(".java") for path in result.artifacts["java_files"])
    assert result.to_dict()["evidence_id"].startswith("tool_")


def test_cfr_decompiles_java_jar(tmp_path):
    tools = ToolRegistry().discover()
    if tools["cfr"].status != "available":
        pytest.skip("cfr is not available")
    jar = _make_jar(tmp_path / "sample.jar")
    result = CfrAdapter().decompile(jar, tmp_path / "cfr")
    assert result.status == "completed"
    assert result.artifacts["java_files"]
    assert result.to_dict()["evidence_id"].startswith("tool_")
