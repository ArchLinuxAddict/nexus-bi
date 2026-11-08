
from nexus.datasets.factory import generate_dataset
from nexus.program.bundle import analyze_binary
from nexus.reconstruction.c.generator import reconstruct_c
from nexus.verification.compiler import compile_c_project


def test_relative_dataset_root_compiles(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = generate_dataset("relative-dataset", fixtures=["arithmetic"], compilers=["gcc"], optimizations=["O0"], debug_modes=[False], pie_modes=[True])
    assert result["summary"]["samples"] == 1
    assert result["summary"]["quarantined"] == 0


def test_jni_entries_are_deduplicated(jni_binary):
    bundle = analyze_binary(jni_binary)
    keys = [(entry["class_name"], entry["method"], entry["descriptor"]) for entry in bundle.jni_report["entries"]]
    assert len(keys) == len(set(keys))


def test_generated_main_is_renamed(simple_binary, tmp_path):
    bundle = analyze_binary(simple_binary)
    result = reconstruct_c(bundle, out_dir=tmp_path / "project")
    assert "src/reconstructed_main.c" in result["files"]
    assert any("reconstructed_main" in content for content in result["files"].values())
    assert compile_c_project(tmp_path / "project")["status"] == "passed"
