
import pytest

from nexus.program.bundle import analyze_binary
from nexus.reconstruction.c.generator import reconstruct_c
from nexus.reconstruction.java.generator import reconstruct_java
from nexus.verification.compiler import compile_c_project, compile_java_project
from nexus.verification.static import verify_reconstruction_structure


def test_c_reconstruction_compiles(simple_binary, tmp_path):
    bundle = analyze_binary(simple_binary)
    result = reconstruct_c(bundle, function_ids=["add"], out_dir=tmp_path / "c-project")
    if result["classification"] == "unresolved":
        pytest.skip("C reconstruction backend is unavailable in this environment")
    assert result["classification"] == "partial_behavioral_reconstruction"
    assert result["functions"][0]["generated_name"] == "add"
    assert "arg1 + arg0" in result["files"]["src/add.c"] or "arg0 + arg1" in result["files"]["src/add.c"]
    assert verify_reconstruction_structure(result, tmp_path / "c-project")["status"] == "passed"
    compile_result = compile_c_project(tmp_path / "c-project")
    assert compile_result["status"] == "passed", compile_result
    assert compile_result["compile_rate"] == 1.0
    assert compile_result["dynamic"]["status"] == "capability_unavailable"


def test_unresolved_function_is_explicit(simple_binary, tmp_path):
    bundle = analyze_binary(simple_binary)
    result = reconstruct_c(bundle, function_ids=["main"], out_dir=tmp_path / "main-project")
    function = result["functions"][0]
    assert function["classification"] == "unresolved"
    assert "not claimed to be behaviorally equivalent" in result["files"]["src/reconstructed_main.c"]
    compile_result = compile_c_project(tmp_path / "main-project")
    assert compile_result["status"] == "passed", compile_result


def test_structured_control_flow_reconstruction(branching_binary, tmp_path):
    bundle = analyze_binary(branching_binary)
    result = reconstruct_c(bundle, function_ids=["classify", "maximum"], out_dir=tmp_path / "structured-project")
    assert all(function["classification"] == "structural_approximation" for function in result["functions"])
    assert "goto bb_" in result["files"]["src/classify.c"]
    assert "goto bb_" in result["files"]["src/maximum.c"]
    compile_result = compile_c_project(tmp_path / "structured-project")
    assert compile_result["status"] == "passed", compile_result


def test_structured_loop_reconstruction(loop_binary, tmp_path):
    bundle = analyze_binary(loop_binary)
    result = reconstruct_c(bundle, function_ids=["sum_to"], out_dir=tmp_path / "loop-project")
    assert result["functions"][0]["classification"] == "structural_approximation"
    assert "goto bb_" in result["files"]["src/sum_to.c"]
    compile_result = compile_c_project(tmp_path / "loop-project")
    assert compile_result["status"] == "passed", compile_result


def test_c_evidence_map(simple_binary):
    bundle = analyze_binary(simple_binary)
    result = reconstruct_c(bundle, function_ids=["add"])
    assert result["evidence_map"]["binary_sha256"] == bundle.elf.sha256
    assert result["evidence_map"]["functions"][0]["function_id"].startswith("fn_")


def test_java_reconstruction_for_non_jni(simple_binary, tmp_path):
    bundle = analyze_binary(simple_binary)
    result = reconstruct_java(bundle, function_ids=["add"], out_dir=tmp_path / "java-project")
    assert result["classification"] == "native_symbol_bridge"
    assert "ReconstructedModule" in result["files"]["src/main/java/ReconstructedModule.java"]
    compile_result = compile_java_project(tmp_path / "java-project")
    assert compile_result["status"] == "passed", compile_result


def test_jni_java_reconstruction(jni_binary, tmp_path):
    bundle = analyze_binary(jni_binary)
    assert bundle.jni_report["is_jni"] is True
    assert len(bundle.jni_report["entries"]) == 2
    assert bundle.jni_report["onload"]
    result = reconstruct_java(bundle, out_dir=tmp_path / "jni-java")
    java_file = result["files"]["src/main/java/com/example/JniDemo.java"]
    assert result["classification"] == "jni_bridge_reconstruction"
    assert "native Object add(long arg0, long arg1)" in java_file
    assert "native Object add(int arg0, int arg1)" in java_file
    assert "System.loadLibrary" in java_file
    compile_result = compile_java_project(tmp_path / "jni-java")
    assert compile_result["status"] == "passed", compile_result
