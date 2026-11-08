import shutil
import subprocess
from pathlib import Path

import pytest

from nexus.data.java_corpus import build_java_samples
from nexus.datasets.advanced import validate_advanced_dataset
from nexus.datasets.dex_factory import build_minimal_dex
from nexus.formats.dex import parse_dex
from nexus.formats.java_class import parse_java_class
from nexus.java.service import analyze_java_artifact
from nexus.jni.correlation import correlate_java_native, jni_symbol_candidates
from nexus.reconstruction.java.classfile import reconstruct_java_class_file
from nexus.reconstruction.java.dex import reconstruct_java_dex
from nexus.verification.compiler import compile_java_project


def _run(command, cwd=None):
    return subprocess.run(command, cwd=cwd, capture_output=True, text=True, timeout=120)


def _compile_class(tmp_path, source: str, name: str = "Calc") -> Path:
    if not shutil.which("javac"):
        pytest.skip("javac is unavailable")
    source_dir = tmp_path / "src" / "com" / "example"
    source_dir.mkdir(parents=True)
    path = source_dir / f"{name}.java"
    path.write_text(source, encoding="utf-8")
    result = _run(["javac", "-d", str(tmp_path / "classes"), str(path)])
    if result.returncode != 0:
        pytest.skip(f"javac failed: {result.stderr[:300]}")
    return tmp_path / "classes" / "com" / "example" / f"{name}.class"


CALC_SOURCE = """
package com.example;
public class Calc {
    private int total;
    public static final int SCALE = 3;
    public int add(int a, int b) { return a + b; }
    public int scaled(int a) { return a * SCALE + total; }
    public int choose(int a, int b) { if (a > b) return a; return b; }
    public native int nativeAdd(int a, int b);
}
"""


def test_class_parser_decodes_bytecode_and_constants(tmp_path):
    path = _compile_class(tmp_path, CALC_SOURCE)
    parsed = parse_java_class(path)
    assert parsed["name"] == "com.example.Calc"
    assert parsed["source_file"] == "Calc.java"
    assert next(field for field in parsed["fields"] if field["name"] == "SCALE")["constant_value"] == "3"
    add = next(method for method in parsed["methods"] if method["name"] == "add")
    assert add["descriptor"] == "(II)I"
    assert [item["mnemonic"] for item in add["code"]["instructions"]] == ["iload_1", "iload_2", "iadd", "ireturn"]
    native = next(method for method in parsed["methods"] if method["name"] == "nativeAdd")
    assert native["native"] is True
    assert native["code"] is None


def test_class_reconstruction_compiles(tmp_path):
    path = _compile_class(tmp_path, CALC_SOURCE)
    result = reconstruct_java_class_file(path, out_dir=tmp_path / "project", project_name="calc")
    source = result["files"]["src/main/java/com/example/Calc.java"]
    assert "int add(int arg0, int arg1)" in source
    assert "arg0 + arg1" in source
    assert "if (arg0 > arg1)" in source
    assert "return arg1;" in source
    assert "native int nativeAdd" in source
    assert "SCALE = 3" in source
    verification = compile_java_project(tmp_path / "project")
    assert verification["status"] == "passed", verification


def test_java_service_analyzes_class(tmp_path):
    path = _compile_class(tmp_path, CALC_SOURCE)
    result = analyze_java_artifact(path)
    assert result["format"] == "class"
    assert result["summary"]["name"] == "com.example.Calc"
    assert result["summary"]["native_methods"] == ["nativeAdd"]


def test_jni_correlation_with_compiled_library(tmp_path, jni_binary):
    if not shutil.which("javac"):
        pytest.skip("javac is unavailable")
    source = tmp_path / "JniDemo.java"
    source.write_text(
        "package com.example;\n"
        "public class JniDemo {\n"
        "    public static native int add(int a, int b);\n"
        "    public static native long add(long a, long b);\n"
        "}\n",
        encoding="utf-8",
    )
    result = _run(["javac", "-d", str(tmp_path / "classes"), str(source)])
    if result.returncode != 0:
        pytest.skip(f"javac failed: {result.stderr[:300]}")
    class_path = tmp_path / "classes" / "com" / "example" / "JniDemo.class"
    correlation = correlate_java_native(class_path, jni_binary)
    assert correlation["summary"]["matched"] == 2
    assert correlation["summary"]["jni_onload"] is True
    symbols = {item["native_symbol"] for item in correlation["matches"]}
    assert "Java_com_example_JniDemo_add__II" in symbols
    assert "Java_com_example_JniDemo_add__JJ" in symbols
    assert jni_symbol_candidates("com.example.JniDemo", "add", "(II)I")[-1] == "Java_com_example_JniDemo_add__II"


def test_dex_parse_and_reconstruction_compiles(tmp_path):
    path = tmp_path / "Add.dex"
    path.write_bytes(build_minimal_dex())
    parsed = parse_dex(path)
    assert parsed["classes"][0]["name"] == "com.example.Add"
    method = parsed["classes"][0]["methods"][0]
    assert method["descriptor"] == "(II)I"
    assert [item["mnemonic"] for item in method["code"]["instructions"]] == ["add-int", "return"]
    result = reconstruct_java_dex(parsed, out_dir=tmp_path / "project", project_name="dex")
    source = result["files"]["src/main/java/com/example/Add.java"]
    assert "public static int add(int arg0, int arg1)" in source
    verification = compile_java_project(tmp_path / "project")
    assert verification["status"] == "passed", verification


def test_java_foundation_samples_are_encodable(tmp_path):
    _compile_class(tmp_path, CALC_SOURCE)
    result = build_java_samples(tmp_path / "classes", out_dir=tmp_path / "foundation")
    assert result["status"] == "completed"
    assert result["train"] + result["validation"] + result["test"] >= 1
    assert (tmp_path / "foundation" / "manifest.json").exists()
    assert (tmp_path / "foundation" / "vocab.json").exists()
    validation = validate_advanced_dataset(tmp_path / "foundation")
    assert validation["status"] == "passed", validation


def test_decode_jni_symbol_decodes_parameter_types():
    from nexus.jni.analysis import decode_jni_symbol

    decoded = decode_jni_symbol("Java_jyvmut_Pfqmjma_getInstalledApplications__Landroid_content_pm_PackageManager_2I")
    assert decoded is not None
    assert decoded["class_name"] == "jyvmut.Pfqmjma"
    assert decoded["method"] == "getInstalledApplications"
    assert decoded["arguments"] == ["android.content.pm.PackageManager", "int"]
    assert decoded["descriptor"] == "(Landroid/content/pm/PackageManager;I)"
    assert decoded["return_type"] == "unknown"


def test_native_java_reconstruction_uses_jni_signatures_and_soname(tmp_path):
    from types import SimpleNamespace

    from nexus.reconstruction.java.generator import reconstruct_java

    bundle = SimpleNamespace(
        elf=SimpleNamespace(
            path="/tmp/deadbeef",
            sha256="a" * 64,
            dependencies=[{"name": "libstub.so", "kind": "soname", "value": 1}],
        ),
        jni_report={
            "entries": [
                {
                    "symbol": "Java_com_example_Native_getInstalledApplications__Landroid_content_pm_PackageManager_2I",
                    "class_name": "com.example.Native",
                    "simple_class": "Native",
                    "package": "com.example",
                    "method": "getInstalledApplications",
                    "descriptor": "(Landroid/content/pm/PackageManager;I)",
                    "arguments": ["android.content.pm.PackageManager", "int"],
                    "return_type": "unknown",
                    "evidence": "ev_test",
                }
            ]
        },
        functions=[],
        type_candidates={},
    )
    result = reconstruct_java(bundle, out_dir=tmp_path / "out", project_name="native")
    source = result["files"]["src/main/java/com/example/Native.java"]
    assert "android.content.pm.PackageManager arg0, int arg1" in source
    assert 'System.loadLibrary("stub");' in source
    assert result["classification"] == "jni_bridge_reconstruction"
    assert result["limitations"][0].startswith("This is JNI semantic reconstruction")
