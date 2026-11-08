from nexus.datasets.apk_factory import build_minimal_android_manifest, build_minimal_apk
from nexus.datasets.dex_factory import build_minimal_dex
from nexus.formats.apk import parse_apk
from nexus.java.service import analyze_java_artifact, reconstruct_java_artifact
from nexus.verification.compiler import compile_java_project


def test_apk_inventory_manifest_dex_and_native(tmp_path, simple_binary):
    apk = build_minimal_apk(
        manifest=build_minimal_android_manifest(),
        dex=build_minimal_dex(),
        native_library=simple_binary.read_bytes(),
    )
    path = tmp_path / "fixture.apk"
    path.write_bytes(apk)
    parsed = parse_apk(path)
    assert parsed["manifest"]["package"] == "com.example.app"
    assert parsed["classes_dex_count"] == 1
    assert parsed["java_classes"] == 1
    assert parsed["native_library_count"] == 1
    entry = parsed["native_libraries"][0]
    assert entry["abi_directory"] == "arm64-v8a"
    assert entry["machine"] == "x86_64"
    assert parsed["native_methods"] == []
    assert parsed["android_manifest_present"] is True


def test_apk_java_reconstruction_compiles(tmp_path):
    apk = build_minimal_apk(manifest=build_minimal_android_manifest(), dex=build_minimal_dex())
    path = tmp_path / "fixture.apk"
    path.write_bytes(apk)
    result = reconstruct_java_artifact(path, out_dir=tmp_path / "project", project_name="apk")
    assert "src/main/java/com/example/Add.java" in result["files"]
    verification = compile_java_project(tmp_path / "project")
    assert verification["status"] == "passed", verification
    analysis = analyze_java_artifact(path)
    assert analysis["format"] == "apk"
    assert analysis["summary"]["package"] == "com.example.app"


def test_tool_first_source_selection_prefers_app_package(tmp_path):
    from nexus.reconstruction.java.toolchain import _collect_sources

    files = {
        "sources/a/a.java": "package a;\npublic class a {}\n",
        "sources/androidx/core/Foo.java": "package androidx.core;\npublic class Foo {}\n",
        "sources/com/example/app/Main.java": "package com.example.app;\npublic class Main {}\n",
        "sources/com/example/app/R$id.java": "package com.example.app;\npublic final class R$id {}\n",
    }
    root = tmp_path / "jadx"
    for relative, content in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    huge = root / "sources/com/example/app/Huge.java"
    huge.write_text("x" * 1_500_001, encoding="utf-8")

    sources, stats = _collect_sources(root, max_classes=10, package_hint="com.example.app")

    assert list(sources)[0].endswith("com/example/app/Main.java")
    assert list(sources)[-1].endswith("R$id.java")
    assert stats["source_files_skipped_large"] == 1
    assert stats["source_files_total"] == 5
    assert stats["source_files_returned"] == 4


def test_java_compile_reports_missing_platform_dependency(tmp_path):
    from nexus.verification.toolchains import available_java

    if not available_java():
        import pytest

        pytest.skip("javac is not available")
    source = tmp_path / "project" / "src" / "main" / "java" / "com" / "example" / "A.java"
    source.parent.mkdir(parents=True)
    source.write_text("package com.example;\nimport android.view.View;\npublic class A {}\n", encoding="utf-8")
    result = compile_java_project(tmp_path / "project")
    assert result["status"] == "unavailable"
    assert "android.view.View" in result["reason"]
