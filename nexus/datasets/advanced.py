"""Validated multi-architecture and Java/JNI pilot datasets.

All source used here is authored for NEXUS-BI and released under CC0-1.0.  The
builders record tool versions, hashes, parse coverage, reconstruction
classifications and explicit limitations.  They do not train a model.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from nexus.core.hashing import sha256_file
from nexus.formats.java_class import parse_java_class
from nexus.java.service import reconstruct_java_artifact
from nexus.program.bundle import analyze_binary
from nexus.verification.compiler import compile_c_project, compile_java_project

C_SOURCE = """
int helper_add(int a, int b) { return a + b; }
int helper_mul(int a, int b) { return a * b; }
int helper_mix(int a, int b, int c) { return (a + b) * c; }
int helper_branch(int a, int b) {
    if (a > b) return a - b;
    return b - a;
}
int helper_memory(const int *values, int count) {
    int total = 0;
    for (int index = 0; index < count; index++) total += values[index];
    return total;
}
""".strip()

JAVA_CALC = """
package com.example;
public class Calc {
    private int total;
    public static final int SCALE = 3;
    public int add(int a, int b) { return a + b; }
    public int scaled(int a) { return a * SCALE + total; }
    public int choose(int a, int b) { if (a > b) return a; return b; }
    public native int nativeAdd(int a, int b);
}
""".strip()

JAVA_JNI = """
package com.example;
public class JniDemo {
    static { System.loadLibrary("jni_demo"); }
    public static native int add(int a, int b);
    public static native long add(long a, long b);
}
""".strip()

JNI_C = """
#include <jni.h>
JNIEXPORT jint JNICALL Java_com_example_JniDemo_add__II(JNIEnv *env, jclass cls, jint a, jint b) {
    (void)env; (void)cls; return a + b;
}
JNIEXPORT jlong JNICALL Java_com_example_JniDemo_add__JJ(JNIEnv *env, jclass cls, jlong a, jlong b) {
    (void)env; (void)cls; return a + b;
}
JNIEXPORT jint JNICALL JNI_OnLoad(JavaVM *vm, void *reserved) {
    (void)vm; (void)reserved; return JNI_VERSION_1_8;
}
""".strip()

TARGETS = {
    "x86_64": "x86_64-linux-gnu",
    "x86": "i386-linux-gnu",
    "armv7": "armv7-linux-gnueabihf",
    "aarch64": "aarch64-linux-gnu",
    "riscv": "riscv64-linux-gnu",
    "riscv32": "riscv32-linux-gnu",
}

JAVA_SOURCES = {
    "src/main/java/com/example/Calc.java": JAVA_CALC,
    "src/main/java/com/example/JniDemo.java": JAVA_JNI,
    "src/main/c/JniDemo.c": JNI_C,
}


def _run(command: list[str], cwd: Path | None = None, timeout: int = 180) -> subprocess.CompletedProcess:
    return subprocess.run(command, cwd=cwd, capture_output=True, text=True, timeout=timeout)


def _version(command: str) -> str | None:
    if not shutil.which(command):
        return None
    result = _run([command, "--version"], timeout=20)
    return (result.stdout or result.stderr).splitlines()[0][:240] if result.stdout or result.stderr else None


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")


def build_multiarche_dataset(out_dir: str | Path) -> dict[str, Any]:
    root = Path(out_dir)
    root.mkdir(parents=True, exist_ok=True)
    builds = root / "builds"
    builds.mkdir(parents=True, exist_ok=True)
    source = root / "source" / "multiarche_pilot.c"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text(C_SOURCE + "\n", encoding="utf-8")
    records = []
    skipped = []
    clang = shutil.which("clang")
    llvm_objdump = shutil.which("llvm-objdump")
    if not clang or not llvm_objdump:
        return {"status": "unavailable", "reason": "clang or llvm-objdump is missing", "dataset": str(root)}
    for name, triple in TARGETS.items():
        output = builds / f"multiarche_{name}.o"
        result = _run([clang, f"--target={triple}", "-O1", "-fno-omit-frame-pointer", "-c", str(source), "-o", str(output)])
        if result.returncode != 0:
            skipped.append({"target": name, "triple": triple, "reason": result.stderr[-800:]})
            continue
        try:
            bundle = analyze_binary(output)
        except Exception as exc:
            skipped.append({"target": name, "triple": triple, "reason": str(exc)})
            continue
        reconstruction = None
        if name in {"aarch64", "armv7", "riscv", "riscv32", "x86", "x86_64"}:
            reconstruction = reconstruct_and_compile(output, root / "reconstructions" / name, f"multiarche_{name}")
        records.append(
            {
                "sample_id": f"multiarche:{name}:O1",
                "target": name,
                "triple": triple,
                "artifact": str(output.relative_to(root)),
                "sha256": sha256_file(output),
                "size": output.stat().st_size,
                "architecture": (bundle.architecture or {}).get("name"),
                "abi": (bundle.architecture or {}).get("abi"),
                "functions": [function.name for function in bundle.functions],
                "instruction_count": len(bundle.disassembly.instructions) if bundle.disassembly else 0,
                "call_edges": len(bundle.callgraph.edges),
                "signatures": {
                    function.name: {
                        "calling_convention": signature.get("calling_convention"),
                        "return_type": signature.get("return_type"),
                        "parameters": [
                            {"name": item.get("name"), "type": item.get("type"), "register": item.get("register")}
                            for item in signature.get("parameters") or []
                        ],
                    }
                    for function in bundle.functions
                    for signature in [bundle.signatures.get(function.function_id, {})]
                },
                "reconstruction": reconstruction,
                "labels": {
                    "architecture": name,
                    "source_file": str(source.relative_to(root)),
                    "language": "c",
                    "function_names": [function.name for function in bundle.functions],
                },
                "provenance": {
                    "kind": "controlled_build",
                    "compiler": _version("clang"),
                    "optimization": "O1",
                    "flags": ["-fno-omit-frame-pointer", "--target=" + triple],
                    "license": "CC0-1.0",
                },
                "limitations": list(bundle.limitations),
            }
        )
    manifest = {
        "schema_version": "nexus.dataset.multiarche/v1",
        "dataset_id": "nexus-multiarche-pilot",
        "created_at": datetime.now(UTC).isoformat(),
        "license": "CC0-1.0",
        "source": str(source.relative_to(root)),
        "toolchain": {"clang": _version("clang"), "llvm_objdump": _version("llvm-objdump"), "objdump": _version("objdump")},
        "records": records,
        "skipped": skipped,
        "coverage": {
            "targets_requested": len(TARGETS),
            "targets_built": len(records),
            "architectures": sorted({item["architecture"] for item in records}),
        },
        "limitations": [
            "Cross-compiled relocatable objects are analyzed; not every target is linked into a shared object on this host.",
            "Reconstruction is compiled on the host and is not behaviorally verified against the target architecture.",
            "The dataset is a small controlled pilot, not a broad real-world corpus.",
        ],
    }
    _write_json(root / "manifest.json", manifest)
    with (root / "samples.jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, default=str) + "\n")
    return {"status": "completed", "dataset": str(root), **manifest["coverage"], "skipped": skipped}


def reconstruct_and_compile(binary: Path, out_dir: Path, project_name: str) -> dict[str, Any]:
    from nexus.reconstruction.c.generator import reconstruct_c
    shutil.rmtree(out_dir, ignore_errors=True)
    bundle = analyze_binary(binary)
    result = reconstruct_c(bundle, out_dir=out_dir, project_name=project_name)
    verification = compile_c_project(out_dir)
    return {
        "classification": result.get("classification"),
        "functions": [{"name": item["name"], "classification": item["classification"], "confidence": item["confidence"]} for item in result.get("functions", [])],
        "compile_status": verification.get("status"),
        "compile_rate": verification.get("compile_rate"),
        "errors": verification.get("errors"),
    }


def _build_java_class(root: Path, class_name: str, source: str) -> Path:
    javac = shutil.which("javac")
    if not javac:
        raise RuntimeError("javac is unavailable")
    source_path = root / "java_src" / "com" / "example" / f"{class_name}.java"
    source_path.parent.mkdir(parents=True, exist_ok=True)
    source_path.write_text(source + "\n", encoding="utf-8")
    classes = root / "java_classes"
    classes.mkdir(parents=True, exist_ok=True)
    result = _run([javac, "-d", str(classes), str(source_path)])
    if result.returncode != 0:
        raise RuntimeError(result.stderr[-1200:])
    return classes / "com" / "example" / f"{class_name}.class"


def _build_jni_library(root: Path) -> Path | None:
    gcc = shutil.which("gcc") or shutil.which("clang")
    if not gcc:
        return None
    java_home = None
    java = shutil.which("java")
    if java:
        resolved = Path(os.path.realpath(java))
        candidate = resolved.parent.parent
        if (candidate / "include" / "jni.h").exists():
            java_home = candidate
    if java_home is None:
        for candidate in (Path("/usr/lib/jvm"), Path("/usr/java")):
            if candidate.exists():
                for child in sorted(candidate.iterdir()):
                    if (child / "include" / "jni.h").exists():
                        java_home = child
                        break
            if java_home:
                break
    if java_home is None:
        return None
    source = root / "src" / "main" / "c" / "JniDemo.c"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text(JNI_C + "\n", encoding="utf-8")
    output = root / "libjni_demo.so"
    result = _run([gcc, "-shared", "-fPIC", "-I" + str(java_home / "include"), "-I" + str(java_home / "include" / "linux"), str(source), "-o", str(output)])
    return output if result.returncode == 0 and output.exists() else None


def build_java_dataset(out_dir: str | Path) -> dict[str, Any]:
    root = Path(out_dir)
    root.mkdir(parents=True, exist_ok=True)
    for relative, content in JAVA_SOURCES.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content + "\n", encoding="utf-8")
    records = []
    try:
        calc_class = _build_java_class(root, "Calc", JAVA_CALC)
        jni_class = _build_java_class(root, "JniDemo", JAVA_JNI)
    except RuntimeError as exc:
        return {"status": "unavailable", "reason": str(exc), "dataset": str(root)}
    jni_library = _build_jni_library(root)
    for class_path in (calc_class, jni_class):
        parsed = parse_java_class(class_path)
        project = root / "reconstructions" / parsed["simple_name"]
        shutil.rmtree(project, ignore_errors=True)
        reconstruction = reconstruct_java_artifact(class_path, out_dir=project, project_name=parsed["simple_name"].lower())
        verification = compile_java_project(project)
        records.append(
            {
                "sample_id": f"java-class:{parsed['name']}",
                "artifact": str(class_path.relative_to(root)),
                "sha256": sha256_file(class_path),
                "size": class_path.stat().st_size,
                "class_name": parsed.get("name"),
                "language": "java",
                "source_file": parsed.get("source_file"),
                "fields": len(parsed.get("fields") or []),
                "methods": len(parsed.get("methods") or []),
                "native_methods": [method.get("name") for method in parsed.get("methods") or [] if method.get("native")],
                "reconstruction": {
                    "classification": reconstruction.get("classification"),
                    "compile_status": verification.get("status"),
                    "compile_rate": verification.get("compile_rate"),
                    "files": len(reconstruction.get("files", {})),
                },
                "labels": {"language": "java", "class_name": parsed.get("name")},
                "provenance": {"kind": "controlled_source", "compiler": _version("javac"), "license": "CC0-1.0"},
                "limitations": parsed.get("limitations", []),
            }
        )
    jni_correlation = None
    if jni_library is not None:
        from nexus.jni.correlation import correlate_java_native
        jni_correlation = correlate_java_native(jni_class, jni_library)
    manifest = {
        "schema_version": "nexus.dataset.java/v1",
        "dataset_id": "nexus-java-jni-pilot",
        "created_at": datetime.now(UTC).isoformat(),
        "license": "CC0-1.0",
        "toolchain": {"javac": _version("javac"), "java": _version("java"), "gcc": _version("gcc")},
        "records": records,
        "jni_correlation": jni_correlation,
        "coverage": {
            "classes": len(records),
            "native_methods": sum(len(item["native_methods"]) for item in records),
            "jni_matches": (jni_correlation or {}).get("summary", {}).get("matched", 0),
        },
        "limitations": [
            "The Java pilot uses authored classes and a hand-written JNI library.",
            "Reconstruction is compiled as a syntax check; it is not behaviorally verified.",
            "DEX/APK coverage is validated separately with synthetic and bounded fixtures because d8 may be unavailable.",
        ],
    }
    _write_json(root / "manifest.json", manifest)
    with (root / "samples.jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, default=str) + "\n")
        if jni_correlation is not None:
            handle.write(json.dumps({"sample_id": "jni-correlation", "correlation": jni_correlation}, default=str) + "\n")
    return {"status": "completed", "dataset": str(root), **manifest["coverage"]}


def validate_advanced_dataset(path: str | Path) -> dict[str, Any]:
    root = Path(path)
    manifest_path = root / "manifest.json"
    if not manifest_path.exists():
        return {"status": "failed", "reason": "manifest.json is missing", "dataset": str(root)}
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    errors = []
    checked = 0
    schema = str(manifest.get("schema_version", ""))
    if schema == "nexus.dataset.java-foundation/v1":
        vocab_path = root / "vocab.json"
        store_manifest_path = root / "samples" / "manifest.json"
        if not vocab_path.exists():
            errors.append({"reason": "vocab.json is missing"})
        else:
            try:
                from nexus.foundation.vocab import Vocabulary
                vocab = Vocabulary.load(vocab_path)
                if manifest.get("vocab_sha256") != vocab.digest():
                    errors.append({"reason": "vocab hash mismatch"})
            except (ValueError, OSError) as exc:
                errors.append({"reason": f"vocab load failed: {exc}"})
        if not store_manifest_path.exists():
            errors.append({"reason": "sample store manifest is missing"})
        else:
            store = json.loads(store_manifest_path.read_text(encoding="utf-8"))
            if store.get("counts") != manifest.get("counts"):
                errors.append({"reason": "sample counts do not match store", "manifest": manifest.get("counts"), "store": store.get("counts")})
            for shard in store.get("shards", []):
                shard_path = root / "samples" / "shards" / shard.get("path", "")
                if not shard_path.exists():
                    errors.append({"reason": "missing shard", "path": shard.get("path")})
                elif sha256_file(shard_path) != shard.get("sha256"):
                    errors.append({"reason": "shard hash mismatch", "path": shard.get("path")})
        return {
            "status": "passed" if not errors else "failed",
            "dataset": str(root),
            "schema_version": schema,
            "records": sum(int(value) for value in (manifest.get("counts") or {}).values()),
            "artifacts_verified": 0 if errors else 1,
            "coverage": manifest.get("counts"),
            "errors": errors,
        }
    records = manifest.get("records", [])
    for record in records:
        artifact = record.get("artifact")
        expected = record.get("sha256")
        if not artifact or not expected:
            errors.append({"sample_id": record.get("sample_id"), "reason": "missing artifact/hash"})
            continue
        path_obj = root / artifact
        if not path_obj.exists():
            errors.append({"sample_id": record.get("sample_id"), "reason": "artifact missing"})
            continue
        actual = sha256_file(path_obj)
        if actual != expected:
            errors.append({"sample_id": record.get("sample_id"), "reason": "hash mismatch", "expected": expected, "actual": actual})
            continue
        checked += 1
        reconstruction = record.get("reconstruction") or {}
        if reconstruction.get("compile_status") not in {None, "passed"}:
            errors.append({"sample_id": record.get("sample_id"), "reason": "reconstruction did not compile", "status": reconstruction.get("compile_status")})
    coverage = manifest.get("coverage") or {}
    if records and checked != len(records):
        errors.append({"reason": "artifact verification incomplete", "checked": checked, "records": len(records)})
    return {
        "status": "passed" if not errors else "failed",
        "dataset": str(root),
        "schema_version": manifest.get("schema_version"),
        "records": len(records),
        "artifacts_verified": checked,
        "coverage": coverage,
        "errors": errors,
    }
