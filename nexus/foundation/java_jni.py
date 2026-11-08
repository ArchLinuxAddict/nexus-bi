"""Java/JNI training-data readiness schema and controlled-fixture loader.

This module prepares provenance-checked samples; it does not train or claim a
Java/JNI capability.  Controlled fixtures are kept distinct from real-world data.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

ALLOWED_LICENSES = {"CC0-1.0", "MIT", "Apache-2.0", "BSD-3-Clause", "BSD-2-Clause"}
REQUIRED_FIELDS = (
    "sample_id",
    "project_id",
    "java_source_path",
    "native_binary_path",
    "class_name",
    "method_name",
    "signature",
    "license",
    "provenance",
    "controlled_fixture",
)


@dataclass
class JavaJNISample:
    sample_id: str
    project_id: str
    java_source_path: str
    native_binary_path: str
    class_name: str
    method_name: str
    signature: str
    license: str
    provenance: dict[str, Any]
    controlled_fixture: bool
    split: str = "train"
    native_entries: list[dict[str, Any]] = field(default_factory=list)
    register_natives: bool = False
    overloaded: bool = False
    parameter_types: list[str] = field(default_factory=list)
    return_type: str = "unknown"
    arrays_and_strings: bool = False
    jni_references: list[str] = field(default_factory=list)
    exception_paths: list[str] = field(default_factory=list)
    callbacks: list[str] = field(default_factory=list)
    cross_language_state: list[str] = field(default_factory=list)
    limitations: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def validate_java_jni_sample(sample: JavaJNISample | dict[str, Any], root: str | Path) -> dict[str, Any]:
    data = sample.to_dict() if isinstance(sample, JavaJNISample) else dict(sample)
    problems = []
    for field_name in REQUIRED_FIELDS:
        if not data.get(field_name):
            problems.append(f"missing field: {field_name}")
    if data.get("license") not in ALLOWED_LICENSES:
        problems.append(f"license is not in the allowlist: {data.get('license')}")
    if data.get("controlled_fixture") is not True:
        problems.append("no real-world Java/JNI corpus is approved; controlled_fixture must be true for this milestone")
    base = Path(root)
    for key in ("java_source_path", "native_binary_path"):
        relative = data.get(key)
        if not relative:
            continue
        path = base / relative
        if not path.exists():
            problems.append(f"missing artifact: {key}={relative}")
        else:
            expected = (data.get("provenance") or {}).get(f"{key}_sha256")
            if expected and _sha(path) != expected:
                problems.append(f"hash mismatch: {key}")
    if not data.get("native_entries"):
        problems.append("no deterministic JNI entries were supplied")
    return {"status": "PASS" if not problems else "FAIL", "problems": problems, "sample": data}


def controlled_fixture_manifest(fixture_root: str | Path = "artifacts/e2e/build/jni") -> dict[str, Any]:
    root = Path(fixture_root)
    native = root / "libjni_demo.so"
    java = Path("tests/fixtures/JniDemo.java")
    c_source = Path("tests/fixtures/JniDemo.c")
    candidates = []
    if java.exists() and native.exists():
        candidates.append(
            JavaJNISample(
                sample_id="controlled_jni_demo_add",
                project_id="controlled_jni_demo",
                java_source_path=str(java),
                native_binary_path=str(native),
                class_name="com.example.JniDemo",
                method_name="add",
                signature="(II)I",
                license="MIT",
                provenance={
                    "java_source_path_sha256": _sha(java),
                    "native_binary_path_sha256": _sha(native),
                    "c_source_sha256": _sha(c_source) if c_source.exists() else None,
                    "repository": "nexus-authored test fixture",
                },
                controlled_fixture=True,
                split="controlled",
                native_entries=[
                    {"kind": "jni_symbol", "name": "Java_com_example_JniDemo_add__II"},
                    {"kind": "jni_symbol", "name": "Java_com_example_JniDemo_add__JJ"},
                ],
                overloaded=True,
                parameter_types=["jint", "jint"],
                return_type="jint",
                jni_references=["JNI_OnLoad"],
                limitations=["Controlled fixture only; no real-world Java/JNI corpus or held-out project."],
            )
        )
    statuses = [validate_java_jni_sample(sample, Path.cwd()) for sample in candidates]
    return {
        "schema_version": "0.1.0",
        "fixture_root": str(root),
        "samples": [sample.to_dict() for sample in candidates],
        "validation": statuses,
        "status": "PASS" if candidates and all(item["status"] == "PASS" for item in statuses) else "INSUFFICIENT_DATA",
        "coverage": {
            "java_methods": bool(candidates),
            "jni_signatures": True,
            "register_natives": False,
            "overloads": True,
            "primitive_parameters": True,
            "object_parameters": False,
            "arrays": False,
            "strings": False,
            "jni_references": False,
            "exception_propagation": False,
            "native_callbacks": False,
            "cross_language_state": False,
        },
        "real_world_corpus": {"status": "INSUFFICIENT_DATA", "reason": "No licensed, provenance-recorded real-world Java/JNI project set is present."},
        "training_readiness": {"status": "INSUFFICIENT_DATA", "reason": "Controlled fixtures alone are not a corpus; do not claim Java/JNI capability."},
    }


def readiness_report() -> dict[str, Any]:
    return {
        "schema_version": "0.1.0",
        "requirements": {
            "java_methods": "Java source/class metadata, method name and descriptor",
            "jni_signatures": "deterministic mangled symbols or RegisterNatives table",
            "register_natives": "JNI_OnLoad and method table evidence",
            "overloads": "descriptor-distinct Java and native methods",
            "parameters": "primitive/object/array/string types and return type",
            "references": "local/global reference creation/use/delete evidence",
            "exceptions": "Throw/ExceptionCheck/ExceptionClear evidence",
            "callbacks": "Java callback invocation and re-entry evidence",
            "state": "cross-language state/object identity evidence",
        },
        "status": "INSUFFICIENT_DATA",
        "reason": "A controlled fixture exists, but no licensed real-world Java/JNI corpus is available.",
        "not_claimed": ["Java reconstruction capability", "JNI signature accuracy", "behavioral JNI equivalence"],
    }
