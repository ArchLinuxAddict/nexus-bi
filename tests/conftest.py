import json
import shutil
import subprocess
from pathlib import Path

import pytest

from nexus.core.config import NexusPaths
from nexus.core.engine import Engine
from nexus.storage.store import Storage


def _run(command, cwd=None):
    return subprocess.run(command, cwd=cwd, capture_output=True, text=True, timeout=120)


@pytest.fixture(scope="session")
def build_root(tmp_path_factory):
    return tmp_path_factory.mktemp("nexus-build")


@pytest.fixture(scope="session")
def simple_binary(build_root):
    if not shutil.which("gcc"):
        pytest.skip("gcc is not installed")
    source = build_root / "simple.c"
    source.write_text(
        "int add(int a, int b) { return a + b; }\n"
        "int multiply(int a, int b) { return a * b; }\n"
        "int main(void) { return add(2, 3) + multiply(4, 5); }\n",
        encoding="utf-8",
    )
    binary = build_root / "simple"
    result = _run(["gcc", "-O0", "-g", "-fno-pie", "-no-pie", "-fno-omit-frame-pointer", "-o", str(binary), str(source)])
    if result.returncode != 0:
        pytest.skip(f"fixture compile failed: {result.stderr[:400]}")
    return binary


@pytest.fixture(scope="session")
def stripped_binary(build_root, simple_binary):
    target = build_root / "simple-stripped"
    shutil.copy2(simple_binary, target)
    if shutil.which("strip"):
        result = _run(["strip", "--strip-all", str(target)])
        if result.returncode != 0:
            pytest.skip(f"strip failed: {result.stderr[:400]}")
    return target


@pytest.fixture(scope="session")
def unsafe_binary(build_root):
    if not shutil.which("gcc"):
        pytest.skip("gcc is not installed")
    source = build_root / "unsafe.c"
    source.write_text(
        "#include <string.h>\n#include <stdlib.h>\n"
        "void copy_input(char *dst, const char *src) { strcpy(dst, src); }\n"
        "int run_command(const char *cmd) { return system(cmd); }\n"
        "int main(int argc, char **argv) { char buffer[16]; copy_input(buffer, argv[0]); return run_command(\"true\"); }\n",
        encoding="utf-8",
    )
    binary = build_root / "unsafe"
    result = _run(["gcc", "-O0", "-g", "-fno-pie", "-no-pie", "-o", str(binary), str(source)])
    if result.returncode != 0:
        pytest.skip(f"unsafe fixture compile failed: {result.stderr[:400]}")
    return binary


@pytest.fixture(scope="session")
def branching_binary(build_root):
    if not shutil.which("gcc"):
        pytest.skip("gcc is not installed")
    source = build_root / "branching.c"
    source.write_text(
        "int classify(int value) { if (value < 0) return -1; if (value == 0) return 0; return 1; }\n"
        "int maximum(int a, int b) { return a > b ? a : b; }\n"
        "int main(void) { return classify(-7) + maximum(4, 9); }\n",
        encoding="utf-8",
    )
    binary = build_root / "branching"
    result = _run(["gcc", "-O0", "-g", "-fno-pie", "-no-pie", "-o", str(binary), str(source)])
    if result.returncode != 0:
        pytest.skip(f"branching fixture compile failed: {result.stderr[:400]}")
    return binary


@pytest.fixture(scope="session")
def loop_binary(build_root):
    if not shutil.which("gcc"):
        pytest.skip("gcc is not installed")
    source = build_root / "loops.c"
    source.write_text(
        "unsigned sum_to(unsigned n) { unsigned total = 0; for (unsigned i = 0; i < n; i++) total += i; return total; }\n"
        "int main(void) { return (int)sum_to(10); }\n",
        encoding="utf-8",
    )
    binary = build_root / "loops"
    result = _run(["gcc", "-O0", "-g", "-fno-pie", "-no-pie", "-o", str(binary), str(source)])
    if result.returncode != 0:
        pytest.skip(f"loop fixture compile failed: {result.stderr[:400]}")
    return binary


@pytest.fixture(scope="session")
def jni_binary(build_root):
    if not shutil.which("gcc") or not shutil.which("javac"):
        pytest.skip("gcc or javac is not installed")
    java_home_result = _run(["bash", "-lc", "dirname $(dirname $(readlink -f $(command -v javac)))"])
    java_home = Path(java_home_result.stdout.strip())
    include = java_home / "include"
    linux_include = include / "linux"
    if not (include / "jni.h").exists() or not linux_include.exists():
        pytest.skip("JNI headers are not installed")
    work = build_root / "jni"
    work.mkdir(exist_ok=True)
    (work / "classes").mkdir(exist_ok=True)
    (work / "include").mkdir(exist_ok=True)
    (work / "JniDemo.java").write_text(
        "package com.example;\n"
        "public class JniDemo {\n"
        "    public static native int add(int a, int b);\n"
        "    public static native long add(long a, long b);\n"
        "    static { System.loadLibrary(\"jni_demo\"); }\n"
        "}\n",
        encoding="utf-8",
    )
    result = _run(["javac", "-d", str(work / "classes"), "-h", str(work / "include"), str(work / "JniDemo.java")], cwd=work)
    if result.returncode != 0:
        pytest.skip(f"javac fixture failed: {result.stderr[:400]}")
    (work / "JniDemo.c").write_text(
        "#include <jni.h>\n"
        "JNIEXPORT jint JNICALL Java_com_example_JniDemo_add__II(JNIEnv *env, jclass cls, jint a, jint b) { return a + b; }\n"
        "JNIEXPORT jlong JNICALL Java_com_example_JniDemo_add__JJ(JNIEnv *env, jclass cls, jlong a, jlong b) { return a + b; }\n"
        "JNIEXPORT jint JNICALL JNI_OnLoad(JavaVM *vm, void *reserved) { return JNI_VERSION_1_8; }\n",
        encoding="utf-8",
    )
    library = work / "libjni_demo.so"
    result = _run(["gcc", "-shared", "-fPIC", "-O2", "-g", f"-I{include}", f"-I{linux_include}", "-o", str(library), str(work / "JniDemo.c")], cwd=work)
    if result.returncode != 0:
        pytest.skip(f"JNI library compile failed: {result.stderr[:400]}")
    return library


@pytest.fixture()
def engine(tmp_path):
    paths = NexusPaths(root=tmp_path / "nexus-home").ensure()
    storage = Storage(paths)
    return Engine(paths, storage)


@pytest.fixture(scope="session")
def dataset_root(build_root):
    from nexus.datasets.factory import generate_dataset
    target = build_root / "dataset"
    result = generate_dataset(
        target,
        fixtures=["arithmetic", "strings"],
        compilers=["gcc"],
        optimizations=["O0"],
        debug_modes=[True, False],
        pie_modes=[True],
    )
    if result["summary"]["samples"] == 0:
        pytest.skip("dataset generation produced no samples")
    return target


@pytest.fixture(scope="session")
def realworld_dataset(build_root):
    import io
    import tarfile

    from nexus.datasets.realworld import BuildVariant, ingest_candidates

    if not shutil.which("gcc"):
        pytest.skip("gcc is not installed")
    root = build_root / "realworld"
    archive = root / "sources" / "example_demo.tar.gz"
    archive.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "w") as handle:
        for name, payload in {
            "demo-1.0": None,
            "demo-1.0/main.c": (
                b"#include <stdio.h>\n"
                b"int add(int a, int b) { return a + b; }\n"
                b"int main(void) { printf(\"%d\\n\", add(2, 3)); return 0; }\n"
            ),
            "demo-1.0/LICENSE": b"MIT License\n",
        }.items():
            info = tarfile.TarInfo(name)
            if payload is None:
                info.type = tarfile.DIRTYPE
                info.mode = 0o755
                handle.addfile(info)
            else:
                info.size = len(payload)
                info.mode = 0o644
                handle.addfile(info, io.BytesIO(payload))
    candidates = build_root / "candidates.json"
    candidates.write_text(
        json.dumps(
            [
                {
                    "full_name": "example/demo",
                    "commit": "0123456789abcdef",
                    "license": "MIT",
                    "description": "small parser demo",
                    "topics": ["parser"],
                }
            ]
        ),
        encoding="utf-8",
    )
    result = ingest_candidates(
        candidates,
        root,
        max_projects=1,
        variants=[BuildVariant("gcc", "O0", True)],
    )
    if result["summary"]["samples"] == 0:
        pytest.skip("real-world fixture build produced no samples")
    return root
