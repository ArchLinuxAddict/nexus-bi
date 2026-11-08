import subprocess
from pathlib import Path
from typing import Any

from nexus.sandbox.gate import differential_test
from nexus.sandbox.gate import status as sandbox_status
from nexus.verification.toolchains import (
    available_compilers,
    available_java,
    detect_toolchains,
)


def _run(command: list[str], cwd: Path, timeout: int = 60) -> dict[str, Any]:
    try:
        result = subprocess.run(command, cwd=cwd, capture_output=True, text=True, timeout=timeout)
        output = (result.stdout or "") + (result.stderr or "")
        return {
            "returncode": result.returncode,
            "stdout": result.stdout[-4000:],
            "stderr": result.stderr[-4000:],
            "warnings": output.count("warning:"),
            "errors": output.count("error:"),
            "output_tail": output[-4000:],
        }
    except subprocess.TimeoutExpired:
        return {"returncode": None, "stdout": "", "stderr": "timeout", "warnings": 0, "errors": 1, "output_tail": "timeout"}
    except OSError as exc:
        return {"returncode": None, "stdout": "", "stderr": str(exc), "warnings": 0, "errors": 1, "output_tail": str(exc)}


def compile_c_project(project_dir: str | Path, compiler: str | None = None, timeout: int = 60) -> dict[str, Any]:
    root = Path(project_dir).resolve()
    if not root.is_dir():
        return {"status": "failed", "reason": "project directory does not exist", "files": [], "dynamic": differential_test()}
    compilers = available_compilers()
    if compiler is None:
        compiler = compilers[0] if compilers else None
    if compiler is None:
        return {"status": "unavailable", "reason": "no C compiler found", "files": [], "dynamic": differential_test()}
    sources = sorted(root.glob("src/**/*.c"))
    if not sources:
        return {"status": "skipped", "reason": "no C source files", "files": [], "dynamic": differential_test()}
    objects = root / "build" / "objects"
    objects.mkdir(parents=True, exist_ok=True)
    files = []
    for source in sources:
        object_path = objects / (source.stem + ".o")
        command = [compiler, "-std=c11", "-Wall", "-Wextra", "-Iinclude", "-c", str(source), "-o", str(object_path)]
        result = _run(command, root, timeout)
        files.append({"path": str(source.relative_to(root)), "command": command, **result})
    smoke = root / "tests" / "smoke.c"
    if smoke.exists():
        command = [compiler, "-std=c11", "-Wall", "-Wextra", "-Iinclude", "-c", str(smoke), "-o", str(objects / "smoke.o")]
        result = _run(command, root, timeout)
        files.append({"path": str(smoke.relative_to(root)), "command": command, **result})
    passed = sum(1 for item in files if item["returncode"] == 0)
    warnings = sum(item["warnings"] for item in files)
    errors = sum(item["errors"] for item in files)
    version = detect_toolchains().get(Path(compiler).name)
    return {
        "status": "passed" if passed == len(files) else "failed",
        "compiler": compiler,
        "compiler_version": version.version if version else None,
        "files": files,
        "compile_rate": passed / len(files) if files else 0.0,
        "warnings": warnings,
        "errors": errors,
        "dynamic": differential_test(),
        "sandbox": sandbox_status().to_dict(),
        "limitations": [
            "Compilation is not behavioral verification.",
            "Generated test sources are compiled but not executed.",
            "Dynamic differential testing is disabled until a verified sandbox exists.",
        ],
    }


EXTERNAL_PLATFORM_IMPORTS = (
    "android.",
    "androidx.",
    "com.google.android.",
    "dalvik.",
    "kotlin.",
    "kotlinx.",
)


def _external_platform_import(sources: list[Path], limit: int = 500) -> str | None:
    for source in sources[:limit]:
        try:
            text = source.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for line in text.splitlines():
            stripped = line.strip()
            if not stripped.startswith("import "):
                continue
            module = stripped[7:].split(";", 1)[0].strip()
            if any(module.startswith(prefix) for prefix in EXTERNAL_PLATFORM_IMPORTS):
                return module
    return None


def compile_java_project(project_dir: str | Path, timeout: int = 60) -> dict[str, Any]:
    root = Path(project_dir).resolve()
    if not root.is_dir():
        return {"status": "failed", "reason": "project directory does not exist", "files": [], "dynamic": differential_test()}
    if not available_java():
        return {"status": "unavailable", "reason": "javac not found", "files": [], "dynamic": differential_test()}
    sources = sorted(root.glob("src/main/java/**/*.java"))
    if not sources:
        return {"status": "skipped", "reason": "no Java source files", "files": [], "dynamic": differential_test()}
    dependency = _external_platform_import(sources)
    if dependency is not None:
        return {
            "status": "unavailable",
            "reason": f"external platform dependency is not installed: {dependency}",
            "compiler": "javac",
            "files": [],
            "compile_rate": 0.0,
            "warnings": 0,
            "errors": 0,
            "dynamic": differential_test(),
            "sandbox": sandbox_status().to_dict(),
            "limitations": [
                "Compilation requires the Android/Kotlin classpath, which is not installed.",
                "Compilation is not behavioral verification.",
                "Dynamic differential testing is disabled until a verified sandbox exists.",
            ],
        }
    classes = root / "build" / "classes"
    classes.mkdir(parents=True, exist_ok=True)
    command = ["javac", "--release", "17", "-d", str(classes), *[str(source) for source in sources]]
    result = _run(command, root, timeout)
    version = detect_toolchains()["javac"]
    files = [{"path": str(source.relative_to(root)), "returncode": result["returncode"], "warnings": result["warnings"], "errors": result["errors"], "output_tail": result["output_tail"]} for source in sources]
    return {
        "status": "passed" if result["returncode"] == 0 else "failed",
        "compiler": "javac",
        "compiler_version": version.version,
        "files": files,
        "compile_rate": 1.0 if result["returncode"] == 0 else 0.0,
        "warnings": result["warnings"],
        "errors": result["errors"],
        "dynamic": differential_test(),
        "sandbox": sandbox_status().to_dict(),
        "limitations": [
            "Compilation is not behavioral verification.",
            "Native libraries are not loaded or executed.",
            "Dynamic differential testing is disabled until a verified sandbox exists.",
        ],
    }


def verify_reconstruction(reconstruction: dict[str, Any], project_dir: str | Path | None = None) -> dict[str, Any]:
    language = reconstruction.get("language")
    if project_dir is None:
        return {"status": "failed", "reason": "project directory is required"}
    if language == "c":
        result = compile_c_project(project_dir)
    elif language == "java":
        result = compile_java_project(project_dir)
    else:
        result = {"status": "unsupported", "reason": f"unknown reconstruction language: {language}"}
    result["language"] = language
    result["classification"] = reconstruction.get("classification")
    return result
