"""Adapters that run external tools and return typed evidence records."""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path

from nexus.toolchain.cache import ToolCache
from nexus.toolchain.models import ToolResult, cache_key
from nexus.toolchain.registry import ToolRegistry


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _now() -> str:
    return datetime.now(UTC).isoformat()


class GhidraAdapter:
    name = "ghidra_headless"

    def __init__(self, cache: ToolCache | None = None) -> None:
        identity = ToolRegistry().discover()[self.name]
        self.identity = identity
        self.cache = cache or ToolCache()

    def decompile(self, path: str | Path, timeout: int = 600, use_cache: bool = True) -> ToolResult:
        from nexus.reconstruction.c.ghidra import decompile_binary

        binary = Path(path).resolve()
        input_sha = _sha256(binary)
        arguments = {"operation": "decompile", "timeout": timeout}
        key = cache_key(input_sha, self.name, self.identity.version, arguments)
        if use_cache:
            cached = self.cache.get_result(key)
            if cached is not None and cached.status == "completed":
                cached.cached = True
                return cached
        started = _now()
        payload = decompile_binary(binary, timeout=timeout)
        finished = _now()
        functions = payload.get("functions") or []
        result = ToolResult(
            tool=self.name,
            version=self.identity.version,
            input_sha256=input_sha,
            arguments=arguments,
            command=[str(self.identity.path or "analyzeHeadless")],
            status=str(payload.get("status") or "unknown"),
            started_at=started,
            finished_at=finished,
            artifacts={
                "functions": [
                    {
                        "name": item.get("name"),
                        "entry": item.get("entry"),
                        "entry_va": item.get("entry_va"),
                        "ghidra_entry": item.get("ghidra_entry"),
                        "source": item.get("source"),
                    }
                    for item in functions
                ],
                "image_base": payload.get("image_base"),
                "elapsed_seconds": payload.get("elapsed_seconds"),
            },
            warnings=[str(payload.get("reason"))] if payload.get("reason") else [],
            limitations=list(payload.get("limitations") or []),
            error=payload.get("reason"),
        )
        if result.status == "completed":
            self.cache.put_result(key, result)
        return result


class JadxAdapter:
    name = "jadx"

    def __init__(self, cache: ToolCache | None = None) -> None:
        identity = ToolRegistry().discover()[self.name]
        self.identity = identity
        self.cache = cache or ToolCache()

    def decompile(self, path: str | Path, output_dir: str | Path, timeout: int = 600, use_cache: bool = True) -> ToolResult:
        source = Path(path)
        output = Path(output_dir).resolve()
        input_sha = _sha256(source)
        threads = max(1, min(8, os.cpu_count() or 4))
        arguments = {
            "operation": "decompile",
            "output_dir": str(output),
            "timeout": timeout,
            "flags": ["--no-res", "--show-bad-code"],
            "threads": threads,
        }
        key = cache_key(input_sha, self.name, self.identity.version, arguments)
        if use_cache:
            cached = self.cache.get_result(key)
            if cached is not None and cached.status in {"completed", "completed_with_errors"}:
                if output.exists() and any(output.rglob("*.java")):
                    cached.cached = True
                    return cached
        if self.identity.path is None:
            return ToolResult(tool=self.name, version=self.identity.version, input_sha256=input_sha, status="unavailable", error="jadx was not found")
        if output.exists():
            shutil.rmtree(output)
        output.mkdir(parents=True, exist_ok=True)
        command = [
            str(self.identity.path),
            "--no-res",
            "--show-bad-code",
            "--threads-count",
            str(threads),
            "-d",
            str(output),
            str(source),
        ]
        started = _now()
        timed_out = False
        try:
            process = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
            returncode = process.returncode
            error = None if returncode == 0 else (process.stderr or process.stdout or "")[-3000:]
        except (OSError, subprocess.TimeoutExpired) as exc:
            returncode = None
            timed_out = isinstance(exc, subprocess.TimeoutExpired)
            error = str(exc)
        files = sorted(str(item.relative_to(output)) for item in output.rglob("*") if item.is_file())
        java_files = [item for item in files if item.endswith(".java")]
        if returncode == 0:
            status = "completed"
        elif java_files:
            status = "completed_with_errors"
        elif timed_out:
            status = "timeout"
        else:
            status = "failed"
        warnings = []
        if status == "completed_with_errors":
            warnings.append("JADX reported decompilation errors for some classes; emitted sources are partial.")
        if timed_out and java_files:
            warnings.append("JADX timed out after emitting partial sources.")
        result = ToolResult(
            tool=self.name,
            version=self.identity.version,
            input_sha256=input_sha,
            arguments=arguments,
            command=command,
            status=status,
            started_at=started,
            finished_at=_now(),
            artifacts={"output_dir": str(output), "java_files": java_files, "file_count": len(files)},
            warnings=warnings,
            limitations=["JADX output is reconstructed Java, not original source.", "Decompilation errors and omitted resources may remain."],
            error=error,
        )
        if status in {"completed", "completed_with_errors"}:
            self.cache.put_result(key, result)
        return result


class CfrAdapter:
    name = "cfr"

    def __init__(self, cache: ToolCache | None = None) -> None:
        identity = ToolRegistry().discover()[self.name]
        self.identity = identity
        self.cache = cache or ToolCache()

    def decompile(self, path: str | Path, output_dir: str | Path, timeout: int = 600, use_cache: bool = True) -> ToolResult:
        source = Path(path)
        output = Path(output_dir).resolve()
        input_sha = _sha256(source)
        arguments = {"operation": "decompile", "output_dir": str(output), "timeout": timeout}
        key = cache_key(input_sha, self.name, self.identity.version, arguments)
        if use_cache:
            cached = self.cache.get_result(key)
            if cached is not None and cached.status in {"completed", "completed_with_errors"}:
                if output.exists() and any(output.rglob("*.java")):
                    cached.cached = True
                    return cached
        if self.identity.path is None:
            return ToolResult(tool=self.name, version=self.identity.version, input_sha256=input_sha, status="unavailable", error="cfr was not found")
        if output.exists():
            shutil.rmtree(output)
        output.mkdir(parents=True, exist_ok=True)
        command = ["java", "-jar", str(self.identity.path), str(source), "--outputdir", str(output)]
        started = _now()
        timed_out = False
        try:
            process = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
            returncode = process.returncode
            error = None if returncode == 0 else (process.stderr or process.stdout or "")[-3000:]
        except (OSError, subprocess.TimeoutExpired) as exc:
            returncode = None
            timed_out = isinstance(exc, subprocess.TimeoutExpired)
            error = str(exc)
        files = sorted(str(item.relative_to(output)) for item in output.rglob("*") if item.is_file())
        java_files = [item for item in files if item.endswith(".java")]
        if returncode == 0:
            status = "completed"
        elif java_files:
            status = "completed_with_errors"
        elif timed_out:
            status = "timeout"
        else:
            status = "failed"
        warnings = []
        if status == "completed_with_errors":
            warnings.append("CFR reported errors for some classes; emitted sources are partial.")
        if timed_out and java_files:
            warnings.append("CFR timed out after emitting partial sources.")
        result = ToolResult(
            tool=self.name,
            version=self.identity.version,
            input_sha256=input_sha,
            arguments=arguments,
            command=command,
            status=status,
            started_at=started,
            finished_at=_now(),
            artifacts={"output_dir": str(output), "java_files": java_files, "file_count": len(files)},
            warnings=warnings,
            limitations=["CFR output is reconstructed Java, not original source.", "Independent translator disagreement is useful evidence, not truth."],
            error=error,
        )
        if status in {"completed", "completed_with_errors"}:
            self.cache.put_result(key, result)
        return result
