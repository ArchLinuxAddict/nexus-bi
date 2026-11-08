import os
import platform
import shutil
import subprocess
from pathlib import Path
from typing import Any


def _read(path: str) -> str:
    try:
        return Path(path).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def _command_version(command: str, args: list[str] | None = None) -> str | None:
    if shutil.which(command) is None:
        return None
    try:
        result = subprocess.run(
            [command, *(args or ["--version"])],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    text = result.stdout.strip() or result.stderr.strip()
    return text.splitlines()[0] if text else None


def _cpu_model() -> str:
    for line in _read("/proc/cpuinfo").splitlines():
        if line.lower().startswith("model name"):
            return line.split(":", 1)[1].strip()
    return platform.processor() or "unknown"


def _memory() -> dict[str, int]:
    values = {}
    for line in _read("/proc/meminfo").splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        if key in {"MemTotal", "MemAvailable", "SwapTotal"}:
            values[key] = int(value.strip().split()[0]) * 1024
    return values


def _nvidia() -> dict[str, Any] | None:
    if shutil.which("nvidia-smi") is None:
        return None
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0 or not result.stdout.strip():
        return None
    rows = []
    for line in result.stdout.strip().splitlines():
        parts = [part.strip() for part in line.split(",")]
        rows.append({"name": parts[0], "memory_mb": parts[1] if len(parts) > 1 else None, "driver": parts[2] if len(parts) > 2 else None})
    return {"source": "nvidia-smi", "gpus": rows}


def _torch_cuda() -> dict[str, Any] | None:
    try:
        import torch
    except Exception:
        return None
    try:
        if not torch.cuda.is_available():
            return None
        gpus = []
        for index in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(index)
            gpus.append({
                "name": props.name,
                "memory_mb": int(props.total_memory / (1024 * 1024)),
                "major": props.major,
                "minor": props.minor,
            })
        return {"source": "torch", "gpus": gpus}
    except Exception:
        return None


def hardware_report() -> dict[str, Any]:
    disk = shutil.disk_usage("/")
    memory = _memory()
    nvidia = _nvidia()
    torch_cuda = _torch_cuda()
    dri = sorted(os.listdir("/dev/dri")) if os.path.isdir("/dev/dri") else []
    qemu = {name: shutil.which(name) for name in ("qemu-x86_64", "qemu-aarch64", "qemu-arm", "qemu-system-x86_64")}
    compilers = {
        "gcc": _command_version("gcc"),
        "clang": _command_version("clang"),
        "javac": _command_version("javac", ["-version"]),
        "mvn": _command_version("mvn", ["-version"]),
        "gradle": _command_version("gradle", ["--version"]),
        "objdump": _command_version("objdump"),
        "readelf": _command_version("readelf"),
        "llvm-objdump": _command_version("llvm-objdump"),
    }
    gpu_present = bool(nvidia or torch_cuda or dri)
    gpu_compute = bool(nvidia or torch_cuda)
    return {
        "host": {
            "platform": platform.platform(),
            "kernel": platform.uname().release,
            "cpu_model": _cpu_model(),
            "logical_cpus": os.cpu_count(),
            "memory_total_bytes": memory.get("MemTotal"),
            "memory_available_bytes": memory.get("MemAvailable"),
            "swap_total_bytes": memory.get("SwapTotal", 0),
            "disk_total_bytes": disk.total,
            "disk_free_bytes": disk.free,
        },
        "gpu": {
            "present": gpu_present,
            "compute_available": gpu_compute,
            "dri": dri,
            "nvidia": nvidia,
            "torch_cuda": torch_cuda,
        },
        "isolation": {
            "kvm": os.path.exists("/dev/kvm"),
            "bubblewrap": shutil.which("bwrap"),
            "qemu": qemu,
        },
        "tools": compilers,
        "large_training_ready": gpu_compute,
        "notes": [
            "GPU compute requires CUDA/ROCm or a validated accelerator stack.",
            "Cross-architecture execution requires QEMU or an equivalent runner.",
            "Untrusted dynamic execution requires a verified KVM sandbox.",
        ],
    }
