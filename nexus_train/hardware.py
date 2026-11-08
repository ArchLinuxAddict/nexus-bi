"""Read-only hardware detection for the standalone runtime."""

from __future__ import annotations

import os
import platform
import resource
import shutil
import subprocess
from pathlib import Path
from typing import Any

import torch


def _read(path: str) -> str | None:
    try:
        return Path(path).read_text(encoding="utf-8").strip()
    except OSError:
        return None


def _meminfo() -> dict[str, int]:
    values: dict[str, int] = {}
    text = _read("/proc/meminfo") or ""
    for line in text.splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        parts = value.strip().split()
        if parts and parts[0].isdigit():
            values[key] = int(parts[0]) * 1024
    return values


def _swap_devices() -> list[dict[str, Any]]:
    result = []
    for line in (_read("/proc/swaps") or "").splitlines()[1:]:
        parts = line.split()
        if len(parts) >= 3:
            result.append({"path": parts[0], "type": parts[1], "size_bytes": int(parts[2]) * 1024, "used_bytes": int(parts[3]) * 1024})
    return result


def _temperatures() -> dict[str, Any]:
    zones = []
    for path in sorted(Path("/sys/class/thermal").glob("thermal_zone*/temp")):
        try:
            zones.append({"zone": path.parent.name, "celsius": int(path.read_text().strip()) / 1000.0})
        except (OSError, ValueError):
            continue
    package = None
    try:
        output = subprocess.run(["sensors"], capture_output=True, text=True, timeout=5).stdout
        for line in output.splitlines():
            if "Package id 0" in line and "°C" in line:
                package = float(line.split("+")[1].split("°")[0])
                break
    except (OSError, subprocess.TimeoutExpired, ValueError, IndexError):
        pass
    return {"zones": zones, "package_celsius": package}


def _cpu() -> dict[str, Any]:
    info = {}
    text = _read("/proc/cpuinfo") or ""
    for line in text.splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            if key.strip() == "model name":
                info["model"] = value.strip()
                break
    info["logical_cpus"] = os.cpu_count() or 0
    try:
        import subprocess as sp

        lscpu = sp.run(["lscpu"], capture_output=True, text=True, timeout=5).stdout
        for line in lscpu.splitlines():
            if "Core(s) per socket" in line:
                info["physical_cores"] = int(line.split(":")[1].strip())
            if "Thread(s) per core" in line:
                info["threads_per_core"] = int(line.split(":")[1].strip())
            if "CPU max MHz" in line:
                info["max_mhz"] = float(line.split(":")[1].strip())
    except Exception:
        pass
    flags = []
    for line in text.splitlines():
        if line.startswith("flags"):
            flags = line.split(":", 1)[1].split()
            break
    info["avx2"] = "avx2" in flags
    info["avx512"] = "avx512f" in flags
    governor = _read("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor")
    info["governor"] = governor
    return info


def detect_hardware() -> dict[str, Any]:
    mem = _meminfo()
    disk = shutil.disk_usage(Path.cwd())
    torch_threads = torch.get_num_threads()
    xpu = getattr(torch, "xpu", None)
    gpu_name = "not_detected"
    try:
        result = subprocess.run(["lspci"], capture_output=True, text=True, timeout=5)
        for line in result.stdout.splitlines():
            if any(key in line.lower() for key in ("vga", "3d controller", "display controller")):
                gpu_name = line.split(": ", 1)[-1].strip()
                break
    except (OSError, subprocess.TimeoutExpired):
        pass
    backend = "cpu_fp32"
    if bool(torch.cuda.is_available()):
        backend = "cuda"
    elif xpu is not None and bool(xpu.is_available()):
        backend = "xpu"
    gpu = {
        "name": gpu_name,
        "torch_version": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "xpu_available": bool(xpu.is_available()) if xpu is not None else False,
        "xpu_device_count": int(xpu.device_count()) if xpu is not None and xpu.is_available() else 0,
        "ipex_available": False,
        "supported_training_backend": backend,
    }
    try:
        import intel_extension_for_pytorch  # type: ignore  # noqa: F401
        gpu["ipex_available"] = True
    except ImportError:
        pass
    return {
        "hostname": platform.node(),
        "os": platform.platform(),
        "python": platform.python_version(),
        "cpu": _cpu(),
        "memory": {"total_bytes": mem.get("MemTotal"), "available_bytes": mem.get("MemAvailable"), "swap_total_bytes": mem.get("SwapTotal", 0), "swap_used_bytes": mem.get("SwapUsed", 0), "swaps": _swap_devices()},
        "disk": {"path": str(Path.cwd()), "total_bytes": disk.total, "free_bytes": disk.free},
        "temperatures": _temperatures(),
        "torch": {"version": torch.__version__, "threads": torch_threads, "interop_threads": torch.get_num_interop_threads(), "mkldnn": torch.backends.mkldnn.is_available()},
        "gpu": gpu,
        "rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0, 2),
        "load_average": list(os.getloadavg()) if hasattr(os, "getloadavg") else [],
    }
