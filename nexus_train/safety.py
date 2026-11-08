"""Hardware safety and resource guards for the standalone trainer."""

from __future__ import annotations

import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from nexus_train.hardware import _meminfo, _temperatures


def _read_int(path: str, default: int = 0) -> int:
    try:
        return int(Path(path).read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return default


def _rss_bytes() -> int:
    status = Path("/proc/self/status")
    try:
        for line in status.read_text(encoding="utf-8").splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024
    except OSError:
        return 0
    return 0


def _pressure_memory() -> float:
    try:
        for line in Path("/proc/pressure/memory").read_text(encoding="utf-8").splitlines():
            if line.startswith("full"):
                for part in line.split():
                    if part.startswith("avg10="):
                        return float(part.split("=")[1])
    except OSError:
        return 0.0
    return 0.0


@dataclass
class SafetyLimits:
    max_rss_gib: float = 12.0
    min_free_disk_gib: float = 10.0
    max_package_temp_c: float = 95.0
    max_swap_used_mib: float = 32.0
    max_memory_pressure_avg10: float = 20.0
    check_interval_seconds: float = 2.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_rss_gib": self.max_rss_gib,
            "min_free_disk_gib": self.min_free_disk_gib,
            "max_package_temp_c": self.max_package_temp_c,
            "max_swap_used_mib": self.max_swap_used_mib,
            "max_memory_pressure_avg10": self.max_memory_pressure_avg10,
            "check_interval_seconds": self.check_interval_seconds,
        }


class SafetyMonitor:
    def __init__(self, limits: SafetyLimits, workdir: str | Path | None = None):
        self.limits = limits
        self.workdir = Path(workdir or Path.cwd())
        self.started = time.monotonic()
        self.last_check = 0.0
        self.last_snapshot: dict[str, Any] | None = None
        self.stop_reason: str | None = None
        self.stop_requested = False

    def snapshot(self) -> dict[str, Any]:
        mem = _meminfo()
        disk = shutil.disk_usage(self.workdir)
        temps = _temperatures()
        swap_used = mem.get("SwapTotal", 0) - mem.get("SwapFree", 0) if "SwapFree" in mem else 0
        snapshot = {
            "rss_mib": round(_rss_bytes() / 1024 / 1024, 2),
            "rss_limit_mib": round(self.limits.max_rss_gib * 1024, 2),
            "available_mib": round(mem.get("MemAvailable", 0) / 1024 / 1024, 2),
            "swap_used_mib": round(swap_used / 1024 / 1024, 2),
            "disk_free_gib": round(disk.free / 1024**3, 2),
            "package_temp_c": temps.get("package_celsius"),
            "memory_pressure_avg10": _pressure_memory(),
            "elapsed_seconds": round(time.monotonic() - self.started, 2),
        }
        self.last_snapshot = snapshot
        return snapshot

    def violations(self) -> list[str]:
        snap = self.snapshot()
        reasons = []
        if snap["rss_mib"] > self.limits.max_rss_gib * 1024:
            reasons.append(f"RSS {snap['rss_mib']} MiB exceeds {self.limits.max_rss_gib} GiB cap")
        if snap["disk_free_gib"] < self.limits.min_free_disk_gib:
            reasons.append(f"free disk {snap['disk_free_gib']} GiB is below {self.limits.min_free_disk_gib} GiB")
        if snap.get("package_temp_c") is not None and snap["package_temp_c"] > self.limits.max_package_temp_c:
            reasons.append(f"package temperature {snap['package_temp_c']} C exceeds {self.limits.max_package_temp_c} C")
        if snap["swap_used_mib"] > self.limits.max_swap_used_mib:
            reasons.append(f"swap use {snap['swap_used_mib']} MiB exceeds {self.limits.max_swap_used_mib} MiB")
        if snap["memory_pressure_avg10"] > self.limits.max_memory_pressure_avg10:
            reasons.append(f"memory pressure {snap['memory_pressure_avg10']} exceeds {self.limits.max_memory_pressure_avg10}")
        return reasons

    def request_stop(self) -> None:
        self.stop_requested = True

    def guard(self, request_stop: bool = False) -> str | None:
        if request_stop or self.stop_requested:
            self.stop_reason = "sigint"
            return self.stop_reason
        now = time.monotonic()
        if now - self.last_check < self.limits.check_interval_seconds:
            return None
        self.last_check = now
        reasons = self.violations()
        if reasons:
            self.stop_reason = "; ".join(reasons)
            return self.stop_reason
        return None
