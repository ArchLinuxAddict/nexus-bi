"""Single compute-authorization policy for the standalone runtime.

The legacy gate in ``nexus/training/check.py`` capped CPU work at 5M estimated
parameters.  The standalone runtime supports a documented *diagnostic pilot*
budget instead of silently bypassing that gate.
"""

from __future__ import annotations

from typing import Any

from nexus_train.hardware import detect_hardware
from nexus_train.model_factory import build_model

PILOT_CPU_PARAMETER_CAP = 15_000_000
SMOKE_CPU_PARAMETER_CAP = 5_000_000
PILOT_CPU_STEP_CAP = 20_000
MAIN_CPU_PARAMETER_CAP = 35_000_000
MAIN_CPU_STEP_CAP = 50_000


def authorize_local_config(local: dict[str, Any], vocab, hardware: dict[str, Any] | None = None) -> dict[str, Any]:
    hardware = hardware or detect_hardware()
    model = build_model(local["model"], vocab)
    parameters = model.count_parameters()
    training = local["training"]
    run_mode = str(training.get("run_mode", "pilot"))
    max_steps = int(training.get("max_steps", 0))
    max_rss = float(local.get("safety", {}).get("max_rss_gib", 0))
    gpu_backend = str((hardware.get("gpu") or {}).get("supported_training_backend", "cpu_fp32"))
    status = "AUTHORIZED_DIAGNOSTIC"
    reason = f"{run_mode} run on CPU fp32 within the documented diagnostic budget"
    if max_rss > 12.0:
        status = "BLOCKED_BY_COMPUTE"
        reason = "max_rss_gib exceeds the audited 12 GiB cap"
    elif run_mode == "final":
        if gpu_backend == "cpu_fp32":
            status = "BLOCKED_BY_COMPUTE"
            reason = "final training requires a supported GPU backend; this host is CPU-only"
        else:
            status = "AUTHORIZED_FINAL"
            reason = "final training authorized on the detected backend"
    elif run_mode == "smoke":
        if parameters > SMOKE_CPU_PARAMETER_CAP:
            status = "BLOCKED_BY_COMPUTE"
            reason = f"smoke model {parameters} exceeds the {SMOKE_CPU_PARAMETER_CAP} CPU smoke cap"
    elif run_mode == "pilot":
        if parameters > PILOT_CPU_PARAMETER_CAP:
            status = "BLOCKED_BY_COMPUTE"
            reason = f"pilot model {parameters} exceeds the {PILOT_CPU_PARAMETER_CAP} CPU pilot cap"
        elif max_steps > PILOT_CPU_STEP_CAP:
            status = "BLOCKED_BY_COMPUTE"
            reason = f"pilot max_steps {max_steps} exceeds the {PILOT_CPU_STEP_CAP} CPU pilot step cap"
    elif run_mode == "main":
        if parameters > MAIN_CPU_PARAMETER_CAP:
            status = "BLOCKED_BY_COMPUTE"
            reason = f"main model {parameters} exceeds the {MAIN_CPU_PARAMETER_CAP} measured CPU cap"
        elif max_steps > MAIN_CPU_STEP_CAP:
            status = "BLOCKED_BY_COMPUTE"
            reason = f"main max_steps {max_steps} exceeds the {MAIN_CPU_STEP_CAP} measured CPU step cap"
        else:
            status = "AUTHORIZED_FINAL"
            reason = f"main run on {gpu_backend} within the measured memory/step budget"
    else:
        status = "BLOCKED_BY_COMPUTE"
        reason = f"unknown run_mode: {run_mode}"
    return {
        "status": status,
        "run_mode": run_mode,
        "parameters": parameters,
        "max_steps": max_steps,
        "max_rss_gib": max_rss,
        "gpu_backend": gpu_backend,
        "reason": reason,
        "policy": {
            "pilot_cpu_parameter_cap": PILOT_CPU_PARAMETER_CAP,
            "smoke_cpu_parameter_cap": SMOKE_CPU_PARAMETER_CAP,
            "pilot_cpu_step_cap": PILOT_CPU_STEP_CAP,
            "main_cpu_parameter_cap": MAIN_CPU_PARAMETER_CAP,
            "main_cpu_step_cap": MAIN_CPU_STEP_CAP,
            "final_requires_gpu": True,
        },
    }
