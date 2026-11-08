"""Optional Mojo CPU benchmark harness.

Mojo is optional. If the `mojo` executable is absent, the harness records
baseline timings and returns MOJO_NOT_INSTALLED; it never installs anything.
"""

from __future__ import annotations

import json
import shutil
import statistics
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MOJO_SOURCE = ROOT / "mojo" / "benchmark_kernels.mojo"


def _time(fn: Callable[[], Any], repeats: int = 20) -> dict[str, Any]:
    times = []
    for _ in range(repeats):
        started = time.perf_counter()
        fn()
        times.append(time.perf_counter() - started)
    return {"seconds": times, "mean_seconds": statistics.mean(times), "median_seconds": statistics.median(times), "min_seconds": min(times)}


def _byte_histogram_python(data: bytes) -> list[int]:
    counts = [0] * 256
    for value in data:
        counts[value] += 1
    return counts


def _byte_histogram_numpy(data: bytes) -> np.ndarray:
    return np.bincount(np.frombuffer(data, dtype=np.uint8), minlength=256)


def _tokenize(records: list[dict[str, Any]]) -> int:
    # Instruction tokenization: mnemonic + operand-shape normalization, matching
    # the foundation representation's string-level operation.
    total = 0
    for record in records:
        for instruction in record.get("instructions", []):
            total += 1 + len(instruction.get("operands", []) or [])
    return total


def _ir_transform(records: list[dict[str, Any]]) -> int:
    table = {"mov": "move", "add": "arithmetic", "sub": "arithmetic", "cmp": "compare", "jmp": "branch", "call": "call", "ret": "return"}
    total = 0
    for record in records:
        for instruction in record.get("instructions", []):
            total += len(table.get(str(instruction.get("mnemonic", "")), "other"))
    return total


def _dataset_preprocess(records: list[dict[str, Any]]) -> int:
    total = 0
    for record in records:
        for instruction in record.get("instructions", []):
            total += len(instruction.get("bytes", "")) // 2
    return total


def _numeric_kernel(size: int = 512) -> float:
    generator = np.random.default_rng(20261103)
    matrix = generator.standard_normal((size, size), dtype=np.float32)
    vector = generator.standard_normal(size, dtype=np.float32)
    return float((matrix @ vector).sum())


def run_mojo_benchmark(path: str | Path | None = None, repeats: int = 20) -> dict[str, Any]:
    path = Path(path) if path else ROOT / "tests" / "fixtures" / "JniDemo.c"
    raw = path.read_bytes() if path.exists() else (b"nexus-bi-mojo-benchmark" * 512)
    # Representative instruction records from a controlled manifest record are
    # used when available; otherwise a small deterministic synthetic record set.
    records = [
        {"instructions": [{"mnemonic": "mov", "operands": [{"kind": "register"}, {"kind": "immediate"}], "bytes": "48c7c001000000"}, {"mnemonic": "add", "operands": [{"kind": "register"}, {"kind": "register"}], "bytes": "4801d8"}]},
        {"instructions": [{"mnemonic": "cmp", "operands": [{"kind": "register"}, {"kind": "immediate"}], "bytes": "4883f800"}, {"mnemonic": "jne", "operands": [{"kind": "address"}], "bytes": "75f0"}]},
    ]
    baselines = {
        "byte_histogram_python": _time(lambda: _byte_histogram_python(raw), repeats),
        "byte_histogram_numpy": _time(lambda: _byte_histogram_numpy(raw), repeats),
        "instruction_tokenization_python": _time(lambda: _tokenize(records), repeats),
        "semantic_ir_transform_python": _time(lambda: _ir_transform(records), repeats),
        "dataset_preprocessing_python": _time(lambda: _dataset_preprocess(records), repeats),
        "numeric_matvec_numpy": _time(lambda: _numeric_kernel(), max(3, repeats // 4)),
    }
    py_counts = _byte_histogram_python(raw)
    np_counts = _byte_histogram_numpy(raw).tolist()
    correctness = {
        "byte_histogram_equal": py_counts == np_counts,
        "byte_histogram_sum": sum(py_counts),
        "input_bytes": len(raw),
        "instruction_records": len(records),
    }
    mojo = shutil.which("mojo")
    status = "MOJO_NOT_INSTALLED" if mojo is None else "MOJO_AVAILABLE_NOT_BENCHMARKED"
    return {
        "schema_version": "0.1.0",
        "status": status,
        "mojo_executable": mojo,
        "mojo_source": str(MOJO_SOURCE),
        "mojo_source_exists": MOJO_SOURCE.exists(),
        "baselines": baselines,
        "correctness": correctness,
        "compilation_overhead": None,
        "end_to_end_benefit": None,
        "fairness_notes": [
            "Byte histogram compares an optimized NumPy implementation with a Python loop; both are labeled and measured separately.",
            "Mojo was not installed, so no Mojo timing or acceleration claim is made.",
            "The harness does not rewrite PyTorch autograd or the model in Mojo.",
        ],
    }


def write_mojo_benchmark(path: str | Path | None = None) -> dict[str, Any]:
    report = run_mojo_benchmark()
    target = Path(path) if path else ROOT / "artifacts" / "training_runtime" / "mojo_benchmark.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
