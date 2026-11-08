import json
import time
from pathlib import Path
from typing import Any

from nexus.program.bundle import analyze_binary
from nexus.program.record import build_program_record
from nexus.verification.static import verify_program_record


def benchmark_binary(path: str | Path, validate: bool = True) -> dict[str, Any]:
    started = time.perf_counter()
    bundle = analyze_binary(path)
    analysis_seconds = time.perf_counter() - started
    record_started = time.perf_counter()
    record = build_program_record(bundle)
    record_seconds = time.perf_counter() - record_started
    validation = verify_program_record(record) if validate else {"status": "skipped"}
    return {
        "path": str(path),
        "sha256": bundle.elf.sha256,
        "size": bundle.elf.size,
        "machine": bundle.elf.header.machine_name,
        "functions": len(bundle.functions),
        "instructions": len(bundle.disassembly.instructions) if bundle.disassembly else 0,
        "call_edges": len(bundle.callgraph.edges),
        "strings": len(bundle.strings),
        "security_findings": len(bundle.security_findings),
        "jni": bundle.jni_report.get("is_jni", False),
        "analysis_seconds": round(analysis_seconds, 4),
        "record_seconds": round(record_seconds, 4),
        "record_bytes": len(json.dumps(record)),
        "validation": validation["status"],
        "limitations": bundle.limitations,
    }


def benchmark_paths(paths: list[str | Path], output_path: str | Path | None = None) -> dict[str, Any]:
    results = [benchmark_binary(path) for path in paths]
    timings = [item["analysis_seconds"] for item in results]
    payload = {
        "status": "completed",
        "scope": "deterministic analysis benchmark; not a model-quality benchmark",
        "binaries": results,
        "summary": {
            "count": len(results),
            "total_functions": sum(item["functions"] for item in results),
            "total_instructions": sum(item["instructions"] for item in results),
            "mean_analysis_seconds": round(sum(timings) / len(timings), 4) if timings else 0.0,
            "max_analysis_seconds": round(max(timings), 4) if timings else 0.0,
            "validation_passed": sum(1 for item in results if item["validation"] == "passed"),
        },
        "limitations": [
            "This measures deterministic extraction, not semantic understanding or reconstruction quality.",
            "Results are host-specific and not comparable to external decompiler benchmarks without a shared corpus.",
        ],
    }
    if output_path is not None:
        target = Path(output_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def benchmark_dataset(manifest_path: str | Path, limit: int = 20) -> dict[str, Any]:
    manifests = [json.loads(line) for line in Path(manifest_path).read_text(encoding="utf-8").splitlines() if line.strip()]
    paths = []
    for manifest in manifests[:limit]:
        for artifact in manifest.get("artifacts", []):
            if artifact.get("media_type") == "application/x-elf":
                paths.append(Path(manifest_path).parent / artifact["path"])
                break
    return benchmark_paths(paths)
