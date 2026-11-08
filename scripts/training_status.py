#!/usr/bin/env python3
"""Human-readable status for a NEXUS-BI training run."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _tail_metrics(path: Path, limit: int = 50) -> list[dict]:
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()[-limit:]
    rows = []
    for line in lines:
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default=None, help="run directory; defaults to the newest realworld_so run")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    if args.run:
        run_dir = Path(args.run)
    else:
        roots = sorted((ROOT / "artifacts" / "v6_runs").glob("*"), key=lambda path: path.stat().st_mtime, reverse=True)
        roots = [path for path in roots if path.is_dir()]
        if not roots:
            raise SystemExit("no realworld_so training run found")
        run_dir = roots[0]

    status = _read_json(run_dir / "status.json")
    manifest = _read_json(run_dir / "manifest.json")
    validation = _read_json(run_dir / "validation.json")
    metrics = _tail_metrics(run_dir / "metrics.jsonl")
    step = int(status.get("step") or (manifest.get("progress") or {}).get("step") or 0)
    total = int((manifest.get("progress") or {}).get("max_steps") or 0)
    elapsed = float(status.get("elapsed_seconds") or 0)
    rate = elapsed / step if step else 0
    remaining = (total - step) * rate if total and step else None
    losses = [float(row.get("loss", 0)) for row in metrics if row.get("loss") is not None]
    recent = losses[-50:]
    recent_loss = statistics.median(recent) if recent else None
    objectives = []
    for item in status.get("objectives") or []:
        objectives.append({
            "objective": item.get("objective_id"),
            "raw": item.get("raw"),
            "count": item.get("count"),
            "weight": item.get("weight"),
        })
    payload = {
        "run": str(run_dir),
        "status": manifest.get("status") or status.get("event"),
        "step": step,
        "max_steps": total,
        "progress_percent": round(step / total * 100, 2) if total else None,
        "elapsed_hours": round(elapsed / 3600, 2),
        "seconds_per_step": round(rate, 3),
        "eta_hours": round(remaining / 3600, 1) if remaining is not None else None,
        "status_loss": status.get("loss"),
        "recent_median_loss": recent_loss,
        "rss_mb": status.get("rss_mb"),
        "learning_rate": status.get("learning_rate"),
        "best_validation_score": manifest.get("best_validation_score") or validation.get("weighted_validation_score"),
        "validation_samples": validation.get("samples"),
        "validation_metrics": validation.get("metrics"),
        "objectives": objectives,
        "checkpoints": {
            "best": (run_dir / "best.pt").exists(),
            "last": (run_dir / "last.pt").exists(),
            "final": (run_dir / "final.pt").exists(),
        },
        "note": "Raw objective losses are batch-composition dependent; validation metrics are the reliable progress signal.",
    }
    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print(f"run: {payload['run']}")
        print(f"status: {payload['status']}")
        print(f"progress: {payload['step']}/{payload['max_steps']} ({payload['progress_percent']}%)")
        print(f"elapsed: {payload['elapsed_hours']} h | rate: {payload['seconds_per_step']} s/step | eta: {payload['eta_hours']} h")
        print(f"loss now: {payload['status_loss']:.4f} | recent median: {payload['recent_median_loss']}")
        print(f"rss: {payload['rss_mb']} MB | lr: {payload['learning_rate']}")
        print(f"best validation score: {payload['best_validation_score']}")
        for key, value in (payload["validation_metrics"] or {}).items():
            print(f"  {key}: {value}")
        print("objectives:")
        for item in objectives:
            raw = item["raw"]
            count = item["count"]
            shown = "n/a" if not count else f"{raw:.6f}"
            print(f"  {item['objective']}: {shown} (count={count}, weight={item['weight']})")
        print("checkpoints:", payload["checkpoints"])
        print(payload["note"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
