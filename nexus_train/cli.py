"""Command-line interface for the standalone NEXUS-BI training runtime."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from nexus_train.authorization import authorize_local_config
from nexus_train.benchmark import run_sweep
from nexus_train.checkpoint import inspect
from nexus_train.config import load_local_config
from nexus_train.data_validation import gate_status, load_gate
from nexus_train.dataset_freeze import DEFAULT_FREEZE_ROOT, freeze_dataset, validate_frozen_dataset
from nexus_train.evaluator import evaluate_frozen, write_evaluation
from nexus_train.hardware import detect_hardware
from nexus_train.recovery import latest_checkpoint
from nexus_train.trainer import StandaloneRun

ROOT = Path(__file__).resolve().parents[1]
V5 = ROOT / "artifacts" / "final_training_package" / "runs" / "nexus-bi-unified-first-run" / "final.pt"
V5_SHA = "e3718e946f16198499a22283ff873b5f623205fb6964d9613809f9079ed8e749"


def _print(payload: Any, as_json: bool = False) -> None:
    if as_json:
        print(json.dumps(payload, indent=2, default=str))
    else:
        if isinstance(payload, dict):
            for key, value in payload.items():
                if isinstance(value, (dict, list)):
                    print(f"{key}: {json.dumps(value, default=str)[:2000]}")
                else:
                    print(f"{key}: {value}")
        else:
            print(payload)


def _frozen_dir_from_args(args) -> str:
    if getattr(args, "frozen_dir", None):
        return str(args.frozen_dir)
    local = load_local_config(args.config)
    return str(local["dataset"]["frozen_dir"])


def cmd_doctor(args) -> int:
    hardware = detect_hardware()
    frozen_dir = _frozen_dir_from_args(args)
    gate = gate_status(frozen_dir)
    v5 = {"path": str(V5), "exists": V5.exists(), "expected_sha256": V5_SHA}
    if V5.exists():
        from nexus.core.hashing import sha256_file

        v5.update({"sha256": sha256_file(V5), "matches_expected": sha256_file(V5) == V5_SHA})
    latest = latest_checkpoint([ROOT / "runs", ROOT / "checkpoints", ROOT / "artifacts" / "final_training_package" / "runs"])
    local = load_local_config(args.config)
    from nexus_train.data_validation import load_gate

    authorization = authorize_local_config(local, load_gate(local["dataset"]["frozen_dir"]).vocab, hardware)
    payload = {
        "status": "PASS" if v5.get("matches_expected") and gate["status"] == "PASS" else "FAIL",
        "hardware": hardware,
        "authorization": authorization,
        "frozen_dataset": gate,
        "v5": v5,
        "latest_checkpoint": str(latest) if latest else None,
        "config": args.config,
        "frozen_dir": frozen_dir,
        "dependencies": {"torch": hardware["torch"], "mojo": _command("mojo"), "yaml": _module_available("yaml")},
        "warnings": ["No GPU backend; CPU fp32 only.", "No swap exists; the 12 GiB RSS cap is a safety limit, not a training guarantee.", "A successful smoke test is not long-duration reliability evidence."],
    }
    _print(payload, args.json)
    return 0 if payload["status"] == "PASS" else 3


def _command(name: str) -> dict[str, Any]:
    import shutil
    import subprocess

    path = shutil.which(name)
    if not path:
        return {"command": name, "available": False}
    try:
        result = subprocess.run([path, "--version"], capture_output=True, text=True, timeout=10)
        return {"command": path, "available": True, "version": (result.stdout or result.stderr).strip()[:500]}
    except Exception as exc:
        return {"command": path, "available": True, "error": str(exc)}


def _module_available(name: str) -> bool:
    import importlib.util

    return importlib.util.find_spec(name) is not None


def cmd_validate_data(args) -> int:
    frozen_dir = Path(_frozen_dir_from_args(args)).resolve()
    manifest_path = frozen_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    if manifest.get("schema_version") == "nexus.corpus-v3/v1":
        from nexus.data.validate_v3 import validate_corpus_v3

        report = validate_corpus_v3(frozen_dir)
    elif manifest.get("schema_version") == "nexus.split-v2/v1":
        from nexus.data.validate_v2 import validate_split_v2

        report = validate_split_v2(frozen_dir)
    else:
        report = validate_frozen_dataset(frozen_dir)
    if args.json:
        _print(report, True)
    else:
        print(f"Frozen dataset: {report['status']}")
        print(f"Root: {report['root']}")
        print(f"Counts: {report.get('counts')}")
        print(f"Protected test retained: {report.get('protected_test_retained')}")
        print(f"Cross-split identity: {report.get('cross_split_identity')}")
        if report.get("problems"):
            print("Problems:")
            for problem in report["problems"]:
                print(f"  - {problem}")
    return 0 if report["status"] == "PASS" else 3


def cmd_readiness(args) -> int:
    from nexus_train.readiness import check_readiness

    report = check_readiness(args.config, args.out)
    _print(report, args.json)
    return 0 if report["status"] in {"TRAINING_PACKAGE_READY", "TRAINING_PACKAGE_READY_BUT_COMPUTE_UNAVAILABLE"} else 3


def cmd_freeze_data(args) -> int:
    result = freeze_dataset(args.out)
    _print(result, args.json)
    return 0


def cmd_benchmark(args) -> int:
    result = run_sweep(args.sweep, args.config, steps=args.steps, threads=args.threads)
    target = Path(args.out)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, indent=2), encoding="utf-8")
    _print({"out": str(target), "candidates": [{key: item.get(key) for key in ("name", "parameter_count", "mean_step_seconds", "samples_per_second", "peak_rss_mb", "command_returncode")} for item in result["candidates"]], "largest": result.get("largest_successfully_benchmarked", {}).get("name")}, args.json)
    return 0 if all(item.get("command_returncode") == 0 for item in result["candidates"]) else 4


def cmd_train(args) -> int:
    if args.max_steps == 0:
        raise ValueError("--max-steps must be positive")
    local = load_local_config(args.config)
    gate = load_gate(local["dataset"]["frozen_dir"])
    authorization = authorize_local_config(local, gate.vocab)
    if authorization["status"].startswith("BLOCKED"):
        raise ValueError(f"training not authorized: {authorization['reason']}")
    if args.dry_run:
        local = load_local_config(args.config)
        gate = load_gate(local["dataset"]["frozen_dir"])
        gate.require_clean()
        from nexus_train.model_factory import model_config_from_dict
        from nexus_train.trainer import training_config_from_local

        model_config = model_config_from_dict(local["model"], gate.vocab)
        training = training_config_from_local(local, gate.identity(), run_name=args.run_name or local["run_name"], max_steps=args.max_steps or None)
        train = gate.build_split("train", model_config, max_functions=int(local["dataset"].get("max_functions", 8)))
        validation = gate.build_split("validation", model_config, max_functions=int(local["dataset"].get("max_functions", 8)))
        init_proof = None
        init_path = args.init_from or args.init_checkpoint
        if init_path:
            from nexus.foundation.checkpoint import validate_initialization_checkpoint
            from nexus.foundation.config import config_for_vocabulary
            init_proof = validate_initialization_checkpoint(init_path, expected_config=config_for_vocabulary(model_config, gate.vocab), expected_vocab=gate.vocab)
            init_proof = {key: init_proof.get(key) for key in ("status", "sha256", "checkpoint_type", "parameter_count", "config_hash", "vocab_sha256", "step", "metadata_schema", "from_scratch_state_verified")}
        _print({"status": "DRY_RUN_PASS", "frozen_dataset": gate.identity(), "authorization": authorization, "model_config": model_config.to_dict(), "parameters": __import__("nexus.foundation.model", fromlist=["build_foundation_model"]).build_foundation_model(model_config).count_parameters(), "train_samples": len(train), "validation_samples": len(validation), "max_steps": training.max_steps, "initialization": init_proof, "note": "No optimizer step was executed."}, args.json)
        return 0
    init_checkpoint = args.init_from or args.init_checkpoint
    run = StandaloneRun(args.config, run_name=args.run_name, max_steps=args.max_steps, verbose=not args.json, init_checkpoint=init_checkpoint)
    result = run.run()
    _print({key: result.get(key) for key in ("status", "run_dir", "steps", "checkpoint", "parameter_update", "resources", "validation", "stop_reason")}, args.json)
    return 0 if result.get("status") == "completed" else 4


def cmd_resume(args) -> int:
    checkpoint = Path(args.checkpoint) if args.checkpoint else latest_checkpoint([ROOT / "runs", ROOT / "checkpoints"])
    if checkpoint is None or not checkpoint.exists():
        print("No checkpoint found to resume.", file=sys.stderr)
        return 4
    run = StandaloneRun(args.config, run_name=args.run_name, max_steps=args.max_steps, verbose=not args.json)
    result = run.run(resume_path=checkpoint)
    _print({"status": result.get("status"), "run_dir": result.get("run_dir"), "steps": result.get("steps"), "checkpoint": result.get("checkpoint"), "resumed_from": str(checkpoint), "resources": result.get("resources")}, args.json)
    return 0 if result.get("status") == "completed" else 4


def cmd_evaluate(args) -> int:
    report = evaluate_frozen(args.checkpoint, args.config, reconstruction_samples=args.reconstruction_samples, split=args.split)
    paths = write_evaluation(report, args.out)
    _print({"out": paths, "checkpoint": report["checkpoint"]["sha256"], "samples": report["samples"], "metrics": report["metrics"], "capability_categories": report["capability_categories"], "frozen_dataset": report.get("frozen_dataset", {}).get("frozen_id")}, args.json)
    return 0


def cmd_status(args) -> int:
    hardware = detect_hardware()
    gate = gate_status(args.frozen_dir)
    latest = latest_checkpoint([ROOT / "runs", ROOT / "checkpoints", ROOT / "artifacts" / "final_training_package" / "runs"])
    info = inspect(latest) if latest else None
    payload = {"frozen_dataset": gate, "latest_checkpoint": info, "hardware": {"cpu": hardware["cpu"], "memory": hardware["memory"], "gpu": hardware["gpu"], "rss_mb": hardware["rss_mb"]}, "run_dirs": sorted(str(path) for path in (ROOT / "runs").glob("*") if path.is_dir())[-10:], "v5": {"exists": V5.exists(), "sha256": V5_SHA}}
    _print(payload, args.json)
    return 0 if gate["status"] == "PASS" else 3


def cmd_checkpoint(args) -> int:
    _print(inspect(args.path), args.json)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="nexus_train", description="Standalone NEXUS-BI local training runtime")
    sub = parser.add_subparsers(dest="command", required=True)
    doctor = sub.add_parser("doctor", help="check hardware, checkpoint, frozen data and dependencies")
    doctor.add_argument("--config", default=str(ROOT / "configs" / "final_training.yaml"))
    doctor.add_argument("--frozen-dir", default=None)
    doctor.add_argument("--json", action="store_true")
    doctor.set_defaults(func=cmd_doctor)
    validate = sub.add_parser("validate-data", help="independently validate the frozen dataset gate")
    validate.add_argument("--config", default=str(ROOT / "configs" / "final_training.yaml"))
    validate.add_argument("--frozen-dir", default=None)
    validate.add_argument("--json", action="store_true")
    validate.set_defaults(func=cmd_validate_data)
    readiness = sub.add_parser("readiness", help="legacy engine audit; canonical readiness is 'nexus foundation readiness'")
    readiness.add_argument("--config", default=str(ROOT / "configs" / "final_training.yaml"))
    readiness.add_argument("--out", default=str(ROOT / "artifacts" / "readiness"))
    readiness.add_argument("--json", action="store_true")
    readiness.set_defaults(func=cmd_readiness)
    freeze = sub.add_parser("freeze-data", help="create the immutable test-protective frozen dataset")
    freeze.add_argument("--out", default=str(DEFAULT_FREEZE_ROOT))
    freeze.add_argument("--json", action="store_true")
    freeze.set_defaults(func=cmd_freeze_data)
    benchmark = sub.add_parser("benchmark", help="run the isolated model-size sweep")
    benchmark.add_argument("--config", default=str(ROOT / "configs" / "local_cpu.yaml"))
    benchmark.add_argument("--sweep", default=str(ROOT / "configs" / "model_sweep.yaml"))
    benchmark.add_argument("--steps", type=int, default=3)
    benchmark.add_argument("--threads", type=int, default=4)
    benchmark.add_argument("--out", default=str(ROOT / "artifacts" / "training_runtime" / "model_benchmark.json"))
    benchmark.add_argument("--json", action="store_true")
    benchmark.set_defaults(func=cmd_benchmark)
    train = sub.add_parser("train", help="legacy engine entry point; use 'nexus foundation train' as the canonical interface")
    train.add_argument("--config", required=True)
    train.add_argument("--run-name")
    train.add_argument("--max-steps", type=int, default=None)
    train.add_argument("--dry-run", action="store_true")
    init_group = train.add_mutually_exclusive_group()
    init_group.add_argument("--init-from", default=None, help="verified random initialization checkpoint")
    init_group.add_argument("--init-checkpoint", default=None, help="deprecated alias for --init-from")
    train.add_argument("--json", action="store_true")
    train.set_defaults(func=cmd_train)
    resume = sub.add_parser("resume", help="resume the latest or an explicit checkpoint")
    resume.add_argument("--config", required=True)
    resume.add_argument("--latest", action="store_true")
    resume.add_argument("--checkpoint")
    resume.add_argument("--run-name", default="nexus-local-resume")
    resume.add_argument("--max-steps", type=int, default=None)
    resume.add_argument("--json", action="store_true")
    resume.set_defaults(func=cmd_resume)
    evaluate = sub.add_parser("evaluate", help="evaluate a checkpoint on the protected frozen test split")
    evaluate.add_argument("--checkpoint", required=True)
    evaluate.add_argument("--config", default=str(ROOT / "configs" / "final_training.yaml"))
    evaluate.add_argument("--reconstruction-samples", type=int, default=2)
    evaluate.add_argument("--split", choices=["test", "extended_test"], default="test")
    evaluate.add_argument("--out", default=str(ROOT / "artifacts" / "training_runtime" / "evaluation.json"))
    evaluate.add_argument("--json", action="store_true")
    evaluate.set_defaults(func=cmd_evaluate)
    status = sub.add_parser("status", help="show frozen data, latest checkpoint and hardware")
    status.add_argument("--frozen-dir", default=str(ROOT / "artifacts" / "training_runtime" / "frozen_dataset" / "v3"))
    status.add_argument("--json", action="store_true")
    status.set_defaults(func=cmd_status)
    checkpoint = sub.add_parser("checkpoint", help="inspect a checkpoint safely")
    checkpoint.add_argument("--path", required=True)
    checkpoint.add_argument("--json", action="store_true")
    checkpoint.set_defaults(func=cmd_checkpoint)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except KeyboardInterrupt:
        print("Interrupted; if a training run was active, the trainer attempted an interrupt checkpoint.", file=sys.stderr)
        return 130
    except (ValueError, FileNotFoundError, FileExistsError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 4
