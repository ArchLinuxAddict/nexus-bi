"""Standalone trainer orchestration: gate, model, safety, telemetry, recovery."""

from __future__ import annotations

import signal
from pathlib import Path
from typing import Any

from nexus.foundation.training import FoundationTrainer, FoundationTrainingConfig
from nexus_train.config import load_local_config
from nexus_train.data_validation import FrozenDatasetGate, load_gate
from nexus_train.model_factory import model_config_from_dict
from nexus_train.safety import SafetyLimits, SafetyMonitor
from nexus_train.telemetry import Telemetry

ROOT = Path(__file__).resolve().parents[1]


def _resolve(path: str | Path) -> Path:
    target = Path(path)
    return target if target.is_absolute() else ROOT / target


def training_config_from_local(local: dict[str, Any], identity: dict[str, Any], *, run_name: str | None = None, max_steps: int | None = None) -> FoundationTrainingConfig:
    training = dict(local["training"])
    if max_steps is not None:
        training["max_steps"] = int(max_steps)
    if run_name:
        training["run_name"] = run_name
    training.setdefault("run_name", local.get("run_name", "nexus-local"))
    training["dataset_identity"] = identity
    return FoundationTrainingConfig(**training)


class StandaloneRun:
    def __init__(
        self,
        config_path: str | Path,
        *,
        run_name: str | None = None,
        max_steps: int | None = None,
        verbose: bool = True,
        init_checkpoint: str | Path | None = None,
        training_overrides: dict[str, Any] | None = None,
        output_dir: str | Path | None = None,
        run_root: str | Path | None = None,
        dataset_filter_labels: set[str] | None = None,
    ):
        self.local = load_local_config(config_path)
        if training_overrides:
            merged = dict(self.local["training"])
            merged.update({key: value for key, value in training_overrides.items() if value is not None})
            self.local["training"] = merged
            validation = _validate_overrides(self.local)
            self.local["_objective_readiness"] = validation
        self.gate: FrozenDatasetGate = load_gate(self.local["dataset"]["frozen_dir"])
        self.gate.require_clean()
        self.vocab = self.gate.vocab
        self.model_config = model_config_from_dict(self.local["model"], self.vocab)
        self.identity = self.gate.identity()
        self.run_name = run_name or self.local.get("run_name", "nexus-local")
        self.max_steps = max_steps
        self.verbose = verbose
        self.init_checkpoint = Path(init_checkpoint) if init_checkpoint else None
        self.run_root = _resolve(run_root) if run_root else None
        self.dataset_filter_labels = dataset_filter_labels
        self.training_config = training_config_from_local(self.local, self.identity, run_name=self.run_name, max_steps=max_steps)
        if output_dir:
            self.training_config.output_dir = str(output_dir)
        self.monitor = SafetyMonitor(SafetyLimits(**self.local["safety"]), workdir=ROOT)
        self.train_dataset = None
        self.validation_dataset = None
        self.telemetry = None

    def prepare(self) -> FoundationTrainer:
        max_functions = int(self.local["dataset"].get("max_functions", 8))
        self.train_dataset = self.gate.build_split("train", self.model_config, max_functions=max_functions, allowed_datasets=self.dataset_filter_labels)
        self.validation_dataset = self.gate.build_split("validation", self.model_config, max_functions=max_functions, allowed_datasets=self.dataset_filter_labels)
        if self.train_dataset.split != "train" or self.validation_dataset.split != "validation":
            raise ValueError("frozen split labels are incorrect")
        run_dir = (self.run_root or _resolve(self.local["telemetry"].get("run_dir", "runs"))) / self.run_name
        if run_dir.exists():
            raise FileExistsError(f"run directory already exists: {run_dir}; choose another --run-name")
        self.telemetry = Telemetry(run_dir)
        trainer = FoundationTrainer(self.model_config, self.training_config, self.train_dataset, self.validation_dataset, self.vocab, run_dir=run_dir)
        if self.init_checkpoint is not None:
            initialized = trainer.initialize_from(self.init_checkpoint)
            self.telemetry.emit("initialized_from", initialized)
        self.telemetry.emit("prepared", {"frozen_dataset": self.identity, "train_samples": len(self.train_dataset), "validation_samples": len(self.validation_dataset), "model_config": self.model_config.to_dict()})
        return trainer

    def run(self, resume_path: str | Path | None = None, *, evaluate: bool = True) -> dict[str, Any]:
        if resume_path is not None and self.init_checkpoint is not None:
            raise ValueError("--init-from and --resume-from are mutually exclusive")
        trainer = self.prepare()
        trainer.resource_guard = self.monitor.guard
        printed = 0

        def callback(record: dict[str, Any]) -> None:
            nonlocal printed
            printed += 1
            self.telemetry.emit("step", record)
            if self.verbose and (printed <= 3 or printed % 10 == 0):
                print(self.telemetry.progress_line(record), flush=True)

        trainer.step_callback = callback
        old_handler = signal.getsignal(signal.SIGINT)

        def request_stop(_signum, _frame):
            print("\nSIGINT received; requesting a checkpointed stop...", flush=True)
            self.monitor.request_stop()

        signal.signal(signal.SIGINT, request_stop)
        try:
            if resume_path is not None:
                loaded = trainer.resume(resume_path)
                self.telemetry.emit("resumed", loaded)
            result = trainer.train(evaluate=evaluate)
        finally:
            signal.signal(signal.SIGINT, old_handler)
        self.telemetry.emit("finished", {"status": result.get("status"), "step": result.get("steps"), "checkpoint": result.get("checkpoint"), "resources": result.get("resources"), "safety": self.monitor.last_snapshot})
        return result


def _validate_overrides(local: dict[str, Any]) -> dict[str, Any]:
    from nexus.foundation.objectives import OBJECTIVES
    from nexus_train.objective_registry import validate_objective_selection

    training = local["training"]
    run_mode = str(training.get("run_mode", "pilot"))
    readiness = validate_objective_selection(list(training.get("objectives", [])), run_mode)
    if readiness["status"] != "PASS":
        raise ValueError(f"objective readiness gate failed: {readiness['problems']}")
    known = {objective.objective_id for objective in OBJECTIVES}
    unknown = set(training.get("objectives", [])) - known
    if unknown:
        raise ValueError(f"unsupported training objectives: {sorted(unknown)}")
    return readiness


def smoke_test(config_path: str | Path, *, max_steps: int = 2, run_name: str = "nexus-local-smoke") -> dict[str, Any]:
    run = StandaloneRun(config_path, run_name=run_name, max_steps=max_steps)
    return run.run()
