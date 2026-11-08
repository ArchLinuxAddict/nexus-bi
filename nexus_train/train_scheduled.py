"""Coverage-scheduled training entry point for subsequent NEXUS-BI runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from nexus_train.config import load_local_config
from nexus_train.data_validation import load_gate
from nexus_train.model_factory import model_config_from_dict
from nexus_train.objective_scheduler import ObjectiveScheduler, ScheduledDataset
from nexus_train.recovery import latest_checkpoint
from nexus_train.trainer import StandaloneRun

ROOT = Path(__file__).resolve().parents[1]


class StandaloneRunScheduled(StandaloneRun):
    """Use the existing trainer with a coverage-scheduled batch stream."""

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.scheduler: ObjectiveScheduler | None = None
        self._resume_path: Path | None = None

    def run(self, resume_path: str | Path | None = None) -> dict[str, Any]:
        self._resume_path = Path(resume_path) if resume_path is not None else None
        try:
            return super().run(resume_path)
        finally:
            self._write_coverage()

    def _write_coverage(self) -> None:
        if self.scheduler is None or self.telemetry is None:
            return
        run_dir = self.telemetry.run_dir
        report = self.scheduler.coverage_report()
        (run_dir / "coverage_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        self.scheduler.save_state(run_dir / "scheduler_state.json")
        self.telemetry.emit("coverage", {"complete": report["complete"], "observed_step": report["observed_step"], "blocking_deficits": report["blocking_deficits"], "skipped_total": report["skipped_total"]})

    def prepare(self):
        trainer = super().prepare()
        sampling = self.local.get("objective_sampling") or {}
        policy_path = Path(sampling.get("policy_path") or "artifacts/objectives/objective_sampling_policy.json")
        if not policy_path.is_absolute():
            policy_path = ROOT / policy_path
        policy = json.loads(policy_path.read_text(encoding="utf-8"))
        objectives = list(trainer.training_config.objectives)
        scheduler = ObjectiveScheduler(
            trainer.train_dataset,
            trainer.model_config,
            policy,
            enabled_objectives=objectives,
            batch_size=trainer.training_config.batch_size,
            seed=trainer.training_config.seed,
            window_steps=sampling.get("window_steps"),
        )
        if self._resume_path is not None:
            sidecar = self._resume_path.parent / "scheduler_state.json"
            if sidecar.exists():
                scheduler.load_state_dict(json.loads(sidecar.read_text(encoding="utf-8")))
        base_dataset = trainer.train_dataset
        trainer.train_dataset = ScheduledDataset(base_dataset, scheduler)
        original_train = trainer.train
        checkpoint_interval = trainer.training_config.checkpoint_interval

        def scheduled_train():
            original_callback = trainer.step_callback

            def wrapped(record: dict[str, Any]) -> None:
                scheduler.observe(record)
                enriched = scheduler.enrich_record(record)
                if original_callback is not None:
                    original_callback(enriched)
                step = int(record.get("step", 0))
                if checkpoint_interval and step % checkpoint_interval == 0:
                    self._write_coverage()

            trainer.step_callback = wrapped
            return original_train()

        trainer.train = scheduled_train
        self.scheduler = scheduler
        self.telemetry.emit("scheduler_prepared", scheduler.pool_summary())
        self._write_coverage()
        return trainer


def dry_run(config_path: str | Path) -> dict[str, Any]:
    local = load_local_config(config_path)
    gate = load_gate(local["dataset"]["frozen_dir"])
    model_config = model_config_from_dict(local["model"], gate.vocab)
    dataset = gate.build_split("train", model_config, max_functions=int(local["dataset"].get("max_functions", 8)))
    sampling = local.get("objective_sampling") or {}
    policy_path = Path(sampling.get("policy_path") or "artifacts/objectives/objective_sampling_policy.json")
    if not policy_path.is_absolute():
        policy_path = ROOT / policy_path
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    scheduler = ObjectiveScheduler(
        dataset,
        model_config,
        policy,
        enabled_objectives=list(local["training"]["objectives"]),
        batch_size=int(local["training"]["batch_size"]),
        seed=int(local["training"]["seed"]),
        window_steps=sampling.get("window_steps"),
    )
    return {"status": "SCHEDULER_DRY_RUN_PASS", "policy": str(policy_path), "pool_summary": scheduler.pool_summary(), "initial_coverage": scheduler.coverage_report()}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="NEXUS-BI coverage-scheduled training")
    parser.add_argument("--config", default=str(ROOT / "configs" / "final_training_v2.yaml"))
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--init-checkpoint", default=None)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--latest", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    if args.dry_run:
        result = dry_run(args.config)
        print(json.dumps(result, indent=2, default=str))
        return 0
    resume_path = Path(args.resume) if args.resume else None
    if args.latest and resume_path is None:
        local = load_local_config(args.config)
        roots = [local.get("telemetry", {}).get("run_dir", "artifacts/final_training_package_v2/runs")]
        resume_path = latest_checkpoint([ROOT / root if not Path(root).is_absolute() else Path(root) for root in roots], require_training_state=True)
        if resume_path is None:
            print(json.dumps({"status": "NO_CHECKPOINT", "roots": [str(root) for root in roots]}, indent=2))
            return 4
    run = StandaloneRunScheduled(args.config, run_name=args.run_name, max_steps=args.max_steps, verbose=not args.json, init_checkpoint=args.init_checkpoint)
    result = run.run(resume_path=resume_path)
    print(json.dumps(result, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
