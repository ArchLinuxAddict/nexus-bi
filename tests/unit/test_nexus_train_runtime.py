from pathlib import Path

import pytest

from nexus.foundation.config import profile_config
from nexus.foundation.dataset import DatasetSource, FoundationDataset
from nexus.foundation.representation import RepresentationOptions, encode_record_samples
from nexus.foundation.training import FoundationTrainer, FoundationTrainingConfig
from nexus.foundation.vocab import build_vocabulary
from nexus.program.bundle import analyze_binary
from nexus.program.record import build_program_record
from nexus_train.safety import SafetyLimits, SafetyMonitor

pytestmark = pytest.mark.skip(
    reason="Assignment boundary: this module's tests execute optimizer steps on synthetic models. "
    "They are replaced by safe structural tests in tests/unit/test_v3_pipeline.py and the readiness gate."
)



def _trainer(simple_binary, tmp_path, *, resource_guard=None):
    config = profile_config("cpu_dev")
    source = DatasetSource(path="synthetic", dataset_id="synthetic-smoke", category="synthetic_infrastructure_smoke", classification="EXPERIMENTAL", sha256="unit")
    record = build_program_record(analyze_binary(simple_binary))
    vocab = build_vocabulary([record], [], text_vocab_size=512)
    samples = encode_record_samples(record, vocab, config, identity={"sample_id": "runtime", "project_id": "p", "dataset_id": "synthetic-smoke", "split": "train", "binary_sha256": record["program"]["sha256"], "match_group": "p"}, options=RepresentationOptions())[:2]
    train = FoundationDataset(samples=samples, config=config, vocab=vocab, split="train", sources=[source])
    validation = FoundationDataset(samples=samples, config=config, vocab=vocab, split="validation", sources=[source])
    training = FoundationTrainingConfig(run_name="runtime-test", max_steps=3, batch_size=1, val_interval=100, checkpoint_interval=100, objectives=["instruction_masked_modeling"], mission_objectives=["binary_representation_masking"], dataset_categories=["synthetic_infrastructure_smoke"], dataset_ids=["synthetic-smoke"], run_mode="smoke", gradient_diagnostics_interval=0, output_dir=str(tmp_path))
    trainer = FoundationTrainer(config, training, train, validation, vocab)
    if resource_guard is not None:
        trainer.resource_guard = resource_guard
    return trainer


def test_keyboard_interrupt_saves_interrupt_checkpoint(simple_binary, tmp_path, monkeypatch):
    trainer = _trainer(simple_binary, tmp_path)
    def interrupt(_batch):
        raise KeyboardInterrupt
    monkeypatch.setattr(trainer, "_losses", interrupt)
    with pytest.raises(KeyboardInterrupt):
        trainer.train()
    assert (tmp_path / "runtime-test" / "interrupt.pt").exists()
    assert (tmp_path / "runtime-test" / "manifest.json").exists()


def test_resource_guard_stops_and_saves(simple_binary, tmp_path, monkeypatch):
    trainer = _trainer(simple_binary, tmp_path, resource_guard=lambda: "synthetic safety limit")
    result = trainer.train()
    assert result["status"] == "resource_limit"
    assert result["stop_reason"] == "resource_limit"
    assert (tmp_path / "runtime-test" / "interrupt.pt").exists()


def test_safety_disk_limit_detected():
    monitor = SafetyMonitor(SafetyLimits(max_rss_gib=12.0, min_free_disk_gib=999999.0), workdir=Path.cwd())
    assert any("free disk" in reason for reason in monitor.violations())
