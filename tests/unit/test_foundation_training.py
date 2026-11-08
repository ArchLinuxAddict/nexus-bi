import json

import pytest
import torch

from nexus.core.errors import MissionPolicyError
from nexus.foundation.checkpoint import inspect_checkpoint, load_foundation_checkpoint
from nexus.foundation.config import profile_config
from nexus.foundation.dataset import DatasetSource, FoundationDataset
from nexus.foundation.representation import RepresentationOptions, encode_record_samples
from nexus.foundation.training import (
    FoundationTrainer,
    FoundationTrainingConfig,
    TrainingFailure,
    build_trainer,
    evaluate_dataset,
)
from nexus.foundation.vocab import build_vocabulary
from nexus.program.bundle import analyze_binary
from nexus.program.record import build_program_record

pytestmark = pytest.mark.skip(
    reason="Assignment boundary: this module's tests execute optimizer steps on synthetic models. "
    "They are replaced by safe structural tests in tests/unit/test_v3_pipeline.py and the readiness gate."
)



def _dataset(binary, split, config, source, limit=3):
    record = build_program_record(analyze_binary(binary))
    vocab = build_vocabulary([record], [], text_vocab_size=512)
    samples = encode_record_samples(
        record,
        vocab,
        config,
        identity={
            "sample_id": f"{split}-unit",
            "project_id": "unit-project",
            "dataset_id": source.dataset_id,
            "split": split,
            "binary_sha256": record["program"]["sha256"],
            "source_family": "fixture",
            "match_group": "unit-project",
            "binary_path": str(binary),
        },
        options=RepresentationOptions(),
    )[:limit]
    dataset = FoundationDataset(samples=samples, config=config, vocab=vocab, split=split, sources=[source])
    return dataset, vocab


def _source():
    return DatasetSource(
        path="synthetic",
        dataset_id="synthetic-smoke",
        category="synthetic_infrastructure_smoke",
        classification="EXPERIMENTAL",
        sha256="unit-test",
    )


def _config(tmp_path, **overrides):
    values = dict(
        run_name="unit-smoke",
        max_steps=2,
        batch_size=2,
        val_interval=1,
        checkpoint_interval=1,
        objectives=["instruction_masked_modeling", "ir_op_prediction", "cfg_edge_prediction"],
        mission_objectives=["binary_representation_masking", "instruction_semantics", "basic_block_correspondence"],
        dataset_categories=["synthetic_infrastructure_smoke"],
        dataset_ids=["synthetic-smoke"],
        run_mode="smoke",
        gradient_diagnostics_interval=0,
        output_dir=str(tmp_path),
    )
    values.update(overrides)
    return FoundationTrainingConfig(**values)


def test_training_updates_parameters_and_writes_integrity_checkpoint(simple_binary, tmp_path):
    config = profile_config("cpu_dev")
    source = _source()
    train, vocab = _dataset(simple_binary, "train", config, source)
    validation, _ = _dataset(simple_binary, "validation", config, source)
    training = _config(tmp_path)
    trainer = build_trainer("cpu_dev", training, train, validation, vocab)
    result = trainer.train()
    assert result["status"] == "completed"
    assert result["steps"] == 2
    assert result["parameter_update"]["parameters_updated"] is True
    assert result["parameter_update"]["delta_l2"] > 0
    assert result["checkpoint"]["sha256"]
    assert result["checkpoint"]["metadata"]["initialization"] == "random"
    assert result["checkpoint"]["metadata"]["pretrained_weights_loaded"] is False
    assert result["checkpoint"]["metadata"]["dataset_fingerprints"]
    assert result["checkpoint"]["metadata"]["objective_config"]
    assert result["checkpoint"]["metadata"]["seeds"]["torch"] == training.seed
    info = inspect_checkpoint(result["checkpoint"]["path"])
    assert info["metadata"]["parameter_count"] == result["parameter_count"]
    assert info["has_optimizer_state"] is True
    assert (tmp_path / "unit-smoke" / "manifest.json").exists()
    assert (tmp_path / "unit-smoke" / "metrics.jsonl").exists()
    summary = json.loads((tmp_path / "unit-smoke" / "manifest.json").read_text())
    assert summary["status"] == "completed"
    assert summary["dataset_splits"]["test_used_for_training"] is False


def test_checkpoint_resume_continues_step_and_optimizer_state(simple_binary, tmp_path):
    config = profile_config("cpu_dev")
    source = _source()
    train, vocab = _dataset(simple_binary, "train", config, source)
    validation, _ = _dataset(simple_binary, "validation", config, source)
    first = build_trainer("cpu_dev", _config(tmp_path, run_name="first"), train, validation, vocab)
    result = first.train()
    assert result["steps"] == 2
    resumed_config = _config(tmp_path, run_name="resumed", max_steps=3)
    second = build_trainer("cpu_dev", resumed_config, train, validation, vocab)
    loaded = second.resume(result["checkpoint"]["path"])
    assert loaded["step"] == 2
    continued = second.train()
    assert continued["status"] == "completed"
    assert continued["steps"] == 3
    assert continued["resumed_from"]["step"] == 2
    assert continued["parameter_update"]["parameters_updated"] is True


def test_resume_rejects_vocabulary_mismatch(simple_binary, tmp_path):
    config = profile_config("cpu_dev")
    source = _source()
    train, vocab = _dataset(simple_binary, "train", config, source)
    validation, _ = _dataset(simple_binary, "validation", config, source)
    trainer = build_trainer("cpu_dev", _config(tmp_path), train, validation, vocab)
    result = trainer.train()
    with pytest.raises(ValueError):
        load_foundation_checkpoint(result["checkpoint"]["path"], expected_vocab_sha256="0" * 64)


def test_test_split_is_refused_as_training_data(simple_binary, tmp_path):
    config = profile_config("cpu_dev")
    source = _source()
    test_dataset, vocab = _dataset(simple_binary, "test", config, source)
    validation, _ = _dataset(simple_binary, "validation", config, source)
    with pytest.raises(ValueError):
        FoundationTrainer(config, _config(tmp_path), test_dataset, validation, vocab)


def test_unclassified_mission_objective_is_refused(simple_binary, tmp_path):
    config = profile_config("cpu_dev")
    source = _source()
    train, vocab = _dataset(simple_binary, "train", config, source)
    validation, _ = _dataset(simple_binary, "validation", config, source)
    training = _config(tmp_path, mission_objectives=["not_a_mission_objective"])
    with pytest.raises(MissionPolicyError):
        FoundationTrainer(config, training, train, validation, vocab)


def test_non_finite_loss_failure_recovery(simple_binary, tmp_path, monkeypatch):
    config = profile_config("cpu_dev")
    source = _source()
    train, vocab = _dataset(simple_binary, "train", config, source)
    validation, _ = _dataset(simple_binary, "validation", config, source)
    trainer = build_trainer("cpu_dev", _config(tmp_path, max_steps=3), train, validation, vocab)
    monkeypatch.setattr(trainer, "_losses", lambda batch: (torch.tensor(float("nan")), [], {}))
    with pytest.raises(TrainingFailure):
        trainer.train()
    assert (tmp_path / "unit-smoke" / "failures.jsonl").exists()


def test_validation_metrics_include_baselines_and_losses(simple_binary, tmp_path):
    config = profile_config("cpu_dev")
    source = _source()
    train, vocab = _dataset(simple_binary, "train", config, source)
    validation, _ = _dataset(simple_binary, "validation", config, source)
    trainer = build_trainer("cpu_dev", _config(tmp_path), train, validation, vocab)
    trainer.train()
    evaluation = evaluate_dataset(trainer.model, validation, trainer.training_config.objectives)
    assert evaluation["samples"] > 0
    assert "metrics" in evaluation
    assert "losses" in evaluation
    assert evaluation["metrics"]["cfg_f1"] >= 0
    assert torch.isfinite(torch.tensor(evaluation["weighted_validation_score"]))


def test_resume_restores_normalized_aggregator_and_scheduler_reference(simple_binary, tmp_path):
    config = profile_config("cpu_dev")
    source = _source()
    train, vocab = _dataset(simple_binary, "train", config, source)
    validation, _ = _dataset(simple_binary, "validation", config, source)
    first = build_trainer("cpu_dev", _config(tmp_path, run_name="state-first"), train, validation, vocab)
    result = first.train()
    running = dict(first.aggregator.running)
    assert running
    second = build_trainer("cpu_dev", _config(tmp_path, run_name="state-second", max_steps=3), train, validation, vocab)
    second.resume(result["checkpoint"]["path"])
    assert second.aggregator.running == running
    assert second.checkpoint_scheduler_config["max_steps"] == 2
    assert second.resumed_scheduler_config["max_steps"] == 3
    assert second.aggregator.export_state()["mode"] == "normalized"


def test_resume_reproduces_the_next_step_loss(simple_binary, tmp_path):
    config = profile_config("cpu_dev")
    source = _source()
    train, vocab = _dataset(simple_binary, "train", config, source, limit=4)
    validation, _ = _dataset(simple_binary, "validation", config, source)
    full = build_trainer("cpu_dev", _config(tmp_path, run_name="full-run", max_steps=2), train, validation, vocab)
    full_result = full.train()
    full_step2 = next(item for item in full_result["history"] if item["step"] == 2)["loss"]
    first = build_trainer("cpu_dev", _config(tmp_path, run_name="first-run", max_steps=1), train, validation, vocab)
    first_result = first.train()
    checkpoint = first_result["checkpoint"]["path"]
    resumed = build_trainer("cpu_dev", _config(tmp_path, run_name="resumed-run", max_steps=2), train, validation, vocab)
    resumed.resume(checkpoint)
    resumed_result = resumed.train()
    resumed_step2 = next(item for item in resumed_result["history"] if item["step"] == 2)["loss"]
    assert abs(full_step2 - resumed_step2) < 1e-6
    assert resumed_result["resumed_from"]["step"] == 1
