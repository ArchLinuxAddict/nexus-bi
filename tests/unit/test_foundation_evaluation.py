import json

from nexus.foundation.config import profile_config
from nexus.foundation.dataset import DatasetSource, FoundationDataset
from nexus.foundation.evaluation import evaluate_checkpoint, markdown_report
from nexus.foundation.representation import RepresentationOptions, encode_record_samples
from nexus.foundation.training import FoundationTrainingConfig, build_trainer
from nexus.foundation.vocab import build_vocabulary
from nexus.program.bundle import analyze_binary
from nexus.program.record import build_program_record


def _dataset(binary, split, config, source, with_source=False):
    record = build_program_record(analyze_binary(binary))
    vocab = build_vocabulary([record], [], text_vocab_size=512)
    samples = encode_record_samples(
        record,
        vocab,
        config,
        identity={
            "sample_id": f"{split}-sample",
            "project_id": f"project-{split}",
            "dataset_id": source.dataset_id,
            "split": split,
            "binary_sha256": record["program"]["sha256"],
            "source_family": "fixture",
            "match_group": f"project-{split}",
            "binary_path": str(binary),
        },
        options=RepresentationOptions(),
    )[:2]
    if with_source and samples:
        source_text = "int add(int a, int b) { return a + b; }"
        samples[0]["source_text"] = source_text
        samples[0]["source_sha256"] = "a" * 64
        samples[0]["source_path"] = "simple.c"
        samples[0]["c_ids"] = vocab.encode_text(source_text, config.max_text_tokens)
        samples[0]["c_mask"] = [1] * len(samples[0]["c_ids"])
        samples[0]["source_ids"] = samples[0]["c_ids"]
        samples[0]["labels"]["source_alignment"] = {"sha256": "a" * 64, "alignment": "unit", "confidence": 1.0}
    return FoundationDataset(samples=samples, config=config, vocab=vocab, split=split, sources=[source]), vocab


def test_evaluation_reports_capability_categories(simple_binary, tmp_path):
    config = profile_config("cpu_dev")
    source = DatasetSource(
        path="synthetic",
        dataset_id="realworld-so-v1",
        category="real_world_system_binary",
        classification="EXPERIMENTAL",
        sha256="unit",
    )
    train, vocab = _dataset(simple_binary, "train", config, source, with_source=True)
    validation, _ = _dataset(simple_binary, "validation", config, source)
    test, _ = _dataset(simple_binary, "test", config, source, with_source=True)
    training = FoundationTrainingConfig(
        run_name="eval-unit",
        max_steps=2,
        batch_size=1,
        val_interval=1,
        checkpoint_interval=2,
        objectives=["instruction_masked_modeling", "ir_op_prediction", "cfg_edge_prediction", "c_reconstruction"],
        mission_objectives=["binary_representation_masking", "instruction_semantics", "basic_block_correspondence", "c_reconstruction"],
        dataset_categories=["real_world_system_binary"],
        dataset_ids=["realworld-so-v1"],
        run_mode="smoke",
        gradient_diagnostics_interval=0,
        output_dir=str(tmp_path),
    )
    trainer = build_trainer("cpu_dev", training, train, validation, vocab)
    result = trainer.train()
    report = evaluate_checkpoint(result["checkpoint"]["path"], test, train, reconstruction_samples=1, permutations=3, workspace=tmp_path)
    assert report["datasets"]["test_used_for_training"] is False
    assert "byte_understanding" in report["capability_categories"]
    assert "cross_architecture_generalization" in report["capability_categories"]
    assert report["capability_categories"]["cross_architecture_generalization"]["status"] == "unavailable"
    assert "metrics" in report and "baselines" in report
    assert "ablations" in report and "full" in report["ablations"]
    assert "negative_witness_random_initialization" in report
    assert report["calibration"]["status"] in {"measured", "not_attempted"}
    assert report["reconstruction"]["status"] in {"measured", "unavailable"}
    if report["reconstruction"]["status"] == "measured":
        summary = report["reconstruction"]["summary"]
        assert summary["behavioral_verification"] == "not_attempted"
        assert summary["universal_equivalence"] == "not_claimed"
    text = markdown_report(report)
    assert "not a foundation-model capability claim" in text
    json.dumps(report, default=str)
