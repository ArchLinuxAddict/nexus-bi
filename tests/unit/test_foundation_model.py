import torch

from nexus.foundation.config import MODEL_PROFILES, profile_config, profile_report
from nexus.foundation.model import build_foundation_model, parameter_report
from nexus.foundation.objectives import OBJECTIVES, LossAggregator, compute_objective_losses, gradient_norms
from nexus.foundation.representation import (
    RepresentationOptions,
    autoregressive_instruction_batch,
    collate_samples,
    encode_record_samples,
    mask_byte_batch,
    mask_evidence_batch,
    mask_instruction_batch,
)
from nexus.foundation.vocab import build_vocabulary
from nexus.program.bundle import analyze_binary
from nexus.program.record import build_program_record


def _samples(binary, config=None, options=None, count=2):
    config = config or profile_config("cpu_dev")
    record = build_program_record(analyze_binary(binary))
    vocab = build_vocabulary([record], [], text_vocab_size=512)
    samples = encode_record_samples(
        record,
        vocab,
        config,
        identity={
            "sample_id": "unit",
            "project_id": "unit",
            "dataset_id": "synthetic-smoke",
            "split": "train",
            "binary_sha256": record["program"]["sha256"],
            "source_family": "fixture",
            "match_group": "unit",
        },
        options=options or RepresentationOptions(),
    )
    return samples[:count], vocab, config


def test_profiles_are_one_architecture_family():
    report = profile_report()
    assert report["architecture_version"] == "nexus-bi/foundation-1"
    assert set(report["profiles"]) == set(MODEL_PROFILES)
    counts = [item["estimated_parameters"] for item in report["profiles"].values()]
    assert counts == sorted(counts)
    assert counts[-1] > counts[0]


def test_random_initialization_is_deterministic_and_not_pretrained():
    config = profile_config("cpu_dev")
    first = build_foundation_model(config, seed=7)
    second = build_foundation_model(config, seed=7)
    other = build_foundation_model(config, seed=8)
    for name, tensor in first.state_dict().items():
        assert torch.equal(tensor, second.state_dict()[name])
    assert any(not torch.equal(tensor, other.state_dict()[name]) for name, tensor in first.state_dict().items())
    assert first.count_parameters() > 0
    assert parameter_report(first)["total"] == first.count_parameters()


def test_unified_forward_heads_and_missing_modalities(simple_binary):
    samples, vocab, config = _samples(simple_binary)
    batch = collate_samples(samples, config)
    model = build_foundation_model(config)
    outputs = model(batch)
    assert outputs["instruction_logits"].shape[:2] == batch["instr_id"].shape
    assert outputs["byte_logits"].shape[:2] == batch["byte_ids"].shape
    assert outputs["cfg_logits"].shape[1:] == (batch["block_mask"].size(1), batch["block_mask"].size(1))
    assert outputs["purpose_logits"].shape[1] == config.num_purpose_classes
    assert outputs["jni_logits"].shape[1:] == (5, config.jni_type_classes)
    assert outputs["function_embedding"].shape == outputs["pooled"].shape
    # Missing bytes/IR/graph must still produce a finite forward pass.
    missing_options = RepresentationOptions(include_bytes=False, include_ir=False, include_graph=False)
    missing_samples, _, _ = _samples(simple_binary, config=config, options=missing_options)
    missing_batch = collate_samples(missing_samples, config)
    with torch.no_grad():
        missing = model(missing_batch)
    assert torch.isfinite(missing["pooled"]).all()


def test_graph_bias_is_typed_and_nonzero(simple_binary):
    _, _, config = _samples(simple_binary)
    model = build_foundation_model(config)
    edges = torch.tensor([[[0, 1], [1, 2]]], dtype=torch.long)
    types = torch.tensor([[0, 2]], dtype=torch.long)
    mask = torch.tensor([[True, True]])
    bias = model._graph_bias(edges, types, mask, 3)
    assert bias.shape == (1, config.n_heads, 3, 3)
    assert float(bias.detach().abs().sum()) > 0


def test_hierarchical_module_and_project_encoding(simple_binary):
    samples, _, config = _samples(simple_binary, count=4)
    model = build_foundation_model(config)
    batch = collate_samples(samples, config)
    index = torch.tensor([0, 0, 1, 1], dtype=torch.long)
    with torch.no_grad():
        module = model.encode_module(batch, index, module_count=2)
    assert module["module_pooled"].shape[0] == 2
    assert module["function_context"].shape[0] == 4
    project = model.encode_project(module["module_vectors"], module["module_mask"])
    assert project.shape == (2, config.d_model)
    assert torch.isfinite(project).all()


def test_reconstruction_decoder_training_and_generation(simple_binary):
    samples, _, config = _samples(simple_binary)
    batch = collate_samples(samples, config)
    batch_size, text_length = len(samples), 8
    batch["c_ids"] = torch.randint(8, min(config.text_vocab, 40), (batch_size, text_length))
    batch["c_ids"][:, 0] = 3
    batch["c_ids"][:, -1] = 4
    batch["c_mask"] = torch.ones((batch_size, text_length), dtype=torch.bool)
    model = build_foundation_model(config)
    outputs = model(batch, run_decoder=True, decoder_language="c")
    assert outputs["c_logits"].shape[:2] == (batch_size, text_length - 1)
    assert outputs["java_logits"].shape == outputs["c_logits"].shape
    generated = model.generate_reconstruction(batch, language="c", max_new_tokens=4)
    assert len(generated["token_ids"]) == batch_size
    assert generated["language"] == "c"


def test_all_objective_losses_backpropagate(simple_binary):
    samples, _, config = _samples(simple_binary)
    batch = collate_samples(samples, config)
    batch = mask_instruction_batch(batch, probability=0.5, seed=3)
    batch = mask_byte_batch(batch, probability=0.5, seed=3)
    batch["c_ids"] = torch.randint(8, min(config.text_vocab, 40), (len(samples), 6))
    batch["c_ids"][:, 0] = 3
    batch["c_ids"][:, -1] = 4
    batch["c_mask"] = torch.ones_like(batch["c_ids"], dtype=torch.bool)
    batch["purpose_target"][:] = 1
    batch["tool_target"][:] = 2
    batch["security_target"][:] = 1
    batch["confidence_mask"][:] = True
    batch["confidence_target"][:] = 0.5
    model = build_foundation_model(config)
    outputs = model(batch, run_decoder=True, decoder_language="c")
    evidence_batch = mask_evidence_batch(batch, seed=5)
    evidence = model(evidence_batch)
    outputs["ir_mlm_logits"] = evidence["ir_logits"]
    outputs["register_read_masked_logits"] = evidence["register_read_logits"]
    outputs["register_write_masked_logits"] = evidence["register_write_logits"]
    outputs["memory_read_masked_logits"] = evidence["memory_read_logits"]
    outputs["memory_write_masked_logits"] = evidence["memory_write_logits"]
    outputs["memory_width_masked_logits"] = evidence["memory_width_logits"]
    outputs["cfg_masked_logits"] = evidence["cfg_logits"]
    batch["ir_targets"] = evidence_batch["ir_targets"]
    batch["ir_mlm_mask"] = evidence_batch["ir_mlm_mask"]
    ar_batch = autoregressive_instruction_batch(batch)
    ar_outputs = model(ar_batch, causal_instructions=True)
    outputs["instr_ar_logits"] = ar_outputs["instruction_logits"]
    enabled = [objective.objective_id for objective in OBJECTIVES if objective.implemented]
    raw = compute_objective_losses(outputs, batch, enabled)
    active = {key: value for key, value in raw.items() if value[1] > 0}
    assert "instruction_masked_modeling" in active
    assert "byte_masked_modeling" in active
    assert "ir_op_prediction" in active
    assert "cfg_edge_prediction" in active
    assert "c_reconstruction" in active
    total = sum(value[0] for value in active.values())
    total.backward()
    assert any(parameter.grad is not None for parameter in model.parameters())
    assert torch.isfinite(total)


def test_loss_aggregator_reports_inactive_objectives_and_gradient_norms(simple_binary):
    samples, _, config = _samples(simple_binary)
    batch = collate_samples(samples, config)
    batch = mask_instruction_batch(batch, probability=0.5, seed=4)
    model = build_foundation_model(config)
    outputs = model(batch)
    raw = compute_objective_losses(outputs, batch, ["instruction_masked_modeling", "c_reconstruction", "purpose_prediction"])
    aggregator = LossAggregator(mode="normalized", enabled=["instruction_masked_modeling", "c_reconstruction", "purpose_prediction"])
    total, results = aggregator(raw, step=0)
    assert torch.isfinite(total)
    statuses = {result.objective_id: result.active for result in results}
    assert statuses["instruction_masked_modeling"] is True
    assert statuses["c_reconstruction"] is False
    norms = gradient_norms(model, raw)
    assert "instruction_masked_modeling" in norms
    assert norms["instruction_masked_modeling"] >= 0


def test_variable_lengths_and_vocab_fallbacks(simple_binary):
    samples, vocab, config = _samples(simple_binary, count=4)
    lengths = {len(sample["instr_id"]) for sample in samples}
    assert len(lengths) >= 1
    batch = collate_samples(samples, config)
    model = build_foundation_model(config)
    with torch.no_grad():
        outputs = model(batch)
    assert outputs["pooled"].shape[0] == len(samples)
    # Unknown symbols must fall back rather than index out of range.
    assert vocab.encode("mnemonic", "definitely_not_an_instruction") == 1
    assert vocab.encode_text("int x = 123456789;", 64)


def test_instruction_masking_hides_identity_not_only_target(simple_binary):
    samples, _, config = _samples(simple_binary)
    batch = collate_samples(samples, config)
    masked = mask_instruction_batch(batch, probability=0.5, seed=11)
    selected = masked["instr_mlm_mask"]
    assert selected.any()
    assert (masked["instr_mnemonic"][selected] == 2).all()
    assert (masked["instr_shape"][selected] == 2).all()
    assert (masked["instr_ir"][selected] == 2).all()
    assert (masked["instr_regs"][selected] == 2).all()
    assert (masked["instr_kinds"][selected] == 2).all()
    assert (masked["instr_access"] == 0).all()
    assert (masked["instr_width"] == 0).all()


def test_ir_evidence_masking_hides_mnemonic_and_shape(simple_binary):
    samples, _, config = _samples(simple_binary)
    batch = collate_samples(samples, config)
    masked = mask_evidence_batch(batch, seed=13)
    selected = masked["ir_mlm_mask"]
    assert selected.any()
    assert (masked["instr_mnemonic"][selected] == 2).all()
    assert (masked["instr_shape"][selected] == 2).all()
    assert (masked["instr_ir"][selected] == 2).all()
