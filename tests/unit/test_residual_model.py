import torch

from nexus.foundation.config import profile_config
from nexus.foundation.residual_data import collate_residual
from nexus.foundation.residual_model import build_residual_model
from nexus.foundation.vocab import build_vocabulary


def test_residual_model_forward_shapes():
    texts = ["int add(int a, int b) { return a + b; }", "class A { int x() { return 1; } }"]
    vocab = build_vocabulary([], texts, text_vocab_size=512)
    config = profile_config("cpu_dev")
    records = [
        {"sample_id": "c1", "language": "c", "tool": "ghidra_headless", "tool_status": "completed", "compile_status": "passed", "tool_text": texts[0], "target_text": texts[0], "failure_label": 0, "candidate_label": 1.0, "confidence_label": 1.0},
        {"sample_id": "j1", "language": "java", "tool": "jadx", "tool_status": "completed", "compile_status": "unknown", "tool_text": texts[1], "target_text": texts[1], "failure_label": 0, "candidate_label": 1.0, "confidence_label": 1.0},
    ]
    batch = collate_residual(records, vocab, config)
    model = build_residual_model(config, seed=1)
    outputs = model(batch, repair_input_ids=batch["repair_input_ids"][:, :-1], repair_input_mask=batch["repair_mask"][:, :-1])
    assert outputs["candidate_logit"].shape == (2,)
    assert outputs["failure_logits"].shape == (2, 2)
    assert outputs["confidence"].shape == (2,)
    assert outputs["repair_logits"].shape[0] == 2
    assert torch.isfinite(outputs["repair_logits"]).all()
    assert model.count_parameters() > 0
