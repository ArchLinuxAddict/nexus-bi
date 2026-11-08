import torch

from nexus.foundation.config import config_for_vocabulary, profile_config
from nexus.foundation.residual_data import collate_residual
from nexus.foundation.residual_model import build_residual_model
from nexus.foundation.vocab import build_vocabulary


def _record(tool_text: str, target_text: str) -> dict:
    return {
        "sample_id": "copy",
        "language": "java",
        "tool": "jni_table",
        "tool_status": "completed",
        "compile_status": "unknown",
        "tool_text": tool_text,
        "target_text": target_text,
        "failure_label": 0,
        "candidate_label": 1.0,
        "confidence_label": 1.0,
    }


def test_tokenizer_preserves_whitespace_roundtrip():
    text = "public static native byte[] f(byte[] arg0);\n    return arg0;"
    vocab = build_vocabulary([], [text], text_vocab_size=4096)
    ids = vocab.encode_text(text, 512)
    assert vocab.decode_text(ids) == text


def test_copy_labels_point_to_full_source_positions_and_model_shapes():
    tool_text = "JNI: (I)V\ncandidate: public void f(int arg0);"
    target_text = "public static native void f(int arg0);"
    vocab = build_vocabulary([], [tool_text + "\n" + target_text], text_vocab_size=4096)
    config = config_for_vocabulary(profile_config("cpu_dev"), vocab)
    batch = collate_residual([_record(tool_text, target_text)], vocab, config)
    labels = batch["copy_labels"][0]
    target_ids = batch["repair_input_ids"][0]
    tool_ids = batch["tool_ids"][0]
    assert (labels != -100).any()
    for position, label in enumerate(labels.tolist()):
        if label < 0:
            continue
        assert 0 <= label < tool_ids.numel()
        assert int(tool_ids[label]) == int(target_ids[position])
    model = build_residual_model(config, seed=1)
    outputs = model(
        batch,
        repair_input_ids=batch["repair_input_ids"][:, :-1],
        repair_input_mask=batch["repair_mask"][:, :-1],
    )
    assert outputs["repair_log_probs"].shape[0] == 1
    assert outputs["repair_log_probs"].shape[-1] == config.text_vocab
    assert torch.isfinite(outputs["repair_log_probs"]).all()
    assert outputs["copy_logits"].shape[:2] == outputs["repair_log_probs"].shape[:2]


def test_copy_can_be_disabled_for_legacy_checkpoints():
    tool_text = "int add(int a, int b)"
    vocab = build_vocabulary([], [tool_text], text_vocab_size=1024)
    config = config_for_vocabulary(profile_config("cpu_dev"), vocab)
    batch = collate_residual([_record(tool_text, tool_text)], vocab, config)
    model = build_residual_model(config, seed=2)
    model.use_copy = False
    outputs = model(batch, repair_input_ids=batch["repair_input_ids"][:, :-1], repair_input_mask=batch["repair_mask"][:, :-1])
    assert "repair_log_probs" not in outputs
    assert outputs["repair_logits"].shape[0] == 1


def test_canonical_jni_declaration_overrides_generated_types():
    from nexus.foundation.java_types import canonical_declaration, canonicalize_generated

    assert canonical_declaration(
        "f",
        "(Landroid/content/pm/PackageManager;I)Ljava/util/List;",
    ) == "public static native java.util.List f(android.content.pm.PackageManager arg0, int arg1);"
    assert canonicalize_generated(
        "garbage output",
        "decryptAssetBlob",
        "([B)[B",
    ) == "public static native byte[] decryptAssetBlob(byte[] arg0);"
