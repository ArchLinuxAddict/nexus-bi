import json

from nexus.foundation.checkpoint import save_foundation_checkpoint
from nexus.foundation.config import config_for_vocabulary, profile_config
from nexus.foundation.model import build_foundation_model
from nexus.foundation.reconstruction import UnifiedReconstructor, check_c_source, structural_similarity
from nexus.foundation.representation import extract_c_function
from nexus.foundation.vocab import build_vocabulary
from nexus.program.bundle import analyze_binary
from nexus.program.record import build_program_record


def _checkpoint(binary, tmp_path):
    record = build_program_record(analyze_binary(binary))
    vocab = build_vocabulary([record], [], text_vocab_size=512)
    config = config_for_vocabulary(profile_config("cpu_dev"), vocab)
    model = build_foundation_model(config, seed=11)
    saved = save_foundation_checkpoint(
        tmp_path / "random.pt",
        model,
        config,
        vocab,
        metadata={"status": "random_initialization", "trained": False, "steps": 0},
        step=0,
        include_training_state=False,
    )
    return saved, config, vocab


def test_c_function_extraction_and_separate_statuses():
    source = "int add(int a, int b) { return a + b; }\nint other(void) { return 0; }\n"
    extracted = extract_c_function(source, "add")
    assert extracted and "return a + b" in extracted
    similarity = structural_similarity(extracted, source)
    assert similarity["status"] == "measured"
    assert 0.0 <= similarity["ratio"] <= 1.0
    result = check_c_source(extracted)
    assert set(result) >= {"syntax", "compilation", "note"}
    assert result["syntax"]["status"] in {"passed", "failed", "not_attempted"}
    # Compilation status must be separate from syntax and must not claim behavior.
    assert "behavior" not in json.dumps(result).lower()


def test_deterministic_reconstruction_preserves_statuses(simple_binary):
    reconstructor = UnifiedReconstructor(workspace=simple_binary.parent)
    bundle = analyze_binary(simple_binary)
    add = next(function for function in bundle.functions if function.name == "add")
    result = reconstructor.reconstruct_function(simple_binary, add.function_id, language="c", include_neural=False)
    assert result["status"] == "completed"
    candidate = result["candidates"][0]
    statuses = candidate["statuses"]
    assert statuses["syntax_valid"] in {"passed", "failed", "not_attempted"}
    assert statuses["compilation"] in {"passed", "failed", "not_attempted"}
    assert statuses["behavioral_agreement"] == "not_attempted"
    assert statuses["finite_domain_verified"] == "not_attempted"
    assert statuses["universal_equivalence"] == "not_claimed"
    assert candidate["method"] == "deterministic_semantic_ir"
    assert candidate["evidence"]["names_used_as_model_input"] is False
    assert candidate["evidence"]["source_used_as_model_input"] is False


def test_neural_reconstruction_uses_binary_only_input_and_does_not_claim_correctness(simple_binary, tmp_path):
    saved, config, vocab = _checkpoint(simple_binary, tmp_path)
    reconstructor = UnifiedReconstructor(checkpoint_path=saved["path"], workspace=simple_binary.parent)
    bundle = analyze_binary(simple_binary)
    add = next(function for function in bundle.functions if function.name == "add")
    result = reconstructor.reconstruct_function(
        simple_binary,
        add.function_id,
        language="c",
        include_deterministic=False,
        include_neural=True,
        max_new_tokens=8,
    )
    assert result["status"] == "completed"
    candidate = result["candidates"][0]
    assert candidate["method"] == "neural_unified_model"
    assert candidate["statuses"]["universal_equivalence"] == "not_claimed"
    assert candidate["statuses"]["behavioral_agreement"] == "not_attempted"
    assert candidate["additional"]["checkpoint_sha256"] == saved["sha256"]
    assert result["model"]["parameter_count"] == reconstructor.model.count_parameters()


def test_project_reconstruction_keeps_unresolved_operations(simple_binary):
    reconstructor = UnifiedReconstructor(workspace=simple_binary.parent)
    result = reconstructor.reconstruct_project(simple_binary, language="c", max_functions=2, include_neural=False)
    assert result["functions_considered"] >= 1
    assert result["functions_with_candidates"] >= 1
    assert result["verification_scope"].startswith("compilation and syntax")
    assert "unresolved_operations" in result
    for candidate_group in result["results"]:
        for candidate in candidate_group["candidates"]:
            assert candidate["statuses"]["universal_equivalence"] == "not_claimed"


def test_jni_signature_prediction_is_labeled_unverified(jni_binary, tmp_path):
    saved, _, _ = _checkpoint(jni_binary, tmp_path)
    reconstructor = UnifiedReconstructor(checkpoint_path=saved["path"], workspace=tmp_path)
    bundle = analyze_binary(jni_binary)
    function = bundle.functions[0]
    result = reconstructor.predict_jni_signature(jni_binary, function.function_id)
    assert result["status"] in {"predicted_unverified", "not_found"}
    if result["status"] == "predicted_unverified":
        assert "not a verified JNI mapping" in result["note"]
        assert "predicted_return_type" in result


def test_reconstruction_budget_is_dynamic_and_bounded():
    from nexus.foundation.evaluation import reconstruction_generation_budget, target_is_truncated
    from nexus.foundation.vocab import build_vocabulary

    assert reconstruction_generation_budget(10, 256) == 96
    assert reconstruction_generation_budget(200, 256) == 216
    assert reconstruction_generation_budget(1000, 256) == 255
    vocab = build_vocabulary([], [], text_vocab_size=512)
    short = "int f(void) { return 0; }"
    long_source = "int f(void) { " + " ".join(["int x = 1;"] * 80) + " return 0; }"
    assert target_is_truncated(short, vocab, 256) is False
    assert target_is_truncated(long_source, vocab, 64) is True
