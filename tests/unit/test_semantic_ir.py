from nexus.program.bundle import analyze_binary
from nexus.program.ir import IR_VERSION, build_program_ir, serialize_program_ir


def test_semantic_ir_preserves_addresses(simple_binary):
    bundle = analyze_binary(simple_binary)
    ir = build_program_ir(bundle)
    assert ir["schema_version"] == IR_VERSION
    assert ir["binary_sha256"] == bundle.elf.sha256
    assert ir["functions"]
    first = ir["functions"][0]
    assert first["instructions"]
    assert first["blocks"]
    instruction_ids = {instruction["ir_id"] for instruction in first["instructions"]}
    for block in first["blocks"]:
        assert set(block["instruction_ids"]) <= instruction_ids
    for instruction in first["instructions"]:
        assert instruction["evidence"].startswith("ev_")
        assert instruction["address"].startswith("0x")


def test_semantic_ir_serialization_is_stable(simple_binary):
    bundle = analyze_binary(simple_binary)
    first = serialize_program_ir(build_program_ir(bundle))
    second = serialize_program_ir(build_program_ir(bundle))
    assert first == second
