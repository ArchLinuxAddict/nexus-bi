from nexus.program.bundle import analyze_binary
from nexus.program.record import build_program_record
from nexus.verification.static import verify_program_record


def test_recover_add_and_main(simple_binary):
    bundle = analyze_binary(simple_binary)
    names = {function.name for function in bundle.functions}
    assert "add" in names
    assert "main" in names
    add = bundle.function_by_name("add")
    assert add is not None
    assert add.confidence > 0.5
    assert add.blocks
    assert add.instructions


def test_call_graph_contains_add(simple_binary):
    bundle = analyze_binary(simple_binary)
    add = bundle.function_by_name("add")
    main = bundle.function_by_name("main")
    assert add and main
    assert any(edge.target == add.function_id for edge in main.callees)


def test_program_record_validates(simple_binary):
    bundle = analyze_binary(simple_binary)
    record = build_program_record(bundle)
    validation = verify_program_record(record)
    assert validation["status"] == "passed", validation
    assert record["program"]["machine"] == "x86_64"
    assert record["functions"]
    assert record["instructions"]
    assert all(len(item["bytes"]) // 2 == item["length"] for item in record["instructions"])


def test_record_evidence_is_complete(simple_binary):
    bundle = analyze_binary(simple_binary)
    record = build_program_record(bundle)
    evidence_ids = {item["evidence_id"] for item in record["evidence"]}
    referenced = set()

    def walk(value):
        if isinstance(value, dict):
            for item in value.values():
                walk(item)
        elif isinstance(value, list):
            for item in value:
                walk(item)
        elif isinstance(value, str) and value.startswith("ev_"):
            referenced.add(value)

    walk(record)
    assert referenced <= evidence_ids


def test_stripped_binary_record(stripped_binary):
    bundle = analyze_binary(stripped_binary)
    record = build_program_record(bundle)
    assert record["program"]["stripped"] is True
    assert record["functions"]
    assert verify_program_record(record)["status"] == "passed"


def test_deterministic_analysis(simple_binary):
    first = analyze_binary(simple_binary)
    second = analyze_binary(simple_binary)
    assert first.elf.sha256 == second.elf.sha256
    assert [item.entry for item in first.functions] == [item.entry for item in second.functions]
    assert [(item.address, item.bytes_hex) for item in first.disassembly.instructions] == [(item.address, item.bytes_hex) for item in second.disassembly.instructions]
