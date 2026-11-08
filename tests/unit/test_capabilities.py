from nexus.architectures import capability_matrix, get_adapter
from nexus.core.capabilities import as_dict
from nexus.core.hardware import hardware_report


def test_capability_registry_has_disabled_gates():
    capabilities = as_dict()
    assert capabilities["dynamic_binary_execution"]["status"] == "disabled"
    assert capabilities["cross_architecture_execution"]["status"] == "disabled"
    assert capabilities["qemu_emulation"]["status"] == "disabled"
    assert capabilities["x86_64_elf_parsing"]["status"] == "available"


def test_architecture_matrix_is_honest():
    matrix = {item["name"]: item for item in capability_matrix()}
    assert matrix["x86_64"]["instruction_decoding"] is True
    assert matrix["aarch64"]["instruction_decoding"] is True
    assert matrix["aarch64"]["cfg"] is True
    assert matrix["armv7"]["cfg"] is True
    assert matrix["riscv"]["instruction_decoding"] is True
    assert matrix["x86"]["cfg"] is True
    assert get_adapter(62).name == "x86_64"


def test_hardware_report_reports_no_compute_gpu():
    report = hardware_report()
    assert "host" in report
    assert "gpu" in report
    assert report["large_training_ready"] is False
