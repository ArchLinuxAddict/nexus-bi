import shutil
from collections.abc import Iterable
from dataclasses import asdict, dataclass

from nexus.core.errors import CapabilityUnavailable
from nexus.core.hardware import hardware_report


@dataclass(frozen=True)
class Capability:
    name: str
    status: str
    reason: str
    evidence: tuple[str, ...] = ()

    def to_dict(self) -> dict:
        value = asdict(self)
        value["evidence"] = list(self.evidence)
        return value


def _status(name: str, status: str, reason: str, evidence: Iterable[str] = ()) -> Capability:
    return Capability(name=name, status=status, reason=reason, evidence=tuple(evidence))


def registry() -> dict[str, Capability]:
    hardware = hardware_report()
    objdump = shutil.which("objdump")
    torch_ok = False
    try:
        import torch  # noqa: F401
        torch_ok = True
    except Exception:
        torch_ok = False
    capabilities = [
        _status("x86_64_elf_parsing", "available", "Pure-Python ELF32/ELF64 parser implemented", ("nexus/elf/parser.py",)),
        _status("x86_64_disassembly", "available" if objdump else "disabled", "GNU objdump backend" if objdump else "objdump not installed", (objdump or "",)),
        _status("multiarch_disassembly", "available", "LLVM objdump backend for x86, x86-64, ARM, AArch64, RISC-V, MIPS and PowerPC", ("nexus/disassembly/llvm_objdump.py", "nexus/program/multiarch.py")),
        _status("abi_signature_inference", "experimental", "Register read-before-write ABI inference for supported architectures", ("nexus/architectures/abi.py",)),
        _status("static_cfg", "available", "Conservative basic-block and CFG construction", ("nexus/program",)),
        _status("static_dataflow", "experimental", "Register def-use and memory-access heuristics; not a sound points-to analysis", ("nexus/dataflow",)),
        _status("type_candidates", "experimental", "Heuristic ABI/width candidates; not verified types", ("nexus/types",)),
        _status("cpu_model_smoke", "available" if torch_ok else "disabled", "PyTorch CPU smoke test" if torch_ok else "torch not installed", ("nexus/foundation",)),
        _status("large_scale_training", "available" if hardware["large_training_ready"] else "disabled", "Requires a supported GPU accelerator" if not hardware["large_training_ready"] else "GPU detected", ("nexus/foundation",)),
        _status("cross_architecture_execution", "disabled" if not any(hardware["isolation"]["qemu"].values()) else "experimental", "QEMU cross-architecture execution is not available" if not any(hardware["isolation"]["qemu"].values()) else "QEMU present; not yet validated", ("nexus/architectures",)),
        _status("qemu_emulation", "disabled" if not any(hardware["isolation"]["qemu"].values()) else "experimental", "QEMU not installed or not validated", ("nexus/sandbox",)),
        _status("dynamic_binary_execution", "disabled", "No verified KVM-backed sandbox; dynamic analysis is disabled", ("nexus/sandbox",)),
        _status("full_expert_reconstruction", "experimental", "Research objective; current reconstruction is limited and evidence-scored", ("nexus/reconstruction",)),
        _status("c_reconstruction", "experimental", "Limited pattern-based reconstruction plus explicit unresolved output", ("nexus/reconstruction/c",)),
        _status("java_class_analysis", "available", "Class-file constant pool, bytecode, line and local-variable parsing", ("nexus/formats/java_class.py",)),
        _status("dex_analysis", "experimental", "DEX tables and a practical Dalvik bytecode subset", ("nexus/formats/dex.py",)),
        _status("apk_analysis", "experimental", "APK manifest (binary XML), DEX, native-library and JNI inventory", ("nexus/formats/apk.py", "nexus/formats/android.py")),
        _status("java_reconstruction", "experimental", "Class/DEX straight-line bytecode reconstruction with explicit unresolved constructs; JNI symbol-driven native mapping", ("nexus/reconstruction/java",)),
        _status("jni_correlation", "experimental", "Java native declarations matched against JNI symbols with overload and RegisterNatives limitations", ("nexus/jni/correlation.py",)),
        _status("research", "available", "Offline curated ABI/ELF/JNI knowledge and local system-header search", ("nexus/research",)),
        _status("jni_analysis", "available", "JNI symbol, JNI_OnLoad, RegisterNatives and import inspection", ("nexus/jni",)),
        _status("security_analysis", "experimental", "Evidence-linked dangerous-API and bounds heuristics; findings are not confirmed exploits", ("nexus/security",)),
        _status("dataset_generation", "available", "Local x86-64 controlled compilation fixtures", ("nexus/datasets",)),
        _status("verification", "available", "Static checks and compiler detection; dynamic verification is capability-gated", ("nexus/verification",)),
        _status("api", "available", "FastAPI application", ("nexus/api",)),
        _status("dashboard", "available", "Static SPA served by the API", ("nexus/dashboard",)),
        _status("cli", "available", "argparse CLI", ("nexus/cli",)),
    ]
    return {item.name: item for item in capabilities}


def get(name: str) -> Capability:
    capabilities = registry()
    if name not in capabilities:
        raise CapabilityUnavailable(f"Unknown capability: {name}")
    return capabilities[name]


def require_available(name: str) -> None:
    capability = get(name)
    if capability.status not in {"available", "experimental"}:
        raise CapabilityUnavailable(
            f"Capability '{name}' is {capability.status}: {capability.reason}",
            details={"capability": capability.to_dict()},
        )


def require_strict(name: str) -> None:
    capability = get(name)
    if capability.status != "available":
        raise CapabilityUnavailable(
            f"Capability '{name}' is {capability.status}: {capability.reason}",
            details={"capability": capability.to_dict()},
        )


def as_dict() -> dict:
    return {name: capability.to_dict() for name, capability in registry().items()}
