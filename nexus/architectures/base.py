"""Architecture and ABI descriptions used across the platform.

These records are deterministic metadata: ELF machine ids, pointer widths,
calling-convention registers, stack/link registers and capability claims.  They
do not decode instructions by themselves.  Instruction decoding is provided by
the selected deterministic backend (GNU objdump for x86, LLVM objdump for the
generic multi-architecture path).
"""

from collections.abc import Iterable
from dataclasses import asdict, dataclass

from nexus.core.errors import CapabilityUnavailable


@dataclass(frozen=True)
class ArchitectureAdapter:
    name: str
    elf_machine: int
    aliases: tuple[str, ...]
    elf_recognition: bool
    instruction_decoding: bool
    cfg: bool
    semantic_analysis: bool
    reconstruction: bool
    verification: bool
    abi: str
    argument_registers: tuple[str, ...] = ()
    return_register: str | None = None
    callee_saved: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()
    pointer_width: int = 64
    endianness: str = "little"
    instruction_alignment: int = 1
    stack_pointer: str | None = None
    frame_pointer: str | None = None
    link_register: str | None = None
    scratch_registers: tuple[str, ...] = ()
    syscall_registers: tuple[str, ...] = ()
    float_argument_registers: tuple[str, ...] = ()
    llvm_triple: str = ""
    function_prologues: tuple[tuple[str, ...], ...] = ()
    decoder: str = "llvm-objdump"

    def to_dict(self) -> dict:
        return asdict(self)


X86_64 = ArchitectureAdapter(
    name="x86_64",
    elf_machine=62,
    aliases=("x86_64", "amd64", "x86-64"),
    elf_recognition=True,
    instruction_decoding=True,
    cfg=True,
    semantic_analysis=True,
    reconstruction=True,
    verification=False,
    abi="System V AMD64",
    argument_registers=("rdi", "rsi", "rdx", "rcx", "r8", "r9"),
    return_register="rax",
    callee_saved=("rbx", "rbp", "r12", "r13", "r14", "r15"),
    pointer_width=64,
    stack_pointer="rsp",
    frame_pointer="rbp",
    scratch_registers=("rax", "rcx", "rdx", "rsi", "rdi", "r8", "r9", "r10", "r11"),
    syscall_registers=("rax", "rdi", "rsi", "rdx", "r10", "r8", "r9"),
    float_argument_registers=("xmm0", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5", "xmm6", "xmm7"),
    llvm_triple="x86_64-linux-gnu",
    function_prologues=(("push", "rbp"), ("endbr64",), ("endbr32",)),
    notes=(
        "Instruction decoding and conservative CFG are validated on x86-64 fixtures.",
        "Semantic and reconstruction capabilities are experimental and evidence-scored.",
        "Dynamic verification is disabled until a verified sandbox exists.",
    ),
)

X86 = ArchitectureAdapter(
    name="x86",
    elf_machine=3,
    aliases=("x86", "i386", "i686"),
    elf_recognition=True,
    instruction_decoding=True,
    cfg=True,
    semantic_analysis=False,
    reconstruction=False,
    verification=False,
    abi="System V i386",
    argument_registers=("rdi", "rsi", "rdx", "rcx", "r8", "r9"),
    return_register="rax",
    callee_saved=("rbx", "rbp", "esi", "edi"),
    pointer_width=32,
    stack_pointer="rsp",
    frame_pointer="rbp",
    scratch_registers=("rax", "rcx", "rdx"),
    syscall_registers=("rax", "rbx", "rcx", "rdx", "rsi", "rdi", "rbp"),
    llvm_triple="i386-linux-gnu",
    function_prologues=(("push", "ebp"),),
    notes=(
        "Generic LLVM decoding and CFG recovery are available.",
        "The C reconstruction translator remains x86-64 oriented; 32-bit output is structural only.",
    ),
)

ARM = ArchitectureAdapter(
    name="armv7",
    elf_machine=40,
    aliases=("arm", "armv7", "thumb", "thumb2", "armhf", "armel"),
    elf_recognition=True,
    instruction_decoding=True,
    cfg=True,
    semantic_analysis=False,
    reconstruction=True,
    verification=False,
    abi="AAPCS32",
    argument_registers=("arm_r0", "arm_r1", "arm_r2", "arm_r3"),
    return_register="arm_r0",
    callee_saved=("arm_r4", "arm_r5", "arm_r6", "arm_r7", "arm_r8", "arm_r9", "arm_r10", "arm_r11"),
    pointer_width=32,
    instruction_alignment=2,
    stack_pointer="arm_sp",
    frame_pointer="arm_r11",
    link_register="arm_lr",
    scratch_registers=("arm_r0", "arm_r1", "arm_r2", "arm_r3", "arm_r12"),
    syscall_registers=("arm_r7",),
    float_argument_registers=("s0", "s1", "s2", "s3", "s4", "s5", "s6", "s7"),
    llvm_triple="armv7-linux-gnueabihf",
    function_prologues=(("push",), ("str", "arm_lr")),
    notes=(
        "LLVM-based decoding, CFG and AAPCS32 signature inference are available.",
        "Thumb/ARM interworking and reconstruction semantics remain experimental.",
    ),
)

AARCH64 = ArchitectureAdapter(
    name="aarch64",
    elf_machine=183,
    aliases=("aarch64", "arm64", "armv8"),
    elf_recognition=True,
    instruction_decoding=True,
    cfg=True,
    semantic_analysis=False,
    reconstruction=True,
    verification=False,
    abi="AAPCS64",
    argument_registers=("a64_x0", "a64_x1", "a64_x2", "a64_x3", "a64_x4", "a64_x5", "a64_x6", "a64_x7"),
    return_register="a64_x0",
    callee_saved=("a64_x19", "a64_x20", "a64_x21", "a64_x22", "a64_x23", "a64_x24", "a64_x25", "a64_x26", "a64_x27", "a64_x28"),
    pointer_width=64,
    instruction_alignment=4,
    stack_pointer="a64_sp",
    frame_pointer="a64_fp",
    link_register="a64_lr",
    scratch_registers=("a64_x0", "a64_x1", "a64_x2", "a64_x3", "a64_x4", "a64_x5", "a64_x6", "a64_x7", "a64_x8", "a64_x9", "a64_x10", "a64_x11", "a64_x12", "a64_x13", "a64_x14", "a64_x15", "a64_x16", "a64_x17"),
    syscall_registers=("a64_x8",),
    float_argument_registers=("v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7"),
    llvm_triple="aarch64-linux-gnu",
    function_prologues=(("stp", "a64_fp"), ("sub", "a64_sp"), ("pacibsp",), ("bti",)),
    notes=(
        "LLVM-based decoding, CFG and AAPCS64 signature inference are available.",
        "Vector/SIMD semantics and source reconstruction remain experimental.",
    ),
)

RISCV = ArchitectureAdapter(
    name="riscv",
    elf_machine=243,
    aliases=("riscv", "riscv64", "rv64"),
    elf_recognition=True,
    instruction_decoding=True,
    cfg=True,
    semantic_analysis=False,
    reconstruction=True,
    verification=False,
    abi="RISC-V LP64 psABI",
    argument_registers=("rv_a0", "rv_a1", "rv_a2", "rv_a3", "rv_a4", "rv_a5", "rv_a6", "rv_a7"),
    return_register="rv_a0",
    callee_saved=("rv_s0", "rv_s1", "rv_s2", "rv_s3", "rv_s4", "rv_s5", "rv_s6", "rv_s7", "rv_s8", "rv_s9", "rv_s10", "rv_s11"),
    pointer_width=64,
    instruction_alignment=2,
    stack_pointer="rv_sp",
    frame_pointer="rv_s0",
    link_register="rv_ra",
    scratch_registers=("rv_a0", "rv_a1", "rv_a2", "rv_a3", "rv_a4", "rv_a5", "rv_a6", "rv_a7", "rv_t0", "rv_t1", "rv_t2", "rv_t3", "rv_t4", "rv_t5", "rv_t6"),
    syscall_registers=("rv_a7", "rv_a0", "rv_a1", "rv_a2", "rv_a3", "rv_a4", "rv_a5"),
    float_argument_registers=("fa0", "fa1", "fa2", "fa3", "fa4", "fa5", "fa6", "fa7"),
    llvm_triple="riscv64-linux-gnu",
    function_prologues=(("addi", "rv_sp"), ("addi", "rv_s0")),
    notes=(
        "LLVM-based decoding, CFG and LP64 signature inference are available.",
        "Compressed instructions are decoded by the backend; reconstruction remains experimental.",
    ),
)

RISCV32 = ArchitectureAdapter(
    name="riscv32",
    elf_machine=243,
    aliases=("riscv32", "rv32"),
    elf_recognition=True,
    instruction_decoding=True,
    cfg=True,
    semantic_analysis=False,
    reconstruction=True,
    verification=False,
    abi="RISC-V ILP32 psABI",
    argument_registers=RISCV.argument_registers,
    return_register=RISCV.return_register,
    callee_saved=RISCV.callee_saved,
    pointer_width=32,
    instruction_alignment=2,
    stack_pointer=RISCV.stack_pointer,
    frame_pointer=RISCV.frame_pointer,
    link_register=RISCV.link_register,
    scratch_registers=RISCV.scratch_registers,
    syscall_registers=RISCV.syscall_registers,
    float_argument_registers=RISCV.float_argument_registers,
    llvm_triple="riscv32-linux-gnu",
    function_prologues=RISCV.function_prologues,
    notes=(
        "LLVM-based decoding, CFG and ILP32 signature inference are available.",
        "Compressed instructions are decoded by the backend; reconstruction remains experimental.",
    ),
)

MIPS = ArchitectureAdapter(
    name="mips",
    elf_machine=8,
    aliases=("mips", "mipsel", "mips32", "mips64"),
    elf_recognition=True,
    instruction_decoding=True,
    cfg=True,
    semantic_analysis=False,
    reconstruction=False,
    verification=False,
    abi="MIPS o32/n32/n64",
    argument_registers=("mips_a0", "mips_a1", "mips_a2", "mips_a3"),
    return_register="mips_v0",
    callee_saved=("mips_s0", "mips_s1", "mips_s2", "mips_s3", "mips_s4", "mips_s5", "mips_s6", "mips_s7"),
    pointer_width=32,
    instruction_alignment=4,
    stack_pointer="mips_sp",
    frame_pointer="mips_fp",
    link_register="mips_ra",
    llvm_triple="mipsel-linux-gnu",
    notes=("LLVM-based decoding and CFG are available; ABI variants are not automatically distinguished.",),
)

POWERPC = ArchitectureAdapter(
    name="powerpc",
    elf_machine=20,
    aliases=("powerpc", "ppc", "ppc64", "powerpc64"),
    elf_recognition=True,
    instruction_decoding=True,
    cfg=True,
    semantic_analysis=False,
    reconstruction=False,
    verification=False,
    abi="PowerPC SysV",
    argument_registers=("r3", "r4", "r5", "r6", "r7", "r8", "r9", "r10"),
    return_register="r3",
    callee_saved=("r14", "r15", "r16", "r17", "r18", "r19", "r20", "r21", "r22", "r23", "r24", "r25", "r26", "r27", "r28", "r29", "r30", "r31"),
    pointer_width=32,
    instruction_alignment=4,
    stack_pointer="r1",
    frame_pointer="r31",
    link_register="lr",
    llvm_triple="powerpc-linux-gnu",
    notes=("LLVM-based decoding and CFG are available; ELFv1/ELFv2 ABI selection is not automatic.",),
)

ADAPTERS: tuple[ArchitectureAdapter, ...] = (X86_64, X86, ARM, AARCH64, RISCV, RISCV32, MIPS, POWERPC)
BY_MACHINE = {adapter.elf_machine: adapter for adapter in ADAPTERS}
BY_NAME = {name: adapter for adapter in ADAPTERS for name in (adapter.name, *adapter.aliases)}
# ELF machine 243 is shared by RV32/RV64; get_adapter defaults to the 64-bit form.
BY_MACHINE[243] = RISCV


def get_adapter(name_or_machine: str | int, *, elf_class: str | None = None) -> ArchitectureAdapter:
    if isinstance(name_or_machine, int):
        adapter = BY_MACHINE.get(name_or_machine)
        if adapter is RISCV and elf_class in {"ELF32", "32"}:
            return RISCV32
    else:
        adapter = BY_NAME.get(str(name_or_machine).lower())
    if adapter is None:
        raise CapabilityUnavailable(f"No architecture adapter for {name_or_machine!r}")
    return adapter


def adapter_for_elf(elf) -> ArchitectureAdapter:
    return get_adapter(elf.header.machine, elf_class=elf.header.elf_class)


def normalize_architecture_name(value: str) -> str:
    adapter = get_adapter(value)
    return adapter.name


def require_decoding(adapter: ArchitectureAdapter) -> None:
    if not adapter.instruction_decoding:
        raise CapabilityUnavailable(
            f"Instruction decoding is not validated for {adapter.name}",
            details={"architecture": adapter.to_dict()},
        )


def all_adapters() -> Iterable[ArchitectureAdapter]:
    return ADAPTERS


def capability_matrix() -> list[dict]:
    return [adapter.to_dict() for adapter in ADAPTERS]
