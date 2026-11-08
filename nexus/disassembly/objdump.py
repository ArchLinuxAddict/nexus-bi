import re
import shutil
import subprocess

from nexus.core.errors import DisassemblyError, ToolUnavailable
from nexus.core.hashing import sha256_text
from nexus.disassembly.model import Disassembly, Instruction, MemoryEffect, Target
from nexus.elf.model import ElfFile

REGISTER_ALIASES = {
    "rax": "rax", "eax": "rax", "ax": "rax", "al": "rax", "ah": "rax",
    "rbx": "rbx", "ebx": "rbx", "bx": "rbx", "bl": "rbx", "bh": "rbx",
    "rcx": "rcx", "ecx": "rcx", "cx": "rcx", "cl": "rcx", "ch": "rcx",
    "rdx": "rdx", "edx": "rdx", "dx": "rdx", "dl": "rdx", "dh": "rdx",
    "rsi": "rsi", "esi": "rsi", "si": "rsi", "sil": "rsi",
    "rdi": "rdi", "edi": "rdi", "di": "rdi", "dil": "rdi",
    "rbp": "rbp", "ebp": "rbp", "bp": "rbp", "bpl": "rbp",
    "rsp": "rsp", "esp": "rsp", "sp": "rsp", "spl": "rsp",
    "rip": "rip", "eip": "rip", "ip": "rip",
    "r8": "r8", "r8d": "r8", "r8w": "r8", "r8b": "r8",
    "r9": "r9", "r9d": "r9", "r9w": "r9", "r9b": "r9",
    "r10": "r10", "r10d": "r10", "r10w": "r10", "r10b": "r10",
    "r11": "r11", "r11d": "r11", "r11w": "r11", "r11b": "r11",
    "r12": "r12", "r12d": "r12", "r12w": "r12", "r12b": "r12",
    "r13": "r13", "r13d": "r13", "r13w": "r13", "r13b": "r13",
    "r14": "r14", "r14d": "r14", "r14w": "r14", "r14b": "r14",
    "r15": "r15", "r15d": "r15", "r15w": "r15", "r15b": "r15",
}
REGISTER_WIDTHS = {
    "al": 8, "ah": 8, "bl": 8, "bh": 8, "cl": 8, "ch": 8, "dl": 8, "dh": 8,
    "sil": 8, "dil": 8, "bpl": 8, "spl": 8,
    "ax": 16, "bx": 16, "cx": 16, "dx": 16, "si": 16, "di": 16, "bp": 16, "sp": 16,
    "eax": 32, "ebx": 32, "ecx": 32, "edx": 32, "esi": 32, "edi": 32, "ebp": 32, "esp": 32,
    "rax": 64, "rbx": 64, "rcx": 64, "rdx": 64, "rsi": 64, "rdi": 64, "rbp": 64, "rsp": 64,
}
for _index in range(8, 16):
    REGISTER_WIDTHS[f"r{_index}"] = 64
    REGISTER_WIDTHS[f"r{_index}d"] = 32
    REGISTER_WIDTHS[f"r{_index}w"] = 16
    REGISTER_WIDTHS[f"r{_index}b"] = 8
CONDITIONAL_JUMPS = {
    "ja", "jae", "jb", "jbe", "jc", "je", "jg", "jge", "jl", "jle",
    "jna", "jnae", "jnb", "jnbe", "jnc", "jne", "jng", "jnge", "jnl",
    "jnle", "jno", "jnp", "jns", "jnz", "jo", "jp", "jpe", "jpo",
    "js", "jz", "jcxz", "jecxz", "jrcxz", "loop", "loope", "loopne",
}
SIZE_PREFIXES = {
    "byte": 8, "word": 16, "dword": 32, "qword": 64, "tbyte": 80,
    "xmmword": 128, "ymmword": 256, "zmmword": 512, "oword": 128,
    "fword": 48,
}


def _canonical_register(value: str) -> str | None:
    name = value.strip().lower().lstrip("%")
    return REGISTER_ALIASES.get(name, name if name.startswith("xmm") or name.startswith("ymm") or name.startswith("zmm") or name.startswith("st") else None)


def _register_width(value: str) -> int | None:
    name = value.strip().lower().lstrip("%")
    return REGISTER_WIDTHS.get(name)


def _registers_in(text: str) -> list[str]:
    registers = []
    for token in re.findall(r"%?[a-zA-Z][a-zA-Z0-9]*", text):
        canonical = _canonical_register(token)
        if canonical:
            registers.append(canonical)
    return registers


def _memory_operand(operand: str, kind: str) -> MemoryEffect | None:
    if "[" not in operand or "]" not in operand:
        return None
    width = None
    prefix = operand.strip().split("PTR", 1)[0].strip().lower()
    if prefix in SIZE_PREFIXES:
        width = SIZE_PREFIXES[prefix]
    else:
        for name in sorted(SIZE_PREFIXES, key=len, reverse=True):
            if prefix.endswith(name):
                width = SIZE_PREFIXES[name]
                break
    inside = operand[operand.find("[") + 1:operand.find("]")]
    segment = None
    if ":" in inside:
        segment, inside = inside.split(":", 1)
        segment = segment.strip().lower()
    base = None
    index = None
    displacement = 0
    parts = re.split(r"([+-])", inside.replace(" ", ""))
    sign = 1
    for part in parts:
        if part == "+":
            sign = 1
            continue
        if part == "-":
            sign = -1
            continue
        if not part:
            continue
        if "*" in part:
            reg, scale = part.split("*", 1)
            if _canonical_register(reg):
                index = _canonical_register(reg)
            if scale.isdigit():
                displacement += 0
            continue
        if _canonical_register(part):
            if base is None:
                base = _canonical_register(part)
            else:
                index = _canonical_register(part)
            continue
        try:
            if part.startswith("0x"):
                displacement += sign * int(part, 16)
            else:
                displacement += sign * int(part, 10)
        except ValueError:
            continue
    return MemoryEffect(kind=kind, address=operand.strip(), width=width, base=base, index=index, displacement=displacement, segment=segment)


def _is_register(operand: str) -> bool:
    return _canonical_register(operand.strip()) is not None and "[" not in operand


def _is_immediate(operand: str) -> bool:
    value = operand.strip()
    return bool(re.match(r"^\$?-?(?:0x[0-9a-fA-F]+|\d+)$", value))


def _is_memory(operand: str) -> bool:
    return "[" in operand or "PTR" in operand.upper()


def _direct_target(operand: str) -> int | None:
    match = re.search(r"\b([0-9a-fA-F]+)\s+<", operand)
    if match:
        return int(match.group(1), 16)
    if re.match(r"^(?:0x)?[0-9a-fA-F]+$", operand.strip()):
        return int(operand.strip(), 0)
    return None


def _operand_register_width(operand: str) -> int | None:
    if _is_register(operand):
        return _register_width(operand)
    for size_name, bits in SIZE_PREFIXES.items():
        if operand.strip().lower().startswith(size_name):
            return bits
    return None


def _effects(instruction: Instruction) -> None:
    mnemonic = instruction.mnemonic
    operands = instruction.operands
    reads: list[str] = []
    writes: list[str] = []
    memory: list[MemoryEffect] = []
    flags_read: list[str] = []
    flags_written: list[str] = []

    def read_operand(operand: str, access: str = "read") -> None:
        for register in _registers_in(operand):
            if register not in reads:
                reads.append(register)
        mem = _memory_operand(operand, access)
        if mem:
            memory.append(mem)
            for register in (mem.base, mem.index):
                if register and register not in reads:
                    reads.append(register)

    def write_operand(operand: str) -> None:
        if _is_register(operand):
            register = _canonical_register(operand)
            if register and register not in writes:
                writes.append(register)
        else:
            mem = _memory_operand(operand, "write")
            if mem:
                memory.append(mem)
                for register in (mem.base, mem.index):
                    if register and register not in reads:
                        reads.append(register)

    if mnemonic in {"nop", "endbr64", "endbr32", "ud2", "hlt", "pause", "lfence", "sfence", "mfence"}:
        pass
    elif mnemonic in {"cs", "data16"} and any("nop" in operand.lower() for operand in operands):
        pass
    elif mnemonic in {"mov", "movabs", "movzx", "movsx", "movsxd", "movbe"}:
        if len(operands) > 1:
            read_operand(operands[1])
        if operands:
            write_operand(operands[0])
    elif mnemonic == "lea":
        if len(operands) > 1:
            for register in _registers_in(operands[1]):
                if register not in reads:
                    reads.append(register)
        if operands:
            write_operand(operands[0])
    elif mnemonic in {"add", "sub", "and", "or", "xor", "shl", "shr", "sar", "sal", "rol", "ror", "rcl", "rcr", "inc", "dec", "neg", "not", "adc", "sbb"}:
        zeroing = (
            mnemonic in {"xor", "sub"}
            and len(operands) == 2
            and _is_register(operands[0])
            and _is_register(operands[1])
            and _canonical_register(operands[0]) == _canonical_register(operands[1])
        )
        for operand in operands:
            if zeroing and _is_register(operand):
                continue
            read_operand(operand)
        if operands:
            write_operand(operands[0])
        flags_written.extend(["CF", "OF", "SF", "ZF", "AF", "PF"])
        if zeroing:
            instruction.zeroing_idiom = True
        if mnemonic in {"adc", "sbb", "rcl", "rcr"}:
            flags_read.append("CF")
    elif mnemonic in {"cmp", "test"}:
        for operand in operands:
            read_operand(operand)
        flags_written.extend(["CF", "OF", "SF", "ZF", "AF", "PF"])
    elif mnemonic == "push":
        if operands:
            read_operand(operands[0])
        if "rsp" not in reads:
            reads.append("rsp")
        if "rsp" not in writes:
            writes.append("rsp")
        memory.append(MemoryEffect(kind="write", address="[rsp-8]", width=64, base="rsp", displacement=-8))
    elif mnemonic == "pop":
        if "rsp" not in reads:
            reads.append("rsp")
        if "rsp" not in writes:
            writes.append("rsp")
        memory.append(MemoryEffect(kind="read", address="[rsp]", width=64, base="rsp"))
        if operands:
            write_operand(operands[0])
    elif mnemonic in {"call", "callq"}:
        if "rsp" not in reads:
            reads.append("rsp")
        if "rsp" not in writes:
            writes.append("rsp")
        memory.append(MemoryEffect(kind="write", address="[rsp-8]", width=64, base="rsp", displacement=-8))
        if operands and "*" not in operands[0] and _direct_target(operands[0]) is None:
            for register in _registers_in(operands[0]):
                if register not in reads:
                    reads.append(register)
        if operands and "*" in operands[0]:
            for register in _registers_in(operands[0]):
                if register not in reads:
                    reads.append(register)
    elif mnemonic in {"jmp", "jmpq"}:
        if operands and "*" in operands[0]:
            for register in _registers_in(operands[0]):
                if register not in reads:
                    reads.append(register)
    elif mnemonic in {"ret", "retq", "iret", "iretq"}:
        if "rsp" not in reads:
            reads.append("rsp")
        if "rsp" not in writes:
            writes.append("rsp")
        memory.append(MemoryEffect(kind="read", address="[rsp]", width=64, base="rsp"))
    elif mnemonic in {"leave"}:
        reads.append("rbp")
        writes.extend(["rsp", "rbp"])
        memory.append(MemoryEffect(kind="read", address="[rbp]", width=64, base="rbp"))
    elif mnemonic.startswith("set"):
        for operand in operands:
            write_operand(operand)
        flags_read.extend(["CF", "OF", "SF", "ZF", "AF", "PF"])
    elif mnemonic.startswith("cmov"):
        if operands:
            read_operand(operands[0])
            write_operand(operands[0])
        if len(operands) > 1:
            read_operand(operands[1])
        flags_read.extend(["CF", "OF", "SF", "ZF", "AF", "PF"])
    elif mnemonic in CONDITIONAL_JUMPS:
        flags_read.extend(["CF", "OF", "SF", "ZF", "AF", "PF"])
    elif mnemonic in {"imul"}:
        for operand in operands:
            read_operand(operand)
        if operands:
            write_operand(operands[0])
        flags_written.extend(["CF", "OF"])
    elif mnemonic in {"div", "idiv"}:
        for register in ("rax", "rdx"):
            if register not in reads:
                reads.append(register)
        if operands:
            read_operand(operands[0])
        writes.extend(["rax", "rdx"])
    elif mnemonic in {"mul"}:
        if "rax" not in reads:
            reads.append("rax")
        if operands:
            read_operand(operands[0])
        writes.extend(["rax", "rdx"])
        flags_written.extend(["CF", "OF"])
    elif mnemonic == "xchg":
        for operand in operands:
            read_operand(operand)
            write_operand(operand)
    elif mnemonic == "cpuid":
        reads.extend(["rax", "rcx"])
        writes.extend(["rax", "rbx", "rcx", "rdx"])
    elif mnemonic == "syscall":
        reads.extend(["rax", "rdi", "rsi", "rdx", "r10", "r8", "r9"])
        writes.extend(["rax", "rcx", "r11"])
    elif mnemonic.startswith("stos"):
        reads.extend(["rdi", "rax"])
        writes.append("rdi")
        memory.append(MemoryEffect(kind="write", address="[rdi]", width=64, base="rdi"))
    elif mnemonic.startswith("lods"):
        reads.extend(["rsi"])
        writes.extend(["rsi", "rax"])
        memory.append(MemoryEffect(kind="read", address="[rsi]", width=64, base="rsi"))
    elif mnemonic.startswith("scas"):
        reads.extend(["rdi", "rax"])
        writes.append("rdi")
        memory.append(MemoryEffect(kind="read", address="[rdi]", width=64, base="rdi"))
    elif mnemonic.startswith("movs"):
        reads.extend(["rsi", "rdi"])
        writes.extend(["rsi", "rdi"])
        memory.append(MemoryEffect(kind="read", address="[rsi]", width=64, base="rsi"))
        memory.append(MemoryEffect(kind="write", address="[rdi]", width=64, base="rdi"))
    else:
        for operand in operands:
            read_operand(operand)
        instruction.confidence = min(instruction.confidence, 0.65)

    instruction.reads = sorted(set(reads))
    instruction.writes = sorted(set(writes))
    instruction.memory = memory
    instruction.flags_read = sorted(set(flags_read))
    instruction.flags_written = sorted(set(flags_written))


def _control_flow(instruction: Instruction) -> None:
    mnemonic = instruction.mnemonic
    operands = instruction.operands
    target = _direct_target(operands[0]) if operands else None
    if mnemonic in {"ret", "retq", "iret", "iretq"}:
        instruction.control_flow = "return"
    elif mnemonic in {"call", "callq"}:
        instruction.control_flow = "indirect_call" if operands and "*" in operands[0] else "call"
        if target is not None:
            instruction.targets.append(Target(target, "direct", 1.0))
        elif operands and "*" in operands[0]:
            instruction.targets.append(Target(None, "indirect", 0.5))
    elif mnemonic in {"jmp", "jmpq"}:
        instruction.control_flow = "indirect_branch" if operands and "*" in operands[0] else "unconditional_branch"
        if target is not None:
            instruction.targets.append(Target(target, "direct", 1.0))
        elif operands and "*" in operands[0]:
            instruction.targets.append(Target(None, "indirect", 0.5))
    elif mnemonic in CONDITIONAL_JUMPS:
        instruction.control_flow = "conditional_branch"
        if target is not None:
            instruction.targets.append(Target(target, "direct", 1.0))
    elif mnemonic in {"syscall", "int", "int3", "ud2", "hlt"}:
        instruction.control_flow = "trap"
    else:
        instruction.control_flow = "fallthrough"


def _parse_instruction_line(line: str, section: str | None, function_label: str | None) -> Instruction | None:
    match = re.match(r"^\s*([0-9a-fA-F]+):\t(.*)$", line)
    if not match:
        return None
    address = int(match.group(1), 16)
    rest = match.group(2)
    if "\t" in rest:
        bytes_text, text = rest.split("\t", 1)
    else:
        bytes_match = re.match(r"^((?:[0-9a-fA-F]{2}\s+)+)(.*)$", rest)
        if not bytes_match:
            return None
        bytes_text, text = bytes_match.group(1), bytes_match.group(2)
    bytes_tokens = bytes_text.strip().split()
    if not bytes_tokens or any(not re.fullmatch(r"[0-9a-fA-F]{2}", token) for token in bytes_tokens):
        return None
    bytes_hex = "".join(token.lower() for token in bytes_tokens)
    text = text.strip()
    comment = None
    if "#" in text:
        text, comment = text.split("#", 1)
        text = text.strip()
        comment = comment.strip()
    if not text:
        return None
    tokens = text.split(None, 1)
    mnemonic = tokens[0].lower()
    if mnemonic in {"lock", "rep", "repz", "repnz", "bnd", "notrack"} and len(tokens) > 1:
        prefix = mnemonic
        tokens = tokens[1].split(None, 1)
        mnemonic = tokens[0].lower()
        text = f"{prefix} {text}"
    operand_text = tokens[1].strip() if len(tokens) > 1 else ""
    operands = [item.strip() for item in operand_text.split(",")] if operand_text else []
    decoded = "(bad)" not in text
    instruction = Instruction(
        address=address,
        bytes_hex=bytes_hex,
        length=len(bytes_hex) // 2,
        text=text,
        mnemonic=mnemonic,
        operands=operands,
        function_label=function_label,
        section=section,
        comment=comment,
        decoded=decoded,
        confidence=1.0 if decoded else 0.0,
    )
    if decoded:
        _effects(instruction)
        _control_flow(instruction)
    return instruction


def disassemble_elf(elf: ElfFile, timeout: int = 120) -> Disassembly:
    if elf.header.machine_name != "x86_64":
        raise DisassemblyError(
            f"disassembly is not implemented for {elf.header.machine_name}; x86-64 is the validated backend"
        )
    objdump = shutil.which("objdump")
    if objdump is None:
        raise ToolUnavailable("objdump is not installed")
    try:
        result = subprocess.run(
            [objdump, "-d", "-M", "intel", elf.path],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        raise DisassemblyError("objdump timed out", details={"timeout": timeout}) from exc
    if result.returncode != 0:
        raise DisassemblyError("objdump failed", details={"stderr": result.stderr[-2000:]})
    instructions: list[Instruction] = []
    functions: dict[str, int] = {}
    sections: list[str] = []
    current_section = None
    current_function = None
    for line in result.stdout.splitlines():
        section_match = re.match(r"^Disassembly of section (.+):$", line.strip())
        if section_match:
            current_section = section_match.group(1)
            if current_section not in sections:
                sections.append(current_section)
            current_function = None
            continue
        function_match = re.match(r"^([0-9a-fA-F]+) <([^>]+)>:$", line.strip())
        if function_match:
            address = int(function_match.group(1), 16)
            current_function = function_match.group(2)
            functions[current_function] = address
            continue
        continuation = re.match(r"^\s*([0-9a-fA-F]+):\t((?:[0-9a-fA-F]{2}[ \t]+)+)$", line)
        if continuation is not None and instructions:
            address = int(continuation.group(1), 16)
            last = instructions[-1]
            if address == last.address + len(last.bytes_hex) // 2:
                extra = "".join(continuation.group(2).split()).lower()
                last.bytes_hex += extra
                last.length = len(last.bytes_hex) // 2
                continue
        instruction = _parse_instruction_line(line, current_section, current_function)
        if instruction is not None:
            instructions.append(instruction)
    version = None
    try:
        version_result = subprocess.run([objdump, "--version"], capture_output=True, text=True, timeout=10)
        version = version_result.stdout.splitlines()[0] if version_result.stdout else None
    except Exception:
        version = None
    return Disassembly(
        backend="gnu-objdump",
        version=version,
        instructions=instructions,
        functions=functions,
        sections=sections,
        raw_output_sha256=sha256_text(result.stdout),
        limitations=[
            "Function labels are derived from objdump symbols; stripped binaries may have none.",
            "Indirect control-flow targets are not resolved.",
            "Register effects are conservative heuristics, not a sound data-flow analysis.",
        ],
    )


def operand_register_width(operand: str) -> int | None:
    return _operand_register_width(operand)


def canonical_register(value: str) -> str | None:
    return _canonical_register(value)
