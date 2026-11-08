"""Evidence-driven x86-64 to C semantic reconstruction emitter.

The emitter is an independent, evidence-driven x86-64 to C translator.  It is
deliberately bounded: unsupported instructions, unresolved flags, unresolved
indirect control flow and unmodelled memory shapes fail explicitly instead of
being approximated as no-ops.  It is used as one strategy in a controlled
comparison with the historical Phase 8 emitters.
"""

import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from nexus.disassembly.model import Instruction
from nexus.program.bundle import AnalysisBundle
from nexus.program.model import FunctionCandidate
from nexus.reconstruction.c.generator import _abi_prototype
from nexus.reconstruction.c.plt import resolve_plt_stub

SCALAR_REGISTERS = (
    "rax", "rbx", "rcx", "rdx", "rsi", "rdi", "rbp", "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15",
)
ARGUMENT_REGISTERS = ("rdi", "rsi", "rdx", "rcx", "r8", "r9")
HIGH_BYTE_REGISTERS = {"ah": "rax", "bh": "rbx", "ch": "rcx", "dh": "rdx"}
CALLER_SAVED = ("rax", "rcx", "rdx", "rsi", "rdi", "r8", "r9", "r10", "r11")
VECTOR_REGISTERS = tuple(f"xmm{index}" for index in range(16))
SIZE_WIDTHS = {"BYTE": 8, "WORD": 16, "DWORD": 32, "QWORD": 64, "XMMWORD": 128}
NOP_MNEMONICS = {"nop", "endbr64", "endbr32", "cs", "data16", "pause", "lfence", "sfence", "mfence"}
CONDITION_ALIASES = {
    "e": "e", "z": "e", "ne": "ne", "nz": "ne",
    "l": "l", "nge": "l", "le": "le", "ng": "le",
    "g": "g", "nle": "g", "ge": "ge", "nl": "ge",
    "b": "b", "nae": "b", "c": "b", "be": "be", "na": "be",
    "a": "a", "nbe": "a", "ae": "ae", "nb": "ae", "nc": "ae",
    "s": "s", "ns": "ns", "o": "o", "no": "no",
    "p": "p", "pe": "p", "np": "np", "po": "np",
}
SIGNED_CONDITIONS = {"l", "le", "g", "ge"}
UNSIGNED_CONDITIONS = {"b", "be", "a", "ae"}
KNOWN_LIBC = {
    "malloc": ("ptr", ["size_t"]),
    "calloc": ("ptr", ["size_t", "size_t"]),
    "realloc": ("ptr", ["ptr", "size_t"]),
    "free": ("void", ["ptr"]),
    "memcpy": ("ptr", ["ptr", "ptr", "size_t"]),
    "memmove": ("ptr", ["ptr", "ptr", "size_t"]),
    "memset": ("ptr", ["ptr", "int", "size_t"]),
    "strlen": ("size_t", ["ptr"]),
    "strcmp": ("int", ["ptr", "ptr"]),
    "strncmp": ("int", ["ptr", "ptr", "size_t"]),
    "strcpy": ("ptr", ["ptr", "ptr"]),
    "puts": ("int", ["ptr"]),
    "printf": ("int", ["ptr"]),
    "abort": ("void", []),
    "exit": ("void", ["int"]),
    "_exit": ("void", ["int"]),
    "__stack_chk_fail": ("void", []),
}
NORETURN = {"abort", "exit", "_exit", "__stack_chk_fail", "__assert_fail"}


class EmitError(Exception):
    def __init__(self, reason: str, address: int | None = None, category: str = "unsupported"):
        super().__init__(reason)
        self.reason = reason
        self.address = address
        self.category = category


@dataclass
class FlagState:
    kind: str
    left: str
    right: str
    result: str
    width: int
    signed: bool = False
    high: str | None = None


@dataclass
class Operand:
    text: str
    kind: str
    width: int = 64
    register: str | None = None
    vector: bool = False
    immediate: int | None = None
    memory: dict[str, Any] | None = None


@dataclass
class EmitResult:
    success: bool
    body: str | None
    reason: str | None
    stats: dict[str, Any] = field(default_factory=dict)


@dataclass
class Prototype:
    return_name: str
    params: str
    return_width: int
    parameter_count: int
    parameter_widths: list[int]


def _sanitize(name: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_]", "_", name)
    if not value:
        value = "reconstructed"
    if value[0].isdigit():
        value = f"fn_{value}"
    return value


def _canonical(name: str) -> str | None:
    from nexus.disassembly.objdump import canonical_register

    value = canonical_register(name.strip())
    if value:
        return value
    lowered = name.strip().lower()
    if re.fullmatch(r"(xmm|ymm|zmm)\d+", lowered):
        return lowered
    return None


def _register_width(name: str) -> int:
    lowered = name.strip().lower()
    if lowered.startswith(("xmm", "ymm", "zmm")):
        return 128 if lowered.startswith("xmm") else (256 if lowered.startswith("ymm") else 512)
    from nexus.disassembly.objdump import operand_register_width

    return operand_register_width(name) or 64


def _is_vector_register(name: str) -> bool:
    return bool(re.fullmatch(r"(xmm|ymm|zmm)\d+", name.strip().lower()))


def _immediate(text: str) -> int | None:
    value = text.strip().removeprefix("$")
    try:
        return int(value, 0)
    except ValueError:
        try:
            return int(value, 10)
        except ValueError:
            return None


def parse_memory(text: str) -> dict[str, Any] | None:
    stripped = text.strip()
    segment = None
    if stripped.lower().startswith("fs:"):
        segment = "fs"
        stripped = stripped[3:]
    if "[" not in stripped or "]" not in stripped:
        return None
    prefix = stripped[: stripped.find("[")].strip()
    width = 64
    for name, bits in SIZE_WIDTHS.items():
        if prefix.upper().startswith(name):
            width = bits
            break
    inside = stripped[stripped.find("[") + 1 : stripped.rfind("]")]
    base = None
    index = None
    scale = 1
    displacement = 0
    rip = False
    sign = 1
    for part in re.split(r"([+\-])", inside.replace(" ", "")):
        if part == "+":
            sign = 1
            continue
        if part == "-":
            sign = -1
            continue
        if not part:
            continue
        if part == "rip":
            rip = True
            continue
        if "*" in part:
            register, scale_text = part.split("*", 1)
            canonical = _canonical(register)
            if canonical is None:
                return None
            index = canonical
            try:
                scale = int(scale_text, 0)
            except ValueError:
                return None
            continue
        canonical = _canonical(part)
        if canonical:
            if base is None:
                base = canonical
            else:
                index = canonical
            continue
        try:
            value = int(part, 0)
            if value >= (1 << 63):
                value -= 1 << 64
            elif value >= (1 << 31) and value < (1 << 32):
                value -= 1 << 32
            displacement += sign * value
        except ValueError:
            return None
    return {
        "width": width,
        "base": base,
        "index": index,
        "scale": scale,
        "disp": displacement,
        "rip": rip,
        "segment": segment,
        "text": text,
    }


def parse_operand(text: str) -> Operand:
    stripped = text.strip()
    lowered = stripped.lower()
    if "fs:" in lowered and "[" not in stripped:
        after = stripped[lowered.find("fs:") + 3 :].strip()
        try:
            displacement = int(after, 0)
        except ValueError:
            displacement = 0
        width = 64
        prefix = stripped[: lowered.find("fs:")].strip().upper()
        for name, bits in SIZE_WIDTHS.items():
            if prefix.startswith(name):
                width = bits
                break
        return Operand(stripped, "memory", width, memory={
            "width": width, "base": None, "index": None, "scale": 1, "disp": displacement,
            "rip": False, "segment": "fs", "text": stripped,
        })
    register = _canonical(stripped) if "[" not in stripped else None
    if register:
        if _is_vector_register(register):
            return Operand(stripped, "vector", _register_width(register), register, True)
        return Operand(stripped, "register", _register_width(stripped), register)
    immediate = _immediate(stripped)
    if immediate is not None:
        return Operand(stripped, "immediate", 64, immediate=immediate)
    memory = parse_memory(stripped)
    if memory:
        return Operand(stripped, "memory", int(memory["width"]), memory=memory)
    return Operand(stripped, "unknown", 64)


def _slot_name(base: str, displacement: int) -> str:
    sign = "m" if displacement < 0 else "p"
    return f"stack_{base}_{sign}{abs(displacement)}"


def _mask(width: int) -> int:
    return (1 << width) - 1


def _c_type(width: int) -> str:
    return {8: "uint8_t", 16: "uint16_t", 32: "uint32_t", 64: "uint64_t"}[width]


def _signed_type(width: int) -> str:
    return {8: "int8_t", 16: "int16_t", 32: "int32_t", 64: "int64_t"}[width]


def _sign_bit(width: int) -> int:
    return 1 << (width - 1)
VECTOR_HELPERS = {
    "paddb": "nexus_v_paddb", "paddw": "nexus_v_paddw", "paddd": "nexus_v_paddd", "paddq": "nexus_v_paddq",
    "psubb": "nexus_v_psubb", "psubw": "nexus_v_psubw", "psubd": "nexus_v_psubd", "psubq": "nexus_v_psubq",
    "pand": "nexus_v_pand", "por": "nexus_v_por", "pandn": "nexus_v_pandn", "pxor": "nexus_v_pxor",
    "andps": "nexus_v_pand", "andpd": "nexus_v_pand", "orps": "nexus_v_por", "orpd": "nexus_v_por",
    "xorps": "nexus_v_pxor", "xorpd": "nexus_v_pxor", "andnps": "nexus_v_pandn", "andnpd": "nexus_v_pandn",
    "pminub": "nexus_v_pminub", "pmaxub": "nexus_v_pmaxub", "pminud": "nexus_v_pminud", "pmaxud": "nexus_v_pmaxud",
    "pcmpeqb": "nexus_v_pcmpeqb", "pcmpeqw": "nexus_v_pcmpeqw", "pcmpeqd": "nexus_v_pcmpeqd", "pcmpeqq": "nexus_v_pcmpeqq",
    "pcmpgtb": "nexus_v_pcmpgtb", "pcmpgtw": "nexus_v_pcmpgtw", "pcmpgtd": "nexus_v_pcmpgtd", "pcmpgtq": "nexus_v_pcmpgtq",
    "pmuludq": "nexus_v_pmuludq", "punpcklqdq": "nexus_v_punpcklqdq", "punpckhqdq": "nexus_v_punpckhqdq",
    "unpcklps": "nexus_v_unpcklps", "unpckhps": "nexus_v_unpckhps", "unpcklpd": "nexus_v_unpcklpd", "unpckhpd": "nexus_v_unpckhpd",
}


def _vector_helper_name(mnemonic: str) -> str | None:
    return VECTOR_HELPERS.get(mnemonic)


class _Translator:
    def __init__(
        self,
        bundle: AnalysisBundle,
        function: FunctionCandidate,
        name: str,
        prototypes: dict[int, Prototype],
        name_map: dict[str, str],
        strategy: str,
    ) -> None:
        self.bundle = bundle
        self.function = function
        self.name = name
        self.prototypes = prototypes
        self.name_map = name_map
        self.strategy = strategy
        self.lines: list[str] = []
        self.prelude: list[str] = []
        self.data_pool: list[str] = []
        self.data_index = 0
        self.externs: dict[str, str] = {}
        self.unsupported: list[dict[str, Any]] = []
        self.stats: Counter = Counter()
        self.flags: FlagState | None = None
        self.block_starts = {block.start for block in function.blocks}
        self.instruction_addresses = {instruction.address for instruction in function.instructions}
        self.functions_by_entry = {item.entry: item for item in bundle.functions}
        self.frame_pointer = self._detect_frame_pointer()
        self.vector_used = False
        for instruction in function.instructions:
            for operand in instruction.operands:
                canonical = _canonical(operand)
                if canonical and _is_vector_register(canonical):
                    self.vector_used = True
                    break
            if self.vector_used:
                break
        self.rsp = 4096
        self.declared_scalar: set[str] = set()
        self.declared_vector: set[str] = set()
        self.current_versions: dict[str, str] = {}
        self.version_counter: dict[str, int] = {}
        self.stack_offsets: set[str] = set()
        self.pointer_registers: set[str] = set()
        self.noreturn_call = False
        self.jump_table: dict[str, Any] | None = None
        self.ssa = strategy in {"ssa", "ssa_informed"}

    def _detect_frame_pointer(self) -> bool:
        for instruction in self.function.instructions:
            if instruction.mnemonic == "mov" and len(instruction.operands) == 2 and instruction.operands[0] == "rbp" and instruction.operands[1] == "rsp":
                return True
            if instruction.mnemonic == "lea" and len(instruction.operands) == 2 and instruction.operands[0] == "rbp" and instruction.operands[1].startswith("[rsp"):
                return True
        return False

    def _proto(self) -> Prototype:
        return_name, params, return_width = _abi_prototype(self.function, self.bundle)
        names = [part.strip() for part in params.split(",") if part.strip()] if params != "void" else []
        widths = []
        for name in names:
            match = re.search(r"uint(\d+)_t", name)
            widths.append(int(match.group(1)) if match else 64)
        return Prototype(return_name, params, return_width, len(names), widths)

    def _arg_expression(self, register: str, width: int) -> str:
        # Argument registers are initialized from the recovered ABI parameters
        # at function entry.  Calls must pass the current register value, not
        # the original source argument, because optimized code reuses them.
        return f"((uint64_t){self._read_register(register)})"

    def _read_register(self, register: str) -> str:
        if self.ssa and register in self.current_versions:
            return self.current_versions[register]
        return register

    def _write_register(self, register: str, expression: str, width: int) -> list[str]:
        if _is_vector_register(register):
            self.declared_vector.add(register)
            return [f"    {register} = ({expression});"]
        self.declared_scalar.add(register)
        target = register
        if self.ssa:
            count = self.version_counter.get(register, 0) + 1
            self.version_counter[register] = count
            target = f"{register}_v{count}"
            self.declared_scalar.add(target)
            self.current_versions[register] = target
        if width == 64:
            return [f"    {target} = (uint64_t)({expression});"]
        if width == 32:
            return [f"    {target} = (uint64_t)(uint32_t)({expression});"]
        old = self._read_register(register)
        mask = _mask(width)
        if width == 16:
            return [f"    {target} = (({old}) & ~UINT64_C(0x{mask:x})) | (uint64_t)((uint16_t)({expression}));"]
        if width == 8:
            return [f"    {target} = (({old}) & ~UINT64_C(0x{mask:x})) | (uint64_t)((uint8_t)({expression}));"]
        raise EmitError(f"unsupported register width {width}")

    def _merge_versions(self) -> list[str]:
        if not self.ssa:
            return []
        return [f"    {register} = {version};" for register, version in list(self.current_versions.items()) if register != version]

    def _address_expression(self, memory: dict[str, Any], as_pointer: bool) -> str:
        if memory["segment"] == "fs":
            if memory["disp"] == 0x28:
                return "((uint64_t)(uintptr_t)&nexus_canary_value)"
            raise EmitError(f"unsupported fs segment displacement {memory['disp']:#x}", category="unsupported_memory")
        if memory["rip"]:
            target = self._instruction_end + int(memory["disp"])
            payload = self.bundle.elf.read_va(target, 256) or b""
            if not payload:
                raise EmitError(f"unreadable rip-relative data at 0x{target:x}", category="unsupported_memory")
            size = 16 if int(memory.get("width") or 64) == 128 else 256
            payload = payload[: max(16, min(size, len(payload)))]
            symbol = self._add_data(payload, align=16 if int(memory.get("width") or 64) == 128 else 8)
            return f"((uint64_t)(uintptr_t){symbol})"
        base = memory["base"]
        terms: list[str] = []
        if base == "rsp":
            terms.append("((uint64_t)rsp)")
        elif base == "rbp" and self.frame_pointer:
            terms.append(f"((uint64_t){self._read_register('rbp')})")
        elif base:
            terms.append(f"((uint64_t){self._read_register(base)})")
        index = memory["index"]
        if index:
            if index == "rsp":
                terms.append(f"((uint64_t)rsp * {memory['scale']})")
            elif index == "rbp" and self.frame_pointer:
                terms.append(f"((uint64_t){self._read_register('rbp')} * {memory['scale']})")
            else:
                terms.append(f"((uint64_t){self._read_register(index)} * {memory['scale']})")
        displacement = int(memory["disp"])
        if displacement:
            terms.append(f"((uint64_t){displacement})" if displacement > 0 else f"-((uint64_t){abs(displacement)})")
        if not terms:
            raise EmitError(f"empty memory address {memory.get('text', '')}", category="unsupported_memory")
        expression = terms[0]
        for term in terms[1:]:
            expression += f" + {term}" if not term.startswith("-") else f" {term}"
        return f"((uint64_t)(uintptr_t)({expression}))"

    def _address(self, operand: Operand) -> str:
        assert operand.memory is not None
        return self._address_expression(operand.memory, as_pointer=True)

    def _add_data(self, payload: bytes, align: int = 8) -> str:
        symbol = f"nexus_data_{self.data_index}"
        self.data_index += 1
        values = ", ".join(f"0x{byte:02x}" for byte in payload)
        self.data_pool.append(
            f"static const uint8_t {symbol}[{len(payload)}] __attribute__((aligned({align}))) = {{{values}}};"
        )
        return symbol

    def _load(self, operand: Operand, width: int | None = None) -> str:
        value_width = width or operand.width
        if operand.kind == "register":
            register = operand.register or ""
            if operand.text.lower() in HIGH_BYTE_REGISTERS and value_width == 8:
                return f"(((uint8_t)(({self._read_register(HIGH_BYTE_REGISTERS[operand.text.lower()])} >> 8) & 0xff)))"
            return f"(({_c_type(value_width)})({self._read_register(register)}))"
        if operand.kind == "vector":
            return self._read_register(operand.register or "xmm0")
        if operand.kind == "immediate":
            return f"(({_c_type(value_width)})UINT64_C({int(operand.immediate or 0) & _mask(value_width)}))"
        if operand.kind == "memory":
            address = self._address(operand)
            if value_width == 128:
                return f"nexus_v_load((const void *)(uintptr_t)({address}))"
            return f"(*(({_c_type(value_width)} *)(uintptr_t)({address})))"
        raise EmitError(f"unsupported operand {operand.text}", category="unsupported_operand")

    def _store(self, operand: Operand, expression: str, width: int | None = None) -> list[str]:
        value_width = width or operand.width
        if operand.kind == "register":
            if operand.text.lower() in HIGH_BYTE_REGISTERS and value_width == 8:
                register = HIGH_BYTE_REGISTERS[operand.text.lower()]
                return [f"    {register} = ({register} & ~UINT64_C(0xff00)) | (((uint64_t)((uint8_t)({expression}))) << 8);"]
            return self._write_register(operand.register or "", expression, value_width)
        if operand.kind == "vector":
            self.declared_vector.add(operand.register or "xmm0")
            return [f"    {operand.register} = ({expression});"]
        if operand.kind == "memory":
            address = self._address(operand)
            if value_width == 128:
                return [f"    nexus_v_store((void *)(uintptr_t)({address}), {expression});"]
            return [f"    *(({_c_type(value_width)} *)(uintptr_t)({address})) = ({_c_type(value_width)})({expression});"]
        raise EmitError(f"unsupported destination {operand.text}", category="unsupported_operand")

    def _condition(self, mnemonic: str) -> str:
        code = mnemonic[1:] if len(mnemonic) > 1 and mnemonic[0] == "j" else mnemonic
        if mnemonic.startswith("set"):
            code = mnemonic[3:]
        elif mnemonic.startswith("cmov"):
            code = mnemonic[4:]
        code = CONDITION_ALIASES.get(code)
        if code is None:
            raise EmitError(f"unsupported condition code {mnemonic}", category="unsupported_condition")
        if code in {"p", "np"}:
            raise EmitError("parity flags are not modelled", category="unsupported_condition")
        flags = self.flags
        if flags is None:
            raise EmitError(f"condition {mnemonic} without a local flag definition", category="unresolved_flags")
        if flags.kind in {"bsr", "bsf"}:
            if code == "e":
                return f"((({_c_type(flags.width)})({flags.result})) == 0)"
            if code == "ne":
                return f"((({_c_type(flags.width)})({flags.result})) != 0)"
            raise EmitError(f"condition {mnemonic} after {flags.kind} is not modelled", category="unsupported_condition")
        if flags.kind == "bt":
            if code == "b":
                return f"((({_c_type(flags.width)})({flags.result})) != 0)"
            if code == "ae":
                return f"((({_c_type(flags.width)})({flags.result})) == 0)"
            raise EmitError(f"condition {mnemonic} after bt is not modelled", category="unsupported_condition")
        width = flags.width
        u = _c_type(width)
        s = _signed_type(width)
        result = flags.result
        left = flags.left
        right = flags.right
        if code == "e":
            return f"((({u})({result})) == 0)"
        if code == "ne":
            return f"((({u})({result})) != 0)"
        if code == "s":
            return f"((({s})({result})) < 0)"
        if code == "ns":
            return f"((({s})({result})) >= 0)"
        if flags.kind in {"cmp", "sub"}:
            if code == "l":
                return f"((({s})({left})) < (({s})({right})))"
            if code == "le":
                return f"((({s})({left})) <= (({s})({right})))"
            if code == "g":
                return f"((({s})({left})) > (({s})({right})))"
            if code == "ge":
                return f"((({s})({left})) >= (({s})({right})))"
            if code == "b":
                return f"((({u})({left})) < (({u})({right})))"
            if code == "be":
                return f"((({u})({left})) <= (({u})({right})))"
            if code == "a":
                return f"((({u})({left})) > (({u})({right})))"
            if code == "ae":
                return f"((({u})({left})) >= (({u})({right})))"
        if flags.kind == "test":
            if code in {"l", "le", "g", "ge"}:
                operator = {"l": "<", "le": "<=", "g": ">", "ge": ">="}[code]
                return f"((({s})({result})) {operator} 0)"
            if code == "a":
                return f"((({u})({result})) != 0)"
            if code == "ae":
                return "1"
            if code == "b":
                return "0"
            if code == "be":
                return f"((({u})({result})) == 0)"
        if flags.kind in {"add", "adc", "sbb", "imul", "mul", "inc", "dec", "neg"} and code in {"l", "le", "g", "ge"}:
            # SF != OF is the exact signed comparison condition; use the
            # individual flag expressions rather than a signed result shortcut.
            sf, of = self._sf_of(flags)
            if code == "l":
                return f"(({sf}) != ({of}))"
            if code == "ge":
                return f"(({sf}) == ({of}))"
            zf = f"((({u})({result})) == 0)"
            if code == "le":
                return f"((({zf}) != 0) || (({sf}) != ({of})))"
            return f"((({zf}) == 0) && (({sf}) == ({of})))"
        if flags.kind in {"add", "sub", "adc", "sbb", "cmp", "inc", "dec", "neg", "imul", "mul"} and code in {"a", "ae", "b", "be"}:
            cf, zf = self._cf_zf(flags)
            if code == "b":
                return f"({cf})"
            if code == "ae":
                return f"(!({cf}))"
            if code == "be":
                return f"(({cf}) || ({zf}))"
            return f"((!({cf})) && (!({zf})))"
        if flags.kind in {"and", "or", "xor", "test"}:
            if code in {"b", "be"}:
                return "0"
            if code in {"a", "ae"}:
                return f"((({u})({result})) != 0)" if code == "a" else "1"
        raise EmitError(f"condition {mnemonic} not modelled for {flags.kind}", category="unsupported_condition")

    def _sf_of(self, flags: FlagState) -> tuple[str, str]:
        width = flags.width
        u = _c_type(width)
        s = _signed_type(width)
        result = f"(({u})({flags.result}))"
        left = f"(({u})({flags.left}))"
        right = f"(({u})({flags.right}))"
        sign = _sign_bit(width)
        sf = f"(({s})({result}) < 0)"
        if flags.kind in {"add", "adc"}:
            of = f"((((({left}) ^ ({result})) & (({right}) ^ ({result}))) & UINT64_C(0x{sign:x})) != 0)"
        elif flags.kind in {"sub", "sbb"}:
            of = f"((((({left}) ^ ({right})) & (({left}) ^ ({result}))) & UINT64_C(0x{sign:x})) != 0)"
        elif flags.kind == "neg":
            of = f"((({left}) & UINT64_C(0x{sign:x})) != 0)"
        elif flags.kind == "inc":
            of = f"((({result}) & UINT64_C(0x{sign:x})) != 0)"
        elif flags.kind == "dec":
            of = f"(((({result}) ^ UINT64_C(0x{sign:x})) == 0))"
        elif flags.kind in {"imul", "mul"}:
            high = flags.high or "0"
            of = f"(({high}) != 0)"
        else:
            raise EmitError("signed condition for non-arithmetic flags", category="unsupported_condition")
        return sf, of

    def _cf_zf(self, flags: FlagState) -> tuple[str, str]:
        width = flags.width
        u = _c_type(width)
        left = f"(({u})({flags.left}))"
        right = f"(({u})({flags.right}))"
        result = f"(({u})({flags.result}))"
        zf = f"(({result}) == 0)"
        if flags.kind in {"add", "adc"}:
            cf = f"((({result}) < ({left})) || (({result}) < ({right})))"
        elif flags.kind in {"sub", "cmp"}:
            cf = f"(({left}) < ({right}))"
        elif flags.kind == "sbb":
            cf = f"(({left}) < ({right}))"
        elif flags.kind == "neg":
            cf = f"(({left}) != 0)"
        elif flags.kind in {"and", "or", "xor", "test"}:
            cf = "0"
        elif flags.kind == "inc":
            cf = "0"
        elif flags.kind == "dec":
            cf = "0"
        elif flags.kind in {"imul", "mul"}:
            cf = f"(({flags.high or '0'}) != 0)"
        elif flags.kind == "rol":
            cf = f"(({result}) & 1)"
        elif flags.kind == "ror":
            cf = f"((({result}) >> {width - 1}) & 1)"
        else:
            raise EmitError("carry condition for unsupported flag producer", category="unsupported_condition")
        return cf, zf

    def _flag_pre(self, left: str, right: str) -> list[str]:
        return [
            f"    nexus_flag_left = (uint64_t)({left});",
            f"    nexus_flag_right = (uint64_t)({right});",
        ]

    def _flag_result(self, result: str, high: str | None = None) -> list[str]:
        lines = [f"    nexus_flag_result = (uint64_t)({result});"]
        if high is not None:
            lines.append(f"    nexus_flag_high = (uint64_t)({high});")
        return lines

    def _flag_state(self, width: int, kind: str, signed: bool = False, high: bool = False) -> None:
        self.flags = FlagState(kind, "nexus_flag_left", "nexus_flag_right", "nexus_flag_result", width, signed, "nexus_flag_high" if high else None)

    def _emit_instruction(self, instruction: Instruction) -> list[str]:
        lines = self._emit_instruction_inner(instruction)
        self._update_stack_provenance(instruction)
        return lines

    def _is_stack_expression(self, memory: dict[str, Any]) -> bool:
        base = memory.get("base")
        index = memory.get("index")
        return (
            base == "rsp"
            or (base == "rbp" and self.frame_pointer)
            or (base in self.stack_offsets)
            or (index in self.stack_offsets)
        )

    def _provenance_transfer(self, offsets: set[str], pointers: set[str], instruction: Instruction) -> tuple[set[str], set[str]]:
        o = set(offsets)
        p = set(pointers)
        mnemonic = instruction.mnemonic
        operands = [parse_operand(text) for text in instruction.operands]
        if mnemonic in {"call", "callq"}:
            for register in CALLER_SAVED:
                o.discard(register)
                p.discard(register)
            return o, p
        if mnemonic == "lea" and operands and operands[0].kind == "register" and len(operands) > 1 and operands[1].kind == "memory":
            destination = operands[0].register or ""
            o.discard(destination)
            p.discard(destination)
            memory = operands[1].memory or {}
            if self._is_stack_expression_for(memory, o, p):
                o.add(destination)
            return o, p
        if mnemonic in {"mov", "movabs", "movzx", "movsx", "movsxd", "movbe", "movq", "movd"} and len(operands) == 2 and operands[0].kind == "register":
            destination = operands[0].register or ""
            source = operands[1]
            o.discard(destination)
            p.discard(destination)
            if source.kind == "register":
                source_register = source.register or ""
                if source_register in p:
                    p.add(destination)
                elif source_register == "rsp" or (source_register == "rbp" and self.frame_pointer) or source_register in o:
                    o.add(destination)
            return o, p
        if mnemonic in {"add", "sub"} and operands and operands[0].kind == "register":
            destination = operands[0].register or ""
            source_registers = [operand.register for operand in operands[1:] if operand.kind == "register"]
            if destination in p:
                return o, p
            if destination in o:
                if mnemonic == "add" and any(register in p for register in source_registers):
                    o.discard(destination)
                    p.add(destination)
                return o, p
            if mnemonic == "add" and any(register in p for register in source_registers):
                p.add(destination)
                return o, p
            if mnemonic == "add" and any(register in o or register == "rsp" or (register == "rbp" and self.frame_pointer) for register in source_registers):
                o.add(destination)
            return o, p
        if mnemonic in {"inc", "dec"} and operands and operands[0].kind == "register":
            return o, p
        if mnemonic == "xchg" and len(operands) == 2:
            left = operands[0].register if operands[0].kind == "register" else None
            right = operands[1].register if operands[1].kind == "register" else None
            if left and right:
                left_kind = "pointer" if left in p else ("offset" if left in o else None)
                right_kind = "pointer" if right in p else ("offset" if right in o else None)
                o.discard(left)
                p.discard(left)
                o.discard(right)
                p.discard(right)
                if right_kind == "pointer":
                    p.add(left)
                elif right_kind == "offset":
                    o.add(left)
                if left_kind == "pointer":
                    p.add(right)
                elif left_kind == "offset":
                    o.add(right)
            return o, p
        if mnemonic in {"pop", "leave"}:
            if mnemonic == "leave":
                o.discard("rbp")
                p.discard("rbp")
            if operands and operands[0].kind == "register":
                o.discard(operands[0].register or "")
                p.discard(operands[0].register or "")
            return o, p
        if mnemonic in {"mul", "div", "idiv"}:
            for register in ("rax", "rdx"):
                o.discard(register)
                p.discard(register)
            return o, p
        if operands and operands[0].kind == "register" and operands[0].register in instruction.writes:
            o.discard(operands[0].register or "")
            p.discard(operands[0].register or "")
        return o, p

    def _is_stack_expression_for(self, memory: dict[str, Any], offsets: set[str], pointers: set[str]) -> bool:
        base = memory.get("base")
        index = memory.get("index")
        return (
            base == "rsp"
            or (base == "rbp" and self.frame_pointer)
            or base in offsets
            or index in offsets
        )

    def _detect_jump_table(self, block) -> dict[str, Any] | None:
        instructions = block.instructions
        if not instructions:
            return None
        jmp = instructions[-1]
        if jmp.mnemonic not in {"jmp", "jmpq"} or len(jmp.operands) != 1:
            return None
        jmp_register = _canonical(jmp.operands[0])
        if jmp_register is None or "[" in jmp.operands[0]:
            return None
        add_index = None
        for index in range(len(instructions) - 2, -1, -1):
            candidate = instructions[index]
            operands = [parse_operand(text) for text in candidate.operands]
            if candidate.mnemonic == "add" and len(operands) == 2 and operands[0].kind == "register" and operands[0].register == jmp_register and operands[1].kind == "register":
                add_index = index
                break
        if add_index is None:
            return None
        add_operands = [parse_operand(text) for text in instructions[add_index].operands]
        base_register = add_operands[1].register or ""
        mov_index = None
        for index in range(add_index - 1, -1, -1):
            candidate = instructions[index]
            operands = [parse_operand(text) for text in candidate.operands]
            if candidate.mnemonic in {"movsxd", "mov", "movslq"} and len(operands) == 2 and operands[0].kind == "register" and operands[0].register == jmp_register and operands[1].kind == "memory":
                memory = operands[1].memory or {}
                if memory.get("base") == base_register and memory.get("scale") == 4:
                    mov_index = index
                    break
        if mov_index is None:
            return None
        mov_operands = [parse_operand(text) for text in instructions[mov_index].operands]
        index_register = (mov_operands[1].memory or {}).get("index")
        if not index_register:
            return None
        lea_index = None
        table_address = None
        for index in range(mov_index - 1, -1, -1):
            candidate = instructions[index]
            operands = [parse_operand(text) for text in candidate.operands]
            if candidate.mnemonic == "lea" and len(operands) == 2 and operands[0].kind == "register" and operands[0].register == base_register and operands[1].kind == "memory":
                memory = operands[1].memory or {}
                if memory.get("rip"):
                    table_address = candidate.end + int(memory.get("disp") or 0)
                    lea_index = index
                    break
        if lea_index is None or table_address is None:
            return None
        bound = None
        default_target = None
        all_instructions = self.function.instructions
        lea_position = next((position for position, candidate in enumerate(all_instructions) if candidate.address == instructions[lea_index].address), 0)
        for position in range(lea_position - 1, -1, -1):
            candidate = all_instructions[position]
            operands = [parse_operand(text) for text in candidate.operands]
            if candidate.mnemonic != "cmp" or len(operands) != 2 or operands[0].kind != "register" or operands[0].register != index_register or operands[1].kind != "immediate" or operands[1].immediate is None:
                continue
            bound = int(operands[1].immediate)
            for follower in all_instructions[position + 1 : position + 3]:
                if follower.mnemonic in {"ja", "jae"} and follower.targets:
                    default_target = follower.targets[0].va
                    if follower.mnemonic == "jae":
                        bound -= 1
                    break
            if default_target is not None:
                break
        if bound is None or bound < 0 or bound > 256 or default_target is None or default_target not in self.block_starts:
            return None
        entries = []
        for index in range(bound + 1):
            payload = self.bundle.elf.read_va(table_address + index * 4, 4)
            if payload is None or len(payload) != 4:
                return None
            offset = int.from_bytes(payload, "little", signed=True)
            target = table_address + offset
            if target not in self.instruction_addresses:
                return None
            entries.append(target)
        return {
            "jmp_address": jmp.address,
            "movsxd_address": instructions[mov_index].address,
            "index_register": index_register,
            "table_address": table_address,
            "count": len(entries),
            "targets": entries,
            "default": default_target,
            "base_register": base_register,
            "method": "relative_32_bit_table",
        }

    def _flag_descriptor(self, instruction: Instruction) -> tuple[str, int, bool, bool] | None | str:
        mnemonic = instruction.mnemonic
        operands = [parse_operand(text) for text in instruction.operands]
        width = operands[0].width if operands else 64
        if mnemonic in {"cmp", "test"}:
            return (mnemonic, width, False, True)
        if mnemonic in {"add", "sub", "adc", "sbb", "and", "or", "xor", "shl", "shr", "sar", "inc", "dec", "neg"}:
            return (mnemonic, width, mnemonic in {"adc", "sbb"}, width in {16, 32, 64})
        if mnemonic == "imul":
            return ("imul", width, True, True)
        if mnemonic == "mul":
            return ("mul", width, False, True)
        if mnemonic == "bt":
            return ("bt", width, False, False)
        if mnemonic in {"div", "idiv", "call", "callq"}:
            return None
        if mnemonic in {"mov", "movabs", "movzx", "movsx", "movsxd", "movbe", "movq", "movd", "lea", "push", "pop", "nop", "endbr64", "endbr32", "leave", "cs", "data16", "cmp", "test", "cmov", "set"} or mnemonic.startswith(("cmov", "set", "j")):
            return "preserve"
        if mnemonic in {"ret", "retq"}:
            return None
        return "preserve"

    def _block_flags(self) -> dict[str, tuple[str, int, bool, bool] | None]:
        blocks = {block.block_id: block for block in self.function.blocks}
        entry_id = next((block.block_id for block in self.function.blocks if block.start == self.function.entry), self.function.blocks[0].block_id)
        in_flags: dict[str, tuple[str, int, bool, bool] | None] = {block_id: None for block_id in blocks}
        out_flags: dict[str, tuple[str, int, bool, bool] | None] = {block_id: None for block_id in blocks}
        incoming: dict[str, list[str]] = {block_id: [] for block_id in blocks}
        for block in blocks.values():
            for successor in block.successors:
                if successor in incoming:
                    incoming[successor].append(block.block_id)
        def transfer(start, block):
            state = start
            for instruction in block.instructions:
                descriptor = self._flag_descriptor(instruction)
                if descriptor is None:
                    state = None
                elif descriptor != "preserve":
                    state = descriptor
            return state
        for _ in range(max(1, len(blocks) * 3)):
            changed = False
            for block_id, block in blocks.items():
                if block_id == entry_id:
                    start = None
                else:
                    predecessors = incoming.get(block_id, [])
                    if not predecessors:
                        start = None
                    else:
                        values = [out_flags[pred] for pred in predecessors]
                        if any(value is None for value in values) or len(set(values)) != 1:
                            start = None
                        else:
                            start = values[0]
                if start != in_flags[block_id]:
                    changed = True
                    in_flags[block_id] = start
                outgoing = transfer(start, block)
                if outgoing != out_flags[block_id]:
                    changed = True
                    out_flags[block_id] = outgoing
            if not changed:
                break
        return in_flags

    def _block_provenance(self) -> tuple[dict[str, set[str]], dict[str, set[str]]]:
        blocks = {block.block_id: block for block in self.function.blocks}
        entry_id = next((block.block_id for block in self.function.blocks if block.start == self.function.entry), self.function.blocks[0].block_id)
        universe = set(SCALAR_REGISTERS) | {"rbp"}
        in_offsets: dict[str, set[str]] = {block_id: set(universe) for block_id in blocks}
        in_pointers: dict[str, set[str]] = {block_id: set(universe) for block_id in blocks}
        out_offsets: dict[str, set[str]] = {block_id: set(universe) for block_id in blocks}
        out_pointers: dict[str, set[str]] = {block_id: set(universe) for block_id in blocks}
        incoming: dict[str, list[str]] = {block_id: [] for block_id in blocks}
        for block in blocks.values():
            for successor in block.successors:
                if successor in incoming:
                    incoming[successor].append(block.block_id)
        for _ in range(max(1, len(blocks) * 2)):
            changed = False
            for block_id, block in blocks.items():
                if block_id == entry_id:
                    start_offsets, start_pointers = set(), set()
                else:
                    predecessors = incoming.get(block_id, [])
                    if not predecessors:
                        start_offsets, start_pointers = set(), set()
                    else:
                        start_offsets = set.intersection(*(out_offsets[pred] for pred in predecessors))
                        start_pointers = set.intersection(*(out_pointers[pred] for pred in predecessors))
                if start_offsets != in_offsets[block_id] or start_pointers != in_pointers[block_id]:
                    changed = True
                    in_offsets[block_id] = start_offsets
                    in_pointers[block_id] = start_pointers
                offsets, pointers = set(start_offsets), set(start_pointers)
                for instruction in block.instructions:
                    offsets, pointers = self._provenance_transfer(offsets, pointers, instruction)
                if offsets != out_offsets[block_id] or pointers != out_pointers[block_id]:
                    changed = True
                    out_offsets[block_id] = offsets
                    out_pointers[block_id] = pointers
            if not changed:
                break
        reachable: set[str] = set()
        queue = [entry_id]
        while queue:
            current = queue.pop()
            if current in reachable or current not in blocks:
                continue
            reachable.add(current)
            queue.extend(block_id for block_id in blocks[current].successors if block_id in blocks)
        for block_id in blocks:
            if block_id not in reachable:
                in_offsets[block_id] = set()
                in_pointers[block_id] = set()
                out_offsets[block_id] = set()
                out_pointers[block_id] = set()
        return in_offsets, in_pointers

    def _update_stack_provenance(self, instruction: Instruction) -> None:
        mnemonic = instruction.mnemonic
        operands = [parse_operand(text) for text in instruction.operands]
        if mnemonic in {"call", "callq"}:
            for register in CALLER_SAVED:
                self.stack_offsets.discard(register)
                self.pointer_registers.discard(register)
            return
        if mnemonic == "lea" and operands and operands[0].kind == "register" and len(operands) > 1 and operands[1].kind == "memory":
            destination = operands[0].register or ""
            self.stack_offsets.discard(destination)
            self.pointer_registers.discard(destination)
            if self._is_stack_expression(operands[1].memory or {}):
                self.stack_offsets.add(destination)
            return
        if mnemonic in {"mov", "movabs", "movzx", "movsx", "movsxd", "movbe", "movq", "movd"} and len(operands) == 2 and operands[0].kind == "register":
            destination = operands[0].register or ""
            source = operands[1]
            self.stack_offsets.discard(destination)
            self.pointer_registers.discard(destination)
            if source.kind == "register":
                source_register = source.register or ""
                if source_register in self.pointer_registers:
                    self.pointer_registers.add(destination)
                elif source_register == "rsp" or (source_register == "rbp" and self.frame_pointer) or source_register in self.stack_offsets:
                    self.stack_offsets.add(destination)
            return
        if mnemonic in {"add", "sub"} and operands and operands[0].kind == "register":
            destination = operands[0].register or ""
            source_registers = [operand.register for operand in operands[1:] if operand.kind == "register"]
            if destination in self.pointer_registers:
                return
            if destination in self.stack_offsets:
                if mnemonic == "add" and any(register in self.pointer_registers for register in source_registers):
                    self.stack_offsets.discard(destination)
                    self.pointer_registers.add(destination)
                return
            if mnemonic == "add" and any(register in self.pointer_registers for register in source_registers):
                self.pointer_registers.add(destination)
                return
            if mnemonic == "add" and any(register in self.stack_offsets or register == "rsp" or (register == "rbp" and self.frame_pointer) for register in source_registers):
                self.stack_offsets.add(destination)
            return
        if mnemonic in {"inc", "dec"} and operands and operands[0].kind == "register":
            return
        if mnemonic in {"xchg"} and len(operands) == 2:
            left = operands[0].register if operands[0].kind == "register" else None
            right = operands[1].register if operands[1].kind == "register" else None
            if left and right:
                left_kind = "pointer" if left in self.pointer_registers else ("offset" if left in self.stack_offsets else None)
                right_kind = "pointer" if right in self.pointer_registers else ("offset" if right in self.stack_offsets else None)
                self.stack_offsets.discard(left)
                self.pointer_registers.discard(left)
                self.stack_offsets.discard(right)
                self.pointer_registers.discard(right)
                if right_kind == "pointer":
                    self.pointer_registers.add(left)
                elif right_kind == "offset":
                    self.stack_offsets.add(left)
                if left_kind == "pointer":
                    self.pointer_registers.add(right)
                elif left_kind == "offset":
                    self.stack_offsets.add(right)
            return
        if mnemonic in {"pop"}:
            if operands and operands[0].kind == "register":
                self.stack_offsets.discard(operands[0].register or "")
                self.pointer_registers.discard(operands[0].register or "")
            return
        if operands and operands[0].kind == "register" and operands[0].register in instruction.writes:
            self.stack_offsets.discard(operands[0].register or "")
            self.pointer_registers.discard(operands[0].register or "")

    def _emit_instruction_inner(self, instruction: Instruction) -> list[str]:
        self._instruction_end = instruction.end
        mnemonic = instruction.mnemonic
        operands = [parse_operand(text) for text in instruction.operands]
        address = instruction.address
        if mnemonic in NOP_MNEMONICS:
            if mnemonic in {"cs", "data16"} and not any("nop" in operand.lower() for operand in instruction.operands):
                raise EmitError(f"prefix instruction {mnemonic} at 0x{address:x} is not a multi-byte nop", address, "unsupported_instruction")
            self.stats["noops"] += 1
            return []
        if mnemonic in {"push", "pop"}:
            return self._emit_push_pop(mnemonic, operands, address)
        if mnemonic in {"leave"}:
            lines = []
            if self.frame_pointer:
                lines.append("    rsp = rbp;")
            lines.append("    rbp = *((uint64_t *)(uintptr_t)(nexus_stack + (ptrdiff_t)rsp));")
            lines.append("    rsp += 8;")
            return lines
        if mnemonic in {"sub", "add"} and len(operands) == 2 and operands[0].kind == "register" and operands[0].register == "rsp":
            if operands[1].kind != "immediate":
                raise EmitError(f"non-constant stack adjustment at 0x{address:x}", address, "unsupported_stack")
            amount = int(operands[1].immediate or 0)
            if mnemonic == "sub":
                return [f"    rsp -= UINT64_C({amount});"]
            return [f"    rsp += UINT64_C({amount});"]
        if mnemonic in {"mov", "movabs", "movzx", "movsx", "movsxd", "movbe", "movd", "movq"}:
            return self._emit_move(mnemonic, operands, address)
        if mnemonic == "lea":
            if len(operands) != 2 or operands[0].kind != "register":
                raise EmitError(f"unsupported lea at 0x{address:x}", address)
            if operands[1].kind != "memory":
                raise EmitError(f"lea without memory operand at 0x{address:x}", address)
            destination = operands[0].register or ""
            memory = operands[1].memory or {}
            if destination == "rsp":
                if memory.get("base") == "rsp" and not memory.get("index") and not memory.get("rip"):
                    displacement = int(memory.get("disp") or 0)
                    if displacement >= 0:
                        return [f"    rsp += UINT64_C({displacement});"]
                    return [f"    rsp -= UINT64_C({abs(displacement)});"]
                raise EmitError(f"unsupported stack lea at 0x{address:x}", address, "unsupported_stack")
            expression = self._address_expression(operands[1].memory or {}, as_pointer=False)
            return self._write_register(destination, expression, operands[0].width)
        if mnemonic in {"add", "sub", "adc", "sbb", "imul", "and", "or", "xor", "shl", "shr", "sar", "rol", "ror"}:
            return self._emit_arithmetic(mnemonic, operands, address)
        if mnemonic in {"inc", "dec", "neg", "not"}:
            return self._emit_unary(mnemonic, operands, address)
        if mnemonic in {"cmp", "test"}:
            return self._emit_compare(mnemonic, operands, address)
        if mnemonic in {"bsr", "bsf"}:
            return self._emit_bit_scan(mnemonic, operands, address)
        if mnemonic == "bt":
            return self._emit_bt(operands, address)
        if mnemonic in {"set", "sete", "setne", "setl", "setle", "setg", "setge", "setb", "setbe", "seta", "setae", "sets", "setns", "seto", "setno"} or mnemonic.startswith("set"):
            return self._emit_setcc(mnemonic, operands, address)
        if mnemonic.startswith("cmov"):
            return self._emit_cmov(mnemonic, operands, address)
        if mnemonic in {"xchg"}:
            return self._emit_xchg(operands, address)
        if mnemonic in {"mul", "imul"} and len(operands) == 1:
            return self._emit_mul(mnemonic, operands[0], address)
        if mnemonic in {"div", "idiv"}:
            return self._emit_div(mnemonic, operands, address)
        if mnemonic.startswith(("movdq", "movap", "movup")):
            return self._emit_vector_move(mnemonic, operands, address)
        if mnemonic in {"movmskps", "movmskpd", "pmovmskb"}:
            return self._emit_vector_mask(mnemonic, operands, address)
        if mnemonic.startswith("v") and mnemonic not in {"verr", "verw"}:
            return self._emit_vex(mnemonic, operands, address)
        if mnemonic in {"pxor", "xorps", "xorpd", "pand", "por", "pandn", "andps", "andpd", "andnps", "andnpd", "orps", "orpd", "paddb", "paddw", "paddd", "paddq", "psubb", "psubw", "psubd", "psubq", "pminub", "pmaxub", "pminud", "pmaxud", "pcmpeqd", "pcmpeqq", "pcmpeqb", "pcmpeqw", "pcmpgtb", "pcmpgtw", "pcmpgtd", "pcmpgtq", "pshufd", "psrldq", "psrlq", "psllq", "pmuludq", "punpcklqdq", "punpckhqdq", "shufps", "shufpd", "pextrw", "pextrd", "pextrq", "pinsrw", "pinsrd", "pinsrq", "pshufb", "unpcklps", "unpckhps", "unpcklpd", "unpckhpd"}:
            return self._emit_vector_op(mnemonic, operands, address)
        if mnemonic in {"cdqe", "cltq", "cwde", "cwtl", "cbw"}:
            if mnemonic in {"cdqe", "cltq"}:
                return self._write_register("rax", "((int64_t)(int32_t)rax)", 64)
            if mnemonic in {"cwde", "cwtl"}:
                return self._write_register("rax", "((int64_t)(int16_t)ax)", 64)
            return self._write_register("rax", "((int64_t)(int8_t)al)", 64)
        if mnemonic in {"bswap"}:
            if len(operands) != 1 or operands[0].kind != "register":
                raise EmitError(f"unsupported bswap at 0x{address:x}", address)
            width = operands[0].width
            if width == 64:
                expression = "__builtin_bswap64(" + self._read_register(operands[0].register or "") + ")"
            elif width == 32:
                expression = "__builtin_bswap32((uint32_t)" + self._read_register(operands[0].register or "") + ")"
            else:
                raise EmitError(f"unsupported bswap width {width} at 0x{address:x}", address)
            return self._write_register(operands[0].register or "", expression, width)
        if mnemonic in {"call", "callq"}:
            return self._emit_call(instruction, address)
        if mnemonic in {"jmp", "jmpq", "ret", "retq"}:
            return []
        if mnemonic in {"ud2", "hlt", "int3", "syscall", "int"}:
            return [f"    /* trap {mnemonic} at 0x{address:x} */", "    return 0;"]
        if mnemonic in {"jmp", "jmpq", "ret", "retq"} or instruction.control_flow in {"conditional_branch", "unconditional_branch", "return", "indirect_branch", "indirect_call"}:
            return []
        raise EmitError(f"instruction {mnemonic} at 0x{address:x} is not modelled", address, "unsupported_instruction")

    def _emit_push_pop(self, mnemonic: str, operands: list[Operand], address: int | None = None) -> list[str]:
        if not self.frame_pointer and mnemonic == "push" and operands and operands[0].kind == "register" and operands[0].register == "rbp":
            # A callee-saved rbp in a frameless function is a normal register.
            pass
        if mnemonic == "push":
            if not operands:
                raise EmitError("push without operand")
            lines = ["    rsp -= 8;"]
            lines.append(f"    *((uint64_t *)(uintptr_t)(rsp)) = {self._load(operands[0], 64)};")
            return lines
        lines = []
        if operands:
            if operands[0].kind == "register":
                target = operands[0].register or ""
                unique = f"__pop_{target}_{address or 0:x}"
                lines.append(
                    f"    {_c_type(operands[0].width)} {unique} = *((uint64_t *)(uintptr_t)(rsp));"
                )
                lines.extend(self._write_register(target, unique, operands[0].width))
            elif operands[0].kind == "memory":
                value = "(*((uint64_t *)(uintptr_t)(rsp)))"
                lines.extend(self._store(operands[0], value, operands[0].width))
            else:
                raise EmitError("unsupported pop destination")
        else:
            raise EmitError("pop without operand")
        lines.append("    rsp += 8;")
        return lines

    def _emit_move(self, mnemonic: str, operands: list[Operand], address: int) -> list[str]:
        if len(operands) != 2:
            raise EmitError(f"unsupported {mnemonic} form at 0x{address:x}", address)
        destination, source = operands
        if mnemonic == "movq":
            if destination.kind == "vector" and source.kind != "vector":
                value = self._load(source, 64)
                return [
                    f"    {destination.register} = nexus_v_zero();",
                    f"    {destination.register}.q[0] = (uint64_t)({value});",
                ]
            if destination.kind != "vector" and source.kind == "vector":
                expression = f"({source.register}.q[0])"
                return self._store(destination, expression, min(destination.width, 64))
        if mnemonic == "movd":
            if destination.kind == "vector":
                value = self._load(source, 32)
                return [
                    f"    {destination.register} = nexus_v_zero();",
                    f"    {destination.register}.d[0] = (uint32_t)({value});",
                ]
            if source.kind == "vector":
                expression = f"({source.register}.d[0])"
                return self._store(destination, expression, 32)
        if destination.kind == "vector" or source.kind == "vector":
            return self._emit_vector_move(mnemonic, operands, address)
        if mnemonic == "movbe":
            value = self._load(source, destination.width)
            if destination.width == 64:
                expression = f"__builtin_bswap64((uint64_t)({value}))"
            elif destination.width == 32:
                expression = f"__builtin_bswap32((uint32_t)({value}))"
            else:
                expression = f"__builtin_bswap16((uint16_t)({value}))"
            return self._store(destination, expression, destination.width)
        if mnemonic in {"movzx", "movsx", "movsxd"}:
            source_width = source.width if source.kind != "register" else _register_width(source.text)
            if source_width >= destination.width:
                source_width = min(source_width, destination.width)
            value = self._load(source, source_width)
            if mnemonic == "movsx":
                expression = f"((int64_t)({_signed_type(source_width)})({value}))"
            elif mnemonic == "movsxd":
                expression = f"((int64_t)(int32_t)({value}))"
            else:
                expression = f"((uint64_t)({_c_type(source_width)})({value}))"
            return self._store(destination, expression, destination.width)
        value = self._load(source, destination.width if destination.width != 128 else 128)
        return self._store(destination, value, destination.width)

    def _emit_arithmetic(self, mnemonic: str, operands: list[Operand], address: int) -> list[str]:
        if len(operands) < 2:
            raise EmitError(f"unsupported arithmetic form at 0x{address:x}", address)
        destination = operands[0]
        if mnemonic == "imul" and len(operands) >= 3:
            left = self._load(operands[1], destination.width)
            right = self._load(operands[2], destination.width)
            expression = f"((uint64_t)({left}) * (uint64_t)({right}))"
            lines = self._flag_pre(left, right)
            lines.extend(self._flag_result(expression, "0"))
            lines.extend(self._store(destination, expression, destination.width))
            self._flag_state(destination.width, "imul", True, True)
            return lines
        width = destination.width
        left = self._load(destination, width)
        right = self._load(operands[1], width)
        if mnemonic in {"rol", "ror"}:
            count_value = self._load(operands[1], 8)
            masked_count = f"(({count_value}) & 0x{width - 1:x})"
            if mnemonic == "rol":
                rotated = f"((({_c_type(width)})({left}) << {masked_count}) | (({_c_type(width)})({left}) >> (({width} - {masked_count}) & 0x{width - 1:x})))"
            else:
                rotated = f"((({_c_type(width)})({left}) >> {masked_count}) | (({_c_type(width)})({left}) << (({width} - {masked_count}) & 0x{width - 1:x})))"
            expression = f"(({masked_count}) == 0 ? (uint64_t)({left}) : ((uint64_t)({rotated})))"
            lines = self._flag_pre(left, count_value)
            lines.extend(self._flag_result(expression))
            lines.extend(self._store(destination, expression, width))
            immediate_zero = operands[1].kind == "immediate" and int(operands[1].immediate or 0) % width == 0
            if not immediate_zero:
                self._flag_state(width, mnemonic, False, True)
            # Zero-count rotates leave all flags unchanged; self.flags already
            # holds the previous state and must not be cleared.
            return lines
        carry = None
        carry_prefix: list[str] = []
        if mnemonic in {"adc", "sbb"}:
            carry = self._carry_expression()
            carry_prefix = [f"    nexus_carry_value = (uint64_t)({carry});"]
            carry = "nexus_carry_value"
        if mnemonic == "xor" and destination.kind == "register" and operands[1].kind == "register" and destination.register == operands[1].register:
            lines = self._flag_pre("0", "0")
            lines.extend(self._flag_result("0", "0"))
            lines.extend(self._store(destination, "0", width))
            self._flag_state(width, "xor", False, True)
            return lines
        if mnemonic in {"shl", "shr", "sar"}:
            right = self._load(operands[1], 8)
            operator = {"shl": "<<", "shr": ">>", "sar": ">>"}[mnemonic]
            if mnemonic == "sar":
                expression = f"((uint64_t)(({_signed_type(width)})({left}) {operator} (({right}) & 0x{width - 1:x})))"
            else:
                expression = f"((uint64_t)(({_c_type(width)})({left}) {operator} (({right}) & 0x{width - 1:x})))"
        elif mnemonic in {"adc", "sbb"}:
            operator = "+" if mnemonic == "adc" else "-"
            expression = f"((uint64_t)(((__uint128_t)({_c_type(width)})({left}) {operator} (__uint128_t)({_c_type(width)})({right}) {operator} (__uint128_t)({carry}))))"
        else:
            operator = {"add": "+", "sub": "-", "imul": "*", "and": "&", "or": "|", "xor": "^"}[mnemonic]
            expression = f"((uint64_t)(({_c_type(width)})({left}) {operator} ({_c_type(width)})({right})))"
        lines = carry_prefix + self._flag_pre(left, right)
        lines.extend(self._flag_result(expression, "0"))
        lines.extend(self._store(destination, expression, width))
        if mnemonic in {"shl", "shr", "sar"}:
            self._flag_state(width, mnemonic, False, True)
        else:
            self._flag_state(width, mnemonic, mnemonic == "imul", True)
        return lines

    def _emit_unary(self, mnemonic: str, operands: list[Operand], address: int) -> list[str]:
        if len(operands) != 1:
            raise EmitError(f"unsupported unary at 0x{address:x}", address)
        destination = operands[0]
        width = destination.width
        left = self._load(destination, width)
        if mnemonic == "inc":
            expression = f"((uint64_t)(({_c_type(width)})({left}) + 1))"
        elif mnemonic == "dec":
            expression = f"((uint64_t)(({_c_type(width)})({left}) - 1))"
        elif mnemonic == "neg":
            expression = f"((uint64_t)(-({_signed_type(width)})({left})))"
        else:
            expression = f"((uint64_t)(~({_c_type(width)})({left})))"
        lines = self._flag_pre(left, "1" if mnemonic in {"inc", "dec"} else left) if mnemonic != "not" else []
        if mnemonic != "not":
            lines.extend(self._flag_result(expression, "0"))
        lines.extend(self._store(destination, expression, width))
        if mnemonic != "not":
            self._flag_state(width, mnemonic, mnemonic == "neg", True)
        return lines

    def _emit_bit_scan(self, mnemonic: str, operands: list[Operand], address: int) -> list[str]:
        if len(operands) != 2 or operands[0].kind != "register":
            raise EmitError(f"unsupported {mnemonic} at 0x{address:x}", address)
        width = operands[0].width
        if width not in {16, 32, 64}:
            raise EmitError(f"unsupported {mnemonic} width {width} at 0x{address:x}", address)
        source = self._load(operands[1], width)
        unique = f"__scan_{mnemonic}_{address:x}"
        if mnemonic == "bsr":
            compute = f"({width - 1} - __builtin_clzll((uint64_t)({source})))" if width == 64 else f"({width - 1} - __builtin_clz((uint32_t)({source})))"
        else:
            compute = f"(__builtin_ctzll((uint64_t)({source})))" if width == 64 else f"(__builtin_ctz((uint32_t)({source})))"
        lines = [
            f"    uint64_t {unique}_src = (uint64_t)({source});",
            f"    uint64_t {unique}_value = {unique}_src ? (uint64_t)({compute}) : 0;",
        ]
        lines.extend(self._write_register(operands[0].register or "", f"{unique}_value", width))
        self.flags = FlagState(mnemonic, f"{unique}_src", "0", f"{unique}_src", width, False, None)
        return lines

    def _emit_bt(self, operands: list[Operand], address: int) -> list[str]:
        if len(operands) != 2:
            raise EmitError(f"unsupported bt at 0x{address:x}", address)
        width = operands[0].width
        if width not in {16, 32, 64}:
            raise EmitError(f"unsupported bt width {width} at 0x{address:x}", address)
        value = self._load(operands[0], width)
        index = self._load(operands[1], 64)
        bit = f"(((({_c_type(width)})({value})) >> (({index}) & 0x{width - 1:x})) & 1)"
        self.flags = FlagState("bt", value, index, bit, width, False, None)
        return []

    def _carry_expression(self) -> str:
        flags = self.flags
        if flags is None:
            raise EmitError("adc/sbb without a modelled carry flag", category="unresolved_flags")
        width = flags.width
        u = _c_type(width)
        left = f"(({u})({flags.left}))"
        right = f"(({u})({flags.right}))"
        result = f"(({u})({flags.result}))"
        if flags.kind in {"cmp", "sub"}:
            return f"(({left}) < ({right}))"
        if flags.kind == "add":
            return f"(({result}) < ({left}))"
        if flags.kind == "adc":
            return f"((({result}) < ({left})) || (({result}) < ({right})))"
        if flags.kind == "sbb":
            return f"(({left}) < ({right}))"
        if flags.kind in {"and", "or", "xor", "test"}:
            return "0"
        if flags.kind == "neg":
            return f"(({left}) != 0)"
        if flags.kind == "bt":
            return f"(({flags.result}) != 0)"
        if flags.kind in {"shl", "shr", "sar"}:
            raise EmitError("carry after a variable shift is not modelled", category="unsupported_condition")
        if flags.kind == "imul" or flags.kind == "mul":
            return f"(({flags.high or '0'}) != 0)"
        raise EmitError(f"carry after {flags.kind} is not modelled", category="unsupported_condition")

    def _emit_compare(self, mnemonic: str, operands: list[Operand], address: int) -> list[str]:
        if len(operands) != 2:
            raise EmitError(f"unsupported {mnemonic} form at 0x{address:x}", address)
        width = operands[0].width if operands[0].kind != "immediate" else operands[1].width
        if width == 128:
            raise EmitError(f"vector compare {mnemonic} at 0x{address:x} is not scalar-modelled", address, "unsupported_instruction")
        left = self._load(operands[0], width)
        right = self._load(operands[1], width)
        if mnemonic == "test":
            result = f"((uint64_t)(({_c_type(width)})({left}) & ({_c_type(width)})({right})))"
            lines = self._flag_pre(left, right) + self._flag_result(result, "0")
            self._flag_state(width, "test", False, True)
            return lines
        result = f"((uint64_t)(({_c_type(width)})({left}) - ({_c_type(width)})({right})))"
        lines = self._flag_pre(left, right) + self._flag_result(result, "0")
        self._flag_state(width, "cmp", False, True)
        return lines

    def _emit_setcc(self, mnemonic: str, operands: list[Operand], address: int) -> list[str]:
        if len(operands) != 1:
            raise EmitError(f"unsupported setcc at 0x{address:x}", address)
        condition = self._condition(mnemonic)
        destination = operands[0]
        return self._store(destination, f"(({condition}) ? 1 : 0)", 8)

    def _emit_cmov(self, mnemonic: str, operands: list[Operand], address: int) -> list[str]:
        if len(operands) != 2:
            raise EmitError(f"unsupported cmov at 0x{address:x}", address)
        condition = self._condition(mnemonic)
        destination = operands[0]
        width = destination.width
        current = self._load(destination, width)
        source = self._load(operands[1], width)
        return self._store(destination, f"(({condition}) ? ({source}) : ({current}))", width)

    def _emit_xchg(self, operands: list[Operand], address: int) -> list[str]:
        if len(operands) != 2:
            raise EmitError(f"unsupported xchg at 0x{address:x}", address)
        left, right = operands
        width = max(left.width, right.width)
        if width not in {8, 16, 32, 64}:
            raise EmitError(f"unsupported xchg width {width} at 0x{address:x}", address)
        temp = f"__xchg_{address:x}"
        lines = [f"    {_c_type(width)} {temp} = {self._load(left, width)};"]
        lines.extend(self._store(left, self._load(right, width), width))
        lines.extend(self._store(right, temp, width))
        return lines

    def _emit_mul(self, mnemonic: str, operand: Operand, address: int) -> list[str]:
        width = operand.width
        if width not in {32, 64}:
            raise EmitError(f"unsupported {mnemonic} width {width} at 0x{address:x}", address, "unsupported_instruction")
        source = self._load(operand, width)
        pre = self._flag_pre("((uint64_t)rax)", source)
        if mnemonic == "imul":
            lines = [
                f"    {{ __int128 __p = (__int128)(({_signed_type(width)})(rax)) * (__int128)(({_signed_type(width)})({source}));",
                f"      rax = (uint64_t)({_c_type(width)})(__p);",
                f"      rdx = (uint64_t)({_c_type(width)})(__p >> {width}); }}",
            ]
        else:
            lines = [
                f"    {{ unsigned __int128 __p = (unsigned __int128)(({_c_type(width)})(rax)) * (unsigned __int128)(({_c_type(width)})({source}));",
                f"      rax = (uint64_t)({_c_type(width)})(__p);",
                f"      rdx = (uint64_t)({_c_type(width)})(__p >> {width}); }}",
            ]
        high = "((uint64_t)rdx)"
        lines = pre + lines
        lines.extend(self._flag_result("((uint64_t)rax)", high))
        self._flag_state(width, "imul" if mnemonic == "imul" else "mul", mnemonic == "imul", True)
        return lines

    def _emit_div(self, mnemonic: str, operands: list[Operand], address: int) -> list[str]:
        if len(operands) != 1:
            raise EmitError(f"unsupported {mnemonic} at 0x{address:x}", address)
        width = operands[0].width
        if width not in {8, 16, 32, 64}:
            raise EmitError(f"unsupported division width {width} at 0x{address:x}", address)
        divisor = self._load(operands[0], width)
        if width == 64:
            dividend = "((unsigned __int128)(((unsigned __int128)(uint64_t)rdx << 64) | (unsigned __int128)(uint64_t)rax))"
            signed_dividend = "((__int128)(((unsigned __int128)(uint64_t)rdx << 64) | (unsigned __int128)(uint64_t)rax))"
        elif width == 32:
            dividend = "((uint64_t)(((uint64_t)(uint32_t)rdx << 32) | (uint32_t)rax))"
            signed_dividend = "((int64_t)(((uint64_t)(uint32_t)rdx << 32) | (uint32_t)rax))"
        else:
            raise EmitError(f"division width {width} is not modelled", address, "unsupported_instruction")
        if mnemonic == "div":
            lines = [
                f"    {{ unsigned __int128 __d = {dividend};",
                f"      rax = (uint64_t)({_c_type(width)})(__d / (unsigned __int128)({_c_type(width)})({divisor}));",
                f"      rdx = (uint64_t)({_c_type(width)})(__d % (unsigned __int128)({_c_type(width)})({divisor})); }}",
            ]
        else:
            lines = [
                f"    {{ __int128 __d = {signed_dividend};",
                f"      rax = (uint64_t)({_c_type(width)})(__d / (__int128)({_signed_type(width)})({divisor}));",
                f"      rdx = (uint64_t)({_c_type(width)})(__d % (__int128)({_signed_type(width)})({divisor})); }}",
            ]
        self.flags = None
        return lines

    def _emit_vector_move(self, mnemonic: str, operands: list[Operand], address: int) -> list[str]:
        if len(operands) != 2:
            raise EmitError(f"unsupported vector move at 0x{address:x}", address)
        destination, source = operands
        if destination.kind == "vector" and source.kind == "memory":
            return [f"    {destination.register} = nexus_v_load((const void *)(uintptr_t)({self._address(source)}));"]
        if destination.kind == "memory" and source.kind == "vector":
            return [f"    nexus_v_store((void *)(uintptr_t)({self._address(destination)}), {source.register});"]
        if destination.kind == "vector" and source.kind == "vector":
            return [f"    {destination.register} = {source.register};"]
        raise EmitError(f"unsupported vector move {mnemonic} at 0x{address:x}", address, "unsupported_instruction")

    def _emit_vector_mask(self, mnemonic: str, operands: list[Operand], address: int) -> list[str]:
        if len(operands) != 2 or operands[0].kind != "register" or operands[1].kind != "vector":
            raise EmitError(f"unsupported {mnemonic} at 0x{address:x}", address)
        helper = {"movmskps": "nexus_v_movmskps", "movmskpd": "nexus_v_movmskpd", "pmovmskb": "nexus_v_pmovmskb"}[mnemonic]
        return self._store(operands[0], f"{helper}({operands[1].register})", 32)

    def _emit_vex(self, mnemonic: str, operands: list[Operand], address: int) -> list[str]:
        base = mnemonic[1:]
        if any(operand.kind == "vector" and operand.width > 128 for operand in operands):
            raise EmitError(f"VEX width >128 is not modelled at 0x{address:x}", address, "unsupported_vector")
        if any("YMMWORD" in operand.text.upper() or "ZMMWORD" in operand.text.upper() for operand in operands):
            raise EmitError(f"VEX memory width >128 is not modelled at 0x{address:x}", address, "unsupported_vector")
        if mnemonic in {"vzeroupper", "vzeroall"}:
            self.stats["vex_upper_zero"] += 1
            return []
        if base in {"movdqa", "movdqu", "movaps", "movups", "movapd", "movupd"}:
            return self._emit_vector_move(base, operands, address)
        if base in {"movd", "movq"}:
            return self._emit_move(base, operands, address)
        if base in {"movmskps", "movmskpd", "pmovmskb"}:
            return self._emit_vector_mask(base, operands, address)
        if base == "pshufb" and len(operands) == 3:
            if operands[0].kind != "vector" or operands[1].kind != "vector":
                raise EmitError(f"unsupported VEX pshufb at 0x{address:x}", address)
            right = operands[2].register if operands[2].kind == "vector" else f"nexus_v_load((const void *)(uintptr_t)({self._address(operands[2])}))"
            return [f"    {operands[0].register} = nexus_v_pshufb({operands[1].register}, {right});"]
        if base in {"psrldq", "pslldq"} and len(operands) == 3 and operands[0].kind == "vector" and operands[1].kind == "vector" and operands[2].immediate is not None:
            helper = "nexus_v_psrldq" if base == "psrldq" else "nexus_v_pslldq"
            return [f"    {operands[0].register} = {helper}({operands[1].register}, {int(operands[2].immediate) & 0xff});"]
        if base in {"psrlq", "psllq", "psrld", "pslld"} and len(operands) == 3 and operands[0].kind == "vector" and operands[1].kind == "vector" and operands[2].immediate is not None:
            helper = {"psrlq": "nexus_v_psrlq", "psllq": "nexus_v_psllq", "psrld": "nexus_v_psrld", "pslld": "nexus_v_pslld"}[base]
            return [f"    {operands[0].register} = {helper}({operands[1].register}, {int(operands[2].immediate) & 0xff});"]
        if base == "pshufd" and len(operands) == 3 and operands[0].kind == "vector" and operands[1].kind == "vector" and operands[2].immediate is not None:
            return [f"    {operands[0].register} = nexus_v_pshufd({operands[1].register}, 0x{int(operands[2].immediate) & 0xff:x});"]
        if base in {"shufps", "shufpd"} and len(operands) == 4 and operands[0].kind == "vector" and operands[3].immediate is not None:
            source = operands[1].register if operands[1].kind == "vector" else f"nexus_v_load((const void *)(uintptr_t)({self._address(operands[1])}))"
            right = operands[2].register if operands[2].kind == "vector" else f"nexus_v_load((const void *)(uintptr_t)({self._address(operands[2])}))"
            helper = "nexus_v_shufps" if base == "shufps" else "nexus_v_shufpd"
            return [f"    {operands[0].register} = {helper}({source}, {right}, 0x{int(operands[3].immediate) & 0xff:x});"]
        vector_helpers = {"pand", "por", "pxor", "pandn", "paddb", "paddw", "paddd", "paddq", "psubb", "psubw", "psubd", "psubq", "pminub", "pmaxub", "pminud", "pmaxud", "pcmpeqb", "pcmpeqw", "pcmpeqd", "pcmpeqq", "pcmpgtb", "pcmpgtw", "pcmpgtd", "pcmpgtq", "andps", "andpd", "orps", "orpd", "xorps", "xorpd", "andnps", "andnpd"}
        if base in vector_helpers and len(operands) == 3:
            if operands[0].kind != "vector" or operands[1].kind != "vector":
                raise EmitError(f"unsupported VEX {mnemonic} at 0x{address:x}", address)
            right = operands[2].register if operands[2].kind == "vector" else f"nexus_v_load((const void *)(uintptr_t)({self._address(operands[2])}))"
            helper = _vector_helper_name(base)
            return [f"    {operands[0].register} = {helper}({operands[1].register}, {right});"]
        if base in vector_helpers and len(operands) == 2:
            return self._emit_vector_op(base, operands, address)
        raise EmitError(f"VEX operation {mnemonic} at 0x{address:x} is not modelled", address, "unsupported_vector")

    def _vector_binary(self, mnemonic: str, destination: Operand, source: Operand, address: int) -> list[str]:
        if destination.kind != "vector" or source.kind not in {"vector", "memory"}:
            raise EmitError(f"unsupported vector operand shape at 0x{address:x}", address, "unsupported_instruction")
        left = destination.register or "xmm0"
        right = source.register if source.kind == "vector" else f"nexus_v_load((const void *)(uintptr_t)({self._address(source)}))"
        helper = _vector_helper_name(mnemonic)
        if helper:
            self.declared_vector.add(left)
            return [f"    {left} = {helper}({left}, {right});"]
        raise EmitError(f"vector operation {mnemonic} at 0x{address:x} is not modelled", address, "unsupported_vector")

    def _emit_vector_op(self, mnemonic: str, operands: list[Operand], address: int) -> list[str]:
        if mnemonic in {"pxor", "xorps", "xorpd", "pand", "por", "pandn", "andps", "andpd", "andnps", "andnpd", "orps", "orpd", "paddb", "paddw", "paddd", "paddq", "psubb", "psubw", "psubd", "psubq", "pminub", "pmaxub", "pminud", "pmaxud", "pcmpeqd", "pcmpeqq", "pcmpeqb", "pcmpeqw", "pcmpgtb", "pcmpgtw", "pcmpgtd", "pcmpgtq", "pmuludq", "punpcklqdq", "punpckhqdq", "unpcklps", "unpckhps", "unpcklpd", "unpckhpd"}:
            if len(operands) != 2:
                raise EmitError(f"unsupported vector op at 0x{address:x}", address)
            if mnemonic in {"pxor", "xorps", "xorpd"} and operands[0].kind == "vector" and operands[1].kind == "vector" and operands[0].register == operands[1].register:
                return [f"    {operands[0].register} = nexus_v_zero();"]
            return self._vector_binary(mnemonic, operands[0], operands[1], address)
        if mnemonic == "pshufd":
            if len(operands) != 3 or operands[0].kind != "vector" or operands[1].kind != "vector":
                raise EmitError(f"unsupported pshufd at 0x{address:x}", address)
            immediate = operands[2].immediate
            if immediate is None:
                raise EmitError(f"pshufd without immediate at 0x{address:x}", address)
            return [f"    {operands[0].register} = nexus_v_pshufd({operands[1].register}, 0x{int(immediate) & 0xff:x});"]
        if mnemonic in {"psrldq"}:
            if len(operands) != 2 or operands[1].immediate is None:
                raise EmitError(f"unsupported psrldq at 0x{address:x}", address)
            return [f"    {operands[0].register} = nexus_v_psrldq({operands[0].register}, {int(operands[1].immediate) & 0xff});"]
        if mnemonic in {"psrlq", "psllq"}:
            if len(operands) != 2 or operands[1].immediate is None:
                raise EmitError(f"unsupported {mnemonic} at 0x{address:x}", address)
            helper = "nexus_v_psrlq" if mnemonic == "psrlq" else "nexus_v_psllq"
            return [f"    {operands[0].register} = {helper}({operands[0].register}, {int(operands[1].immediate) & 0xff});"]
        if mnemonic in {"shufps", "shufpd"}:
            if len(operands) != 3 or operands[2].immediate is None:
                raise EmitError(f"unsupported {mnemonic} at 0x{address:x}", address)
            helper = "nexus_v_shufps" if mnemonic == "shufps" else "nexus_v_shufpd"
            source = operands[1]
            right = source.register if source.kind == "vector" else f"nexus_v_load((const void *)(uintptr_t)({self._address(source)}))"
            return [f"    {operands[0].register} = {helper}({operands[0].register}, {right}, 0x{int(operands[2].immediate) & 0xff:x});"]
        if mnemonic == "pextrw":
            if len(operands) != 3 or operands[1].kind != "vector" or operands[2].immediate is None:
                raise EmitError(f"unsupported pextrw at 0x{address:x}", address)
            return self._store(operands[0], f"((uint64_t){operands[1].register}.w[{int(operands[2].immediate) & 7}])", operands[0].width)
        if mnemonic in {"pextrd", "pextrq"}:
            if len(operands) != 3 or operands[1].kind != "vector" or operands[2].immediate is None:
                raise EmitError(f"unsupported {mnemonic} at 0x{address:x}", address)
            lane = int(operands[2].immediate)
            if mnemonic == "pextrd":
                expression = f"((uint64_t){operands[1].register}.d[{lane & 3}])"
            else:
                expression = f"((uint64_t){operands[1].register}.q[{lane & 1}])"
            return self._store(operands[0], expression, operands[0].width)
        if mnemonic in {"pinsrw", "pinsrd", "pinsrq"}:
            if len(operands) != 3 or operands[0].kind != "vector" or operands[2].immediate is None:
                raise EmitError(f"unsupported {mnemonic} at 0x{address:x}", address)
            lane = int(operands[2].immediate)
            value = self._load(operands[1], 32 if mnemonic == "pinsrd" else (64 if mnemonic == "pinsrq" else 16))
            if mnemonic == "pinsrw":
                return [f"    {operands[0].register}.w[{lane & 7}] = (uint16_t)({value});"]
            if mnemonic == "pinsrd":
                return [f"    {operands[0].register}.d[{lane & 3}] = (uint32_t)({value});"]
            return [f"    {operands[0].register}.q[{lane & 1}] = (uint64_t)({value});"]
        if mnemonic == "pshufb":
            if len(operands) != 2 or operands[0].kind != "vector" or operands[1].kind not in {"vector", "memory"}:
                raise EmitError(f"unsupported pshufb at 0x{address:x}", address)
            right = operands[1].register if operands[1].kind == "vector" else f"nexus_v_load((const void *)(uintptr_t)({self._address(operands[1])}))"
            return [f"    {operands[0].register} = nexus_v_pshufb({operands[0].register}, {right});"]
        raise EmitError(f"vector operation {mnemonic} at 0x{address:x} is not modelled", address, "unsupported_vector")

    def _call_target(self, instruction: Instruction) -> tuple[int | None, str | None, bool]:
        target = instruction.targets[0].va if instruction.targets else None
        if target is None:
            return None, None, False
        callee = self.functions_by_entry.get(target)
        if callee is None:
            return target, None, False
        name = callee.name or ""
        lowered = name.lower()
        clean_name = re.sub(r"(@plt|\.plt|_plt)$", "", name)
        internal = next((item for item in self.bundle.functions if item.name == clean_name and item.blocks), None)
        if internal is not None:
            return target, internal.name, False
        stub = resolve_plt_stub(self.bundle, callee)
        if stub.get("resolved"):
            return target, str(stub["symbol"]), True
        external = lowered.endswith(("@plt", ".plt", "_plt")) or not callee.blocks
        return target, name, external

    def _emit_call(self, instruction: Instruction, address: int) -> list[str]:
        target, callee_name, external = self._call_target(instruction)
        if callee_name is None:
            raise EmitError(f"unresolved call target at 0x{address:x}", address, "unresolved_call")
        clean = re.sub(r"(@plt|\.plt|_plt)$", "", callee_name)
        if clean.startswith("__stack_chk_fail"):
            self.stats["noreturn_calls"] += 1
            self.noreturn_call = True
            self.flags = None
            return ["    __stack_chk_fail();", "    return 0;"]
        if external or clean in KNOWN_LIBC:
            signature = KNOWN_LIBC.get(clean)
            if signature is None:
                self.stats["guessed_prototypes"] += 1
                symbol = f"nexus_extern_{_sanitize(clean)}"
                self.externs[symbol] = f"extern uint64_t {symbol}(uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t);"
                arguments = [self._arg_expression(register, 64) for register in ARGUMENT_REGISTERS]
                lines = [f"    rax = {symbol}({', '.join(arguments)});"]
                self.flags = None
                self.stats["calls"] += 1
                return lines
            return_type, kinds = signature
            arguments = []
            for index, kind in enumerate(kinds):
                register = ARGUMENT_REGISTERS[index]
                value = self._arg_expression(register, 64)
                if kind == "ptr":
                    arguments.append(f"(void *)(uintptr_t)({value})")
                elif kind == "int":
                    arguments.append(f"(int)({value})")
                else:
                    arguments.append(f"(size_t)({value})")
            call = f"{clean}({', '.join(arguments)})"
            self.stats["calls"] += 1
            self.flags = None
            if return_type == "void":
                return [f"    {call};"]
            if return_type == "ptr":
                return [f"    rax = (uint64_t)(uintptr_t){call};"]
            if return_type == "int":
                return [f"    rax = (uint64_t)(int64_t){call};"]
            return [f"    rax = (uint64_t){call};"]
        callee = self.functions_by_entry.get(target or 0)
        if callee is None or not callee.blocks or (callee.name or "").lower().endswith(("@plt", ".plt", "_plt")):
            internal = next((item for item in self.bundle.functions if item.name == clean and item.blocks), None)
            if internal is not None:
                callee = internal
        if callee is None:
            raise EmitError(f"unknown call target {clean} at 0x{address:x}", address, "unresolved_call")
        proto = self.prototypes.get(callee.entry)
        if proto is None:
            proto = self._proto_for(callee)
            self.prototypes[callee.entry] = proto
        if proto.parameter_count > 6:
            raise EmitError(f"call with {proto.parameter_count} arguments at 0x{address:x}", address, "unsupported_call")
        generated = self.name_map.get(callee.function_id, _sanitize(callee.name))
        arguments = [self._arg_expression(ARGUMENT_REGISTERS[index], 64) for index in range(proto.parameter_count)]
        self.stats["calls"] += 1
        self.flags = None
        if proto.return_width == 0:
            return [f"    {generated}({', '.join(arguments)});"]
        return [f"    rax = (uint64_t){generated}({', '.join(arguments)});"]

    def _proto_for(self, function: FunctionCandidate) -> Prototype:
        return_name, params, return_width = _abi_prototype(function, self.bundle)
        names = [part.strip() for part in params.split(",") if part.strip()] if params != "void" else []
        widths = []
        for name in names:
            match = re.search(r"uint(\d+)_t", name)
            widths.append(int(match.group(1)) if match else 64)
        return Prototype(return_name, params, return_width, len(names), widths)

    def _fallthrough(self, end: int) -> FunctionCandidate | None:
        candidates = [block for block in self.function.blocks if end <= block.start <= end + 16]
        return min(candidates, key=lambda block: block.start) if candidates else None

    def _tail_call(self, target: int, address: int) -> list[str]:
        callee = self.functions_by_entry.get(target)
        if callee is None:
            raise EmitError(f"tail call target at 0x{address:x} is unknown", address, "unresolved_call")
        clean = re.sub(r"(@plt|\.plt|_plt)$", "", callee.name or "")
        if clean in KNOWN_LIBC or (callee.name or "").lower().endswith(("@plt", ".plt", "_plt")):
            signature = KNOWN_LIBC.get(clean)
            if signature is None:
                symbol = f"nexus_extern_{_sanitize(clean)}"
                self.externs[symbol] = f"extern uint64_t {symbol}(uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t);"
                arguments = [self._arg_expression(register, 64) for register in ARGUMENT_REGISTERS]
                return [f"    return {symbol}({', '.join(arguments)});"]
            _, kinds = signature
            arguments = []
            for index, kind in enumerate(kinds):
                value = self._arg_expression(ARGUMENT_REGISTERS[index], 64)
                arguments.append(f"(void *)(uintptr_t)({value})" if kind == "ptr" else (f"(int)({value})" if kind == "int" else f"(size_t)({value})"))
            return [f"    return (uint64_t){clean}({', '.join(arguments)});"]
        proto = self.prototypes.get(callee.entry) or self._proto_for(callee)
        generated = self.name_map.get(callee.function_id, _sanitize(callee.name))
        arguments = [self._arg_expression(ARGUMENT_REGISTERS[index], 64) for index in range(proto.parameter_count)]
        return [f"    return (uint64_t){generated}({', '.join(arguments)});"]


def _is_frame_pointer_function(function: FunctionCandidate) -> bool:
    for instruction in function.instructions[:12]:
        if instruction.mnemonic == "push" and instruction.operands[:1] == ["rbp"]:
            return True
        if instruction.mnemonic == "mov" and len(instruction.operands) == 2 and instruction.operands[0] == "rbp" and instruction.operands[1] == "rsp":
            return True
    return False


# Vector prelude helpers are only emitted when vector registers are used.
VECTOR_PRELUDE = r"""
typedef union {
    uint8_t b[16];
    uint16_t w[8];
    uint32_t d[4];
    uint64_t q[2];
} nexus_v128;

static inline nexus_v128 nexus_v_zero(void) {
    nexus_v128 value;
    for (int index = 0; index < 16; index++) value.b[index] = 0;
    return value;
}
static inline nexus_v128 nexus_v_load(const void *pointer) {
    nexus_v128 value;
    memcpy(&value, pointer, 16);
    return value;
}
static inline void nexus_v_store(void *pointer, nexus_v128 value) {
    memcpy(pointer, &value, 16);
}
static inline nexus_v128 nexus_v_paddq(nexus_v128 a, nexus_v128 b) {
    nexus_v128 r;
    r.q[0] = a.q[0] + b.q[0];
    r.q[1] = a.q[1] + b.q[1];
    return r;
}
static inline nexus_v128 nexus_v_paddd(nexus_v128 a, nexus_v128 b) {
    nexus_v128 r;
    for (int i = 0; i < 4; i++) r.d[i] = a.d[i] + b.d[i];
    return r;
}
static inline nexus_v128 nexus_v_psubq(nexus_v128 a, nexus_v128 b) {
    nexus_v128 r;
    r.q[0] = a.q[0] - b.q[0];
    r.q[1] = a.q[1] - b.q[1];
    return r;
}
static inline nexus_v128 nexus_v_psubd(nexus_v128 a, nexus_v128 b) {
    nexus_v128 r;
    for (int i = 0; i < 4; i++) r.d[i] = a.d[i] - b.d[i];
    return r;
}
static inline nexus_v128 nexus_v_pand(nexus_v128 a, nexus_v128 b) {
    nexus_v128 r;
    for (int i = 0; i < 16; i++) r.b[i] = a.b[i] & b.b[i];
    return r;
}
static inline nexus_v128 nexus_v_por(nexus_v128 a, nexus_v128 b) {
    nexus_v128 r;
    for (int i = 0; i < 16; i++) r.b[i] = a.b[i] | b.b[i];
    return r;
}
static inline nexus_v128 nexus_v_pandn(nexus_v128 a, nexus_v128 b) {
    nexus_v128 r;
    for (int i = 0; i < 16; i++) r.b[i] = (uint8_t)(~a.b[i]) & b.b[i];
    return r;
}
static inline nexus_v128 nexus_v_pxor(nexus_v128 a, nexus_v128 b) {
    nexus_v128 r;
    for (int i = 0; i < 16; i++) r.b[i] = a.b[i] ^ b.b[i];
    return r;
}
static inline nexus_v128 nexus_v_pcmpeqb(nexus_v128 a, nexus_v128 b) {
    nexus_v128 r;
    for (int i = 0; i < 16; i++) r.b[i] = (a.b[i] == b.b[i]) ? 0xff : 0;
    return r;
}
static inline nexus_v128 nexus_v_pcmpeqw(nexus_v128 a, nexus_v128 b) {
    nexus_v128 r;
    for (int i = 0; i < 8; i++) r.w[i] = (a.w[i] == b.w[i]) ? 0xffff : 0;
    return r;
}
static inline nexus_v128 nexus_v_pcmpeqd(nexus_v128 a, nexus_v128 b) {
    nexus_v128 r;
    for (int i = 0; i < 4; i++) r.d[i] = (a.d[i] == b.d[i]) ? 0xffffffffu : 0;
    return r;
}
static inline nexus_v128 nexus_v_pcmpeqq(nexus_v128 a, nexus_v128 b) {
    nexus_v128 r;
    for (int i = 0; i < 2; i++) r.q[i] = (a.q[i] == b.q[i]) ? UINT64_MAX : 0;
    return r;
}
static inline nexus_v128 nexus_v_pcmpgtb(nexus_v128 a, nexus_v128 b) {
    nexus_v128 r;
    for (int i = 0; i < 16; i++) r.b[i] = ((int8_t)a.b[i] > (int8_t)b.b[i]) ? 0xff : 0;
    return r;
}
static inline nexus_v128 nexus_v_pcmpgtw(nexus_v128 a, nexus_v128 b) {
    nexus_v128 r;
    for (int i = 0; i < 8; i++) r.w[i] = ((int16_t)a.w[i] > (int16_t)b.w[i]) ? 0xffff : 0;
    return r;
}
static inline nexus_v128 nexus_v_pcmpgtd(nexus_v128 a, nexus_v128 b) {
    nexus_v128 r;
    for (int i = 0; i < 4; i++) r.d[i] = ((int32_t)a.d[i] > (int32_t)b.d[i]) ? 0xffffffffu : 0;
    return r;
}
static inline nexus_v128 nexus_v_pcmpgtq(nexus_v128 a, nexus_v128 b) {
    nexus_v128 r;
    for (int i = 0; i < 2; i++) r.q[i] = ((int64_t)a.q[i] > (int64_t)b.q[i]) ? UINT64_MAX : 0;
    return r;
}
static inline nexus_v128 nexus_v_pmuludq(nexus_v128 a, nexus_v128 b) {
    nexus_v128 r;
    r.q[0] = (uint64_t)a.d[0] * (uint64_t)b.d[0];
    r.q[1] = (uint64_t)a.d[2] * (uint64_t)b.d[2];
    return r;
}
static inline nexus_v128 nexus_v_punpcklqdq(nexus_v128 a, nexus_v128 b) {
    nexus_v128 r;
    r.q[0] = a.q[0];
    r.q[1] = b.q[0];
    return r;
}
static inline nexus_v128 nexus_v_punpckhqdq(nexus_v128 a, nexus_v128 b) {
    nexus_v128 r;
    r.q[0] = a.q[1];
    r.q[1] = b.q[1];
    return r;
}
static inline nexus_v128 nexus_v_unpcklps(nexus_v128 a, nexus_v128 b) {
    nexus_v128 r;
    r.d[0] = a.d[0]; r.d[1] = b.d[0]; r.d[2] = a.d[1]; r.d[3] = b.d[1];
    return r;
}
static inline nexus_v128 nexus_v_unpckhps(nexus_v128 a, nexus_v128 b) {
    nexus_v128 r;
    r.d[0] = a.d[2]; r.d[1] = b.d[2]; r.d[2] = a.d[3]; r.d[3] = b.d[3];
    return r;
}
static inline nexus_v128 nexus_v_unpcklpd(nexus_v128 a, nexus_v128 b) {
    nexus_v128 r;
    r.q[0] = a.q[0]; r.q[1] = b.q[0];
    return r;
}
static inline nexus_v128 nexus_v_unpckhpd(nexus_v128 a, nexus_v128 b) {
    nexus_v128 r;
    r.q[0] = a.q[1]; r.q[1] = b.q[1];
    return r;
}
static inline nexus_v128 nexus_v_paddb(nexus_v128 a, nexus_v128 b) {
    nexus_v128 r;
    for (int i = 0; i < 16; i++) r.b[i] = (uint8_t)(a.b[i] + b.b[i]);
    return r;
}
static inline nexus_v128 nexus_v_paddw(nexus_v128 a, nexus_v128 b) {
    nexus_v128 r;
    for (int i = 0; i < 8; i++) r.w[i] = (uint16_t)(a.w[i] + b.w[i]);
    return r;
}
static inline nexus_v128 nexus_v_psubb(nexus_v128 a, nexus_v128 b) {
    nexus_v128 r;
    for (int i = 0; i < 16; i++) r.b[i] = (uint8_t)(a.b[i] - b.b[i]);
    return r;
}
static inline nexus_v128 nexus_v_psubw(nexus_v128 a, nexus_v128 b) {
    nexus_v128 r;
    for (int i = 0; i < 8; i++) r.w[i] = (uint16_t)(a.w[i] - b.w[i]);
    return r;
}
static inline nexus_v128 nexus_v_pminub(nexus_v128 a, nexus_v128 b) {
    nexus_v128 r;
    for (int i = 0; i < 16; i++) r.b[i] = a.b[i] < b.b[i] ? a.b[i] : b.b[i];
    return r;
}
static inline nexus_v128 nexus_v_pmaxub(nexus_v128 a, nexus_v128 b) {
    nexus_v128 r;
    for (int i = 0; i < 16; i++) r.b[i] = a.b[i] > b.b[i] ? a.b[i] : b.b[i];
    return r;
}
static inline nexus_v128 nexus_v_pminud(nexus_v128 a, nexus_v128 b) {
    nexus_v128 r;
    for (int i = 0; i < 4; i++) r.d[i] = a.d[i] < b.d[i] ? a.d[i] : b.d[i];
    return r;
}
static inline nexus_v128 nexus_v_pmaxud(nexus_v128 a, nexus_v128 b) {
    nexus_v128 r;
    for (int i = 0; i < 4; i++) r.d[i] = a.d[i] > b.d[i] ? a.d[i] : b.d[i];
    return r;
}
static inline uint32_t nexus_v_movmskps(nexus_v128 a) {
    return (uint32_t)(((a.d[0] >> 31) & 1u) | (((a.d[1] >> 31) & 1u) << 1) | (((a.d[2] >> 31) & 1u) << 2) | (((a.d[3] >> 31) & 1u) << 3));
}
static inline uint32_t nexus_v_movmskpd(nexus_v128 a) {
    return (uint32_t)(((a.q[0] >> 63) & 1u) | (((a.q[1] >> 63) & 1u) << 1));
}
static inline uint32_t nexus_v_pmovmskb(nexus_v128 a) {
    uint32_t value = 0;
    for (int i = 0; i < 16; i++) value |= (uint32_t)((a.b[i] >> 7) & 1u) << i;
    return value;
}
static inline nexus_v128 nexus_v_pshufb(nexus_v128 a, nexus_v128 mask) {
    nexus_v128 r;
    for (int i = 0; i < 16; i++) {
        uint8_t control = mask.b[i];
        r.b[i] = (control & 0x80) ? 0 : a.b[control & 0x0f];
    }
    return r;
}
static inline nexus_v128 nexus_v_pshufd(nexus_v128 a, unsigned immediate) {
    nexus_v128 r;
    for (int i = 0; i < 4; i++) r.d[i] = a.d[(immediate >> (2 * i)) & 3];
    return r;
}
static inline nexus_v128 nexus_v_psrldq(nexus_v128 a, unsigned count) {
    nexus_v128 r;
    for (int i = 0; i < 16; i++) r.b[i] = (i + count < 16) ? a.b[i + count] : 0;
    return r;
}
static inline nexus_v128 nexus_v_pslldq(nexus_v128 a, unsigned count) {
    nexus_v128 r;
    for (int i = 0; i < 16; i++) r.b[i] = (i >= count) ? a.b[i - count] : 0;
    return r;
}
static inline nexus_v128 nexus_v_psrld(nexus_v128 a, unsigned count) {
    nexus_v128 r;
    for (int i = 0; i < 4; i++) r.d[i] = (count >= 32) ? 0 : (a.d[i] >> count);
    return r;
}
static inline nexus_v128 nexus_v_pslld(nexus_v128 a, unsigned count) {
    nexus_v128 r;
    for (int i = 0; i < 4; i++) r.d[i] = (count >= 32) ? 0 : (a.d[i] << count);
    return r;
}
static inline nexus_v128 nexus_v_psrlq(nexus_v128 a, unsigned count) {
    nexus_v128 r;
    for (int i = 0; i < 2; i++) r.q[i] = (count >= 64) ? 0 : (a.q[i] >> count);
    return r;
}
static inline nexus_v128 nexus_v_psllq(nexus_v128 a, unsigned count) {
    nexus_v128 r;
    for (int i = 0; i < 2; i++) r.q[i] = (count >= 64) ? 0 : (a.q[i] << count);
    return r;
}
static inline nexus_v128 nexus_v_shufps(nexus_v128 a, nexus_v128 b, unsigned immediate) {
    nexus_v128 r;
    r.d[0] = a.d[immediate & 3];
    r.d[1] = a.d[(immediate >> 2) & 3];
    r.d[2] = b.d[(immediate >> 4) & 3];
    r.d[3] = b.d[(immediate >> 6) & 3];
    return r;
}
static inline nexus_v128 nexus_v_shufpd(nexus_v128 a, nexus_v128 b, unsigned immediate) {
    nexus_v128 r;
    r.q[0] = (immediate & 1) ? a.q[1] : a.q[0];
    r.q[1] = (immediate & 2) ? b.q[1] : b.q[0];
    return r;
}
"""


def emit_function(
    bundle: AnalysisBundle,
    function: FunctionCandidate,
    generated_name: str,
    prototypes: dict[int, Prototype] | None = None,
    name_map: dict[str, str] | None = None,
    strategy: str = "semantic_ir",
) -> EmitResult:
    if not function.blocks:
        return EmitResult(False, None, "no basic blocks", {"instructions": len(function.instructions)})
    translator = _Translator(bundle, function, generated_name, prototypes or {}, name_map or {}, strategy)
    proto = translator._proto()
    body_lines: list[str] = []
    translator.lines = body_lines
    entry = next((block for block in function.blocks if block.start == function.entry), function.blocks[0])
    in_offsets, in_pointers = translator._block_provenance()
    in_flags = translator._block_flags()
    translator.jump_targets = set()
    for candidate_block in function.blocks:
        info = translator._detect_jump_table(candidate_block)
        if info:
            translator.jump_targets.update(info["targets"])
    body_lines.append(f"    goto bb_{entry.start:x};")
    ordered = sorted(function.blocks, key=lambda item: item.start)
    for block in ordered:
        body_lines.append(f"bb_{block.start:x}:")
        descriptor = in_flags.get(block.block_id)
        translator.flags = None if descriptor is None else FlagState(descriptor[0], "nexus_flag_left", "nexus_flag_right", "nexus_flag_result", descriptor[1], descriptor[2], "nexus_flag_high" if descriptor[3] else None)
        translator.current_versions = {}
        translator.stack_offsets = set(in_offsets.get(block.block_id, set()))
        translator.pointer_registers = set(in_pointers.get(block.block_id, set()))
        translator.noreturn_call = False
        translator.jump_table = translator._detect_jump_table(block)
        last = block.instructions[-1] if block.instructions else None
        for instruction in block.instructions:
            if instruction.address in translator.jump_targets and instruction.address != block.start:
                body_lines.append(f"bb_{instruction.address:x}:")
            if translator.jump_table and instruction.address == translator.jump_table["movsxd_address"]:
                index_register = translator.jump_table["index_register"]
                body_lines.append(f"    nexus_jump_index = (uint64_t)({translator._read_register(index_register)});")
            if instruction is not last and instruction.control_flow in {"conditional_branch", "unconditional_branch", "return", "indirect_branch", "indirect_call"}:
                return EmitResult(False, None, f"control transfer at 0x{instruction.address:x} is not the end of block 0x{block.start:x}", {"unsupported": translator.unsupported})
            try:
                emitted = translator._emit_instruction(instruction)
            except EmitError as error:
                translator.unsupported.append({
                    "address": f"0x{instruction.address:x}",
                    "mnemonic": instruction.mnemonic,
                    "operands": instruction.operands,
                    "category": error.category,
                    "reason": error.reason,
                })
                return EmitResult(False, None, f"{error.reason} (0x{instruction.address:x})", {
                    "instructions": len(function.instructions),
                    "translated": len(body_lines),
                    "unsupported": translator.unsupported,
                    "stats": dict(translator.stats),
                })
            body_lines.extend(emitted)
            if translator.noreturn_call:
                break
        body_lines.extend(translator._merge_versions())
        if last is None:
            return EmitResult(False, None, "empty block", {"unsupported": translator.unsupported})
        if translator.noreturn_call:
            continue
        if last.control_flow == "return":
            if proto.return_width == 32:
                body_lines.append("    return (uint32_t)rax;")
            elif proto.return_width == 16:
                body_lines.append("    return (uint16_t)rax;")
            elif proto.return_width == 8:
                body_lines.append("    return (uint8_t)rax;")
            else:
                body_lines.append("    return rax;")
            continue
        if last.mnemonic in {"call", "callq"}:
            fallthrough = translator._fallthrough(last.end)
            if fallthrough is None:
                body_lines.append("    return rax;")
            else:
                body_lines.append(f"    goto bb_{fallthrough.start:x};")
            continue
        if last.control_flow == "unconditional_branch" and translator.jump_table and last.address == translator.jump_table["jmp_address"]:
            body_lines.append("    switch (nexus_jump_index) {")
            for index, target in enumerate(translator.jump_table["targets"]):
                body_lines.append(f"    case {index}: goto bb_{target:x};")
            body_lines.append(f"    default: goto bb_{translator.jump_table['default']:x};")
            body_lines.append("    }")
            continue
        if last.control_flow == "unconditional_branch":
            target = last.targets[0].va if last.targets else None
            if target is not None and target in translator.block_starts:
                body_lines.append(f"    goto bb_{target:x};")
            elif target is not None:
                try:
                    body_lines.extend(translator._tail_call(target, last.address))
                except EmitError as error:
                    translator.unsupported.append({
                        "address": f"0x{last.address:x}", "mnemonic": last.mnemonic, "operands": last.operands,
                        "category": error.category, "reason": error.reason,
                    })
                    return EmitResult(False, None, error.reason, {"unsupported": translator.unsupported, "stats": dict(translator.stats)})
            else:
                return EmitResult(False, None, f"unresolved unconditional branch at 0x{last.address:x}", {"unsupported": translator.unsupported})
            continue
        if last.control_flow == "conditional_branch":
            target = last.targets[0].va if last.targets else None
            fallthrough = translator._fallthrough(last.end)
            if target is None or target not in translator.block_starts or fallthrough is None:
                return EmitResult(False, None, f"unresolved conditional branch at 0x{last.address:x}", {"unsupported": translator.unsupported})
            try:
                condition = translator._condition(last.mnemonic)
            except EmitError as error:
                translator.unsupported.append({
                    "address": f"0x{last.address:x}", "mnemonic": last.mnemonic, "operands": last.operands,
                    "category": error.category, "reason": error.reason,
                })
                return EmitResult(False, None, error.reason, {"unsupported": translator.unsupported, "stats": dict(translator.stats)})
            body_lines.append(f"    if ({condition}) goto bb_{target:x};")
            body_lines.append(f"    goto bb_{fallthrough.start:x};")
            continue
        if last.control_flow in {"indirect_branch", "indirect_call"}:
            return EmitResult(False, None, f"indirect control flow at 0x{last.address:x} is not modelled", {"unsupported": translator.unsupported})
        fallthrough = translator._fallthrough(last.end)
        if fallthrough is None:
            body_lines.append("    return 0;")
        else:
            body_lines.append(f"    goto bb_{fallthrough.start:x};")
    lines: list[str] = []
    lines.extend(translator.data_pool)
    if translator.data_pool:
        lines.append("")
    if translator.vector_used:
        lines.append(VECTOR_PRELUDE.strip())
        lines.append("")
    lines.append("extern void __stack_chk_fail(void);")
    for declaration in sorted(translator.externs.values()):
        lines.append(declaration)
    if translator.externs or True:
        lines.append("")
    lines.append(f"{proto.return_name} {generated_name}({proto.params}) {{")
    lines.append("    uint8_t nexus_stack[65536] = {0};")
    lines.append("    const uint64_t nexus_canary_value = UINT64_C(0x4e45585553424939);")
    lines.append("    uint64_t rsp = (uint64_t)(uintptr_t)(nexus_stack + 32768);")
    lines.append("    uint64_t nexus_flag_left = 0, nexus_flag_right = 0, nexus_flag_result = 0, nexus_flag_high = 0;")
    lines.append("    uint64_t nexus_carry_value = 0;")
    lines.append("    uint64_t nexus_jump_index = 0;")
    if translator.frame_pointer:
        lines.append("    uint64_t rbp = 0;")
    lines.append("    (void)nexus_stack;")
    lines.append("    (void)nexus_canary_value;")
    for register in SCALAR_REGISTERS:
        if register == "rbp" and translator.frame_pointer:
            continue
        lines.append(f"    uint64_t {register} = 0;")
    for register in sorted(translator.declared_scalar - set(SCALAR_REGISTERS)):
        if register not in VECTOR_REGISTERS and register not in {"rsp", "rip"} and not (register == "rbp" and translator.frame_pointer):
            lines.append(f"    uint64_t {register} = 0;")
    if translator.vector_used:
        for register in VECTOR_REGISTERS:
            lines.append(f"    nexus_v128 {register} = nexus_v_zero();")
    for index in range(proto.parameter_count):
        if index < len(ARGUMENT_REGISTERS):
            lines.append(f"    {ARGUMENT_REGISTERS[index]} = (uint64_t)arg{index};")
    lines.extend(body_lines)
    lines.append("}")
    body = "\n".join(lines)
    return EmitResult(True, body, None, {
        "instructions": len(function.instructions),
        "translated": len(function.instructions),
        "unsupported": translator.unsupported,
        "stats": dict(translator.stats),
        "frame_pointer": translator.frame_pointer,
        "vector": translator.vector_used,
        "feature": "register_versioned" if translator.ssa else "direct_value",
        "externs": sorted(translator.externs.values()),
        "region_analysis": __import__("nexus.reconstruction.c.regions", fromlist=["analyze_regions"]).analyze_regions(function) if strategy == "region" else None,
    })
