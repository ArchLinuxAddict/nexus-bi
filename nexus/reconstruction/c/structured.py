import re
from dataclasses import dataclass
from typing import Any

from nexus.disassembly.model import Instruction
from nexus.program.bundle import AnalysisBundle
from nexus.program.model import FunctionCandidate
from nexus.reconstruction.c.generator import (
    _abi_prototype,
    _immediate,
    _reg,
    _sanitize,
    _slot,
    _width_for_reg,
)


@dataclass
class StructuredResult:
    success: bool
    body: str | None
    reason: str | None
    evidence: list[int]


CONDITION_OPERATORS = {
    "je": "==", "jz": "==", "jne": "!=", "jnz": "!=",
    "jl": "<", "jnge": "<", "jle": "<=", "jng": "<=",
    "jg": ">", "jnle": ">", "jge": ">=", "jnl": ">=",
    "jb": "<", "jnae": "<", "jbe": "<=", "jna": "<=",
    "ja": ">", "jnbe": ">", "jae": ">=", "jnb": ">=", "jnc": ">=",
}
SIGNED_JUMPS = {"jl", "jnge", "jle", "jng", "jg", "jnle", "jge", "jnl"}
UNSIGNED_JUMPS = {"jb", "jnae", "jbe", "jna", "ja", "jnbe", "jae", "jnb", "jnc"}
CONDITIONAL_JUMPS = set(CONDITION_OPERATORS) | {"js", "jns", "jo", "jno", "jp", "jnp", "jpe", "jpo", "jcxz", "jecxz", "jrcxz", "loop", "loope", "loopne"}


def _slot_name(slot: tuple[str, int]) -> str:
    base, displacement = slot
    if displacement < 0:
        return f"var_{base}_m{abs(displacement)}"
    return f"var_{base}_{displacement}"


def _memory_width(operand: str) -> int:
    upper = operand.upper()
    if "QWORD PTR" in upper:
        return 64
    if "DWORD PTR" in upper:
        return 32
    if "WORD PTR" in upper:
        return 16
    if "BYTE PTR" in upper:
        return 8
    return 64


def _memory_address(operand: str) -> str | None:
    if "[" not in operand or "]" not in operand or "rip" in operand:
        return None
    inside = operand[operand.find("[") + 1 : operand.find("]")].replace(" ", "")
    terms = []
    sign = 1
    for part in re.split(r"([+-])", inside):
        if part == "+":
            sign = 1
            continue
        if part == "-":
            sign = -1
            continue
        if not part:
            continue
        if part in {"rbp", "rsp"}:
            part = "((uintptr_t)stack_frame + 512)"
            terms.append(part)
            continue
        if "*" in part:
            base, scale = part.split("*", 1)
            if _reg(base) is None:
                return None
            if base in {"rbp", "rsp"}:
                base = "((uintptr_t)stack_frame + 512)"
            terms.append(f"{sign}*{scale}*{base}" if sign < 0 else f"{scale}*{base}")
            continue
        if _reg(part) is not None:
            terms.append(f"{sign}*{part}" if sign < 0 else part)
            continue
        try:
            value = int(part, 0)
            terms.append(f"{sign}*{value}" if sign < 0 else str(value))
        except ValueError:
            return None
    if not terms:
        return None
    text = terms[0].lstrip("+")
    for term in terms[1:]:
        text += term if term.startswith("-") else "+" + term
    return text


def _memory_operand(operand: str) -> tuple[str, int] | None:
    address = _memory_address(operand)
    if address is None:
        return None
    return address, _memory_width(operand)


def _operand_expr(operand: str, registers: set[str], slots: dict[tuple[str, int], int], width: int | None = None) -> str | None:
    register = _reg(operand)
    if register:
        if register in {"rbp", "rsp"}:
            return "((uint64_t)(uintptr_t)stack_frame + 512)"
        operand_width = _width_for_reg(operand)
        return f"((uint{operand_width}_t){register})" if operand_width else register
    immediate = _immediate(operand)
    if immediate is not None:
        return str(immediate)
    slot = _slot(operand)
    if slot:
        name = _slot_name(slot)
        slot_width = slots.get(slot)
        if slot_width:
            return f"((uint{slot_width}_t){name})"
        return name
    memory = _memory_operand(operand)
    if memory:
        address, memory_width = memory
        return f"(*((uint{memory_width}_t *)(uintptr_t)({address})))"
    if "rip" in operand:
        return None
    return None


def _truncate_expr(expr: str, width: int) -> str:
    return f"((uint{width}_t)({expr}))"


def _store_register(register: str, value: str, width: int) -> str:
    if width < 64:
        return f"{register} = {_truncate_expr(value, width)};"
    return f"{register} = {value};"


def _statement(instruction: Instruction, registers: set[str], slots: dict[tuple[str, int], int]) -> tuple[str | None, str | None, dict[str, Any] | None]:
    mnemonic = instruction.mnemonic
    operands = instruction.operands
    if any("fs:0x28" in operand for operand in operands):
        return None, None, None
    if mnemonic in {"endbr64", "endbr32", "nop", "leave", "data16"}:
        return None, None, None
    if mnemonic in {"jmp", "jmpq", "ret", "retq"} or mnemonic in CONDITIONAL_JUMPS:
        return None, None, None
    if mnemonic in {"push", "pop"}:
        return None, None, None
    if mnemonic in {"sub", "add"} and len(operands) == 2 and _reg(operands[0]) == "rsp":
        return None, None, None
    if mnemonic == "mov" and len(operands) == 2 and _reg(operands[0]) == "rbp" and _reg(operands[1]) == "rsp":
        return None, None, None
    if mnemonic in {"mov", "movabs", "movzx", "movsx", "movsxd"}:
        if len(operands) != 2:
            return None, f"unsupported mov form at 0x{instruction.address:x}", None
        destination, source = operands
        destination_register = _reg(destination)
        if destination_register:
            destination_width = _width_for_reg(destination)
            source_value = _operand_expr(source, registers, slots, destination_width)
            if source_value is None:
                return None, f"unsupported source operand at 0x{instruction.address:x}", None
            if mnemonic in {"movzx", "movsx", "movsxd"}:
                source_width = _width_for_reg(source) if _reg(source) else destination_width
                if mnemonic == "movsx":
                    source_value = f"((int{source_width}_t)({source_value}))"
                elif mnemonic == "movsxd":
                    source_value = f"((int32_t)({source_value}))"
                else:
                    source_value = f"((uint{source_width}_t)({source_value}))"
            return _store_register(destination_register, source_value, destination_width), None, None
        slot = _slot(destination)
        if slot is not None:
            slot_width = slots.get(slot)
            if slot_width is None:
                return None, f"unknown stack width at 0x{instruction.address:x}", None
            source_value = _operand_expr(source, registers, slots, slot_width)
            if source_value is None:
                return None, f"unsupported stack source at 0x{instruction.address:x}", None
            return f"{_slot_name(slot)} = {_truncate_expr(source_value, slot_width)};", None, None
        memory = _memory_operand(destination)
        if memory is None:
            return None, f"unsupported memory destination at 0x{instruction.address:x}", None
        address, memory_width = memory
        source_value = _operand_expr(source, registers, slots, memory_width)
        if source_value is None:
            return None, f"unsupported memory source at 0x{instruction.address:x}", None
        return f"*((uint{memory_width}_t *)(uintptr_t)({address})) = {_truncate_expr(source_value, memory_width)};", None, None
    if mnemonic == "lea":
        if len(operands) != 2:
            return None, f"unsupported lea at 0x{instruction.address:x}", None
        destination = _reg(operands[0])
        if not destination:
            return None, f"unsupported lea destination at 0x{instruction.address:x}", None
        inside = operands[1][operands[1].find("[") + 1:operands[1].find("]")].replace(" ", "")
        if "rbp" in inside or "rsp" in inside:
            address = _memory_address(operands[1])
            if address is None:
                return None, f"unsupported lea address at 0x{instruction.address:x}", None
            width = _width_for_reg(operands[0])
            return _store_register(destination, f"(uint64_t)(uintptr_t)({address})", width), None, None
        if "rip" in inside:
            return None, f"unsupported lea address at 0x{instruction.address:x}", None
        parts = re.split(r"([+-])", inside)
        sign = 1
        terms = []
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
                register, scale = part.split("*", 1)
                if register not in registers:
                    return None, f"unknown lea register at 0x{instruction.address:x}", None
                terms.append(f"{scale}*{register}")
            elif part in registers:
                terms.append(f"{sign}{part}" if sign < 0 else part)
            else:
                try:
                    terms.append(f"{sign}{int(part, 0)}" if sign < 0 else str(int(part, 0)))
                except ValueError:
                    return None, f"unsupported lea term at 0x{instruction.address:x}", None
        expression = "+".join(terms).replace("+-", "-")
        width = _width_for_reg(operands[0])
        return _store_register(destination, expression, width), None, None
    if mnemonic in {"div", "idiv"}:
        if len(operands) != 1:
            return None, f"unsupported division form at 0x{instruction.address:x}", None
        right = _operand_expr(operands[0], registers, slots, 64)
        if right is None:
            return None, f"unsupported division operand at 0x{instruction.address:x}", None
        if "rax" not in registers or "rdx" not in registers:
            return None, f"division without rax/rdx at 0x{instruction.address:x}", None
        return f"{{ uint64_t dividend = rax; rax = dividend / ({right}); rdx = dividend % ({right}); }}", None, None
    if mnemonic in {"add", "sub", "imul", "and", "or", "xor", "shl", "shr", "sar"}:
        if len(operands) < 2:
            return None, f"unsupported arithmetic at 0x{instruction.address:x}", None
        destination = _reg(operands[0])
        slot = _slot(operands[0])
        memory = _memory_operand(operands[0])
        if destination:
            width = _width_for_reg(operands[0])
            left = f"((uint{width}_t){destination})"
        elif slot and slot in slots:
            width = slots[slot]
            left = f"((uint{width}_t){_slot_name(slot)})"
        elif memory:
            address, width = memory
            left = f"(*((uint{width}_t *)(uintptr_t)({address})))"
        else:
            return None, f"unsupported arithmetic destination at 0x{instruction.address:x}", None
        right = _operand_expr(operands[1], registers, slots, width)
        if right is None:
            return None, f"unsupported arithmetic source at 0x{instruction.address:x}", None
        if mnemonic == "imul" and len(operands) >= 3:
            third = _operand_expr(operands[2], registers, slots, width)
            if third is None:
                return None, f"unsupported imul operand at 0x{instruction.address:x}", None
            right = f"({right} * {third})"
        operator = {"add": "+", "sub": "-", "imul": "*", "and": "&", "or": "|", "xor": "^", "shl": "<<", "shr": ">>", "sar": ">>"}[mnemonic]
        if mnemonic == "sar":
            expression = f"((uint{width}_t)(((int{width}_t)({left})) >> ({right})))"
        else:
            expression = f"({left} {operator} {right})"
        flags = {"mnemonic": "result_flags", "register": destination, "width": width} if destination else None
        if destination:
            return _store_register(destination, expression, width), None, flags
        if slot is not None:
            return f"{_slot_name(slot)} = {_truncate_expr(expression, width)};", None, None
        if memory:
            address, _ = memory
            return f"*((uint{width}_t *)(uintptr_t)({address})) = {_truncate_expr(expression, width)};", None, None
        return None, f"unsupported arithmetic destination at 0x{instruction.address:x}", None
    if mnemonic in {"inc", "dec", "neg", "not"}:
        if len(operands) != 1:
            return None, f"unsupported unary at 0x{instruction.address:x}", None
        destination = _reg(operands[0])
        if not destination:
            return None, f"unsupported unary destination at 0x{instruction.address:x}", None
        width = _width_for_reg(operands[0])
        left = f"((uint{width}_t){destination})"
        if mnemonic == "inc":
            expression = f"({left} + 1)"
        elif mnemonic == "dec":
            expression = f"({left} - 1)"
        elif mnemonic == "neg":
            expression = f"(-{left})"
            return _store_register(destination, expression, width), None, {"mnemonic": "neg", "operands": [operands[0]], "instruction": instruction}
        else:
            expression = f"(~{left})"
        return _store_register(destination, expression, width), None, {"mnemonic": "result_flags", "register": destination, "width": width}
    if mnemonic in {"cmp", "test"}:
        return None, None, {"mnemonic": mnemonic, "operands": operands, "instruction": instruction}
    if mnemonic.startswith("set") or mnemonic.startswith("cmov"):
        return None, f"conditional instruction requires a preceding comparison at 0x{instruction.address:x}", None
    return None, f"instruction {mnemonic} at 0x{instruction.address:x} is not translated", None


def _condition_from_compare(last_compare: dict[str, Any] | None, mnemonic: str) -> str | None:
    if not last_compare:
        return None
    operands = last_compare.get("operands", [])
    if last_compare.get("mnemonic") == "result_flags" and last_compare.get("register"):
        register = last_compare["register"]
        width = last_compare.get("width", 64)
        if mnemonic in {"je", "jz"}:
            return f"((uint{width}_t)({register}) == 0)"
        if mnemonic in {"jne", "jnz"}:
            return f"((uint{width}_t)({register}) != 0)"
        if mnemonic in {"js", "jns"}:
            return f"((int{width}_t)({register}) {'<' if mnemonic == 'js' else '>='} 0)"
        return None
    if last_compare.get("mnemonic") == "neg" and mnemonic in {"js", "jns"}:
        source_operand = operands[0] if operands else last_compare.get("operands", [None])[0]
        left = _operand_expr(source_operand, set(), {}, None) if source_operand else None
        if left is None:
            return None
        width = _width_for_reg(source_operand) or 32
        operator = "<" if mnemonic == "js" else ">="
        return f"((int{width}_t)({left}) {operator} 0)"
    if mnemonic in {"js", "jns"} and operands:
        left_operand = operands[0]
        left = _operand_expr(left_operand, set(), {}, None)
        if left is None:
            return None
        width = _width_for_reg(left_operand) or 32
        cast = f"int{width}_t"
        operator = "<" if mnemonic == "js" else ">="
        return f"((({cast})({left})) {operator} 0)"
    if len(operands) != 2:
        return None
    left_operand, right_operand = operands
    left = _operand_expr(left_operand, set(), {}, None)
    right = _operand_expr(right_operand, set(), {}, None)
    if left is None or right is None:
        return None
    memory_width = None
    compare_instruction = last_compare.get("instruction")
    if compare_instruction is not None:
        for operand in (left_operand, right_operand):
            for effect in compare_instruction.memory:
                if effect.address == operand and effect.width:
                    memory_width = effect.width
    width = memory_width or _memory_width(left_operand) if "[" in left_operand else memory_width or _width_for_reg(left_operand) or _width_for_reg(right_operand) or 32
    if "[" in right_operand:
        width = _memory_width(right_operand)
    signed = mnemonic in SIGNED_JUMPS
    cast = f"int{width}_t" if signed else f"uint{width}_t"
    if last_compare["mnemonic"] == "test":
        if mnemonic in {"je", "jz"}:
            return f"((({cast})({left}) & ({cast})({right})) == 0)"
        if mnemonic in {"jne", "jnz"}:
            return f"((({cast})({left}) & ({cast})({right})) != 0)"
        if mnemonic in {"jg", "jge", "jl", "jle"}:
            signed_left = f"((int{width}_t)({left}))"
            operator = {"jg": ">", "jge": ">=", "jl": "<", "jle": "<="}[mnemonic]
            return f"({signed_left} {operator} 0)"
        if mnemonic == "ja":
            return f"(({cast})({left}) != 0)"
        if mnemonic == "jae":
            return "1"
        if mnemonic == "jb":
            return "0"
        if mnemonic == "jbe":
            return f"(({cast})({left}) == 0)"
        return None
    operator = CONDITION_OPERATORS.get(mnemonic)
    if operator is None:
        return None
    return f"((({cast})({left})) {operator} (({cast})({right})))"


def translate_structured(bundle: AnalysisBundle, function: FunctionCandidate, generated_name: str | None = None) -> StructuredResult:
    if not function.blocks:
        return StructuredResult(False, None, "no basic blocks", [])
    return_name, params, return_width = _abi_prototype(function, bundle)
    functions_by_entry = {item.entry: item for item in bundle.functions}

    types = bundle.type_candidates.get(function.function_id, [])
    parameter_types = sorted([item for item in types if item.get("role") == "parameter"], key=lambda item: item.get("parameter_index", 0))
    registers: set[str] = set()
    slots: dict[tuple[str, int], int] = {}
    for instruction in function.instructions:
        registers.update(register for register in instruction.reads + instruction.writes if register not in {"rsp", "rbp", "rip"})
        for memory in instruction.memory:
            if memory.base in {"rbp", "rsp"} and memory.displacement is not None:
                slots[(memory.base, memory.displacement)] = memory.width or 64
    if not registers:
        return StructuredResult(False, None, "no registers to translate", [])
    registers.update({"rdi", "rsi", "rdx", "rcx", "r8", "r9"})
    global_lines = []
    lines = [f"{return_name} {generated_name or _sanitize(function.name)}({params}) {{"]
    lines.append("    uint8_t stack_frame[4096] = {0};")
    declarations = []
    for register in sorted(registers):
        declarations.append(f"    uint64_t {register} = 0;")
    for slot in sorted(slots):
        width = slots[slot] or 64
        declarations.append(f"    uint{width}_t {_slot_name(slot)} = 0;")
    initialized = {"rdi", "rsi", "rdx", "rcx", "r8", "r9"}
    written: set[str] = set(initialized)
    for instruction in function.instructions:
        if instruction.mnemonic in {"push", "pop"}:
            written.update(instruction.writes)
            continue
        for register in instruction.reads:
            if register not in written and register not in {"rsp", "rbp", "rip"}:
                return StructuredResult(False, None, f"read of uninitialized register {register} at 0x{instruction.address:x}", [])
        written.update(instruction.writes)
    for index, parameter in enumerate(parameter_types):
        register = parameter.get("register")
        if register:
            declarations.append(f"    {register} = arg{index};")
    lines.extend(global_lines)
    lines.extend(declarations)
    entry_block = next((block for block in function.blocks if block.start == function.entry), function.blocks[0])
    block_starts = {block.start for block in function.blocks}
    lines.append(f"    goto bb_{entry_block.start:x};")
    for block in sorted(function.blocks, key=lambda item: item.start):
        lines.append(f"bb_{block.start:x}:")
        last_compare = None
        last_instruction = None
        for instruction in block.instructions:
            last_instruction = instruction
            if instruction.control_flow == "call":
                target = instruction.targets[0].va if instruction.targets else None
                callee = functions_by_entry.get(target) if target is not None else None
                if callee is None:
                    return StructuredResult(False, None, f"external or unresolved call at 0x{instruction.address:x}", [])
                callee_name = _sanitize(callee.name)
                if "stack_chk" in callee_name:
                    continue
                if callee_name.endswith("_plt"):
                    external = callee_name[:-4]
                    arity = {"malloc": 1, "free": 1, "calloc": 2, "realloc": 2, "memcpy": 3, "memset": 3, "strlen": 1}.get(external)
                    if arity is not None:
                        argument_registers = ["rdi", "rsi", "rdx"]
                        arguments = [_operand_expr(argument_registers[index], registers, slots) for index in range(arity)]
                        if all(value is not None for value in arguments):
                            if external == "free":
                                lines.append(f"    free((void *)(uintptr_t)({arguments[0]}));")
                            elif external in {"malloc", "calloc", "realloc"}:
                                lines.append(f"    rax = (uint64_t)(uintptr_t){external}({', '.join(arguments)});")
                            elif external == "memcpy":
                                cast_arguments = [f"(void *)(uintptr_t)({arguments[0]})", f"(const void *)(uintptr_t)({arguments[1]})", arguments[2]]
                                lines.append(f"    (void){external}({', '.join(cast_arguments)});")
                            elif external == "memset":
                                cast_arguments = [f"(void *)(uintptr_t)({arguments[0]})", arguments[1], arguments[2]]
                                lines.append(f"    (void){external}({', '.join(cast_arguments)});")
                            else:
                                lines.append(f"    rax = (uint64_t)(uintptr_t){external}({', '.join(arguments)});")
                            continue
                _, callee_params, _ = _abi_prototype(callee, bundle)
                count = 6
                argument_registers = ["rdi", "rsi", "rdx", "rcx", "r8", "r9"]
                arguments = []
                for index in range(count):
                    argument = _operand_expr(argument_registers[index], registers, slots)
                    if argument is None:
                        return StructuredResult(False, None, f"unresolved call argument at 0x{instruction.address:x}", [])
                    arguments.append(argument)
                lines.append(f"    rax = {callee_name}({', '.join(arguments)});")
                continue
            if instruction.mnemonic.startswith("cmov") or instruction.mnemonic.startswith("set"):
                suffix = instruction.mnemonic[4:] if instruction.mnemonic.startswith("cmov") else instruction.mnemonic[3:]
                condition = _condition_from_compare(last_compare, "j" + suffix)
                if condition is None:
                    return StructuredResult(False, None, f"unresolved conditional instruction at 0x{instruction.address:x}", [])
                if not instruction.operands:
                    return StructuredResult(False, None, f"missing conditional operand at 0x{instruction.address:x}", [])
                destination = _reg(instruction.operands[0])
                if not destination:
                    return StructuredResult(False, None, f"unsupported conditional destination at 0x{instruction.address:x}", [])
                width = _width_for_reg(instruction.operands[0])
                if instruction.mnemonic.startswith("set"):
                    statement = _store_register(destination, f"({condition}) ? 1 : 0", width)
                else:
                    source_value = _operand_expr(instruction.operands[1], registers, slots, width) if len(instruction.operands) > 1 else None
                    if source_value is None:
                        return StructuredResult(False, None, f"unsupported conditional source at 0x{instruction.address:x}", [])
                    statement = f"{destination} = ({condition}) ? {source_value} : {destination};"
                lines.append(f"    {statement}")
                continue
            if instruction.mnemonic == "lea" and len(instruction.operands) == 2 and "rip" in instruction.operands[1]:
                destination = _reg(instruction.operands[0])
                if not destination:
                    return StructuredResult(False, None, f"unsupported rip lea destination at 0x{instruction.address:x}", [])
                match = re.search(r"rip([+-]0x[0-9a-fA-F]+|[+-]\d+)", instruction.operands[1])
                if not match:
                    return StructuredResult(False, None, f"unresolved rip displacement at 0x{instruction.address:x}", [])
                target = instruction.end + int(match.group(1), 0)
                payload = bundle.elf.read_va(target, 128) or b""
                text = payload.split(b"\x00", 1)[0]
                if not text or any(byte < 9 or byte > 126 for byte in text):
                    return StructuredResult(False, None, f"rip data is not a printable string at 0x{instruction.address:x}", [])
                name = f"nexus_data_{len(global_lines)}"
                escaped = text.decode("ascii").replace("\\", "\\\\").replace('"', '\"').replace("\n", "\\n").replace("\t", "\\t")
                global_lines.append(f"static const char {name}[] = \"{escaped}\";")
                lines.insert(1, global_lines[-1])
                lines.append(f"    {destination} = (uint64_t)(uintptr_t){name};")
                continue
            statement, reason, compare = _statement(instruction, registers, slots)
            if reason:
                return StructuredResult(False, None, reason, [])
            if compare:
                last_compare = compare
            if statement:
                lines.append(f"    {statement}")
        if last_instruction is None:
            return StructuredResult(False, None, "empty block", [])
        flow = last_instruction.control_flow
        if flow == "return":
            if "rax" in registers:
                lines.append(f"    return {_truncate_expr('rax', return_width)};")
            else:
                lines.append("    return 0;")
        elif flow == "unconditional_branch":
            target = last_instruction.targets[0].va if last_instruction.targets else None
            if target is None or target not in block_starts:
                return StructuredResult(False, None, "unconditional branch target is outside recovered blocks", [])
            lines.append(f"    goto bb_{target:x};")
        elif flow == "conditional_branch":
            target = last_instruction.targets[0].va if last_instruction.targets else None
            if target is not None:
                target_block = next((other for other in function.blocks if other.start == target), None)
                fallthrough_probe = next((other for other in sorted(function.blocks, key=lambda item: item.start) if last_instruction.end <= other.start <= last_instruction.end + 16), None)
                def has_stack_check(block):
                    return bool(block and any(instruction.mnemonic == "call" and instruction.targets and any("stack_chk" in (functions_by_entry.get(item.va).name if functions_by_entry.get(item.va) else "") for item in instruction.targets) for instruction in block.instructions))
                if has_stack_check(target_block) and fallthrough_probe is not None:
                    lines.append(f"    goto bb_{fallthrough_probe.start:x};")
                    continue
                if has_stack_check(fallthrough_probe):
                    lines.append(f"    goto bb_{target:x};")
                    continue
            condition = _condition_from_compare(last_compare, last_instruction.mnemonic)
            if condition is None:
                return StructuredResult(False, None, f"unresolved condition at 0x{last_instruction.address:x}", [])
            if target is None or target not in block_starts:
                return StructuredResult(False, None, "conditional target is outside recovered blocks", [])
            fallthrough_block = next((other for other in function.blocks if other.start == last_instruction.end), None)
            if fallthrough_block is None:
                fallthrough_block = next((other for other in sorted(function.blocks, key=lambda item: item.start) if last_instruction.end <= other.start <= last_instruction.end + 16), None)
            if fallthrough_block is None:
                return StructuredResult(False, None, "unresolved fallthrough block", [])
            lines.append(f"    if ({condition}) goto bb_{target:x};")
            lines.append(f"    goto bb_{fallthrough_block.start:x};")
        elif flow == "call":
            fallthrough_block = next((other for other in function.blocks if other.start == last_instruction.end), None)
            if fallthrough_block is not None:
                lines.append(f"    goto bb_{fallthrough_block.start:x};")
            else:
                lines.append("    return 0;")
        elif flow in {"indirect_call", "indirect_branch", "trap"}:
            return StructuredResult(False, None, f"{flow} is not translated", [])
        else:
            fallthrough_block = next((other for other in function.blocks if other.start == last_instruction.end), None)
            if fallthrough_block is None:
                fallthrough_block = next((other for other in sorted(function.blocks, key=lambda item: item.start) if last_instruction.end <= other.start <= last_instruction.end + 16), None)
            if fallthrough_block is None:
                return StructuredResult(False, None, "fallthrough target is outside recovered blocks", [])
            lines.append(f"    goto bb_{fallthrough_block.start:x};")
    lines.append("}")
    lines.append("")
    return StructuredResult(True, "\n".join(lines), None, [function.entry])
