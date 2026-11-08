"""Read-only Java class-file parser with bytecode decoding.

The parser is deliberately static: it does not load, link, verify or execute
classes.  It resolves the constant pool, fields, methods, descriptors, code
attributes, line/local-variable tables and exception tables so that Java source
reconstruction can be evidence-linked.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

CP_UTF8 = 1
CP_INTEGER = 3
CP_FLOAT = 4
CP_LONG = 5
CP_DOUBLE = 6
CP_CLASS = 7
CP_STRING = 8
CP_FIELDREF = 9
CP_METHODREF = 10
CP_INTERFACE_METHODREF = 11
CP_NAME_AND_TYPE = 12
CP_METHOD_HANDLE = 15
CP_METHOD_TYPE = 16
CP_DYNAMIC = 17
CP_INVOKE_DYNAMIC = 18
CP_MODULE = 19
CP_PACKAGE = 20

ACCESS_FLAGS = {
    0x0001: "public",
    0x0002: "private",
    0x0004: "protected",
    0x0008: "static",
    0x0010: "final",
    0x0020: "synchronized",
    0x0040: "volatile",
    0x0080: "transient",
    0x0100: "native",
    0x0200: "interface",
    0x0400: "abstract",
    0x0800: "strictfp",
    0x1000: "synthetic",
    0x2000: "annotation",
    0x4000: "enum",
    0x8000: "module",
}

PRIMITIVES = {
    "B": "byte", "C": "char", "D": "double", "F": "float",
    "I": "int", "J": "long", "S": "short", "Z": "boolean", "V": "void",
}


def _u16(data: bytes, offset: int) -> int:
    return struct.unpack_from(">H", data, offset)[0]


def _u32(data: bytes, offset: int) -> int:
    return struct.unpack_from(">I", data, offset)[0]


def _u64(data: bytes, offset: int) -> int:
    return struct.unpack_from(">Q", data, offset)[0]


def _u8(data: bytes, offset: int) -> int:
    return data[offset]


def access_names(flags: int) -> list[str]:
    return [name for bit, name in ACCESS_FLAGS.items() if flags & bit]


def access_names_for(kind: str, flags: int) -> list[str]:
    allowed = {
        "class": {"public", "final", "interface", "abstract", "synthetic", "annotation", "enum", "module"},
        "field": {"public", "private", "protected", "static", "final", "volatile", "transient", "synthetic", "enum"},
        "method": {"public", "private", "protected", "static", "final", "synchronized", "varargs", "native", "abstract", "strictfp", "synthetic"},
    }.get(kind, set(ACCESS_FLAGS.values()))
    names = [name for bit, name in ACCESS_FLAGS.items() if flags & bit and name in allowed]
    if kind == "class" and flags & 0x0020:
        names.append("super")
    if kind == "method" and flags & 0x0040:
        names.append("bridge")
    if kind == "method" and flags & 0x0080:
        names.append("varargs")
    return names


def descriptor_to_java(descriptor: str) -> str:
    if not descriptor:
        return "unknown"
    index = 0
    depth = 0
    while index < len(descriptor) and descriptor[index] == "[":
        depth += 1
        index += 1
    if index >= len(descriptor):
        return "unknown"
    token = descriptor[index]
    if token == "L":
        end = descriptor.find(";", index)
        name = descriptor[index + 1:end if end >= 0 else len(descriptor)]
        value = name.replace("/", ".")
    else:
        value = PRIMITIVES.get(token, "unknown")
    return value + "[]" * depth


def parse_method_descriptor(descriptor: str) -> tuple[str, list[dict[str, str]]]:
    if not descriptor.startswith("("):
        return "unknown", []
    end = descriptor.find(")")
    if end < 0:
        return "unknown", []
    params: list[dict[str, str]] = []
    index = 1
    while index < end:
        start = index
        depth = 0
        while index < end and descriptor[index] == "[":
            depth += 1
            index += 1
        if index >= end:
            break
        if descriptor[index] == "L":
            semicolon = descriptor.find(";", index)
            if semicolon < 0 or semicolon > end:
                break
            index = semicolon + 1
        else:
            index += 1
        raw = descriptor[start:index]
        params.append({"descriptor": raw, "type": descriptor_to_java(raw), "slot_width": 2 if raw in {"J", "D"} else 1, "array_depth": depth})
    return descriptor_to_java(descriptor[end + 1:]), params


@dataclass
class ConstantPool:
    entries: dict[int, Any] = field(default_factory=dict)

    def utf8(self, index: int) -> str:
        value = self.entries.get(index)
        return value if isinstance(value, str) else ""

    def class_name(self, index: int) -> str:
        entry = self.entries.get(index)
        if isinstance(entry, dict) and entry.get("tag") == CP_CLASS:
            return self.utf8(entry.get("index", 0)).replace("/", ".")
        return ""

    def ref(self, index: int) -> dict[str, Any]:
        entry = self.entries.get(index)
        if not isinstance(entry, dict):
            return {}
        tag = entry.get("tag")
        if tag in {CP_FIELDREF, CP_METHODREF, CP_INTERFACE_METHODREF, CP_NAME_AND_TYPE}:
            name_type = entry
            if tag != CP_NAME_AND_TYPE:
                name_type = self.entries.get(entry.get("b", 0), {})
            return {
                "kind": {CP_FIELDREF: "field", CP_METHODREF: "method", CP_INTERFACE_METHODREF: "interface_method", CP_NAME_AND_TYPE: "name_and_type"}.get(tag, "unknown"),
                "class": self.class_name(entry.get("a", 0)) if tag != CP_NAME_AND_TYPE else "",
                "name": self.utf8(name_type.get("a", 0)),
                "descriptor": self.utf8(name_type.get("b", 0)),
                "tag": tag,
            }
        if tag in {CP_METHOD_HANDLE, CP_METHOD_TYPE, CP_DYNAMIC, CP_INVOKE_DYNAMIC, CP_STRING, CP_CLASS, CP_MODULE, CP_PACKAGE}:
            return {"kind": "constant", "tag": tag, "index": entry.get("index")}
        return {"kind": "constant", "tag": tag, "value": entry}


OPCODE_NAMES: list[str] = [
    "nop", "aconst_null", "iconst_m1", "iconst_0", "iconst_1", "iconst_2", "iconst_3", "iconst_4", "iconst_5",
    "lconst_0", "lconst_1", "fconst_0", "fconst_1", "fconst_2", "dconst_0", "dconst_1",
    "bipush", "sipush", "ldc", "ldc_w", "ldc2_w",
    "iload", "lload", "fload", "dload", "aload",
    "iload_0", "iload_1", "iload_2", "iload_3",
    "lload_0", "lload_1", "lload_2", "lload_3",
    "fload_0", "fload_1", "fload_2", "fload_3",
    "dload_0", "dload_1", "dload_2", "dload_3",
    "aload_0", "aload_1", "aload_2", "aload_3",
    "iaload", "laload", "faload", "daload", "aaload", "baload", "caload", "saload",
    "istore", "lstore", "fstore", "dstore", "astore",
    "istore_0", "istore_1", "istore_2", "istore_3",
    "lstore_0", "lstore_1", "lstore_2", "lstore_3",
    "fstore_0", "fstore_1", "fstore_2", "fstore_3",
    "dstore_0", "dstore_1", "dstore_2", "dstore_3",
    "astore_0", "astore_1", "astore_2", "astore_3",
    "iastore", "lastore", "fastore", "dastore", "aastore", "bastore", "castore", "sastore",
    "pop", "pop2", "dup", "dup_x1", "dup_x2", "dup2", "dup2_x1", "dup2_x2", "swap",
    "iadd", "ladd", "fadd", "dadd", "isub", "lsub", "fsub", "dsub",
    "imul", "lmul", "fmul", "dmul", "idiv", "ldiv", "fdiv", "ddiv",
    "irem", "lrem", "frem", "drem", "ineg", "lneg", "fneg", "dneg",
    "ishl", "lshl", "ishr", "lshr", "iushr", "lushr", "iand", "land", "ior", "lor", "ixor", "lxor",
    "iinc", "i2l", "i2f", "i2d", "l2i", "l2f", "l2d", "f2i", "f2l", "f2d", "d2i", "d2l", "d2f",
    "i2b", "i2c", "i2s", "lcmp", "fcmpl", "fcmpg", "dcmpl", "dcmpg",
    "ifeq", "ifne", "iflt", "ifge", "ifgt", "ifle", "if_icmpeq", "if_icmpne", "if_icmplt", "if_icmpge",
    "if_icmpgt", "if_icmple", "if_acmpeq", "if_acmpne", "goto", "jsr", "ret",
    "tableswitch", "lookupswitch", "ireturn", "lreturn", "freturn", "dreturn", "areturn", "return",
    "getstatic", "putstatic", "getfield", "putfield", "invokevirtual", "invokespecial", "invokestatic",
    "invokeinterface", "invokedynamic", "new", "newarray", "anewarray", "arraylength", "athrow",
    "checkcast", "instanceof", "monitorenter", "monitorexit", "wide", "multianewarray", "ifnull",
    "ifnonnull", "goto_w", "jsr_w", "breakpoint",
]
OPCODE_NAMES.extend([f"impdep{index}" for index in (1, 2)])

CONSTANT_OPS = {
    2: "null", 3: "-1", 4: "0", 5: "1", 6: "2", 7: "3", 8: "4", 9: "5",
    10: "0L", 11: "1L", 12: "0.0f", 13: "1.0f", 14: "2.0f", 15: "0.0d", 16: "1.0d",
}
CONSTANT_OPCODES = {2, *range(3, 17), *range(26, 46), *range(59, 79)}


def opcode_name(opcode: int) -> str:
    return OPCODE_NAMES[opcode] if 0 <= opcode < len(OPCODE_NAMES) else f"op_{opcode:02x}"


def _signed(value: int, bits: int) -> int:
    sign = 1 << (bits - 1)
    return value - (1 << bits) if value & sign else value


def _branch_target(code: bytes, pc: int, offset: int, opcode: int) -> int:
    return pc + offset


def decode_bytecode(data: bytes, pool: ConstantPool | None = None) -> list[dict[str, Any]]:
    instructions: list[dict[str, Any]] = []
    pc = 0
    while pc < len(data):
        start = pc
        opcode = data[pc]
        pc += 1
        mnemonic = opcode_name(opcode)
        operands: list[str] = []
        branch_targets: list[int] = []
        length = 1
        try:
            if opcode == 16:  # bipush
                value = _signed(data[pc], 8)
                operands.append(str(value))
                length = 2
            elif opcode == 17:  # sipush
                value = _signed(_u16(data, pc), 16)
                operands.append(str(value))
                length = 3
            elif opcode == 18:  # ldc
                index = data[pc]
                operands.append(_constant_operand(pool, index))
                length = 2
            elif opcode in {19, 20, 178, 179, 180, 181, 182, 183, 184, 187, 189, 192, 193}:
                index = _u16(data, pc)
                if opcode in {178, 179, 180, 181, 182, 183, 184}:
                    ref = pool.ref(index) if pool else {}
                    operands.append(f"{ref.get('class', '?')}.{ref.get('name', '?')}:{ref.get('descriptor', '?')}")
                elif opcode == 187:
                    operands.append(pool.class_name(index) if pool else str(index))
                elif opcode == 189:
                    operands.append(pool.class_name(index) if pool else str(index))
                elif opcode == 192:
                    operands.append(pool.class_name(index) if pool else str(index))
                elif opcode == 193:
                    operands.append(pool.class_name(index) if pool else str(index))
                else:
                    operands.append(_constant_operand(pool, index))
                length = 3
            elif opcode in {21, 22, 23, 24, 25, 54, 55, 56, 57, 58, 169}:
                operands.append(str(data[pc]))
                length = 2
            elif opcode == 132:  # iinc
                operands.extend([str(data[pc]), str(_signed(data[pc + 1], 8))])
                length = 3
            elif opcode in {153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 198, 199}:
                offset = _signed(_u16(data, pc), 16)
                target = _branch_target(data, start, offset, opcode)
                operands.append(f"-> {target}")
                branch_targets.append(target)
                length = 3
            elif opcode in {200, 201}:
                offset = _signed(_u32(data, pc), 32)
                target = _branch_target(data, start, offset, opcode)
                operands.append(f"-> {target}")
                branch_targets.append(target)
                length = 5
            elif opcode == 185:  # invokeinterface
                index = _u16(data, pc)
                count = data[pc + 2]
                ref = pool.ref(index) if pool else {}
                operands.append(f"{ref.get('class', '?')}.{ref.get('name', '?')}:{ref.get('descriptor', '?')} count={count}")
                length = 5
            elif opcode == 186:  # invokedynamic
                index = _u16(data, pc)
                ref = pool.ref(index) if pool else {}
                operands.append(f"#{index} {ref.get('name', '')}")
                length = 5
            elif opcode == 188:  # newarray
                types = {4: "boolean", 5: "char", 6: "float", 7: "double", 8: "byte", 9: "short", 10: "int", 11: "long"}
                operands.append(types.get(data[pc], f"atype_{data[pc]}"))
                length = 2
            elif opcode == 196:  # wide
                wide_opcode = data[pc]
                pc += 1
                if wide_opcode == 132:
                    index = _u16(data, pc)
                    value = _signed(_u16(data, pc + 2), 16)
                    operands.extend([str(index), str(value)])
                    length = 6
                    mnemonic = "wide iinc"
                else:
                    index = _u16(data, pc)
                    operands.append(str(index))
                    length = 4
                    mnemonic = f"wide {opcode_name(wide_opcode)}"
            elif opcode == 197:  # multianewarray
                index = _u16(data, pc)
                dimensions = data[pc + 2]
                operands.append(f"{pool.class_name(index) if pool else index} dims={dimensions}")
                length = 4
            elif opcode == 170:  # tableswitch
                padding = (4 - (pc % 4)) % 4
                pc += padding
                default = _signed(_u32(data, pc), 32)
                low = _signed(_u32(data, pc + 4), 32)
                high = _signed(_u32(data, pc + 8), 32)
                count = high - low + 1 if high >= low else 0
                offsets = [_signed(_u32(data, pc + 12 + index * 4), 32) for index in range(count)]
                length = pc - start + 12 + count * 4
                operands.extend([f"default -> {start + default}", f"low={low}", f"high={high}"])
                branch_targets.extend(start + offset for offset in offsets)
            elif opcode == 171:  # lookupswitch
                padding = (4 - (pc % 4)) % 4
                pc += padding
                default = _signed(_u32(data, pc), 32)
                pairs = _signed(_u32(data, pc + 4), 32)
                length = pc - start + 8 + max(0, pairs) * 8
                operands.append(f"default -> {start + default}")
                for index in range(max(0, min(pairs, 10000))):
                    key = _signed(_u32(data, pc + 8 + index * 8), 32)
                    offset = _signed(_u32(data, pc + 12 + index * 8), 32)
                    operands.append(f"{key} -> {start + offset}")
                    branch_targets.append(start + offset)
            elif opcode in CONSTANT_OPCODES:
                if opcode in CONSTANT_OPS and opcode >= 2:
                    pass
        except (IndexError, struct.error):
            instructions.append({"pc": start, "opcode": opcode, "mnemonic": mnemonic, "operands": operands, "branch_targets": branch_targets, "length": max(1, length), "truncated": True})
            break
        instructions.append(
            {
                "pc": start,
                "opcode": opcode,
                "mnemonic": mnemonic,
                "operands": operands,
                "branch_targets": branch_targets,
                "length": max(1, length),
                "truncated": False,
            }
        )
        pc = start + max(1, length)
        if pc > len(data):
            break
    return instructions


def _constant_operand(pool: ConstantPool | None, index: int) -> str:
    if pool is None:
        return f"#{index}"
    entry = pool.entries.get(index)
    if isinstance(entry, str):
        return json_string(entry)
    if isinstance(entry, dict) and entry.get("tag") == CP_STRING:
        return json_string(pool.utf8(entry.get("index", 0)))
    if isinstance(entry, int):
        return str(entry)
    if isinstance(entry, dict):
        return f"#{index}"
    return f"#{index}"


def json_string(value: str) -> str:
    escaped = value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n").replace("\r", "\\r").replace("\t", "\\t")
    return f'"{escaped}"'


@dataclass
class ClassAttribute:
    name: str
    data: bytes


def _read_attributes(data: bytes, cursor: int, pool: ConstantPool) -> tuple[list[ClassAttribute], int]:
    count = _u16(data, cursor)
    cursor += 2
    attributes = []
    for _ in range(count):
        name = pool.utf8(_u16(data, cursor))
        length = _u32(data, cursor + 2)
        cursor += 6
        payload = data[cursor:cursor + length]
        if len(payload) != length:
            raise ValueError("truncated attribute")
        cursor += length
        attributes.append(ClassAttribute(name=name, data=payload))
    return attributes, cursor


def _parse_code_attribute(payload: bytes, pool: ConstantPool) -> dict[str, Any]:
    max_stack = _u16(payload, 0)
    max_locals = _u16(payload, 2)
    code_length = _u32(payload, 4)
    code = payload[8:8 + code_length]
    cursor = 8 + code_length
    instructions = decode_bytecode(code, pool)
    exception_table = []
    try:
        exception_count = _u16(payload, cursor)
        cursor += 2
        for _ in range(exception_count):
            start_pc, end_pc, handler_pc, catch_type = struct.unpack_from(">HHHH", payload, cursor)
            cursor += 8
            exception_table.append(
                {
                    "start_pc": start_pc,
                    "end_pc": end_pc,
                    "handler_pc": handler_pc,
                    "catch_type": pool.class_name(catch_type) if catch_type else "any",
                }
            )
    except (IndexError, struct.error):
        pass
    line_numbers = []
    local_variables = []
    try:
        attributes, _ = _read_attributes(payload, cursor, pool)
        for attribute in attributes:
            if attribute.name == "LineNumberTable":
                count = _u16(attribute.data, 0)
                for index in range(count):
                    offset = 2 + index * 4
                    if offset + 4 <= len(attribute.data):
                        line_numbers.append({"start_pc": _u16(attribute.data, offset), "line": _u16(attribute.data, offset + 2)})
            elif attribute.name == "LocalVariableTable":
                count = _u16(attribute.data, 0)
                for index in range(count):
                    offset = 2 + index * 10
                    if offset + 10 <= len(attribute.data):
                        start_pc, length, name_index, descriptor_index, slot = struct.unpack_from(">HHHHH", attribute.data, offset)
                        local_variables.append(
                            {
                                "start_pc": start_pc,
                                "length": length,
                                "name": pool.utf8(name_index),
                                "descriptor": pool.utf8(descriptor_index),
                                "type": descriptor_to_java(pool.utf8(descriptor_index)),
                                "slot": slot,
                            }
                        )
    except (ValueError, struct.error):
        pass
    return {
        "max_stack": max_stack,
        "max_locals": max_locals,
        "code_length": code_length,
        "code_hex": code.hex(),
        "instructions": instructions,
        "exception_table": exception_table,
        "line_numbers": line_numbers,
        "local_variables": local_variables,
        "decode_truncated": len(code) != code_length,
    }


def parse_java_class(path: str | Path) -> dict[str, Any]:
    target = Path(path)
    data = target.read_bytes()
    result: dict[str, Any] = {
        "path": str(target),
        "format": "Java class",
        "size": len(data),
        "methods": [],
        "fields": [],
        "interfaces": [],
        "limitations": [
            "Static class-file parsing only; no JVM loading, verification, linking, execution or decompilation.",
        ],
    }
    if len(data) < 10 or _u32(data, 0) != 0xCAFEBABE:
        raise ValueError("not a Java class file")
    minor, major = _u16(data, 4), _u16(data, 6)
    result["version"] = {"minor": minor, "major": major}
    count = _u16(data, 8)
    cursor = 10
    pool = ConstantPool()
    index = 1
    while index < count:
        if cursor >= len(data):
            raise ValueError("truncated constant pool")
        tag = data[cursor]
        cursor += 1
        if tag == CP_UTF8:
            length = _u16(data, cursor)
            cursor += 2
            if cursor + length > len(data):
                raise ValueError("truncated UTF-8 constant")
            pool.entries[index] = data[cursor:cursor + length].decode("utf-8", errors="replace")
            cursor += length
        elif tag in {CP_INTEGER, CP_FLOAT}:
            pool.entries[index] = _u32(data, cursor)
            cursor += 4
        elif tag in {CP_LONG, CP_DOUBLE}:
            pool.entries[index] = _u64(data, cursor)
            cursor += 8
            index += 1
        elif tag in {CP_CLASS, CP_STRING, CP_METHOD_TYPE, CP_MODULE, CP_PACKAGE}:
            pool.entries[index] = {"tag": tag, "index": _u16(data, cursor)}
            cursor += 2
        elif tag in {CP_FIELDREF, CP_METHODREF, CP_INTERFACE_METHODREF, CP_NAME_AND_TYPE, CP_DYNAMIC, CP_INVOKE_DYNAMIC}:
            pool.entries[index] = {"tag": tag, "a": _u16(data, cursor), "b": _u16(data, cursor + 2)}
            cursor += 4
        elif tag == CP_METHOD_HANDLE:
            pool.entries[index] = {"tag": tag, "kind": data[cursor], "index": _u16(data, cursor + 1)}
            cursor += 3
        else:
            raise ValueError(f"unsupported constant-pool tag {tag}")
        index += 1
    result["constant_pool_entries"] = len(pool.entries)
    access_flags = _u16(data, cursor)
    this_class = _u16(data, cursor + 2)
    super_class = _u16(data, cursor + 4)
    interfaces_count = _u16(data, cursor + 6)
    cursor += 8
    result["access_flags"] = access_flags
    result["access"] = access_names_for("class", access_flags)
    result["this_class"] = pool.class_name(this_class)
    result["name"] = result["this_class"]
    result["package"] = result["this_class"].rpartition(".")[0] if "." in result["this_class"] else ""
    result["simple_name"] = result["this_class"].rpartition(".")[2]
    result["superclass"] = pool.class_name(super_class) if super_class else None
    for _ in range(interfaces_count):
        result["interfaces"].append(pool.class_name(_u16(data, cursor)))
        cursor += 2
    fields_count = _u16(data, cursor)
    cursor += 2
    for _ in range(fields_count):
        flags = _u16(data, cursor)
        name = pool.utf8(_u16(data, cursor + 2))
        descriptor = pool.utf8(_u16(data, cursor + 4))
        field_attributes, cursor = _read_attributes(data, cursor + 6, pool)
        constant_value = None
        for attribute in field_attributes:
            if attribute.name == "ConstantValue" and len(attribute.data) >= 2:
                constant_value = _constant_operand(pool, _u16(attribute.data, 0))
        result["fields"].append(
            {
                "name": name,
                "descriptor": descriptor,
                "type": descriptor_to_java(descriptor),
                "access_flags": flags,
                "access": access_names_for("field", flags),
                "static": bool(flags & 0x0008),
                "final": bool(flags & 0x0010),
                "volatile": bool(flags & 0x0040),
                "transient": bool(flags & 0x0080),
                "synthetic": bool(flags & 0x1000),
                "constant_value": constant_value,
                "signature": next((pool.utf8(_u16(item.data, 0)) for item in field_attributes if item.name == "Signature" and len(item.data) >= 2), None),
            }
        )
    methods_count = _u16(data, cursor)
    cursor += 2
    for _ in range(methods_count):
        flags = _u16(data, cursor)
        name = pool.utf8(_u16(data, cursor + 2))
        descriptor = pool.utf8(_u16(data, cursor + 4))
        method_attributes, cursor = _read_attributes(data, cursor + 6, pool)
        return_type, parameters = parse_method_descriptor(descriptor)
        code = None
        for attribute in method_attributes:
            if attribute.name == "Code":
                try:
                    code = _parse_code_attribute(attribute.data, pool)
                except (ValueError, IndexError, struct.error) as exc:
                    code = {"error": str(exc), "instructions": []}
                break
        result["methods"].append(
            {
                "name": name,
                "descriptor": descriptor,
                "return_type": return_type,
                "parameters": parameters,
                "access_flags": flags,
                "access": access_names_for("method", flags),
                "static": bool(flags & 0x0008),
                "final": bool(flags & 0x0010),
                "synchronized": bool(flags & 0x0020),
                "native": bool(flags & 0x0100),
                "abstract": bool(flags & 0x0400),
                "synthetic": bool(flags & 0x1000),
                "bridge": bool(flags & 0x0040),
                "constructor": name == "<init>",
                "class_initializer": name == "<clinit>",
                "signature": next((pool.utf8(_u16(item.data, 0)) for item in method_attributes if item.name == "Signature" and len(item.data) >= 2), None),
                "code": code,
                "attribute_names": [item.name for item in method_attributes],
            }
        )
    attributes, cursor = _read_attributes(data, cursor, pool)
    for attribute in attributes:
        if attribute.name == "SourceFile" and len(attribute.data) >= 2:
            result["source_file"] = pool.utf8(_u16(attribute.data, 0))
        elif attribute.name == "Signature" and len(attribute.data) >= 2:
            result["signature_attribute"] = pool.utf8(_u16(attribute.data, 0))
        elif attribute.name == "InnerClasses":
            result["inner_classes"] = True
        elif attribute.name == "BootstrapMethods":
            result["bootstrap_methods"] = True
    result["attribute_names"] = [item.name for item in attributes]
    result["method_count"] = len(result["methods"])
    result["field_count"] = len(result["fields"])
    result["is_interface"] = bool(access_flags & 0x0200)
    result["is_abstract"] = bool(access_flags & 0x0400)
    result["is_enum"] = bool(access_flags & 0x4000)
    result["is_annotation"] = bool(access_flags & 0x2000)
    result["is_module"] = bool(access_flags & 0x8000)
    return result
