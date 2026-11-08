"""Read-only DEX parser and Dalvik bytecode decoder.

The parser resolves strings, types, prototypes, fields, methods, class data,
code items, try/catch tables and a practical subset of Dalvik instructions.  It
does not execute Dalvik/ART code, deoptimize or claim exact source recovery.
"""

from __future__ import annotations

import struct
from pathlib import Path
from typing import Any

from nexus.formats.java_class import access_names_for


def _uleb128(data: bytes, offset: int) -> tuple[int, int]:
    result = 0
    shift = 0
    for _ in range(5):
        if offset >= len(data):
            raise ValueError("truncated uleb128")
        value = data[offset]
        offset += 1
        result |= (value & 0x7F) << shift
        if not value & 0x80:
            return result, offset
        shift += 7
    raise ValueError("uleb128 too long")


def _sleb128(data: bytes, offset: int) -> tuple[int, int]:
    result = 0
    shift = 0
    value = 0
    for _ in range(5):
        if offset >= len(data):
            raise ValueError("truncated sleb128")
        value = data[offset]
        offset += 1
        result |= (value & 0x7F) << shift
        shift += 7
        if not value & 0x80:
            if value & 0x40:
                result |= -(1 << shift)
            return result, offset
    raise ValueError("sleb128 too long")


def _mutf8(data: bytes, offset: int) -> tuple[str, int]:
    length, cursor = _uleb128(data, offset)
    end = cursor + length
    if end > len(data):
        end = len(data)
    raw = data[cursor:end]
    try:
        return raw.decode("utf-8", errors="replace"), end
    except Exception:
        return raw.decode("latin-1", errors="replace"), end


def _type_name(descriptor: str) -> str:
    if descriptor.startswith("L") and descriptor.endswith(";"):
        return descriptor[1:-1].replace("/", ".")
    return descriptor


def _signed(value: int, bits: int) -> int:
    sign = 1 << (bits - 1)
    return value - (1 << bits) if value & sign else value


# Dalvik opcodes: name, format.  Formats determine the encoded length.
DALVIK_OPS: dict[int, tuple[str, str]] = {
    0x00: ("nop", "10x"), 0x01: ("move", "12x"), 0x02: ("move/from16", "22x"), 0x03: ("move/16", "32x"),
    0x04: ("move-wide", "12x"), 0x05: ("move-wide/from16", "22x"), 0x06: ("move-wide/16", "32x"),
    0x07: ("move-object", "12x"), 0x08: ("move-object/from16", "22x"), 0x09: ("move-object/16", "32x"),
    0x0A: ("move-result", "11x"), 0x0B: ("move-result-wide", "11x"), 0x0C: ("move-result-object", "11x"),
    0x0D: ("move-exception", "11x"), 0x0E: ("return-void", "10x"), 0x0F: ("return", "11x"),
    0x10: ("return-wide", "11x"), 0x11: ("return-object", "11x"), 0x12: ("const/4", "11n"),
    0x13: ("const/16", "21s"), 0x14: ("const", "31i"), 0x15: ("const/high16", "21h"),
    0x16: ("const-wide/16", "21s"), 0x17: ("const-wide/32", "31i"), 0x18: ("const-wide", "51l"),
    0x19: ("const-wide/high16", "21h"), 0x1A: ("const-string", "21c"), 0x1B: ("const-string/jumbo", "31c"),
    0x1C: ("const-class", "21c"), 0x1D: ("monitor-enter", "11x"), 0x1E: ("monitor-exit", "11x"),
    0x1F: ("check-cast", "21c"), 0x20: ("instance-of", "22c"), 0x21: ("array-length", "12x"),
    0x22: ("new-instance", "21c"), 0x23: ("new-array", "22c"), 0x24: ("filled-new-array", "35c"),
    0x25: ("filled-new-array/range", "3rc"), 0x26: ("fill-array-data", "31t"), 0x27: ("throw", "11x"),
    0x28: ("goto", "10t"), 0x29: ("goto/16", "20t"), 0x2A: ("goto/32", "30t"),
    0x2B: ("packed-switch", "31t"), 0x2C: ("sparse-switch", "31t"),
    0x2D: ("cmpl-float", "23x"), 0x2E: ("cmpg-float", "23x"), 0x2F: ("cmpl-double", "23x"),
    0x30: ("cmpg-double", "23x"), 0x31: ("cmp-long", "23x"),
    0x32: ("if-eq", "22t"), 0x33: ("if-ne", "22t"), 0x34: ("if-lt", "22t"), 0x35: ("if-ge", "22t"),
    0x36: ("if-gt", "22t"), 0x37: ("if-le", "22t"), 0x38: ("if-eqz", "21t"), 0x39: ("if-nez", "21t"),
    0x3A: ("if-ltz", "21t"), 0x3B: ("if-gez", "21t"), 0x3C: ("if-gtz", "21t"), 0x3D: ("if-lez", "21t"),
}
for opcode, name in enumerate(
    ["aget", "aget-wide", "aget-object", "aget-boolean", "aget-byte", "aget-char", "aget-short",
     "aput", "aput-wide", "aput-object", "aput-boolean", "aput-byte", "aput-char", "aput-short"], start=0x44
):
    DALVIK_OPS[opcode] = (name, "23x")
for opcode, name in enumerate(
    ["iget", "iget-wide", "iget-object", "iget-boolean", "iget-byte", "iget-char", "iget-short",
     "iput", "iput-wide", "iput-object", "iput-boolean", "iput-byte", "iput-char", "iput-short"], start=0x52
):
    DALVIK_OPS[opcode] = (name, "22c")
for opcode, name in enumerate(
    ["sget", "sget-wide", "sget-object", "sget-boolean", "sget-byte", "sget-char", "sget-short",
     "sput", "sput-wide", "sput-object", "sput-boolean", "sput-byte", "sput-char", "sput-short"], start=0x60
):
    DALVIK_OPS[opcode] = (name, "21c")
DALVIK_OPS.update({
    0x6E: ("invoke-virtual", "35c"), 0x6F: ("invoke-super", "35c"), 0x70: ("invoke-direct", "35c"),
    0x71: ("invoke-static", "35c"), 0x72: ("invoke-interface", "35c"),
    0x74: ("invoke-virtual/range", "3rc"), 0x75: ("invoke-super/range", "3rc"),
    0x76: ("invoke-direct/range", "3rc"), 0x77: ("invoke-static/range", "3rc"),
    0x78: ("invoke-interface/range", "3rc"),
})
for opcode, name in enumerate(
    ["neg-int", "not-int", "neg-long", "not-long", "neg-float", "neg-double",
     "int-to-long", "int-to-float", "int-to-double", "long-to-int", "long-to-float", "long-to-double",
     "float-to-int", "float-to-long", "float-to-double", "double-to-int", "double-to-long", "double-to-float",
     "int-to-byte", "int-to-char", "int-to-short"], start=0x7B
):
    DALVIK_OPS[opcode] = (name, "12x")
for opcode, name in enumerate(
    ["add-int", "sub-int", "mul-int", "div-int", "rem-int", "and-int", "or-int", "xor-int",
     "shl-int", "shr-int", "ushr-int", "add-long", "sub-long", "mul-long", "div-long", "rem-long",
     "and-long", "or-long", "xor-long", "shl-long", "shr-long", "ushr-long",
     "add-float", "sub-float", "mul-float", "div-float", "rem-float",
     "add-double", "sub-double", "mul-double", "div-double", "rem-double"], start=0x90
):
    DALVIK_OPS[opcode] = (name, "23x")
for opcode, name in enumerate(
    [f"{name}/2addr" for name in
     ["add-int", "sub-int", "mul-int", "div-int", "rem-int", "and-int", "or-int", "xor-int",
      "shl-int", "shr-int", "ushr-int", "add-long", "sub-long", "mul-long", "div-long", "rem-long",
      "and-long", "or-long", "xor-long", "shl-long", "shr-long", "ushr-long",
      "add-float", "sub-float", "mul-float", "div-float", "rem-float",
      "add-double", "sub-double", "mul-double", "div-double", "rem-double"]], start=0xB0
):
    DALVIK_OPS[opcode] = (name, "12x")
for opcode, name in enumerate(
    ["add-int/lit16", "rsub-int", "mul-int/lit16", "div-int/lit16", "rem-int/lit16",
     "and-int/lit16", "or-int/lit16", "xor-int/lit16"], start=0xD0
):
    DALVIK_OPS[opcode] = (name, "22s")
for opcode, name in enumerate(
    ["add-int/lit8", "rsub-int/lit8", "mul-int/lit8", "div-int/lit8", "rem-int/lit8",
     "and-int/lit8", "or-int/lit8", "xor-int/lit8", "shl-int/lit8", "shr-int/lit8", "ushr-int/lit8"], start=0xD8
):
    DALVIK_OPS[opcode] = (name, "22b")
DALVIK_OPS.update({
    0xFA: ("invoke-polymorphic", "45cc"), 0xFB: ("invoke-polymorphic/range", "4rcc"),
    0xFC: ("invoke-custom", "35c"), 0xFD: ("invoke-custom/range", "3rc"),
    0xFE: ("const-method-handle", "21c"), 0xFF: ("const-method-type", "21c"),
})

FORMAT_LENGTH = {
    "10x": 1, "12x": 1, "11n": 1, "11x": 1, "10t": 1,
    "20t": 2, "22x": 2, "21t": 2, "21s": 2, "21h": 2, "21c": 2, "23x": 2, "22b": 2, "22t": 2, "22s": 2, "22c": 2,
    "32x": 3, "30t": 3, "31t": 3, "31i": 3, "31c": 3, "35c": 3, "3rc": 3,
    "45cc": 4, "4rcc": 4, "51l": 5,
}


def _operand_index(kind: str | None, index: int, strings, types, fields, methods, protos) -> str:
    if kind == "string" and 0 <= index < len(strings):
        return strings[index]
    if kind == "type" and 0 <= index < len(types):
        return types[index]
    if kind == "field" and 0 <= index < len(fields):
        field = fields[index]
        return f"{field['class']}.{field['name']}:{field['descriptor']}"
    if kind == "method" and 0 <= index < len(methods):
        method = methods[index]
        return f"{method['class']}.{method['name']}:{method['descriptor']}"
    if kind == "proto" and 0 <= index < len(protos):
        return protos[index]["descriptor"]
    return f"#{index}"


def decode_dalvik(
    insns: bytes,
    *,
    strings: list[str] | None = None,
    types: list[str] | None = None,
    fields: list[dict[str, Any]] | None = None,
    methods: list[dict[str, Any]] | None = None,
    protos: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    strings = strings or []
    types = types or []
    fields = fields or []
    methods = methods or []
    protos = protos or []
    units = list(struct.unpack("<" + "H" * (len(insns) // 2), insns[: len(insns) // 2 * 2]))
    instructions = []
    index = 0
    while index < len(units):
        start = index
        unit = units[index]
        opcode = unit & 0xFF
        high = (unit >> 8) & 0xFF
        name, fmt = DALVIK_OPS.get(opcode, (f"op_{opcode:02x}", "10x"))
        length = FORMAT_LENGTH.get(fmt, 1)
        operands: list[str] = []
        registers: list[int] = []
        branch_targets: list[int] = []
        index_kind = None
        index_value = None
        try:
            if fmt == "12x":
                a, b = high & 0xF, high >> 4
                operands = [f"v{a}", f"v{b}"]
                registers = [a, b]
            elif fmt == "11n":
                a, literal = high & 0xF, _signed(high >> 4, 4)
                operands = [f"v{a}", f"#{literal}"]
                registers = [a]
            elif fmt == "11x":
                operands = [f"v{high}"]
                registers = [high]
            elif fmt == "10t":
                offset = _signed(high, 8)
                operands = [f"-> {start + offset}"]
                branch_targets = [start + offset]
            elif fmt == "20t":
                offset = _signed(units[index + 1], 16) if index + 1 < len(units) else 0
                operands = [f"-> {start + offset}"]
                branch_targets = [start + offset]
            elif fmt == "22x":
                operands = [f"v{high}", f"v{units[index + 1]}"]
                registers = [high, units[index + 1]]
            elif fmt == "21t":
                offset = _signed(units[index + 1], 16)
                operands = [f"v{high}", f"-> {start + offset}"]
                registers = [high]
                branch_targets = [start + offset]
            elif fmt == "21s":
                operands = [f"v{high}", f"#{_signed(units[index + 1], 16)}"]
                registers = [high]
            elif fmt == "21h":
                value = _signed(units[index + 1], 16) << 16
                operands = [f"v{high}", f"#{value}"]
                registers = [high]
            elif fmt == "21c":
                index_kind, index_value = ({0x1A: "string", 0x1C: "type"}.get(opcode, "type"), units[index + 1])
                operands = [f"v{high}", _operand_index(index_kind, index_value, strings, types, fields, methods, protos)]
                registers = [high]
            elif fmt == "23x":
                a, b, c = high, units[index + 1] & 0xFF, units[index + 1] >> 8
                operands = [f"v{a}", f"v{b}", f"v{c}"]
                registers = [a, b, c]
            elif fmt == "22b":
                a, b, literal = high, units[index + 1] & 0xFF, _signed(units[index + 1] >> 8, 8)
                operands = [f"v{a}", f"v{b}", f"#{literal}"]
                registers = [a, b]
            elif fmt == "22t":
                a, b, offset = high & 0xF, high >> 4, _signed(units[index + 1], 16)
                operands = [f"v{a}", f"v{b}", f"-> {start + offset}"]
                registers = [a, b]
                branch_targets = [start + offset]
            elif fmt == "22s":
                a, b, literal = high & 0xF, high >> 4, _signed(units[index + 1], 16)
                operands = [f"v{a}", f"v{b}", f"#{literal}"]
                registers = [a, b]
            elif fmt == "22c":
                a, b, field_index = high & 0xF, high >> 4, units[index + 1]
                field = fields[field_index] if 0 <= field_index < len(fields) else {}
                operands = [f"v{a}", f"v{b}", f"{field.get('class', '?')}.{field.get('name', '?')}:{field.get('descriptor', '?')}"]
                registers = [a, b]
                index_kind, index_value = "field", field_index
            elif fmt == "32x":
                operands = [f"v{units[index + 1]}", f"v{units[index + 2]}"]
                registers = [units[index + 1], units[index + 2]]
            elif fmt == "30t":
                offset = _signed((units[index + 1] << 16) | units[index + 2], 32)
                operands = [f"-> {start + offset}"]
                branch_targets = [start + offset]
            elif fmt == "31t":
                offset = _signed((units[index + 1] << 16) | units[index + 2], 32)
                operands = [f"v{high}", f"-> {start + offset}"]
                registers = [high]
                branch_targets = [start + offset]
            elif fmt == "31i":
                value = _signed((units[index + 1] << 16) | units[index + 2], 32)
                operands = [f"v{high}", f"#{value}"]
                registers = [high]
            elif fmt == "31c":
                index_kind, index_value = ("string", (units[index + 1] << 16) | units[index + 2])
                operands = [f"v{high}", _operand_index(index_kind, index_value, strings, types, fields, methods, protos)]
                registers = [high]
            elif fmt == "35c":
                count = (high >> 4) & 0xF
                g = high & 0xF
                method_index = units[index + 1]
                register_word = units[index + 2]
                regs = [register_word & 0xF, (register_word >> 4) & 0xF, (register_word >> 8) & 0xF, (register_word >> 12) & 0xF, g]
                registers = regs[:count]
                operands = [", ".join(f"v{reg}" for reg in registers), _operand_index("method", method_index, strings, types, fields, methods, protos)]
                index_kind, index_value = "method", method_index
            elif fmt == "3rc":
                count = high
                method_index = units[index + 1]
                first = units[index + 2]
                registers = list(range(first, first + count))
                operands = [", ".join(f"v{reg}" for reg in registers), _operand_index("method", method_index, strings, types, fields, methods, protos)]
                index_kind, index_value = "method", method_index
            elif fmt == "45cc":
                count = (high >> 4) & 0xF
                g = high & 0xF
                method_index = units[index + 1]
                proto_index = units[index + 2]
                register_word = units[index + 3]
                registers = [register_word & 0xF, (register_word >> 4) & 0xF, (register_word >> 8) & 0xF, (register_word >> 12) & 0xF, g][:count]
                operands = [", ".join(f"v{reg}" for reg in registers), _operand_index("method", method_index, strings, types, fields, methods, protos), _operand_index("proto", proto_index, strings, types, fields, methods, protos)]
                index_kind, index_value = "method", method_index
            elif fmt == "4rcc":
                count = high
                method_index = units[index + 1]
                proto_index = units[index + 2]
                first = units[index + 3]
                registers = list(range(first, first + count))
                operands = [", ".join(f"v{reg}" for reg in registers), _operand_index("method", method_index, strings, types, fields, methods, protos), _operand_index("proto", proto_index, strings, types, fields, methods, protos)]
                index_kind, index_value = "method", method_index
            elif fmt == "51l":
                value = _signed(
                    units[index + 1] | (units[index + 2] << 16) | (units[index + 3] << 32) | (units[index + 4] << 48), 64
                )
                operands = [f"v{high}", f"#{value}"]
                registers = [high]
            else:
                operands = []
        except (IndexError, ValueError):
            instructions.append({"pc": start, "opcode": opcode, "mnemonic": name, "operands": operands, "registers": registers, "length": length, "branch_targets": branch_targets, "truncated": True})
            break
        instructions.append(
            {
                "pc": start,
                "opcode": opcode,
                "mnemonic": name,
                "operands": operands,
                "registers": registers,
                "length": length,
                "branch_targets": branch_targets,
                "index_kind": index_kind,
                "index": index_value,
                "truncated": False,
            }
        )
        index += length
    return instructions


def _parse_code_item(data: bytes, offset: int, strings, types, fields, methods, protos) -> dict[str, Any]:
    registers_size, ins_size, outs_size, tries_size = struct.unpack_from("<HHHH", data, offset)
    debug_info_off, insns_size = struct.unpack_from("<II", data, offset + 8)
    cursor = offset + 16
    insns = data[cursor:cursor + insns_size * 2]
    cursor += insns_size * 2
    if tries_size and insns_size % 2:
        cursor += 2
    tries = []
    for _ in range(min(tries_size, 10000)):
        if cursor + 8 > len(data):
            break
        start_addr, insn_count, handler_off = struct.unpack_from("<IHH", data, cursor)
        cursor += 8
        tries.append({"start_addr": start_addr, "insn_count": insn_count, "handler_off": handler_off})
    instructions = decode_dalvik(insns, strings=strings, types=types, fields=fields, methods=methods, protos=protos)
    return {
        "registers_size": registers_size,
        "ins_size": ins_size,
        "outs_size": outs_size,
        "tries_size": tries_size,
        "debug_info_off": debug_info_off,
        "insns_size": insns_size,
        "instructions": instructions,
        "tries": tries,
        "decode_truncated": len(insns) != insns_size * 2,
    }


def parse_dex(path: str | Path) -> dict[str, Any]:
    target = Path(path)
    data = target.read_bytes()
    result: dict[str, Any] = {
        "path": str(target),
        "format": "DEX",
        "size": len(data),
        "classes": [],
        "limitations": [
            "Static DEX table parsing only; no Dalvik execution, ART verification, deoptimization or exact source recovery.",
            "Dalvik bytecode is decoded for common opcodes; unknown opcodes remain explicit.",
        ],
    }
    if not data.startswith(b"dex\n") or len(data) < 112:
        raise ValueError("not a DEX image")
    result["version"] = data[4:7].decode("ascii", errors="replace")
    checksum = data[8:12].hex()
    file_size = struct.unpack_from("<I", data, 32)[0]
    header_size = struct.unpack_from("<I", data, 36)[0]
    string_ids_size, string_ids_off = struct.unpack_from("<II", data, 56)
    type_ids_size, type_ids_off = struct.unpack_from("<II", data, 64)
    proto_ids_size, proto_ids_off = struct.unpack_from("<II", data, 72)
    field_ids_size, field_ids_off = struct.unpack_from("<II", data, 80)
    method_ids_size, method_ids_off = struct.unpack_from("<II", data, 88)
    class_defs_size, class_defs_off = struct.unpack_from("<II", data, 96)
    result["header"] = {
        "checksum": checksum,
        "file_size": file_size,
        "header_size": header_size,
        "string_ids_size": string_ids_size,
        "type_ids_size": type_ids_size,
        "proto_ids_size": proto_ids_size,
        "field_ids_size": field_ids_size,
        "method_ids_size": method_ids_size,
        "class_defs_size": class_defs_size,
    }
    strings: list[str] = []
    for index in range(min(string_ids_size, 500000)):
        offset = struct.unpack_from("<I", data, string_ids_off + index * 4)[0]
        if offset >= len(data):
            strings.append("")
            continue
        value, _ = _mutf8(data, offset)
        strings.append(value)
    types: list[str] = []
    for index in range(min(type_ids_size, 500000)):
        descriptor_idx = struct.unpack_from("<I", data, type_ids_off + index * 4)[0]
        types.append(strings[descriptor_idx] if descriptor_idx < len(strings) else "")
    protos: list[dict[str, Any]] = []
    for index in range(min(proto_ids_size, 200000)):
        shorty_idx, return_type_idx, parameters_off = struct.unpack_from("<III", data, proto_ids_off + index * 12)
        parameter_types = []
        if parameters_off:
            try:
                count = struct.unpack_from("<I", data, parameters_off)[0]
                cursor = parameters_off + 4
                for _ in range(min(count, 10000)):
                    type_idx = struct.unpack_from("<H", data, cursor)[0]
                    cursor += 2
                    if type_idx < len(types):
                        parameter_types.append(types[type_idx])
            except (struct.error, ValueError):
                pass
        return_type = _type_name(types[return_type_idx]) if return_type_idx < len(types) else "unknown"
        protos.append(
            {
                "proto_id": index,
                "shorty": strings[shorty_idx] if shorty_idx < len(strings) else "",
                "return_type": return_type,
                "parameters": parameter_types,
                "descriptor": "(" + "".join(parameter_types) + ")" + return_type,
            }
        )
    fields: list[dict[str, Any]] = []
    for index in range(min(field_ids_size, 500000)):
        class_idx, type_idx, name_idx = struct.unpack_from("<HHI", data, field_ids_off + index * 8)
        fields.append(
            {
                "field_id": index,
                "class": _type_name(types[class_idx]) if class_idx < len(types) else "",
                "type": _type_name(types[type_idx]) if type_idx < len(types) else "unknown",
                "descriptor": types[type_idx] if type_idx < len(types) else "?",
                "name": strings[name_idx] if name_idx < len(strings) else "",
            }
        )
    methods: list[dict[str, Any]] = []
    for index in range(min(method_ids_size, 500000)):
        class_idx, proto_idx, name_idx = struct.unpack_from("<HHI", data, method_ids_off + index * 8)
        proto = protos[proto_idx] if proto_idx < len(protos) else {"return_type": "unknown", "parameters": [], "descriptor": "()?"}
        methods.append(
            {
                "method_id": index,
                "class": _type_name(types[class_idx]) if class_idx < len(types) else "",
                "name": strings[name_idx] if name_idx < len(strings) else "",
                "proto_idx": proto_idx,
                "descriptor": proto.get("descriptor", "()?"),
                "return_type": proto.get("return_type", "unknown"),
                "parameters": proto.get("parameters", []),
            }
        )
    method_by_class: dict[str, list[dict[str, Any]]] = {}
    for method in methods:
        method_by_class.setdefault(method["class"], []).append(method)
    for index in range(min(class_defs_size, 200000)):
        offset = class_defs_off + index * 32
        if offset + 32 > len(data):
            break
        (
            class_idx, access_flags, superclass_idx, interfaces_off, source_file_idx,
            annotations_off, class_data_off, static_values_off,
        ) = struct.unpack_from("<IIIIIIII", data, offset)
        class_name = _type_name(types[class_idx]) if class_idx < len(types) else ""
        super_name = _type_name(types[superclass_idx]) if superclass_idx < len(types) and superclass_idx != 0xFFFFFFFF else None
        source_file = strings[source_file_idx] if source_file_idx < len(strings) and source_file_idx != 0xFFFFFFFF else None
        interfaces = []
        if interfaces_off:
            try:
                count, cursor = _uleb128(data, interfaces_off)
                for _ in range(min(count, 10000)):
                    type_idx, cursor = _uleb128(data, cursor)
                    if type_idx < len(types):
                        interfaces.append(_type_name(types[type_idx]))
            except ValueError:
                pass
        class_fields: list[dict[str, Any]] = []
        class_methods: list[dict[str, Any]] = []
        if class_data_off:
            try:
                cursor = class_data_off
                static_fields, cursor = _uleb128(data, cursor)
                instance_fields, cursor = _uleb128(data, cursor)
                direct_methods, cursor = _uleb128(data, cursor)
                virtual_methods, cursor = _uleb128(data, cursor)
                field_index = 0
                for field_count, is_static in ((static_fields, True), (instance_fields, False)):
                    for _ in range(min(field_count, 50000)):
                        diff, cursor = _uleb128(data, cursor)
                        access, cursor = _uleb128(data, cursor)
                        field_index += diff
                        field = fields[field_index] if field_index < len(fields) else {}
                        class_fields.append(
                            {
                                "name": field.get("name", ""),
                                "type": _type_name(field.get("type", "?")),
                                "descriptor": field.get("descriptor", "?"),
                                "access_flags": access,
                                "access": access_names_for("field", access),
                                "static": is_static,
                                "final": bool(access & 0x10),
                                "volatile": bool(access & 0x40),
                                "transient": bool(access & 0x80),
                            }
                        )
                method_index = 0
                for method_count, is_direct in ((direct_methods, True), (virtual_methods, False)):
                    for _ in range(min(method_count, 50000)):
                        diff, cursor = _uleb128(data, cursor)
                        access, cursor = _uleb128(data, cursor)
                        code_off, cursor = _uleb128(data, cursor)
                        method_index += diff
                        method = methods[method_index] if method_index < len(methods) else {}
                        code = None
                        if code_off:
                            try:
                                code = _parse_code_item(data, code_off, strings, types, fields, methods, protos)
                            except (ValueError, struct.error) as exc:
                                code = {"error": str(exc), "instructions": []}
                        class_methods.append(
                            {
                                "name": method.get("name", ""),
                                "descriptor": method.get("descriptor", "()?"),
                                "return_type": method.get("return_type", "unknown"),
                                "parameters": [{"descriptor": item, "type": _type_name(item), "slot_width": 2 if item in {"J", "D"} else 1, "array_depth": item.count("[")} for item in method.get("parameters", [])],
                                "access_flags": access,
                                "access": access_names_for("method", access),
                                "static": bool(access & 0x8),
                                "final": bool(access & 0x10),
                                "synchronized": bool(access & 0x20),
                                "native": bool(access & 0x100),
                                "abstract": bool(access & 0x400),
                                "synthetic": bool(access & 0x1000),
                                "constructor": method.get("name") == "<init>",
                                "class_initializer": method.get("name") == "<clinit>",
                                "direct": is_direct,
                                "code": code,
                            }
                        )
            except ValueError:
                result["limitations"].append(f"class_data truncated at class {class_name}")
        result["classes"].append(
            {
                "class_id": index,
                "name": class_name,
                "simple_name": class_name.rpartition(".")[2],
                "package": class_name.rpartition(".")[0] if "." in class_name else "",
                "superclass": super_name,
                "interfaces": interfaces,
                "access_flags": access_flags,
                "access": access_names_for("class", access_flags),
                "is_interface": bool(access_flags & 0x200),
                "is_abstract": bool(access_flags & 0x400),
                "is_enum": bool(access_flags & 0x4000),
                "is_annotation": bool(access_flags & 0x2000),
                "source_file": source_file,
                "fields": class_fields,
                "methods": class_methods,
                "method_count": len(class_methods),
                "field_count": len(class_fields),
            }
        )
    result["method_count"] = len(methods)
    result["field_count"] = len(fields)
    result["class_count"] = len(result["classes"])
    result["strings_count"] = len(strings)
    result["types_count"] = len(types)
    return result
