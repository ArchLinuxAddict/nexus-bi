"""Deterministic minimal DEX fixture builder for tests and dataset pilots.

The generated image contains one class and one straight-line static method.  It
is intentionally small: it validates table, prototype and bytecode handling
without requiring d8/ART or external Android tooling.
"""

from __future__ import annotations

import hashlib
import struct
import zlib


def _uleb(value: int) -> bytes:
    result = b""
    while True:
        byte = value & 0x7F
        value >>= 7
        if value:
            result += bytes([byte | 0x80])
        else:
            return result + bytes([byte])


def build_minimal_dex(
    *,
    class_descriptor: str = "Lcom/example/Add;",
    method_name: str = "add",
    method_descriptor: str = "(II)I",
    source_file: str = "Add.java",
    insns: bytes | None = None,
    access_flags: int = 0x9,
) -> bytes:
    strings = [class_descriptor, "Ljava/lang/Object;", method_name, method_descriptor, "I", "II", source_file]
    string_data = b"".join(_uleb(len(item)) + item.encode() + b"\0" for item in strings)
    string_ids_off = 112
    string_data_off = string_ids_off + 4 * len(strings)
    type_ids_off = string_data_off + len(string_data)
    types = [0, 1, 4]
    proto_ids_off = type_ids_off + 4 * len(types)
    type_list_off = proto_ids_off + 12
    type_list = struct.pack("<I", 2) + struct.pack("<HH", 2, 2)
    field_ids_off = type_list_off + len(type_list)
    method_ids_off = field_ids_off
    method_ids = struct.pack("<HHI", 0, 0, 2)
    class_defs_off = method_ids_off + len(method_ids)
    data_off = class_defs_off + 32
    if insns is None:
        insns = struct.pack("<HHH", 0x0090, 0x0100, 0x000F)
    code_item = struct.pack("<HHHHII", 2, 2, 0, 0, 0, len(insns) // 2) + insns
    class_data_off = data_off + len(code_item)
    class_data = _uleb(0) + _uleb(0) + _uleb(1) + _uleb(0) + _uleb(0) + _uleb(access_flags) + _uleb(data_off)
    data = code_item + class_data
    class_def = struct.pack("<IIIIIIII", 0, 1, 1, 0, 6, 0, class_data_off, 0)
    file_size = data_off + len(data)
    header = bytearray(112)
    header[0:8] = b"dex\n035\0"
    struct.pack_into("<I", header, 32, file_size)
    struct.pack_into("<I", header, 36, 112)
    struct.pack_into("<I", header, 40, 0x12345678)
    struct.pack_into("<II", header, 56, len(strings), string_ids_off)
    struct.pack_into("<II", header, 64, len(types), type_ids_off)
    struct.pack_into("<II", header, 72, 1, proto_ids_off)
    struct.pack_into("<II", header, 80, 0, field_ids_off)
    struct.pack_into("<II", header, 88, 1, method_ids_off)
    struct.pack_into("<II", header, 96, 1, class_defs_off)
    struct.pack_into("<II", header, 104, len(data), data_off)
    string_offsets = []
    cursor = string_data_off
    for item in strings:
        string_offsets.append(cursor)
        cursor += len(_uleb(len(item)) + item.encode() + b"\0")
    blob = bytearray(
        bytes(header)
        + b"".join(struct.pack("<I", offset) for offset in string_offsets)
        + string_data
        + b"".join(struct.pack("<I", item) for item in types)
        + struct.pack("<III", 5, 2, type_list_off)
        + type_list
        + method_ids
        + class_def
        + data
    )
    if len(blob) != file_size:
        raise ValueError("internal DEX size mismatch")
    blob[12:32] = hashlib.sha1(bytes(blob[32:])).digest()
    struct.pack_into("<I", blob, 8, zlib.adler32(bytes(blob[12:])) & 0xFFFFFFFF)
    return bytes(blob)


__all__ = ["build_minimal_dex"]
