"""Deterministic minimal APK/AXML fixture builder for validation tests."""

from __future__ import annotations

import io
import struct
import zipfile


def _string_pool(strings: list[str]) -> bytes:
    encoded = []
    for value in strings:
        raw = value.encode("utf-8")
        encoded.append(bytes([len(value), len(raw)]) + raw + b"\0")
    offsets = []
    cursor = 0
    for item in encoded:
        offsets.append(cursor)
        cursor += len(item)
    data = b"".join(encoded)
    header_size = 28
    strings_start = header_size + 4 * len(strings)
    chunk_size = strings_start + len(data)
    header = struct.pack("<HHIIIIII", 0x0001, header_size, chunk_size, len(strings), 0, 0x100, strings_start, 0)
    return header + b"".join(struct.pack("<I", offset) for offset in offsets) + data


def _start_tag(name_index: int, attributes: list[tuple[int, int, int, int]]) -> bytes:
    attribute_size = 20
    ext_size = 20
    chunk_size = 16 + ext_size + attribute_size * len(attributes)
    header = struct.pack("<HHIII", 0x0102, 16, chunk_size, 0, 0xFFFFFFFF)
    ext = struct.pack("<IIHHHHHH", 0xFFFFFFFF, name_index, ext_size, attribute_size, len(attributes), 0, 0, 0)
    body = b""
    for attr_name, raw_value, data_type, data in attributes:
        body += struct.pack("<III", 0xFFFFFFFF, attr_name, raw_value)
        body += struct.pack("<HBB", attribute_size, 0, data_type) + struct.pack("<I", data)
    return header + ext + body


def _end_tag(name_index: int) -> bytes:
    return struct.pack("<HHIII", 0x0103, 16, 24, 0, 0xFFFFFFFF) + struct.pack("<II", 0xFFFFFFFF, name_index)


def build_minimal_android_manifest(
    package: str = "com.example.app",
    activity: str = ".MainActivity",
    version_code: int = 1,
) -> bytes:
    strings = ["manifest", "package", package, "versionCode", "application", "activity", activity, "name"]
    index = {value: position for position, value in enumerate(strings)}
    manifest = _start_tag(
        index["manifest"],
        [
            (index["package"], index[package], 0x03, index[package]),
            (index["versionCode"], 0xFFFFFFFF, 0x10, version_code),
        ],
    )
    application = _start_tag(index["application"], [])
    activity_tag = _start_tag(index["activity"], [(index["name"], index[activity], 0x03, index[activity])])
    body = manifest + application + activity_tag + _end_tag(index["activity"]) + _end_tag(index["application"]) + _end_tag(index["manifest"])
    pool = _string_pool(strings)
    total = 8 + len(pool) + len(body)
    return struct.pack("<HHI", 0x0003, 8, total) + pool + body


def build_minimal_apk(
    *,
    manifest: bytes,
    dex: bytes | None = None,
    native_library: bytes | None = None,
    native_name: str = "libjni_demo.so",
) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("AndroidManifest.xml", manifest)
        if dex is not None:
            archive.writestr("classes.dex", dex)
        if native_library is not None:
            archive.writestr(f"lib/arm64-v8a/{native_name}", native_library)
        archive.writestr("META-INF/CERT.RSA", b"fixture-not-a-real-signature")
        archive.writestr("resources.arsc", b"fixture")
    return buffer.getvalue()


__all__ = ["build_minimal_android_manifest", "build_minimal_apk"]
