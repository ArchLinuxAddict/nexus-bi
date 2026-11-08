import struct

from nexus.formats.android import manifest_summary, parse_axml


def _string_pool(strings):
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


def _start_tag(name_index, attributes):
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


def _end_tag(name_index):
    chunk_size = 24
    header = struct.pack("<HHIII", 0x0103, 16, chunk_size, 0, 0xFFFFFFFF)
    ext = struct.pack("<II", 0xFFFFFFFF, name_index)
    return header + ext


def test_binary_manifest_decodes_package_and_components():
    strings = ["manifest", "package", "com.example.app", "versionCode", "application", "activity", ".MainActivity", "android.permission.INTERNET", "name"]
    index = {value: position for position, value in enumerate(strings)}
    manifest = _start_tag(
        index["manifest"],
        [
            (index["package"], index["com.example.app"], 0x03, index["com.example.app"]),
            (index["versionCode"], 0xFFFFFFFF, 0x10, 7),
        ],
    )
    application = _start_tag(index["application"], [])
    activity = _start_tag(index["activity"], [(index["name"], index[".MainActivity"], 0x03, index[".MainActivity"])])
    end_activity = _end_tag(index["activity"])
    end_application = _end_tag(index["application"])
    end_manifest = _end_tag(index["manifest"])
    pool = _string_pool(strings)
    body = manifest + application + activity + end_activity + end_application + end_manifest
    total = 8 + len(pool) + len(body)
    data = struct.pack("<HHI", 0x0003, 8, total) + pool + body
    parsed = parse_axml(data)
    assert parsed["root"]["name"] == "manifest"
    summary = manifest_summary(parsed)
    assert summary["package"] == "com.example.app"
    assert summary["version_code"] == 7
    assert summary["application"] == {}
    assert summary["components"]["activity"][0]["name"] == ".MainActivity"
