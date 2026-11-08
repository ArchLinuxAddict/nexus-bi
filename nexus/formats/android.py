"""Binary AndroidManifest.xml (AXML) decoding.

This is a bounded, read-only chunk parser.  It recovers the manifest element
tree and typed attribute values used by Java/native correlation: package name,
components, permissions, libraries and application metadata.  Resource-table
resolution and compiled resource semantics are not claimed.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from typing import Any

RES_STRING_POOL_TYPE = 0x0001
RES_XML_TYPE = 0x0003
RES_XML_START_NAMESPACE_TYPE = 0x0100
RES_XML_END_NAMESPACE_TYPE = 0x0101
RES_XML_START_ELEMENT_TYPE = 0x0102
RES_XML_END_ELEMENT_TYPE = 0x0103
RES_XML_CDATA_TYPE = 0x0104
RES_XML_RESOURCE_MAP_TYPE = 0x0180

UTF8_FLAG = 0x00000100
TYPE_NULL = 0x00
TYPE_REFERENCE = 0x01
TYPE_ATTRIBUTE = 0x02
TYPE_STRING = 0x03
TYPE_FLOAT = 0x04
TYPE_DIMENSION = 0x05
TYPE_FRACTION = 0x06
TYPE_INT_DEC = 0x10
TYPE_INT_HEX = 0x11
TYPE_INT_BOOLEAN = 0x12


@dataclass
class AxmlElement:
    name: str
    namespace: str | None
    attributes: dict[str, Any] = field(default_factory=dict)
    children: list["AxmlElement"] = field(default_factory=list)
    line: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "namespace": self.namespace,
            "attributes": self.attributes,
            "children": [child.to_dict() for child in self.children],
            "line": self.line,
        }


def _u16(data: bytes, offset: int) -> int:
    return struct.unpack_from("<H", data, offset)[0]


def _u32(data: bytes, offset: int) -> int:
    return struct.unpack_from("<I", data, offset)[0]


def _decode_utf8_length(data: bytes, offset: int) -> tuple[int, int]:
    length = data[offset]
    offset += 1
    if length & 0x80:
        length = ((length & 0x7F) << 8) | data[offset]
        offset += 1
    return length, offset


def _decode_utf16_length(data: bytes, offset: int) -> tuple[int, int]:
    length = _u16(data, offset)
    offset += 2
    if length & 0x8000:
        length = ((length & 0x7FFF) << 16) | _u16(data, offset)
        offset += 2
    return length, offset


def parse_string_pool(data: bytes, offset: int) -> tuple[list[str], int]:
    chunk_type = _u16(data, offset)
    header_size = _u16(data, offset + 2)
    chunk_size = _u32(data, offset + 4)
    if chunk_type != RES_STRING_POOL_TYPE or chunk_size < header_size:
        raise ValueError("not a string pool chunk")
    string_count = _u32(data, offset + 8)
    flags = _u32(data, offset + 16)
    strings_start = _u32(data, offset + 20)
    utf8 = bool(flags & UTF8_FLAG)
    offsets_base = offset + header_size
    strings: list[str] = []
    for index in range(min(string_count, 1_000_000)):
        string_offset = _u32(data, offsets_base + index * 4)
        cursor = offset + strings_start + string_offset
        if cursor >= offset + chunk_size or cursor >= len(data):
            strings.append("")
            continue
        try:
            if utf8:
                _, cursor = _decode_utf8_length(data, cursor)
                byte_length, cursor = _decode_utf8_length(data, cursor)
                raw = data[cursor:cursor + byte_length]
                strings.append(raw.decode("utf-8", errors="replace"))
            else:
                char_length, cursor = _decode_utf16_length(data, cursor)
                raw = data[cursor:cursor + char_length * 2]
                strings.append(raw.decode("utf-16-le", errors="replace"))
        except (IndexError, struct.error):
            strings.append("")
    return strings, offset + chunk_size


def _string(strings: list[str], index: int) -> str | None:
    if index is None:
        return None
    if index < len(strings):
        return strings[index]
    return None


def parse_axml(data: bytes) -> dict[str, Any]:
    if len(data) < 8 or _u16(data, 0) != RES_XML_TYPE:
        raise ValueError("not a binary XML document")
    header_size = _u16(data, 2)
    total_size = _u32(data, 4)
    cursor = header_size
    strings: list[str] = []
    resource_map: list[int] = []
    root: AxmlElement | None = None
    stack: list[AxmlElement] = []
    elements: list[AxmlElement] = []
    while cursor + 8 <= min(len(data), total_size):
        chunk_type = _u16(data, cursor)
        chunk_header = _u16(data, cursor + 2)
        chunk_size = _u32(data, cursor + 4)
        if chunk_size < 8 or cursor + chunk_size > len(data):
            break
        if chunk_type == RES_STRING_POOL_TYPE:
            try:
                strings, _ = parse_string_pool(data, cursor)
            except ValueError:
                pass
        elif chunk_type == RES_XML_RESOURCE_MAP_TYPE:
            count = max(0, (chunk_size - chunk_header) // 4)
            resource_map = [_u32(data, cursor + chunk_header + index * 4) for index in range(count)]
        elif chunk_type == RES_XML_START_ELEMENT_TYPE:
            if chunk_header < 16 or cursor + chunk_header + 20 > len(data):
                cursor += chunk_size
                continue
            line = _u32(data, cursor + 8)
            namespace_index = _u32(data, cursor + 16)
            name_index = _u32(data, cursor + 20)
            attribute_start = _u16(data, cursor + 24)
            attribute_size = _u16(data, cursor + 26)
            attribute_count = _u16(data, cursor + 28)
            element = AxmlElement(
                name=_string(strings, name_index) or "",
                namespace=_string(strings, namespace_index),
                line=line,
            )
            attributes_base = cursor + chunk_header + attribute_start
            for index in range(min(attribute_count, 10000)):
                base = attributes_base + index * attribute_size
                if base + 20 > cursor + chunk_size:
                    break
                attr_name = _string(strings, _u32(data, base + 4)) or ""
                raw_value = _u32(data, base + 8)
                data_type = data[base + 15]
                typed_data = _u32(data, base + 16)
                value: Any
                if data_type == TYPE_STRING:
                    value = _string(strings, typed_data)
                elif data_type == TYPE_REFERENCE:
                    value = {"reference": f"0x{typed_data:08x}"}
                elif data_type == TYPE_ATTRIBUTE:
                    value = {"attribute": f"0x{typed_data:08x}"}
                elif data_type == TYPE_INT_BOOLEAN:
                    value = typed_data != 0
                elif data_type == TYPE_INT_DEC:
                    value = typed_data if typed_data < 0x80000000 else typed_data - 0x100000000
                elif data_type == TYPE_INT_HEX:
                    value = f"0x{typed_data:08x}"
                elif data_type == TYPE_FLOAT:
                    value = struct.unpack("<f", struct.pack("<I", typed_data))[0]
                elif data_type == TYPE_NULL:
                    value = None
                else:
                    value = _string(strings, raw_value) if raw_value != 0xFFFFFFFF else typed_data
                key = attr_name
                if key in element.attributes:
                    existing = element.attributes[key]
                    if not isinstance(existing, list):
                        existing = [existing]
                    existing.append(value)
                    element.attributes[key] = existing
                else:
                    element.attributes[key] = value
            if stack:
                stack[-1].children.append(element)
            elif root is None:
                root = element
            elements.append(element)
            stack.append(element)
        elif chunk_type == RES_XML_END_ELEMENT_TYPE:
            if stack:
                stack.pop()
        cursor += chunk_size
    return {
        "strings": strings,
        "resource_map": resource_map,
        "root": root.to_dict() if root else None,
        "elements": [element.to_dict() for element in elements],
        "string_count": len(strings),
        "limitations": [
            "Binary manifest parsing only; resource references are not resolved against resources.arsc.",
            "Unknown data types preserve their numeric value rather than being guessed.",
        ],
    }


def _find(element: dict[str, Any], name: str) -> list[dict[str, Any]]:
    found: list[dict[str, Any]] = []
    if element.get("name") == name:
        found.append(element)
    for child in element.get("children", []) or []:
        found.extend(_find(child, name))
    return found


def manifest_summary(axml: dict[str, Any]) -> dict[str, Any]:
    root = axml.get("root") or {}
    attributes = root.get("attributes") or {}
    permissions = []
    components: dict[str, list[dict[str, Any]]] = {"activity": [], "service": [], "receiver": [], "provider": []}
    uses_library = []
    for element in axml.get("elements", []) or []:
        name = element.get("name")
        attrs = element.get("attributes") or {}
        if name == "uses-permission" and attrs.get("name"):
            permissions.append(attrs.get("name"))
        elif name == "uses-permission-sdk-23" and attrs.get("name"):
            permissions.append(attrs.get("name"))
        elif name in components and attrs.get("name"):
            components[name].append(
                {
                    "name": attrs.get("name"),
                    "exported": attrs.get("exported"),
                    "permission": attrs.get("permission"),
                    "process": attrs.get("process"),
                }
            )
        elif name == "uses-library" and attrs.get("name"):
            uses_library.append({"name": attrs.get("name"), "required": attrs.get("required", True)})
    application = next((item for item in _find(root, "application") if item), {})
    return {
        "package": attributes.get("package") or root.get("package"),
        "version_code": attributes.get("versionCode"),
        "version_name": attributes.get("versionName"),
        "compile_sdk_version": attributes.get("compileSdkVersion"),
        "min_sdk_version": attributes.get("minSdkVersion"),
        "target_sdk_version": attributes.get("targetSdkVersion"),
        "permissions": permissions,
        "components": components,
        "uses_library": uses_library,
        "application": application.get("attributes") if application else {},
        "limitations": [
            "Values are observed attribute values; resource references remain unresolved.",
            "Intent filters and component semantics are not fully interpreted.",
        ],
    }
