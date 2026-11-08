from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

from nexus.elf.model import ElfFile

if TYPE_CHECKING:
    from nexus.program.model import CallGraph, FunctionCandidate

PRIMITIVES = {
    "V": "void", "Z": "boolean", "B": "byte", "C": "char", "S": "short",
    "I": "int", "J": "long", "F": "float", "D": "double",
}


def _split_jni_parts(value: str) -> list[str]:
    # A separator is a bare underscore; _1/_2/_3 and _0xxxx are escapes.
    return re.split(r"_(?![123]|0[0-9a-fA-F]{4})", value)


def decode_jni_class(name: str) -> tuple[str, str]:
    parts = _split_jni_parts(name)
    if len(parts) == 1:
        return "", _unescape_jni(parts[0])
    package = ".".join(_unescape_jni(part) for part in parts[:-1])
    return package, _unescape_jni(parts[-1])


def _parse_type_list(value: str) -> list[str]:
    result = []
    index = 0
    while index < len(value):
        array_depth = 0
        while index < len(value) and value[index] == "[":
            array_depth += 1
            index += 1
        if index >= len(value):
            break
        token = value[index]
        if token == "L":
            semicolon = value.find(";", index)
            if semicolon < 0:
                break
            type_name = value[index + 1:semicolon].replace("/", ".")
            index = semicolon + 1
        else:
            type_name = PRIMITIVES.get(token, "unknown")
            index += 1
        result.append(type_name + "[]" * array_depth)
    return result


def decode_jni_descriptor(descriptor: str) -> tuple[list[str], str]:
    if not descriptor.startswith("("):
        return _parse_type_list(descriptor), "unknown"
    end = descriptor.find(")")
    if end < 0:
        return _parse_type_list(descriptor), "unknown"
    arguments = descriptor[1:end]
    returns = descriptor[end + 1:]
    return _parse_type_list(arguments), decode_jni_type(returns)


def decode_jni_type(descriptor: str) -> str:
    depth = 0
    while descriptor.startswith("["):
        depth += 1
        descriptor = descriptor[1:]
    if descriptor.startswith("L") and descriptor.endswith(";"):
        name = descriptor[1:-1].replace("/", ".")
    else:
        name = PRIMITIVES.get(descriptor, "unknown")
    return name + "[]" * depth


def _unescape_jni(value: str) -> str:
    result = value.replace("_1", "_").replace("_2", ";").replace("_3", "[")
    return re.sub(r"_0([0-9a-fA-F]{4})", lambda match: chr(int(match.group(1), 16)), result)


def _normalize_encoded_parameters(value: str) -> str:
    decoded = _unescape_jni(value)
    result: list[str] = []
    index = 0
    while index < len(decoded):
        while index < len(decoded) and decoded[index] == "[":
            result.append("[")
            index += 1
        if index >= len(decoded):
            break
        token = decoded[index]
        if token == "L":
            end = decoded.find(";", index)
            if end < 0:
                result.append(decoded[index:])
                break
            name = decoded[index + 1:end].replace("_", "/")
            result.append("L" + name + ";")
            index = end + 1
        else:
            result.append(token)
            index += 1
    return "".join(result)


def _valid_encoded_parameters(value: str) -> bool:
    if not value:
        return True
    normalized = _normalize_encoded_parameters(value)
    if not normalized or any(token in normalized for token in {" ", "\n", ";(", ")"}):
        return False
    arguments, _ = decode_jni_descriptor(f"({normalized})")
    return bool(arguments) and all(argument != "unknown" for argument in arguments)


def _split_jni_body(body: str) -> tuple[str, str | None]:
    if "__" not in body:
        return body, None
    candidates: list[tuple[str, str]] = []
    start = 0
    while True:
        index = body.find("__", start)
        if index < 0:
            break
        name_part = body[:index]
        encoded = body[index + 2:]
        if _valid_encoded_parameters(encoded):
            candidates.append((name_part, encoded))
        start = index + 1
    if candidates:
        return max(candidates, key=lambda item: len(item[0]))
    name_part, encoded = body.split("__", 1)
    return name_part, encoded


def decode_jni_symbol(symbol: str) -> dict[str, Any] | None:
    if not symbol.startswith("Java_"):
        return None
    body = symbol[5:]
    name_part, encoded_parameters = _split_jni_body(body)
    if encoded_parameters is None:
        descriptor = None
    else:
        descriptor = "(" + _normalize_encoded_parameters(encoded_parameters) + ")"
    if "_" not in name_part:
        return None
    parts = _split_jni_parts(name_part)
    if len(parts) < 2:
        return None
    class_part = "_".join(parts[:-1])
    method = parts[-1]
    package, simple = decode_jni_class(class_part)
    class_name = f"{package}.{simple}" if package else simple
    args, return_type = decode_jni_descriptor(descriptor) if descriptor else ([], "unknown")
    return {
        "symbol": symbol,
        "class_name": class_name,
        "simple_class": simple,
        "package": package,
        "method": _unescape_jni(method),
        "descriptor": descriptor,
        "arguments": args,
        "return_type": return_type,
        "confidence": 0.62 if descriptor else 0.45,
    }


def analyze_jni(elf: ElfFile, functions: list[FunctionCandidate], callgraph: CallGraph, strings: list[dict[str, Any]]) -> dict[str, Any]:
    symbols = []
    for symbol in list(elf.symbols) + list(elf.dynamic_symbols):
        if not symbol.name:
            continue
        if symbol.name.startswith("Java_") or symbol.name in {"JNI_OnLoad", "JNI_OnUnload"}:
            symbols.append(symbol)
    entries = []
    onload = []
    seen_entries = set()
    seen_onload = set()
    for symbol in symbols:
        if symbol.name in {"JNI_OnLoad", "JNI_OnUnload"}:
            key = (symbol.name, symbol.value)
            if key not in seen_onload:
                seen_onload.add(key)
                onload.append({"symbol": symbol.name, "address": f"0x{symbol.value:x}", "size": symbol.size})
            continue
        decoded = decode_jni_symbol(symbol.name)
        if decoded:
            key = (decoded.get("class_name"), decoded.get("method"), decoded.get("descriptor"), symbol.value)
            if key in seen_entries:
                continue
            seen_entries.add(key)
            decoded["address"] = f"0x{symbol.value:x}"
            decoded["size"] = symbol.size
            decoded["evidence"] = f"ev_{elf.sha256[:12]}_jni_{symbol.value:x}"
            entries.append(decoded)
    imports = sorted({
        symbol.name for symbol in elf.dynamic_symbols
        if symbol.name and (
            symbol.name.startswith("Get") or symbol.name.startswith("Release") or
            symbol.name.startswith("New") or symbol.name.startswith("Set") or
            symbol.name.startswith("Call") or symbol.name.startswith("Exception") or
            symbol.name.startswith("Throw") or symbol.name in {"FindClass", "RegisterNatives", "GetJavaVM"}
        )
    })
    register_natives = []
    for function in functions:
        for instruction in function.instructions:
            if instruction.control_flow not in {"call", "indirect_call"}:
                continue
            for operand in instruction.operands:
                if "RegisterNatives" in operand:
                    register_natives.append({"function_id": function.function_id, "address": f"0x{instruction.address:x}"})
    from nexus.jni.registry import guess_class_name, scan_static_tables

    guessed_class = guess_class_name(elf)
    for table_entry in scan_static_tables(elf, class_name=guessed_class):
        class_name = (table_entry.get("class_name") or guessed_class or "UnassignedNatives").replace("/", ".")
        package, _, simple = class_name.rpartition(".")
        descriptor = table_entry["signature"]
        arguments, return_type = decode_jni_descriptor(descriptor)
        address = table_entry["address"]
        key = (class_name, table_entry["name"], descriptor, address)
        if key in seen_entries:
            continue
        seen_entries.add(key)
        entries.append({
            "symbol": f"registered:{class_name}:{table_entry['name']}{descriptor}",
            "class_name": class_name,
            "simple_class": simple,
            "package": package,
            "method": table_entry["name"],
            "descriptor": descriptor,
            "arguments": arguments,
            "return_type": return_type,
            "confidence": 0.58,
            "address": f"0x{address:x}",
            "size": None,
            "evidence": f"ev_{elf.sha256[:12]}_reg_{address:x}",
            "registered": True,
            "table_address": f"0x{table_entry['table_address']:x}" if table_entry.get("table_address") is not None else None,
        })
        register_natives.append({
            "function_id": None,
            "address": f"0x{address:x}",
            "table_address": f"0x{table_entry['table_address']:x}" if table_entry.get("table_address") is not None else None,
            "class_name": class_name,
            "method": table_entry["name"],
            "descriptor": descriptor,
            "source": "JNINativeMethod table",
        })
    return {
        "is_jni": bool(entries or onload or imports or register_natives),
        "entries": entries,
        "onload": onload,
        "register_natives": register_natives,
        "imports": imports,
        "reference_management": {
            "get_calls": [name for name in imports if name.startswith("Get")],
            "release_calls": [name for name in imports if name.startswith("Release")],
            "new_calls": [name for name in imports if name.startswith("New")],
        },
        "limitations": [
            "JNI function-table calls are indirect and are not resolved to individual JNI methods.",
            "Dynamically registered methods are recovered from JNINativeMethod tables; stack-built tables need decompiler evidence at reconstruction time.",
            "Class/method names are reconstructed from symbol mangling or nearby class strings and may be ambiguous for unusual names.",
            "Java behavior is not executed or dynamically verified.",
        ],
        "strings": [item for item in strings if "jni" in item["value"].lower()][:20],
    }
