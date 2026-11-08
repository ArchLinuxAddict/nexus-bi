"""Recover dynamically registered JNI native methods.

Many Android libraries do not export ``Java_*`` symbols. They call
``RegisterNatives`` from ``JNI_OnLoad`` with a ``JNINativeMethod`` table. This
module recovers those tables from static ELF data and from the decompiled
``JNI_OnLoad`` source (including stack-built tables), so reconstruction can see
the real Java class/method/signature mapping.
"""

from __future__ import annotations

import re
import struct
from typing import Any

from nexus.elf.model import ElfFile

JNINATIVE_METHOD_SIZE = 24
REGISTER_NATIVES_OFFSET = 0x6B8
FIND_CLASS_OFFSET = 0x30


def _is_executable(elf: ElfFile, address: int) -> bool:
    for section in elf.sections:
        if not section.name:
            continue
        if section.addr <= address < section.addr + section.size:
            return section.flags & 0x4 != 0
    return False


def _read_cstring(elf: ElfFile, address: int | None) -> str | None:
    if address is None:
        return None
    offset = elf.va_to_offset(address)
    if offset is None or offset < 0 or offset >= len(elf.raw):
        return None
    end = elf.raw.find(b"\0", offset, min(len(elf.raw), offset + 4096))
    if end < 0:
        return None
    try:
        text = elf.raw[offset:end].decode("utf-8")
    except UnicodeDecodeError:
        return None
    return text if text.isprintable() else None


def _resolve_pointer(elf: ElfFile, address: int, relocations: dict[int, Any], symbols_by_index: dict[int, Any]) -> int | None:
    relocation = relocations.get(address)
    if relocation is not None:
        if relocation.addend is not None:
            return relocation.addend
        symbol = symbols_by_index.get(relocation.symbol_index)
        if symbol is not None:
            return symbol.value
    raw = elf.read_va(address, 8)
    if raw is None:
        return None
    value = struct.unpack_from(elf.endian + "Q", raw, 0)[0]
    return value or None


def _symbol_maps(elf: ElfFile) -> tuple[dict[int, Any], dict[int, Any]]:
    relocations = {relocation.offset: relocation for relocation in elf.relocations}
    symbols_by_index = {}
    for index, symbol in enumerate(elf.dynamic_symbols):
        symbols_by_index[index] = symbol
    for index, symbol in enumerate(elf.symbols):
        symbols_by_index[index] = symbol
    return relocations, symbols_by_index


def _valid_signature(value: str | None) -> bool:
    if not value or not value.startswith("(") or ")" not in value:
        return False
    if any(token in value for token in {" ", "\n", "\r", "\t"}):
        return False
    return value.count("(") == 1 and value.count(")") == 1


def scan_static_tables(elf: ElfFile, *, class_name: str | None = None) -> list[dict[str, Any]]:
    relocations, symbols_by_index = _symbol_maps(elf)
    candidates: list[dict[str, Any]] = []
    for section in elf.sections:
        if section.name not in {".data.rel.ro", ".data", ".rodata"}:
            continue
        if section.size < JNINATIVE_METHOD_SIZE:
            continue
        start = section.addr
        end = section.addr + section.size
        address = start
        while address + JNINATIVE_METHOD_SIZE <= end:
            name = _read_cstring(elf, _resolve_pointer(elf, address, relocations, symbols_by_index))
            signature = _read_cstring(elf, _resolve_pointer(elf, address + 8, relocations, symbols_by_index))
            function = _resolve_pointer(elf, address + 16, relocations, symbols_by_index)
            if (
                name
                and re.fullmatch(r"[A-Za-z_$][\w$]*", name)
                and _valid_signature(signature)
                and function is not None
                and function != 0
                and _is_executable(elf, function)
            ):
                candidates.append({
                    "name": name,
                    "signature": signature,
                    "address": function,
                    "table_address": address,
                    "class_name": class_name,
                })
                address += JNINATIVE_METHOD_SIZE
                continue
            address += 8
    return candidates


def _class_names(source: str) -> list[str]:
    values = []
    for match in re.finditer(r'\+ 0x30\)\)\([^,]+,"([^"]+)"\)', source):
        value = match.group(1)
        if "/" in value and not value.startswith("("):
            values.append(value)
    for match in re.finditer(r'"(com/[A-Za-z0-9_$/]+|[A-Za-z0-9_$]+/[A-Za-z0-9_$/]+)"', source):
        value = match.group(1)
        if value not in values and not value.startswith("("):
            values.append(value)
    return values


def _string_from_symbol_name(source: str, elf: ElfFile, image_base: int | None) -> list[str]:
    values: list[str] = []
    for match in re.finditer(r"s_([A-Za-z0-9_]+)_([0-9a-fA-F]+)", source):
        symbol_text = match.group(1)
        address = int(match.group(2), 16)
        candidates = [address]
        if image_base:
            candidates.append(address - image_base)
        for candidate in candidates:
            value = _read_cstring(elf, candidate)
            if value and value not in values:
                values.append(value)
                break
        else:
            # Fall back to the readable part of the symbol name.
            if symbol_text and symbol_text not in values:
                values.append(symbol_text)
    return values


def _split_args(value: str) -> list[str]:
    result = []
    current = []
    depth = 0
    quote = False
    escape = False
    for character in value:
        if quote:
            current.append(character)
            if escape:
                escape = False
            elif character == "\\":
                escape = True
            elif character == '"':
                quote = False
            continue
        if character == '"':
            quote = True
            current.append(character)
        elif character in "([{":
            depth += 1
            current.append(character)
        elif character in ")]}":
            depth -= 1
            current.append(character)
        elif character == "," and depth == 0:
            result.append("".join(current).strip())
            current = []
        else:
            current.append(character)
    tail = "".join(current).strip()
    if tail:
        result.append(tail)
    return result


def _function_address(value: str, image_base: int | None) -> int | None:
    match = re.search(r"(?:FUN_|0x)([0-9a-fA-F]+)", value)
    if not match:
        return None
    address = int(match.group(1), 16)
    if image_base and address >= image_base:
        address -= image_base
    return address


def recover_from_onload(
    elf: ElfFile,
    source: str,
    *,
    image_base: int | None = None,
) -> list[dict[str, Any]]:
    class_names = _class_names(source)
    default_class = class_names[0] if class_names else None
    strings = _string_from_symbol_name(source, elf, image_base)
    entries: list[dict[str, Any]] = []
    for match in re.finditer(r"\(\*\*\(code \*\*\)\(\*\w+ \+ 0x6b8\)\)\(([^;]*)\)\s*;", source, re.S):
        args = _split_args(match.group(1))
        if len(args) < 4:
            continue
        table_arg = args[2].strip()
        count_text = args[3].strip()
        try:
            count = int(count_text, 0)
        except ValueError:
            continue
        class_name = default_class
        for candidate in class_names:
            if candidate in source[: match.start()]:
                class_name = candidate
        if table_arg.startswith("&"):
            table_name = table_arg[1:]
            table_fields = _stack_table_fields(source, table_name, count, elf, image_base, strings)
            for field in table_fields:
                entries.append({
                    "name": field["name"],
                    "signature": field["signature"],
                    "address": field["address"],
                    "table_address": None,
                    "class_name": class_name,
                })
        else:
            table_address = None
            memcpy = re.search(
                rf"memcpy\(\s*{re.escape(table_arg)},\s*&?PTR_s_[A-Za-z0-9_]+_([0-9a-fA-F]+)",
                source,
            )
            if memcpy:
                table_address = int(memcpy.group(1), 16)
                if image_base and table_address >= image_base:
                    table_address -= image_base
            else:
                table_address = _function_address(table_arg, image_base)
            if table_address is None:
                continue
            table_entries = [
                entry for entry in scan_static_tables(elf, class_name=class_name)
                if entry.get("table_address") is not None and entry["table_address"] >= table_address
            ]
            for entry in table_entries[:count]:
                entries.append(entry)
    return entries


def _stack_table_fields(
    source: str,
    table_name: str,
    count: int,
    elf: ElfFile,
    image_base: int | None,
    symbol_strings: list[str],
) -> list[dict[str, Any]]:
    names = [value for value in symbol_strings if re.fullmatch(r"[A-Za-z$][\w$]*", value)]
    signatures = [value for value in symbol_strings if _valid_signature(value)]
    fields: list[dict[str, Any]] = []
    table_offset = _stack_offset(table_name)
    if table_offset is None:
        return fields
    for index in range(count):
        function_offset = table_offset - (index * JNINATIVE_METHOD_SIZE + 16)
        function_var = _find_stack_local(source, function_offset)
        address = None
        if function_var:
            assignment = re.search(rf"\b{re.escape(function_var)}\s*=\s*(\w+)", source)
            if assignment:
                address = _function_address(assignment.group(1), image_base)
        if address is None:
            continue
        name = names[index] if index < len(names) else None
        signature = signatures[index] if index < len(signatures) else None
        if name and signature:
            fields.append({"name": name, "signature": signature, "address": address})
    return fields


def _stack_offset(name: str) -> int | None:
    match = re.search(r"(?:local|pcStack|uStack|lStack|auStack)_([0-9a-fA-F]+)", name)
    if not match:
        return None
    return int(match.group(1), 16)


def _find_stack_local(source: str, offset: int) -> str | None:
    if offset < 0:
        return None
    names = re.findall(r"\b((?:local|pcStack|uStack|lStack|auStack)_[0-9a-fA-F]+)\b", source)
    for name in names:
        if _stack_offset(name) == offset:
            return name
    return None


def registered_entries(
    elf: ElfFile,
    *,
    onload_source: str | None = None,
    image_base: int | None = None,
    class_name: str | None = None,
) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    if onload_source:
        entries.extend(recover_from_onload(elf, onload_source, image_base=image_base))
    if not entries:
        entries.extend(scan_static_tables(elf, class_name=class_name))
    return entries


def guess_class_name(elf: ElfFile) -> str | None:
    """Guess the owning Java class for a recovered native table.

    This is used when JNI_OnLoad cannot be decompiled (packed/protected
    libraries) or when only the static table is available. The decompiled
    FindClass string always takes precedence when it exists.
    """
    framework_prefixes = ("java/", "javax/", "android/", "androidx/", "kotlin/", "kotlinx/", "dalvik/")
    best: str | None = None
    best_score = -1
    for section in elf.sections:
        if section.name not in {".rodata", ".data"}:
            continue
        data = elf.read_va(section.addr, section.size)
        if not data:
            continue
        for match in re.finditer(rb"[ -~]{3,200}", data):
            try:
                value = match.group(0).decode("ascii")
            except UnicodeDecodeError:
                continue
            if not re.fullmatch(r"[A-Za-z_$][\w$]*(?:/[A-Za-z_$][\w$]*)+", value):
                continue
            if value.startswith(framework_prefixes) or value.endswith((".so", ".dex", ".jar")):
                continue
            score = len(value)
            if value.startswith(("com/", "org/", "net/", "io/")):
                score += 20
            if any(marker in value for marker in ("/test/", "/debug/", "/internal/", "/junk", "/hidden", "/fake", "/decoy")):
                score -= 40
            if any(marker in value for marker in ("/shell", "/protector", "/main", "/app")):
                score += 10
            if score > best_score:
                best = value
                best_score = score
    return best


STRONG_DECRYPTION_MARKERS = (
    "PAS1 decrypt",
    "ensure_decrypted",
    "business so",
    "so_plain",
    "prepatchExtractedDexes",
)
WEAK_DECRYPTION_MARKERS = (
    "RC4",
    "ciphertext",
    "dexes.zip",
    "VMP bad dex",
)
CAPABILITY_MARKERS = {
    "dynamic-class-loading": (
        "ClassLoader.loadClass",
        "InMemoryDexClassLoader",
        "DexClassLoader",
    ),
    "xz-decompression": ("XzUnpacker",),
    "protector": ("protector",),
}
PACKER_MARKERS = {
    "tosks": ("libtosks", "com.tosks", "Tosks Packer", "TosksUnpack"),
    "iapp": ("libygsiyu.so", "mian.iyu", "assets/extra_conf1g.xml"),
    "np-resource": ("np/protect", "NPResource", "np资源"),
    "jiagu": ("libjiagu", "360加固", "libjiagu.so"),
    "arksafe": ("ark-native-payload", "AKP1", "yunzhuru"),
}


def _elf_strings(elf: ElfFile) -> list[str]:
    values: list[str] = []
    read_va = getattr(elf, "read_va", None)
    if not callable(read_va):
        return values
    for section in getattr(elf, "sections", []) or []:
        if section.name not in {".rodata", ".data"}:
            continue
        data = read_va(section.addr, section.size)
        if not data:
            continue
        for match in re.finditer(rb"[ -~]{4,160}", data):
            try:
                values.append(match.group(0).decode("ascii"))
            except UnicodeDecodeError:
                continue
    return values


def protection_profile(elf: ElfFile) -> dict[str, Any]:
    strings = _elf_strings(elf)
    joined = "\n".join(strings)
    strong = [marker for marker in STRONG_DECRYPTION_MARKERS if marker in joined]
    weak = [marker for marker in WEAK_DECRYPTION_MARKERS if marker in joined]
    capabilities = [
        label
        for label, markers in CAPABILITY_MARKERS.items()
        if label != "protector" and any(marker in joined for marker in markers)
    ]
    packers = [
        label
        for label, markers in PACKER_MARKERS.items()
        if any(marker in joined for marker in markers)
    ]
    ark_markers = [marker for marker in ARKSAFE_MARKERS if marker in joined]
    ark_strong = [marker for marker in ark_markers if marker in {"AKP1", "ark-native-payload", "memfd_create"}]
    tail = tail_payload_offsets(getattr(elf, "raw", b"") or b"")
    ark_tail = [item for item in tail if item["magic"] == "AKP1" and item.get("near_tail")]
    if ark_strong or ark_tail:
        label = "ark-safe-vmp"
        evidence = sorted(set(ark_markers + [item["magic"] for item in ark_tail]))
    elif strong or len(weak) >= 2:
        label = "runtime-decryption"
        evidence = strong + weak
    elif any(marker in joined for marker in CAPABILITY_MARKERS["protector"]):
        label = "protector"
        evidence = ["protector"]
    elif packers:
        label = "packer"
        evidence = packers
    else:
        label = "none"
        evidence = []
    return {
        "label": label,
        "evidence": evidence,
        "capabilities": capabilities,
        "packers": packers,
        "runtime_decryption": label in {"runtime-decryption", "ark-safe-vmp"},
        "anti_debug": [marker for marker in ANTI_DEBUG_MARKERS if marker in joined],
        "integrity_checks": [marker for marker in INTEGRITY_MARKERS if marker in joined],
        "tail_payloads": tail,
        "ark_safe": label == "ark-safe-vmp",
    }


TAIL_PAYLOAD_MAGICS = (b"AKP1", b"PAS1", b"ARK1", b"VMP1")
ARKSAFE_MARKERS = ("AKP1", "ark-native-payload", "/proc/self/maps", "memfd_create")
ANTI_DEBUG_MARKERS = (
    "ptrace", "TracerPid", "debugger", "frida", "xposed", "magisk",
    "zygisk", "riru", "lsposed", "substrate", "hook detect", "hookDetect",
)
INTEGRITY_MARKERS = (
    "crc", "adler32", "checksum", "signature", "PackageManager",
    "getPackageInfo", "checkSignature", "SHA-256",
)


def tail_payload_offsets(data: bytes, max_tail: int = 8 * 1024 * 1024) -> list[dict[str, Any]]:
    tail = data[-max_tail:]
    base = len(data) - len(tail)
    found: list[dict[str, Any]] = []
    for magic in TAIL_PAYLOAD_MAGICS:
        start = 0
        while True:
            index = tail.find(magic, start)
            if index < 0:
                break
            absolute = base + index
            found.append({
                "magic": magic.decode("ascii", errors="replace"),
                "offset": absolute,
                "tail_offset": index,
                "near_tail": absolute >= len(data) - max(1024 * 1024, len(data) // 20),
            })
            start = index + 1
            if len(found) >= 32:
                break
    return found
