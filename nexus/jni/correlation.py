"""Correlate Java native declarations with JNI symbols in native libraries.

The correlation is symbolic and evidence-linked.  It distinguishes short and
long JNI symbol forms, reports overload ambiguity and never claims that a
symbol match proves runtime binding: ``RegisterNatives`` can bind names
dynamically.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from nexus.architectures import adapter_for_elf
from nexus.elf.parser import parse_elf


def escape_jni_identifier(value: str) -> str:
    result = []
    for character in value:
        if character == "_":
            result.append("_1")
        elif character == ";":
            result.append("_2")
        elif character == "[":
            result.append("_3")
        elif character == "/":
            result.append("_")
        elif character == ".":
            result.append("_")
        elif character.isalnum() or character == "$":
            result.append(character)
        else:
            result.append(f"_0{ord(character):04x}")
    return "".join(result)


def _escape_descriptor_parameter(text: str) -> str:
    result = []
    for character in text:
        if character == "_":
            result.append("_1")
        elif character == ";":
            result.append("_2")
        elif character == "[":
            result.append("_3")
        elif character == "/":
            result.append("_")
        else:
            result.append(character)
    return "".join(result)


def jni_symbol_candidates(class_name: str, method: str, descriptor: str | None = None) -> list[str]:
    base = f"Java_{escape_jni_identifier(class_name)}_{escape_jni_identifier(method)}"
    candidates = [base]
    if descriptor and descriptor.startswith("("):
        end = descriptor.find(")")
        if end > 0:
            parameters = descriptor[1:end]
            candidates.append(base + "__" + _escape_descriptor_parameter(parameters))
    return candidates


def decode_jni_symbol_name(symbol: str) -> dict[str, Any] | None:
    from nexus.jni.analysis import decode_jni_symbol
    return decode_jni_symbol(symbol)


def _collect_java_methods(java_path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    from nexus.formats.apk import parse_apk
    from nexus.formats.dex import parse_dex
    from nexus.formats.java_class import parse_java_class

    suffix = java_path.suffix.lower()
    methods: list[dict[str, Any]] = []
    source: dict[str, Any] = {"path": str(java_path), "format": suffix.lstrip(".")}
    if suffix == ".class":
        parsed = parse_java_class(java_path)
        source["class"] = parsed.get("name")
        source["classes"] = [parsed.get("name")]
        for method in parsed.get("methods") or []:
            if method.get("native"):
                methods.append(
                    {
                        "class_name": parsed.get("name"),
                        "package": parsed.get("package"),
                        "method": method.get("name"),
                        "descriptor": method.get("descriptor"),
                        "return_type": method.get("return_type"),
                        "parameters": [item.get("type") for item in method.get("parameters") or []],
                        "source": str(java_path),
                    }
                )
    elif suffix == ".dex":
        parsed = parse_dex(java_path)
        source["classes"] = [item.get("name") for item in parsed.get("classes", [])]
        for class_item in parsed.get("classes", []):
            for method in class_item.get("methods") or []:
                if method.get("native"):
                    methods.append(
                        {
                            "class_name": class_item.get("name"),
                            "package": class_item.get("package"),
                            "method": method.get("name"),
                            "descriptor": method.get("descriptor"),
                            "return_type": method.get("return_type"),
                            "parameters": [item.get("type") for item in method.get("parameters") or []],
                            "source": str(java_path),
                        }
                    )
    elif suffix == ".apk":
        parsed = parse_apk(java_path)
        source["manifest"] = parsed.get("manifest")
        source["classes"] = [name for entry in parsed.get("dex", []) for name in entry.get("class_names", [])]
        methods.extend(parsed.get("native_methods") or [])
    elif suffix in {".jar", ".zip"}:
        import zipfile

        with zipfile.ZipFile(java_path) as archive:
            for info in archive.infolist():
                if not info.filename.endswith(".class"):
                    continue
                with archive.open(info) as handle:
                    import tempfile

                    with tempfile.NamedTemporaryFile(suffix=".class") as temporary:
                        temporary.write(handle.read())
                        temporary.flush()
                        parsed = parse_java_class(temporary.name)
                for method in parsed.get("methods") or []:
                    if method.get("native"):
                        methods.append(
                            {
                                "class_name": parsed.get("name"),
                                "package": parsed.get("package"),
                                "method": method.get("name"),
                                "descriptor": method.get("descriptor"),
                                "return_type": method.get("return_type"),
                                "parameters": [item.get("type") for item in method.get("parameters") or []],
                                "source": f"{java_path}!{info.filename}",
                            }
                        )
    else:
        raise ValueError(f"unsupported Java artifact: {java_path.suffix}")
    return methods, source


def _native_symbols(native_path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    elf = parse_elf(native_path)
    symbols = []
    seen = set()
    for symbol in list(elf.symbols) + list(elf.dynamic_symbols):
        if not symbol.name:
            continue
        if not (symbol.name.startswith("Java_") or symbol.name in {"JNI_OnLoad", "JNI_OnUnload"}):
            continue
        key = (symbol.name, symbol.value)
        if key in seen:
            continue
        seen.add(key)
        decoded = decode_jni_symbol_name(symbol.name)
        symbols.append(
            {
                "symbol": symbol.name,
                "address": f"0x{symbol.value:x}",
                "size": symbol.size,
                "binding": symbol.binding,
                "table": symbol.table,
                "decoded": decoded,
                "evidence": f"ev_{elf.sha256[:12]}_sym_{symbol.value:x}",
            }
        )
    try:
        abi = adapter_for_elf(elf).abi
    except Exception:
        abi = None
    report = {
        "path": str(native_path),
        "sha256": elf.sha256,
        "machine": elf.header.machine_name,
        "class": elf.header.elf_class,
        "abi": abi,
        "jni_onload": any(item["symbol"] == "JNI_OnLoad" for item in symbols),
        "imports": sorted({symbol.name for symbol in elf.dynamic_symbols if symbol.name and symbol.name.startswith(("Get", "Release", "New", "Call", "Exception", "Throw", "FindClass", "RegisterNatives"))}),
    }
    return report, symbols


def correlate_java_native(java_path: str | Path, native_path: str | Path) -> dict[str, Any]:
    java = Path(java_path)
    native = Path(native_path)
    methods, source = _collect_java_methods(java)
    native_report, native_symbols = _native_symbols(native)
    by_name: dict[str, list[dict[str, Any]]] = {}
    for symbol in native_symbols:
        by_name.setdefault(symbol["symbol"], []).append(symbol)
    name_counts: dict[tuple[str, str], int] = {}
    for method in methods:
        key = (str(method.get("class_name")), str(method.get("method")))
        name_counts[key] = name_counts.get(key, 0) + 1
    matches = []
    unmatched_java = []
    matched_symbols: set[str] = set()
    for method in methods:
        short, long = jni_symbol_candidates(str(method.get("class_name")), str(method.get("method")), str(method.get("descriptor")))
        candidates = [short, long] if name_counts[(str(method.get("class_name")), str(method.get("method")))] > 1 else [long, short]
        found = None
        for candidate in candidates:
            if candidate in by_name:
                found = by_name[candidate][0]
                break
        if found:
            matched_symbols.add(found["symbol"])
            matches.append(
                {
                    **method,
                    "status": "matched",
                    "native_symbol": found["symbol"],
                    "native_address": found["address"],
                    "native_evidence": found["evidence"],
                    "confidence": 0.9 if found["symbol"] == long else 0.72,
                    "ambiguity": "short_form_used_for_overload" if found["symbol"] == short and long != short else None,
                }
            )
        else:
            unmatched_java.append(
                {
                    **method,
                    "status": "missing_native_symbol",
                    "expected_symbols": candidates,
                    "confidence": 0.1,
                    "note": "A dynamic RegisterNatives binding or a different library may provide this method.",
                }
            )
    unmatched_native = []
    for symbol in native_symbols:
        if symbol["symbol"] in matched_symbols or symbol["symbol"] in {"JNI_OnLoad", "JNI_OnUnload"}:
            continue
        unmatched_native.append(symbol)
    return {
        "schema_version": "nexus.jni-correlation/v1",
        "java": source,
        "native": native_report,
        "java_native_methods": methods,
        "native_symbols": native_symbols,
        "matches": matches,
        "unmatched_java_methods": unmatched_java,
        "unmatched_native_symbols": unmatched_native,
        "summary": {
            "java_native_methods": len(methods),
            "native_jni_symbols": len(native_symbols),
            "matched": len(matches),
            "unmatched_java": len(unmatched_java),
            "unmatched_native": len(unmatched_native),
            "jni_onload": native_report.get("jni_onload", False),
        },
        "limitations": [
            "Symbol matching is static; RegisterNatives can bind Java methods to differently named native functions.",
            "Short-name JNI symbols are ambiguous when Java overloads share a method name.",
            "A missing symbol does not prove the Java method has no native implementation.",
            "No JNI call, reference lifetime, exception or object-identity behavior is executed or verified.",
        ],
    }
