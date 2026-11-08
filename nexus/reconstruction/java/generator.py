import json
import re
from pathlib import Path
from typing import Any

from nexus.jni.analysis import decode_jni_descriptor
from nexus.program.bundle import AnalysisBundle
from nexus.program.model import FunctionCandidate
from nexus.reconstruction.java.jni_semantics import SemanticMethod, analyze_method


def _sanitize(name: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_]", "_", name)
    if not value:
        value = "Reconstructed"
    if value[0].isdigit():
        value = f"C_{value}"
    return value


JAVA_KEYWORDS = {
    "class", "interface", "enum", "public", "private", "protected", "static",
    "final", "void", "int", "long", "short", "byte", "char", "boolean",
    "float", "double", "return", "if", "else", "for", "while", "switch",
    "case", "new", "this", "super", "extends", "implements", "package",
}

NATIVE_FRAMEWORK_PREFIXES = (
    "android.",
    "androidx.",
    "dalvik.",
    "java.",
    "javax.",
    "kotlin.",
    "kotlinx.",
)


def _class_priority(package: str) -> int:
    if any(package.startswith(prefix) for prefix in NATIVE_FRAMEWORK_PREFIXES):
        return 2
    return 1


def _entry_address(entry: dict[str, Any]) -> int | None:
    try:
        return int(str(entry.get("address") or ""), 16)
    except ValueError:
        return None


def _java_identifier(name: str) -> str:
    value = _sanitize(name)
    if value in JAVA_KEYWORDS:
        value = f"native_{value}"
    return value


def _function_params(bundle: AnalysisBundle, function: FunctionCandidate) -> list[str]:
    types = bundle.type_candidates.get(function.function_id, [])
    params = sorted([item for item in types if item.get("role") == "parameter"], key=lambda item: item.get("parameter_index", 0))
    return [f"long arg{index}" for index in range(len(params))]


def _library_name(bundle: AnalysisBundle) -> tuple[str, bool]:
    for dependency in bundle.elf.dependencies:
        if dependency.get("kind") != "soname":
            continue
        filename = Path(str(dependency.get("name") or "")).name
        if not filename:
            continue
        stem = filename.split(".so", 1)[0] if ".so" in filename else Path(filename).stem
        if stem:
            return (stem.removeprefix("lib") or stem), True
    candidate = Path(bundle.elf.path).name
    stem = candidate.split(".so", 1)[0] if ".so" in candidate else Path(candidate).stem
    if stem.startswith("C_") or re.fullmatch(r"[0-9a-f]{32,64}", stem):
        return bundle.elf.sha256[:16], False
    return (stem.removeprefix("lib") or stem), False


def _collect_semantics(
    bundle: AnalysisBundle,
    entries: list[dict[str, Any]],
    ghidra_result: Any = None,
) -> dict[str, SemanticMethod]:
    if not entries:
        return {}
    if ghidra_result is None:
        try:
            from nexus.toolchain.adapters import GhidraAdapter

            ghidra_result = GhidraAdapter().decompile(bundle.path)
        except Exception:
            return {}
    functions = ghidra_result.artifacts.get("functions") or []
    by_name = {item.get("name"): item for item in functions}
    image_base = ghidra_result.artifacts.get("image_base") or 0
    by_entry: dict[int, dict[str, Any]] = {}
    for function in functions:
        try:
            entry_value = int(function.get("entry") or -1)
        except (TypeError, ValueError):
            continue
        by_entry[entry_value] = function
        if image_base:
            by_entry[entry_value - image_base] = function
    semantics: dict[str, SemanticMethod] = {}
    for entry in entries:
        symbol = entry.get("symbol")
        function = by_name.get(symbol)
        if function is None and entry.get("address"):
            try:
                address = int(str(entry["address"]), 16)
            except ValueError:
                address = -1
            function = by_entry.get(address) or by_entry.get(address + image_base)
        source = (function or {}).get("source")
        semantic = analyze_method(entry, source, elf=bundle.elf, image_base=image_base)
        if semantic is not None:
            semantics[symbol] = semantic
    return semantics


FIELD_TYPES = {
    "extraDataMap": "androidx.collection.SimpleArrayMap",
    "lifecycleRegistry": "androidx.lifecycle.LifecycleRegistry",
}


def _java_class_for_jni(
    bundle: AnalysisBundle,
    class_name: str,
    package: str,
    entries: list[dict[str, Any]],
    semantics: dict[str, SemanticMethod] | None = None,
) -> tuple[str, str]:
    semantics = semantics or {}
    simple = class_name.split(".")[-1] if class_name else "JniReconstructed"
    values = [semantics.get(entry.get("symbol")) for entry in entries]
    instance = any(value is not None and value.instance for value in values)
    has_constructor = any(entry.get("method") == "<init>" for entry in entries)
    extends = ""
    if any(value is not None and value.template in {"key_dispatch", "super_dispatch"} for value in values):
        extends = " extends android.app.Activity"
    elif any(value is not None and "android/app/Application" in str(value.operations) for value in values):
        extends = " extends android.app.Application"
    implements = ""
    if any(value is not None and "lifecycleRegistry" in value.body for value in values):
        implements = " implements androidx.lifecycle.LifecycleOwner"
    class_prefix = "public class" if instance or has_constructor else "public final class"
    lines = []
    if package:
        lines.append(f"package {package};")
        lines.append("")
    lines.append(f"{class_prefix} {_java_identifier(simple)}{extends}{implements} {{")
    if not instance and not has_constructor:
        lines.append("    private " + _java_identifier(simple) + "() {")
        lines.append("    }")
        lines.append("")
    fields: set[str] = set()
    for value in values:
        if value is None:
            continue
        fields.update(re.findall(r"this\.([A-Za-z_$][\w$]*)\b(?!\s*\()", value.body))
    for field in sorted(fields):
        field_type = FIELD_TYPES.get(field, "Object")
        initializer = ""
        if field == "extraDataMap":
            initializer = f" = new {field_type}()"
        elif field == "lifecycleRegistry" and "LifecycleOwner" in implements:
            initializer = f" = new {field_type}(this)"
        lines.append(f"    private {field_type} {field}{initializer};")
    if fields:
        lines.append("")
    class_fqn = f"{package}.{simple}" if package else simple
    if any(value is not None and f"{class_fqn}.ExtraData" in (value.return_type + "\n" + value.body) for value in values):
        lines.append("    public static class ExtraData {")
        lines.append("    }")
        lines.append("")
    seen_methods = set()
    for entry, semantic in zip(entries, values):
        arguments = entry.get("arguments") or []
        signature_key = (entry.get("method"), tuple(arguments))
        if signature_key in seen_methods:
            continue
        seen_methods.add(signature_key)
        method_name = _java_identifier(entry["method"])
        params = ", ".join(f"{argument} arg{index}" for index, argument in enumerate(arguments))
        if entry.get("method") == "<init>":
            lines.append(f"    public {_java_identifier(simple)}({params}) {{")
            if semantic is not None:
                for body_line in semantic.body.splitlines():
                    if body_line.strip() in {"return null;", "return false;", "return;"}:
                        continue
                    lines.append(f"        {body_line}")
            lines.append("    }")
            lines.append("")
            continue
        entry_return = entry.get("return_type")
        declared_return = entry_return if entry_return not in {None, "", "unknown", "Object"} else (semantic.return_type if semantic is not None else (entry_return or "Object"))
        if semantic is not None and semantic.template != "trace":
            modifier = "public" if semantic.instance else "public static"
            lines.append(f"    // JNI symbol: {entry.get('symbol')}")
            lines.append(f"    // template: {semantic.template}; confidence: {semantic.confidence:.2f}")
            lines.append(f"    {modifier} {declared_return} {method_name}({params}) {{")
            for body_line in semantic.body.splitlines():
                lines.append(f"        {body_line}")
            lines.append("    }")
        else:
            if semantic is not None:
                lines.append(f"    // JNI symbol: {entry.get('symbol')}")
                lines.append(f"    // recovered operation trace; confidence: {semantic.confidence:.2f}")
                for body_line in semantic.body.splitlines():
                    text = body_line.strip()
                    if text in {"return null;", "return false;", "return;"}:
                        continue
                    if text.startswith("// "):
                        text = text[3:]
                    lines.append(f"    // {text}")
            return_type = declared_return
            if return_type in {"unknown", "Object", None, ""}:
                return_type = "Object"
            lines.append(f"    public static native {return_type} {method_name}({params});")
        lines.append("")
    library_name, inferred = _library_name(bundle)
    lines.append("    static {")
    if not inferred:
        lines.append("        // no SONAME in the ELF; replace with the deployed library name")
    lines.append(f'        System.loadLibrary("{_sanitize(library_name)}");')
    lines.append("    }")
    lines.append("}")
    lines.append("")
    path_name = (_java_identifier(simple) + ".java")
    path = f"src/main/java/{package.replace('.', '/') + '/' if package else ''}{path_name}"
    return path, "\n".join(lines)


def _java_module(bundle: AnalysisBundle, functions: list[FunctionCandidate]) -> str:
    lines = ["public final class ReconstructedModule {", "    private ReconstructedModule() {", "    }", ""]
    seen_names: set[str] = set()
    for function in functions:
        if function.name in {"main", "_start"} or function.name.startswith("_"):
            continue
        name = _java_identifier(function.name)
        if name in seen_names:
            continue
        seen_names.add(name)
        params = _function_params(bundle, function)
        lines.append(f"    public static long {name}({', '.join(params)}) {{")
        lines.append(f"        // structural reconstruction; binary evidence: 0x{function.entry:x}")
        lines.append("        return 0L;")
        lines.append("    }")
        lines.append("")
    lines.append("}")
    lines.append("")
    return "\n".join(lines)


def _registered_to_entry(item: dict[str, Any], bundle: AnalysisBundle) -> dict[str, Any]:
    class_name = (item.get("class_name") or "UnassignedNatives").replace("/", ".")
    package, _, simple = class_name.rpartition(".")
    descriptor = item["signature"]
    arguments, return_type = decode_jni_descriptor(descriptor)
    address = int(item["address"])
    return {
        "symbol": f"registered:{class_name}:{item['name']}{descriptor}",
        "class_name": class_name,
        "simple_class": simple,
        "package": package,
        "method": item["name"],
        "descriptor": descriptor,
        "arguments": arguments,
        "return_type": return_type,
        "confidence": 0.58,
        "address": f"0x{address:x}",
        "size": None,
        "evidence": f"ev_{bundle.elf.sha256[:12]}_reg_{address:x}",
        "registered": True,
    }


def reconstruct_java(bundle: AnalysisBundle, function_ids: list[str] | None = None, out_dir: str | Path | None = None, project_name: str = "nexus_java_reconstruction") -> dict[str, Any]:
    files: dict[str, str] = {}
    jni_report = bundle.jni_report
    entries = jni_report.get("entries", [])
    if function_ids:
        ids = set(function_ids)
        selected_functions = [function for function in bundle.functions if function.function_id in ids or function.name in ids]
        addresses = {function.entry for function in selected_functions}
        filtered = [entry for entry in entries if _entry_address(entry) in addresses]
        if filtered:
            entries = filtered
    ghidra_result = None
    if bundle.jni_report.get("onload") or entries:
        try:
            from nexus.toolchain.adapters import GhidraAdapter

            ghidra_result = GhidraAdapter().decompile(bundle.path)
        except Exception:
            ghidra_result = None
    if ghidra_result is not None:
        from nexus.jni.registry import guess_class_name, protection_profile, registered_entries

        functions = {item.get("name"): item for item in (ghidra_result.artifacts.get("functions") or [])}
        onload_source = (functions.get("JNI_OnLoad") or {}).get("source")
        recovered = registered_entries(
            bundle.elf,
            onload_source=onload_source,
            image_base=ghidra_result.artifacts.get("image_base"),
            class_name=guess_class_name(bundle.elf),
        )
        existing = {
            (entry.get("class_name"), entry.get("method"), entry.get("descriptor"), entry.get("address"))
            for entry in entries
        }
        for item in recovered:
            candidate = _registered_to_entry(item, bundle)
            key = (candidate["class_name"], candidate["method"], candidate["descriptor"], candidate["address"])
            if key in existing:
                continue
            existing.add(key)
            entries.append(candidate)
    classes: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for entry in entries:
        package = entry.get("package") or ""
        class_name = entry.get("class_name") or entry.get("simple_class") or "JniReconstructed"
        classes.setdefault((package, class_name), []).append(entry)
    ordered_classes = sorted(
        classes.items(),
        key=lambda item: (_class_priority(item[0][0]), item[0][0], item[0][1]),
    )
    semantics = _collect_semantics(bundle, entries, ghidra_result) if classes else {}
    evidence_map = {
        "binary_sha256": bundle.elf.sha256,
        "project": project_name,
        "source_format": "native",
        "backend": "jni_symbols",
        "java_bytecode_available": False,
        "classes": [],
        "jni": jni_report,
        "claims": [],
    }
    from nexus.jni.registry import protection_profile

    evidence_map["protection"] = protection_profile(bundle.elf)
    if classes:
        recovered = 0
        for (package, class_name), class_entries in ordered_classes:
            path, content = _java_class_for_jni(bundle, class_name, package, class_entries, semantics)
            files[path] = content
            class_semantics = {
                entry["method"]: semantics[entry["symbol"]].to_dict()
                for entry in class_entries
                if entry.get("symbol") in semantics
            }
            recovered += sum(1 for value in class_semantics.values() if value.get("template") != "trace")
            evidence_map["classes"].append({
                "class_name": class_name,
                "package": package,
                "source_file": path,
                "methods": [entry["method"] for entry in class_entries],
                "evidence": [entry.get("evidence") for entry in class_entries],
                "jni_semantics": class_semantics,
                "confidence": 0.6 if class_semantics else 0.4,
            })
        classification = "jni_semantic_reconstruction" if recovered else "jni_bridge_reconstruction"
        evidence_map["backend"] = "jni_semantics" if recovered else "jni_symbols"
        evidence_map["semantic_methods"] = recovered
        evidence_map["semantic_trace_methods"] = sum(
            1
            for value in semantics.values()
            if value.template == "trace"
        )
    else:
        selected = bundle.functions
        if function_ids:
            ids = set(function_ids)
            selected = [function for function in bundle.functions if function.function_id in ids or function.name in ids]
        else:
            selected = [function for function in bundle.functions if function.confidence >= 0.5 and not function.name.startswith("_")]
        files["src/main/java/ReconstructedModule.java"] = _java_module(bundle, selected)
        evidence_map["classes"].append({
            "class_name": "ReconstructedModule",
            "package": "",
            "source_file": "src/main/java/ReconstructedModule.java",
            "methods": [function.name for function in selected],
            "evidence": [f"ev_{bundle.elf.sha256[:12]}_func_{function.entry:x}" for function in selected],
            "confidence": 0.25,
        })
        classification = "native_symbol_bridge"
    files["pom.xml"] = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<project xmlns="http://maven.apache.org/POM/4.0.0" '
        'xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" '
        'xsi:schemaLocation="http://maven.apache.org/POM/4.0.0 https://maven.apache.org/xsd/maven-4.0.0.xsd">\n'
        '  <modelVersion>4.0.0</modelVersion>\n'
        f'  <groupId>local.nexus</groupId>\n  <artifactId>{project_name}</artifactId>\n  <version>0.1.0</version>\n'
        '  <properties><maven.compiler.release>17</maven.compiler.release></properties>\n'
        '</project>\n'
    )
    files["build.gradle"] = (
        "plugins { id 'java' }\n"
        "group = 'local.nexus'\n"
        "version = '0.1.0'\n"
        "java { toolchain { languageVersion = JavaLanguageVersion.of(17) } }\n"
    )
    files["README.md"] = (
        f"# {project_name}\n\n"
        "Java behavior reconstructed from the JNI call sequence recovered by Ghidra. The analyzed artifact is native "
        "code, so it contains no Java bytecode and JADX/CFR cannot decompile it. Upload the containing APK/JAR to the "
        "Java/APK tab for a real JADX/CFR decompilation. Methods in this project are semantic approximations: class, "
        "method, field and signature names come from the native JNI calls, while control flow and unknown return types "
        "are approximated. It is not original source and is not behaviorally verified.\n"
    )
    limitations = [
        "This is JNI semantic reconstruction, not decompiled Java; the analyzed artifact contains no Java bytecode.",
        "JADX/CFR require an APK/JAR/DEX/class artifact and cannot decompile a native library.",
        "Class/method/field/signature names come from Ghidra-recovered JNI calls; control flow is approximated.",
        "Native symbols encode parameter types but not return types; unknown returns use Object placeholders.",
        "JNI class/method names are decoded from mangled symbols and may be ambiguous.",
        "Native behavior is not executed or differentially verified.",
    ]
    if evidence_map.get("protection", {}).get("runtime_decryption"):
        limitations.insert(
            0,
            "Runtime decryption/unpacking detected; the encrypted payload is not present in this static artifact.",
        )
    files["evidence_map.json"] = json.dumps(evidence_map, indent=2) + "\n"
    if out_dir is not None:
        root = Path(out_dir)
        for relative, content in files.items():
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
    return {
        "language": "java",
        "project_name": project_name,
        "classification": classification,
        "jni": jni_report,
        "files": files,
        "evidence_map": evidence_map,
        "limitations": limitations,
    }
