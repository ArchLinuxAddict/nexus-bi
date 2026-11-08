"""Approximate Java source reconstruction from DEX bytecode.

Dalvik registers are untyped.  The translator infers provisional types from
constants, field descriptors, invocation descriptors and arithmetic opcodes,
and refuses control flow, exceptions and object construction it cannot justify.
The generated Java is a structural approximation, not a decompilation claim.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from nexus.formats.java_class import descriptor_to_java
from nexus.reconstruction.java.classfile import (
    JavaTranslation,
    _default_value,
    _identifier,
    _primitive,
    _simple_type,
)


def _java_type_from_descriptor(descriptor: str) -> str:
    if not descriptor:
        return "Object"
    if descriptor.startswith("L") and descriptor.endswith(";"):
        return descriptor[1:-1].replace("/", ".")
    if descriptor.startswith("["):
        return descriptor_to_java(descriptor)
    return {"V": "void", "Z": "boolean", "B": "byte", "C": "char", "S": "short", "I": "int", "J": "long", "F": "float", "D": "double"}.get(descriptor, descriptor)


def _register_names(method: dict[str, Any]) -> tuple[dict[int, tuple[str, str]], dict[str, str], int]:
    code = method.get("code") or {}
    registers_size = int(code.get("registers_size") or 0)
    ins_size = int(code.get("ins_size") or 0)
    first_in = max(0, registers_size - ins_size)
    registers: dict[int, tuple[str, str]] = {}
    declarations: dict[str, str] = {}
    position = first_in
    if not method.get("static"):
        registers[position] = ("this", "this")
        position += 1
    for index, parameter in enumerate(method.get("parameters") or []):
        descriptor = parameter.get("descriptor", "")
        type_name = _java_type_from_descriptor(descriptor)
        registers[position] = (f"arg{index}", type_name)
        position += 2 if descriptor in {"J", "D"} else 1
    return registers, declarations, first_in


def _register(state: dict[int, tuple[str, str]], number: int, default: str = "long") -> tuple[str, str]:
    if number in state:
        return state[number]
    name = f"v{number}"
    state[number] = (name, default)
    return state[number]


def _binary_operator(name: str) -> str | None:
    base = name.split("/", 1)[0]
    return {
        "add-int": "+", "sub-int": "-", "mul-int": "*", "div-int": "/", "rem-int": "%",
        "and-int": "&", "or-int": "|", "xor-int": "^", "shl-int": "<<", "shr-int": ">>", "ushr-int": ">>>",
        "add-long": "+", "sub-long": "-", "mul-long": "*", "div-long": "/", "rem-long": "%",
        "and-long": "&", "or-long": "|", "xor-long": "^", "shl-long": "<<", "shr-long": ">>", "ushr-long": ">>>",
        "add-float": "+", "sub-float": "-", "mul-float": "*", "div-float": "/", "rem-float": "%",
        "add-double": "+", "sub-double": "-", "mul-double": "*", "div-double": "/", "rem-double": "%",
    }.get(base)


def _result_type(name: str) -> str:
    base = name.split("/", 1)[0]
    if base.endswith("-long"):
        return "long"
    if base.endswith("-float"):
        return "float"
    if base.endswith("-double"):
        return "double"
    return "int"


def _conversion_type(name: str) -> str:
    return {
        "int-to-long": "long", "int-to-float": "float", "int-to-double": "double",
        "long-to-int": "int", "long-to-float": "float", "long-to-double": "double",
        "float-to-int": "int", "float-to-long": "long", "float-to-double": "double",
        "double-to-int": "int", "double-to-long": "long", "double-to-float": "float",
        "int-to-byte": "byte", "int-to-char": "char", "int-to-short": "short",
    }.get(name, "int")


def translate_dex_method(parsed: dict[str, Any], method: dict[str, Any], *, class_name: str) -> JavaTranslation:
    code = method.get("code") or {}
    instructions = code.get("instructions") or []
    if not instructions:
        return JavaTranslation(False, reason="no code item")
    state, declarations, _ = _register_names(method)
    statements: list[str] = []
    evidence: list[int] = []
    last_result: tuple[str, str] | None = None
    return_type = method.get("return_type") or "void"

    def ensure(number: int, default: str = "long") -> tuple[str, str]:
        value = _register(state, number, default)
        if value[0].startswith("v") and value[0] not in declarations:
            # Provisional declaration; unknown Dalvik register typing is
            # annotated rather than silently treated as the original type.
            declarations[value[0]] = value[1]
        return value

    for instruction in instructions:
        mnemonic = instruction.get("mnemonic", "")
        pc = int(instruction.get("pc", 0))
        ops = instruction.get("operands") or []
        registers = instruction.get("registers") or []
        evidence.append(pc)
        try:
            if mnemonic in {"nop"}:
                continue
            if mnemonic in {"return-void"}:
                return JavaTranslation(True, statements, None, return_type, None, declarations, evidence)
            if mnemonic in {"return", "return-wide", "return-object"}:
                if not registers:
                    return JavaTranslation(False, statements, None, return_type, f"return at pc {pc} has no register", declarations, evidence)
                expression, _ = ensure(registers[0], "int" if mnemonic == "return" else "long" if mnemonic == "return-wide" else "java.lang.Object")
                return JavaTranslation(True, statements, expression, return_type, None, declarations, evidence)
            if mnemonic in {"const/4", "const/16", "const", "const/high16", "const-wide/16", "const-wide/32", "const-wide", "const-wide/high16"}:
                if len(ops) < 2:
                    return JavaTranslation(False, statements, None, return_type, f"malformed const at pc {pc}", declarations, evidence)
                register = registers[0] if registers else 0
                name = f"v{register}"
                mnemonic_type = "long" if mnemonic.startswith("const-wide") else "int"
                state[register] = (ops[1], mnemonic_type)
                declarations[name] = mnemonic_type
                continue
            if mnemonic in {"const-string", "const-string/jumbo"}:
                register = registers[0] if registers else 0
                value = ops[1] if len(ops) > 1 else '""'
                if not value.startswith('"'):
                    value = '"' + value.replace('"', '\\"') + '"'
                state[register] = (value, "java.lang.String")
                declarations[f"v{register}"] = "java.lang.String"
                continue
            if mnemonic in {"const-class"}:
                register = registers[0] if registers else 0
                value = ops[1] if len(ops) > 1 else "java.lang.Object"
                if value.startswith("L") and value.endswith(";"):
                    value = value[1:-1].replace("/", ".")
                state[register] = (f"{value}.class", "java.lang.Class")
                declarations[f"v{register}"] = "java.lang.Class"
                continue
            if mnemonic.startswith("move-result"):
                register = registers[0] if registers else 0
                if last_result is None:
                    return JavaTranslation(False, statements, None, return_type, f"move-result without a preceding invocation at pc {pc}", declarations, evidence)
                state[register] = last_result
                declarations[f"v{register}"] = last_result[1]
                last_result = None
                continue
            if mnemonic in {"move", "move/from16", "move/16", "move-wide", "move-wide/from16", "move-wide/16", "move-object", "move-object/from16", "move-object/16"}:
                if len(registers) < 2:
                    return JavaTranslation(False, statements, None, return_type, f"malformed move at pc {pc}", declarations, evidence)
                source = ensure(registers[1], "long" if "wide" in mnemonic else "int" if "object" not in mnemonic else "java.lang.Object")
                state[registers[0]] = source
                declarations[f"v{registers[0]}"] = source[1]
                continue
            if mnemonic in {"add-int", "sub-int", "mul-int", "div-int", "rem-int", "and-int", "or-int", "xor-int", "shl-int", "shr-int", "ushr-int", "add-long", "sub-long", "mul-long", "div-long", "rem-long", "and-long", "or-long", "xor-long", "shl-long", "shr-long", "ushr-long", "add-float", "sub-float", "mul-float", "div-float", "rem-float", "add-double", "sub-double", "mul-double", "div-double", "rem-double"}:
                if len(registers) < 3:
                    return JavaTranslation(False, statements, None, return_type, f"malformed {mnemonic} at pc {pc}", declarations, evidence)
                left = ensure(registers[1], _result_type(mnemonic))[0]
                right = ensure(registers[2], _result_type(mnemonic))[0]
                operator = _binary_operator(mnemonic) or "+"
                result = f"({left} {operator} {right})"
                state[registers[0]] = (result, _result_type(mnemonic))
                declarations[f"v{registers[0]}"] = _result_type(mnemonic)
                continue
            if mnemonic.endswith("/2addr"):
                base = mnemonic[:-7]
                if len(registers) < 2:
                    return JavaTranslation(False, statements, None, return_type, f"malformed {mnemonic} at pc {pc}", declarations, evidence)
                left = ensure(registers[0], _result_type(base))[0]
                right = ensure(registers[1], _result_type(base))[0]
                operator = _binary_operator(base) or "+"
                state[registers[0]] = (f"({left} {operator} {right})", _result_type(base))
                declarations[f"v{registers[0]}"] = _result_type(base)
                continue
            if mnemonic in {"add-int/lit16", "rsub-int", "mul-int/lit16", "div-int/lit16", "rem-int/lit16", "and-int/lit16", "or-int/lit16", "xor-int/lit16", "add-int/lit8", "rsub-int/lit8", "mul-int/lit8", "div-int/lit8", "rem-int/lit8", "and-int/lit8", "or-int/lit8", "xor-int/lit8", "shl-int/lit8", "shr-int/lit8", "ushr-int/lit8"}:
                if len(registers) < 2 or len(ops) < 3:
                    return JavaTranslation(False, statements, None, return_type, f"malformed {mnemonic} at pc {pc}", declarations, evidence)
                source = ensure(registers[1], "int")[0]
                immediate = ops[2]
                base = mnemonic.rsplit("/", 1)[0]
                operator = _binary_operator(base) or "+"
                if base == "rsub-int":
                    result = f"({immediate} - {source})"
                else:
                    result = f"({source} {operator} {immediate})"
                state[registers[0]] = (result, "int")
                declarations[f"v{registers[0]}"] = "int"
                continue
            if mnemonic in {"neg-int", "not-int", "neg-long", "not-long", "neg-float", "neg-double"}:
                source = ensure(registers[1], _result_type(mnemonic.replace("neg", "add").replace("not", "and")))[0]
                operator = "~" if mnemonic.startswith("not") else "-"
                state[registers[0]] = (f"({operator}{source})", _result_type(mnemonic.replace("neg", "add").replace("not", "and")))
                declarations[f"v{registers[0]}"] = state[registers[0]][1]
                continue
            if mnemonic in {"int-to-long", "int-to-float", "int-to-double", "long-to-int", "long-to-float", "long-to-double", "float-to-int", "float-to-long", "float-to-double", "double-to-int", "double-to-long", "double-to-float", "int-to-byte", "int-to-char", "int-to-short"}:
                source = ensure(registers[1], "int")[0]
                target = _conversion_type(mnemonic)
                state[registers[0]] = (f"(({target})({source}))", target)
                declarations[f"v{registers[0]}"] = target
                continue
            if mnemonic.startswith(("iget", "sget")):
                if len(ops) < 3:
                    return JavaTranslation(False, statements, None, return_type, f"malformed field access at pc {pc}", declarations, evidence)
                target = ops[2]
                if ":" not in target:
                    return JavaTranslation(False, statements, None, return_type, f"unresolved field at pc {pc}", declarations, evidence)
                owner_field, descriptor = target.rsplit(":", 1)
                owner, _, field_name = owner_field.rpartition(".")
                field_type = _java_type_from_descriptor(descriptor)
                if mnemonic.startswith("iget"):
                    receiver = ensure(registers[1], "java.lang.Object")[0]
                    expression = f"{receiver}.{field_name}"
                else:
                    expression = f"{owner}.{field_name}"
                state[registers[0]] = (expression, field_type)
                declarations[f"v{registers[0]}"] = field_type
                continue
            if mnemonic.startswith(("iput", "sput")):
                if len(ops) < 3:
                    return JavaTranslation(False, statements, None, return_type, f"malformed field store at pc {pc}", declarations, evidence)
                target = ops[2]
                if ":" not in target:
                    return JavaTranslation(False, statements, None, return_type, f"unresolved field at pc {pc}", declarations, evidence)
                owner_field, _ = target.rsplit(":", 1)
                owner, _, field_name = owner_field.rpartition(".")
                value = ensure(registers[0], "int")[0]
                if mnemonic.startswith("iput"):
                    receiver = ensure(registers[1], "java.lang.Object")[0]
                    statements.append(f"{receiver}.{field_name} = {value};")
                else:
                    statements.append(f"{owner}.{field_name} = {value};")
                continue
            if mnemonic.startswith("invoke-"):
                if len(ops) < 2 or ":" not in ops[1]:
                    return JavaTranslation(False, statements, None, return_type, f"unresolved invocation at pc {pc}", declarations, evidence)
                target, descriptor = ops[1].rsplit(":", 1)
                owner, _, method_name = target.rpartition(".")
                from nexus.formats.java_class import parse_method_descriptor
                call_return, parameters = parse_method_descriptor(descriptor)
                provided = registers
                if mnemonic.startswith(("invoke-virtual", "invoke-super", "invoke-direct", "invoke-interface")):
                    if not provided:
                        return JavaTranslation(False, statements, None, return_type, f"missing receiver at pc {pc}", declarations, evidence)
                    receiver = ensure(provided[0], "java.lang.Object")[0]
                    arguments = [ensure(item, "int")[0] for item in provided[1:]]
                    call = f"{receiver}.{method_name}({', '.join(arguments)})"
                else:
                    arguments = [ensure(item, "int")[0] for item in provided]
                    call = f"{owner}.{method_name}({', '.join(arguments)})"
                if method_name == "<init>":
                    statements.append(f"/* constructor call omitted: {owner} */")
                    last_result = None
                    continue
                if call_return == "void":
                    statements.append(call + ";")
                    last_result = None
                else:
                    last_result = (call, _java_type_from_descriptor(descriptor.rsplit(")", 1)[1]))
                continue
            if mnemonic in {"check-cast"}:
                value = ensure(registers[0], "java.lang.Object")[0]
                target = ops[1] if len(ops) > 1 else "java.lang.Object"
                target = _java_type_from_descriptor(target) if target.startswith(("L", "[")) else target
                state[registers[0]] = (f"(({target})({value}))", target)
                declarations[f"v{registers[0]}"] = target
                continue
            if mnemonic == "instance-of":
                value = ensure(registers[1], "java.lang.Object")[0]
                target = ops[1] if len(ops) > 1 else "java.lang.Object"
                target = _java_type_from_descriptor(target) if target.startswith(("L", "[")) else target
                state[registers[0]] = (f"({value} instanceof {target})", "boolean")
                declarations[f"v{registers[0]}"] = "boolean"
                continue
            if mnemonic == "array-length":
                value = ensure(registers[1], "java.lang.Object")[0]
                state[registers[0]] = (f"{value}.length", "int")
                declarations[f"v{registers[0]}"] = "int"
                continue
            if mnemonic.startswith("aget"):
                array = ensure(registers[1], "java.lang.Object")[0]
                index = ensure(registers[2], "int")[0]
                result_type = {"aget-wide": "long", "aget-object": "java.lang.Object", "aget-boolean": "boolean", "aget-byte": "byte", "aget-char": "char", "aget-short": "short"}.get(mnemonic, "int")
                state[registers[0]] = (f"{array}[{index}]", result_type)
                declarations[f"v{registers[0]}"] = result_type
                continue
            if mnemonic.startswith("aput"):
                value = ensure(registers[0], "int")[0]
                array = ensure(registers[1], "java.lang.Object")[0]
                index = ensure(registers[2], "int")[0]
                statements.append(f"{array}[{index}] = {value};")
                continue
            if mnemonic in {"if-eq", "if-ne", "if-lt", "if-ge", "if-gt", "if-le", "if-eqz", "if-nez", "if-ltz", "if-gez", "if-gtz", "if-lez", "goto", "goto/16", "goto/32", "packed-switch", "sparse-switch"}:
                return JavaTranslation(False, statements, None, return_type, f"control flow {mnemonic} at pc {pc} is unresolved", declarations, evidence)
            if mnemonic in {"new-instance", "new-array", "filled-new-array", "filled-new-array/range", "fill-array-data", "throw", "monitor-enter", "monitor-exit", "move-exception", "invoke-custom", "invoke-custom/range", "invoke-polymorphic", "invoke-polymorphic/range"}:
                return JavaTranslation(False, statements, None, return_type, f"{mnemonic} at pc {pc} is unresolved", declarations, evidence)
            return JavaTranslation(False, statements, None, return_type, f"unsupported instruction {mnemonic} at pc {pc}", declarations, evidence)
        except (IndexError, KeyError, ValueError) as exc:
            return JavaTranslation(False, statements, None, return_type, f"{exc} at pc {pc}", declarations, evidence)
    if return_type == "void":
        return JavaTranslation(True, statements, None, return_type, None, declarations, evidence)
    return JavaTranslation(False, statements, None, return_type, "return value is not proven", declarations, evidence)


def _method_signature(method: dict[str, Any], class_name: str) -> tuple[str, list[str]]:
    modifiers = [item for item in method.get("access", []) if item in {"public", "private", "protected", "static", "final", "synchronized", "native", "abstract"}]
    parameters = []
    for index, parameter in enumerate(method.get("parameters") or []):
        descriptor = parameter.get("descriptor", "") or parameter.get("type") or "I"
        type_name = _java_type_from_descriptor(descriptor)
        parameters.append(f"{type_name} arg{index}")
    return " ".join(modifiers), parameters


def _emit_method(parsed: dict[str, Any], method: dict[str, Any], *, class_name: str, is_interface: bool, is_annotation: bool) -> tuple[str, dict[str, Any]]:
    name = method.get("name", "unknown")
    return_type = _java_type_from_descriptor(method.get("return_type") or "V")
    if name == "<clinit>":
        return "", {"name": name, "classification": "omitted_class_initializer", "confidence": 0.0, "reason": "static initializer omitted"}
    if name == "<init>":
        return "", {"name": name, "classification": "omitted_constructor", "confidence": 0.0, "reason": "constructor omitted"}
    if method.get("synthetic"):
        return "", {"name": name, "classification": "omitted_synthetic", "confidence": 0.0, "reason": "synthetic method omitted"}
    modifiers, parameters = _method_signature(method, class_name)
    signature = f"    {modifiers + ' ' if modifiers else ''}{return_type} {_identifier(name)}({', '.join(parameters)})"
    if is_interface or is_annotation or method.get("abstract"):
        return signature + ";\n", {"name": name, "classification": "declaration", "confidence": 0.8, "return_type": return_type}
    if method.get("native"):
        return signature + ";\n", {"name": name, "classification": "native_declaration", "confidence": 0.9, "return_type": return_type}
    translation = translate_dex_method(parsed, method, class_name=class_name)
    if translation.success:
        lines = [signature + " {"]
        for local_name, local_type in sorted(translation.declarations.items()):
            lines.append(f"        {_simple_type(local_type)} {local_name} = {_default_value(local_type)};")
        for statement in translation.statements:
            lines.append(f"        {statement}")
        if return_type == "void":
            lines.append("        return;")
        elif translation.return_expr is not None:
            expression = translation.return_expr
            if not _primitive(return_type) and expression != "null":
                expression = f"({return_type})({expression})"
            lines.append(f"        return {expression};")
        else:
            lines.append('        throw new UnsupportedOperationException("NEXUS-BI: return value not proven");')
        lines.append("    }")
        return "\n".join(lines) + "\n", {"name": name, "classification": "straight_line_reconstruction", "confidence": 0.4, "return_type": return_type, "bytecode_pcs": translation.evidence}
    body = [
        signature + " {",
        f"        // NEXUS-BI unresolved: {translation.reason}",
        f"        // Dalvik evidence PCs: {', '.join(str(pc) for pc in translation.evidence[:24])}",
    ]
    for instruction in (method.get("code") or {}).get("instructions", [])[:24]:
        body.append(f"        //   {instruction.get('pc')}: {instruction.get('mnemonic')} {' '.join(instruction.get('operands') or [])}")
    body.append('        throw new UnsupportedOperationException("NEXUS-BI: unresolved Dalvik reconstruction");')
    body.append("    }")
    return "\n".join(body) + "\n", {"name": name, "classification": "unresolved", "confidence": 0.12, "reason": translation.reason, "return_type": return_type, "bytecode_pcs": translation.evidence[:24]}


def reconstruct_java_dex_class_source(parsed_class: dict[str, Any], *, project_name: str = "nexus_java_reconstruction") -> tuple[str, dict[str, Any]]:
    class_name = parsed_class.get("simple_name") or "Reconstructed"
    dotted = parsed_class.get("name") or class_name
    package = parsed_class.get("package") or ""
    is_interface = bool(parsed_class.get("is_interface"))
    is_annotation = bool(parsed_class.get("is_annotation"))
    is_enum = bool(parsed_class.get("is_enum"))
    access = parsed_class.get("access") or []
    class_modifiers = [item for item in access if item in {"public", "final", "abstract", "static"}]
    lines = []
    if package:
        lines.append(f"package {package};")
        lines.append("")
    lines.append(f"// NEXUS-BI approximate DEX reconstruction of {dotted}.")
    lines.append(f"// Source: {parsed_class.get('source_file') or 'unknown'}; Dalvik bytecode translated only for a straight-line subset.")
    lines.append("")
    keyword = "@interface" if is_annotation else "interface" if is_interface else "enum" if is_enum else "class"
    declaration = " ".join(class_modifiers + [keyword, class_name])
    extends = parsed_class.get("superclass")
    if extends and extends not in {"java.lang.Object", "java.lang.Enum", "java.lang.annotation.Annotation"}:
        declaration += (" extends " if is_interface and not is_annotation else " extends ") + _simple_type(extends)
    interfaces = [_simple_type(item) for item in parsed_class.get("interfaces") or [] if item]
    if interfaces and not is_annotation:
        declaration += (" extends " if is_interface else " implements ") + ", ".join(interfaces)
    lines.append(declaration + " {")
    if is_enum:
        lines.append("    ;")
    field_lines = []
    for field in parsed_class.get("fields") or []:
        if field.get("synthetic"):
            continue
        modifiers = [item for item in field.get("access", []) if item in {"public", "private", "protected", "static", "final", "volatile", "transient"}]
        field_type = _java_type_from_descriptor(field.get("type") or "Ljava/lang/Object;")
        initializer = ""
        if field.get("final") or field.get("static"):
            initializer = f" = {_default_value(field_type)}"
        field_lines.append(f"    {' '.join(modifiers + [field_type, _identifier(field.get('name', 'field'))])}{initializer};")
    if field_lines:
        lines.extend(field_lines)
        lines.append("")
    methods = parsed_class.get("methods") or []
    if not any(method.get("name") == "<init>" for method in methods) and not is_interface and not is_annotation and not is_enum and extends in {None, "java.lang.Object"}:
        lines.append(f"    public {class_name}() {{")
        lines.append("        super();")
        lines.append("    }")
        lines.append("")
    method_evidence = []
    for method in methods:
        emitted, evidence = _emit_method(parsed_class, method, class_name=dotted, is_interface=is_interface, is_annotation=is_annotation)
        method_evidence.append(evidence)
        if emitted:
            lines.append(emitted.rstrip())
            lines.append("")
    lines.append("}")
    lines.append("")
    evidence = {
        "class_name": dotted,
        "package": package,
        "source_file": parsed_class.get("source_file"),
        "methods": method_evidence,
        "fields": [{"name": item.get("name"), "type": item.get("type"), "descriptor": item.get("descriptor")} for item in parsed_class.get("fields") or []],
        "limitations": [
            "Dalvik registers are untyped; local declarations are provisional.",
            "Control flow, exceptions, object construction, lambdas and polymorphic calls are unresolved.",
            "Constructors and class initializers are omitted.",
            "Compilation is a syntax check, not behavioral verification.",
        ],
        "confidence": 0.4 if any(item.get("classification") == "straight_line_reconstruction" for item in method_evidence) else 0.18,
    }
    return "\n".join(lines), evidence


def reconstruct_java_dex(
    dex: dict[str, Any],
    *,
    out_dir: str | Path | None = None,
    project_name: str = "nexus_java_reconstruction",
    max_classes: int = 2000,
) -> dict[str, Any]:
    files: dict[str, str] = {}
    classes = dex.get("classes") or []
    evidence_classes = []
    for parsed_class in classes[:max_classes]:
        source, evidence = reconstruct_java_dex_class_source(parsed_class, project_name=project_name)
        package = parsed_class.get("package") or ""
        class_name = parsed_class.get("simple_name") or "Reconstructed"
        path = f"src/main/java/{package.replace('.', '/') + '/' if package else ''}{class_name}.java"
        files[path] = source
        evidence_classes.append(evidence)
    files["pom.xml"] = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<project xmlns="http://maven.apache.org/POM/4.0.0" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" '
        'xsi:schemaLocation="http://maven.apache.org/POM/4.0.0 https://maven.apache.org/xsd/maven-4.0.0.xsd">\n'
        "  <modelVersion>4.0.0</modelVersion>\n  <groupId>local.nexus</groupId>\n"
        f"  <artifactId>{project_name}</artifactId>\n  <version>0.1.0</version>\n"
        "  <properties><maven.compiler.release>17</maven.compiler.release></properties>\n</project>\n"
    )
    files["build.gradle"] = (
        "plugins { id 'java' }\n"
        "group = 'local.nexus'\nversion = '0.1.0'\n"
        "java { toolchain { languageVersion = JavaLanguageVersion.of(17) } }\n"
    )
    files["README.md"] = (
        f"# {project_name}\n\n"
        "Approximate Java reconstruction from DEX bytecode. Declarations and a straight-line Dalvik subset are preserved. "
        "Unresolved control flow and runtime constructs are explicit. No behavioral equivalence is claimed.\n"
    )
    evidence = {
        "source_format": "DEX",
        "dex_version": dex.get("version"),
        "classes_requested": len(classes),
        "classes_reconstructed": min(len(classes), max_classes),
        "classes": evidence_classes,
        "limitations": [
            "Only a bounded number of classes is emitted per project.",
            "Dalvik register types and original names are not fully recoverable.",
            "No execution or behavioral verification.",
        ],
    }
    files["evidence_map.json"] = json.dumps(evidence, indent=2) + "\n"
    if out_dir is not None:
        root = Path(out_dir)
        for relative, content in files.items():
            target = root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
    return {
        "language": "java",
        "source_format": "DEX",
        "project_name": project_name,
        "classification": "structural_dex_reconstruction",
        "dex": dex,
        "files": files,
        "evidence_map": evidence,
        "limitations": evidence["limitations"],
    }
