"""Approximate Java source reconstruction from class-file bytecode.

The emitter preserves class/field/method declarations and translates a
straight-line bytecode subset into Java statements.  Control-flow, exceptions,
invokedynamic, lambdas and vector/string-concat machinery are explicit
unresolved constructs.  The output is compiled as a syntax check; compilation is
not behavioral verification.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from nexus.formats.java_class import descriptor_to_java, parse_method_descriptor

JAVA_KEYWORDS = {
    "abstract", "assert", "boolean", "break", "byte", "case", "catch", "char", "class", "const", "continue",
    "default", "do", "double", "else", "enum", "extends", "final", "finally", "float", "for", "goto", "if",
    "implements", "import", "instanceof", "int", "interface", "long", "native", "new", "package", "private",
    "protected", "public", "return", "short", "static", "strictfp", "super", "switch", "synchronized", "this",
    "throw", "throws", "transient", "try", "void", "volatile", "while", "true", "false", "null",
}


def _identifier(value: str, fallback: str = "value") -> str:
    text = re.sub(r"[^A-Za-z0-9_$]", "_", value or "")
    if not text:
        text = fallback
    if text[0].isdigit():
        text = f"_{text}"
    if text in JAVA_KEYWORDS:
        text = f"{text}_"
    return text


def _simple_type(java_type: str) -> str:
    return java_type.replace("$", ".")


def _default_value(java_type: str) -> str:
    if java_type in {"byte", "short", "int", "long", "float", "double"}:
        return "0"
    if java_type == "boolean":
        return "false"
    if java_type == "char":
        return "'\\0'"
    return "null"


def _primitive(java_type: str) -> bool:
    return java_type in {"byte", "short", "int", "long", "float", "double", "boolean", "char"}


@dataclass
class Local:
    name: str
    type: str
    declared: bool = False


@dataclass
class JavaTranslation:
    success: bool
    statements: list[str] = field(default_factory=list)
    return_expr: str | None = None
    return_type: str = "void"
    reason: str | None = None
    declarations: dict[str, str] = field(default_factory=dict)
    evidence: list[int] = field(default_factory=list)


def _local_name(raw: str | None, index: int, fallback: str) -> str:
    if raw:
        name = _identifier(raw, fallback)
        if name not in JAVA_KEYWORDS:
            return name
    return fallback


def _descriptor_type(value: str) -> str:
    return descriptor_to_java(value) if value else "Object"


def _instruction_operand(instruction: dict[str, Any], index: int = 0) -> str:
    operands = instruction.get("operands") or []
    return str(operands[index]) if index < len(operands) else ""


def _load_index(mnemonic: str, instruction: dict[str, Any]) -> tuple[int, str]:
    if "_" in mnemonic:
        suffix = mnemonic.rsplit("_", 1)[1]
        if suffix.isdigit():
            return int(suffix), mnemonic.split("_", 1)[0]
    return int(_instruction_operand(instruction) or "0"), mnemonic


def _stack_type_for_load(prefix: str) -> str:
    return {"iload": "int", "lload": "long", "fload": "float", "dload": "double", "aload": "java.lang.Object"}.get(prefix, "int")


def _statement(expression: str) -> str:
    return expression if expression.endswith(";") else expression + ";"


def _eval_linear(instructions: list[dict[str, Any]], locals_: dict[int, Local], prefixes: tuple[str, ...]) -> tuple[list[tuple[str, str]], str | None]:
    stack: list[tuple[str, str]] = []
    for instruction in instructions:
        mnemonic = instruction.get("mnemonic", "")
        if mnemonic in {"iload", "lload", "fload", "dload", "aload"} or mnemonic.startswith(("iload_", "lload_", "fload_", "dload_", "aload_")):
            slot, prefix = _load_index(mnemonic, instruction)
            local = locals_.get(slot)
            if local is None:
                return stack, f"unresolved local {slot}"
            type_name = local.type if local.type not in {"java.lang.Object", "unknown"} else _stack_type_for_load(prefix)
            stack.append((local.name, type_name))
        elif mnemonic in {"iconst_m1", "iconst_0", "iconst_1", "iconst_2", "iconst_3", "iconst_4", "iconst_5"}:
            stack.append((mnemonic.rsplit("_", 1)[1], "int"))
        elif mnemonic in {"lconst_0", "lconst_1"}:
            stack.append((mnemonic.rsplit("_", 1)[1] + "L", "long"))
        elif mnemonic in {"fconst_0", "fconst_1", "fconst_2"}:
            stack.append((mnemonic.rsplit("_", 1)[1] + ".0f", "float"))
        elif mnemonic in {"dconst_0", "dconst_1"}:
            stack.append((mnemonic.rsplit("_", 1)[1] + ".0d", "double"))
        elif mnemonic in {"bipush", "sipush"}:
            stack.append((_instruction_operand(instruction), "int"))
        elif mnemonic in {"ldc", "ldc_w", "ldc2_w"}:
            value = _instruction_operand(instruction)
            if value.startswith("#"):
                return stack, f"unresolved constant {value}"
            stack.append((value, "java.lang.String" if value.startswith('"') else "int"))
        elif mnemonic in {"iadd", "isub", "imul", "idiv", "irem", "iand", "ior", "ixor", "ishl", "ishr", "iushr", "ladd", "lsub", "lmul", "ldiv", "lrem", "land", "lor", "lxor", "lshl", "lshr", "lushr"}:
            if len(stack) < 2:
                return stack, f"{mnemonic} underflow"
            right = stack.pop()[0]
            left = stack.pop()[0]
            operator = {
                "iadd": "+", "ladd": "+", "isub": "-", "lsub": "-", "imul": "*", "lmul": "*",
                "idiv": "/", "ldiv": "/", "irem": "%", "lrem": "%", "iand": "&", "land": "&",
                "ior": "|", "lor": "|", "ixor": "^", "lxor": "^", "ishl": "<<", "lshl": "<<",
                "ishr": ">>", "lshr": ">>", "iushr": ">>>", "lushr": ">>>",
            }[mnemonic]
            result_type = "long" if mnemonic.startswith("l") else "int"
            stack.append((f"({left} {operator} {right})", result_type))
        elif mnemonic in {"i2l", "i2f", "i2d", "l2i", "l2f", "l2d", "f2i", "f2l", "f2d", "d2i", "d2l", "d2f"}:
            if not stack:
                return stack, f"{mnemonic} underflow"
            value = stack.pop()[0]
            target = {"i2l": "long", "i2f": "float", "i2d": "double", "l2i": "int", "l2f": "float", "l2d": "double", "f2i": "int", "f2l": "long", "f2d": "double", "d2i": "int", "d2l": "long", "d2f": "float"}[mnemonic]
            stack.append((f"(({target})({value}))", target))
        elif mnemonic in {"ineg", "lneg", "fneg", "dneg"}:
            if not stack:
                return stack, f"{mnemonic} underflow"
            value, type_name = stack.pop()
            stack.append((f"(-{value})", type_name))
        elif mnemonic in {"ireturn", "lreturn", "freturn", "dreturn", "areturn", "return"}:
            return stack, None
        elif mnemonic == "nop":
            continue
        else:
            return stack, f"unsupported expression instruction {mnemonic}"
    return stack, "block does not end in return"


def _try_if_return(parsed: dict[str, Any], method: dict[str, Any], class_name: str) -> JavaTranslation | None:
    code = method.get("code") or {}
    instructions = code.get("instructions") or []
    if len(instructions) < 5:
        return None
    branch_index = None
    for index, instruction in enumerate(instructions):
        if instruction.get("mnemonic", "").startswith("if") and instruction.get("branch_targets"):
            if branch_index is not None:
                return None
            branch_index = index
    if branch_index is None:
        return None
    branch = instructions[branch_index]
    target = int(branch["branch_targets"][0])
    target_index = next((index for index, item in enumerate(instructions) if int(item.get("pc", 0)) == target), None)
    if target_index is None or target_index <= branch_index + 1:
        return None
    locals_ = _method_parameter_names(method, class_name)
    prefix_stack, prefix_error = _eval_linear(instructions[:branch_index], locals_, ("iload", "lload", "fload", "dload", "aload"))
    if prefix_error and prefix_error != "block does not end in return":
        return None
    if len(prefix_stack) < 2 and branch["mnemonic"] not in {"ifeq", "ifne", "iflt", "ifge", "ifgt", "ifle", "ifnull", "ifnonnull"}:
        return None
    true_instructions = instructions[branch_index + 1:target_index]
    false_instructions = instructions[target_index:]
    true_stack, true_error = _eval_linear(true_instructions, locals_, ("iload", "lload", "fload", "dload", "aload"))
    false_stack, false_error = _eval_linear(false_instructions, locals_, ("iload", "lload", "fload", "dload", "aload"))
    if true_error is not None or false_error is not None or not true_stack or not false_stack:
        return None
    if len(true_stack) != 1 or len(false_stack) != 1:
        return None
    return_type = method.get("return_type") or "void"
    if return_type == "void":
        return None
    left, right = (prefix_stack[-2][0], prefix_stack[-1][0]) if len(prefix_stack) >= 2 else (prefix_stack[-1][0], "null")
    inverse = {
        "if_icmpeq": "!=", "if_icmpne": "==", "if_icmplt": ">=", "if_icmpge": "<",
        "if_icmpgt": "<=", "if_icmple": ">", "if_acmpeq": "!=", "if_acmpne": "==",
        "ifeq": "!=", "ifne": "==", "iflt": ">=", "ifge": "<", "ifgt": "<=", "ifle": ">",
        "ifnull": "!=", "ifnonnull": "==",
    }.get(branch["mnemonic"])
    if inverse is None:
        return None
    comparison_right = "null" if branch["mnemonic"] in {"ifnull", "ifnonnull"} else ("0" if branch["mnemonic"].startswith("if") and not branch["mnemonic"].startswith("if_") else right)
    condition = f"{left} {inverse} {comparison_right}"
    statements = [f"if ({condition}) {{ return {true_stack[0][0]}; }}"]
    evidence = [int(item.get("pc", 0)) for item in instructions[: branch_index + 1]] + [int(item.get("pc", 0)) for item in true_instructions] + [int(item.get("pc", 0)) for item in false_instructions]
    return JavaTranslation(True, statements, false_stack[0][0], return_type, None, {}, evidence)


def _method_parameter_names(method: dict[str, Any], class_name: str) -> dict[int, Local]:
    code = method.get("code") or {}
    table = {item.get("slot"): item for item in code.get("local_variables", [])}
    locals_: dict[int, Local] = {}
    slot = 0
    if not method.get("static"):
        locals_[0] = Local("this", class_name, True)
        slot = 1
    for index, parameter in enumerate(method.get("parameters") or []):
        descriptor = parameter.get("descriptor", "")
        raw = table.get(slot, {})
        type_name = raw.get("type") or _descriptor_type(descriptor)
        name = _local_name(raw.get("name"), index, f"arg{index}")
        locals_[slot] = Local(name, type_name, True)
        slot += int(parameter.get("slot_width") or 1)
    for item in code.get("local_variables", []) or []:
        item_slot = int(item.get("slot", 0))
        if item_slot in locals_:
            continue
        type_name = item.get("type") or _descriptor_type(item.get("descriptor", ""))
        locals_[item_slot] = Local(_local_name(item.get("name"), item_slot, f"local{item_slot}"), type_name, False)
    return locals_


def translate_method(
    parsed: dict[str, Any],
    method: dict[str, Any],
    *,
    class_name: str,
    superclass: str | None = None,
) -> JavaTranslation:
    code = method.get("code") or {}
    instructions = code.get("instructions") or []
    if not instructions:
        return JavaTranslation(False, reason="no code attribute")
    locals_ = _method_parameter_names(method, class_name)
    stack: list[tuple[str, str]] = []
    statements: list[str] = []
    evidence: list[int] = []
    declarations: dict[str, str] = {}

    def ensure_local(slot: int, prefix: str) -> Local:
        if slot in locals_:
            return locals_[slot]
        type_name = _stack_type_for_load(prefix) if prefix.endswith("load") else {
            "istore": "int", "lstore": "long", "fstore": "float", "dstore": "double", "astore": "java.lang.Object",
        }.get(prefix, "int")
        local = Local(f"local{slot}", type_name, False)
        locals_[slot] = local
        declarations[local.name] = type_name
        return local

    def pop() -> tuple[str, str]:
        if not stack:
            raise ValueError("operand stack underflow")
        return stack.pop()

    def pop_type(expected: str | None = None) -> tuple[str, str]:
        value = pop()
        if expected and value[1] not in {expected, "java.lang.Object"}:
            return value
        return value

    def binary(operator: str, result_type: str = "int", cast: str | None = None) -> None:
        right = pop()[0]
        left = pop()[0]
        expression = f"({left} {operator} {right})"
        if cast:
            expression = f"(({cast})({expression}))"
        stack.append((expression, result_type))

    def unary(operator: str, result_type: str) -> None:
        value = pop()[0]
        stack.append((f"({operator}{value})", result_type))

    simple_if = _try_if_return(parsed, method, class_name)
    if simple_if is not None:
        return simple_if

    def method_args(descriptor: str) -> list[str]:
        _, parameters = parse_method_descriptor(descriptor)
        values = []
        for _ in parameters:
            values.append(pop()[0])
        values.reverse()
        return values

    try:
        for instruction in instructions:
            mnemonic = instruction.get("mnemonic", "")
            pc = int(instruction.get("pc", 0))
            evidence.append(pc)
            if mnemonic == "nop":
                continue
            if mnemonic in {"aconst_null"}:
                stack.append(("null", "java.lang.Object"))
                continue
            if mnemonic in {"iconst_m1", "iconst_0", "iconst_1", "iconst_2", "iconst_3", "iconst_4", "iconst_5"}:
                stack.append((mnemonic.rsplit("_", 1)[1], "int"))
                continue
            if mnemonic in {"lconst_0", "lconst_1"}:
                stack.append((mnemonic.rsplit("_", 1)[1] + "L", "long"))
                continue
            if mnemonic in {"fconst_0", "fconst_1", "fconst_2"}:
                stack.append((mnemonic.rsplit("_", 1)[1] + ".0f", "float"))
                continue
            if mnemonic in {"dconst_0", "dconst_1"}:
                stack.append((mnemonic.rsplit("_", 1)[1] + ".0d", "double"))
                continue
            if mnemonic in {"bipush", "sipush"}:
                stack.append((_instruction_operand(instruction), "int"))
                continue
            if mnemonic in {"ldc", "ldc_w", "ldc2_w"}:
                value = _instruction_operand(instruction)
                if value.startswith("#"):
                    return JavaTranslation(False, statements, None, method.get("return_type", "void"), f"unresolved constant at pc {pc}", declarations, evidence)
                java_type = "java.lang.String" if value.startswith('"') else "long" if value.endswith(("L", "l")) else "double" if value.endswith(("D", "d")) else "float" if value.endswith("f") else "int"
                stack.append((value, java_type))
                continue
            if mnemonic.startswith(("iload", "lload", "fload", "dload", "aload")):
                slot, prefix = _load_index(mnemonic, instruction)
                local = ensure_local(slot, prefix if prefix.endswith("load") else "iload")
                type_name = local.type if local.type not in {"java.lang.Object", "unknown"} else _stack_type_for_load(prefix)
                stack.append((local.name, type_name))
                continue
            if mnemonic.startswith(("istore", "lstore", "fstore", "dstore", "astore")):
                slot, prefix = _load_index(mnemonic, instruction)
                value, value_type = pop()
                local = ensure_local(slot, prefix if prefix.endswith("store") else "istore")
                if local.name in declarations and value_type == "java.lang.Object":
                    declarations[local.name] = "java.lang.Object"
                statements.append(f"{local.name} = {value};")
                continue
            if mnemonic == "iinc":
                slot = int(_instruction_operand(instruction, 0) or 0)
                amount = _instruction_operand(instruction, 1) or "0"
                local = ensure_local(slot, "iload")
                statements.append(f"{local.name} += {amount};")
                continue
            if mnemonic in {"iadd", "ladd", "fadd", "dadd"}:
                binary("+", mnemonic[0] + "add" if False else {"iadd": "int", "ladd": "long", "fadd": "float", "dadd": "double"}[mnemonic])
                continue
            if mnemonic in {"isub", "lsub", "fsub", "dsub"}:
                binary("-", {"isub": "int", "lsub": "long", "fsub": "float", "dsub": "double"}[mnemonic])
                continue
            if mnemonic in {"imul", "lmul", "fmul", "dmul"}:
                binary("*", {"imul": "int", "lmul": "long", "fmul": "float", "dmul": "double"}[mnemonic])
                continue
            if mnemonic in {"idiv", "ldiv", "fdiv", "ddiv"}:
                binary("/", {"idiv": "int", "ldiv": "long", "fdiv": "float", "ddiv": "double"}[mnemonic])
                continue
            if mnemonic in {"irem", "lrem", "frem", "drem"}:
                binary("%", {"irem": "int", "lrem": "long", "frem": "float", "drem": "double"}[mnemonic])
                continue
            if mnemonic in {"ishl", "lshl"}:
                binary("<<", "long" if mnemonic == "lshl" else "int")
                continue
            if mnemonic in {"ishr", "lshr"}:
                binary(">>", "long" if mnemonic == "lshr" else "int")
                continue
            if mnemonic in {"iushr", "lushr"}:
                binary(">>>", "long" if mnemonic == "lushr" else "int")
                continue
            if mnemonic in {"iand", "land"}:
                binary("&", "long" if mnemonic == "land" else "int")
                continue
            if mnemonic in {"ior", "lor"}:
                binary("|", "long" if mnemonic == "lor" else "int")
                continue
            if mnemonic in {"ixor", "lxor"}:
                binary("^", "long" if mnemonic == "lxor" else "int")
                continue
            if mnemonic in {"ineg", "lneg", "fneg", "dneg"}:
                unary("-", {"ineg": "int", "lneg": "long", "fneg": "float", "dneg": "double"}[mnemonic])
                continue
            if mnemonic in {"i2l", "i2f", "i2d", "l2i", "l2f", "l2d", "f2i", "f2l", "f2d", "d2i", "d2l", "d2f", "i2b", "i2c", "i2s"}:
                value = pop()[0]
                target = {"i2l": "long", "i2f": "float", "i2d": "double", "l2i": "int", "l2f": "float", "l2d": "double", "f2i": "int", "f2l": "long", "f2d": "double", "d2i": "int", "d2l": "long", "d2f": "float", "i2b": "byte", "i2c": "char", "i2s": "short"}[mnemonic]
                stack.append((f"(({target})({value}))", target))
                continue
            if mnemonic in {"lcmp", "fcmpl", "fcmpg", "dcmpl", "dcmpg"}:
                right = pop()[0]
                left = pop()[0]
                stack.append((f"({left} < {right} ? -1 : ({left} == {right} ? 0 : 1))", "int"))
                continue
            if mnemonic in {"getstatic", "getfield"}:
                operand = _instruction_operand(instruction)
                if ":" not in operand:
                    return JavaTranslation(False, statements, None, method.get("return_type", "void"), f"unresolved field at pc {pc}", declarations, evidence)
                owner_field, descriptor = operand.rsplit(":", 1)
                owner, _, field_name = owner_field.rpartition(".")
                if mnemonic == "getstatic":
                    stack.append((f"{owner}.{field_name}", _descriptor_type(descriptor)))
                else:
                    receiver = pop()[0]
                    stack.append((f"{receiver}.{field_name}", _descriptor_type(descriptor)))
                continue
            if mnemonic in {"putstatic", "putfield"}:
                operand = _instruction_operand(instruction)
                if ":" not in operand:
                    return JavaTranslation(False, statements, None, method.get("return_type", "void"), f"unresolved field at pc {pc}", declarations, evidence)
                owner_field, _ = operand.rsplit(":", 1)
                owner, _, field_name = owner_field.rpartition(".")
                value = pop()[0]
                if mnemonic == "putstatic":
                    statements.append(f"{owner}.{field_name} = {value};")
                else:
                    receiver = pop()[0]
                    statements.append(f"{receiver}.{field_name} = {value};")
                continue
            if mnemonic in {"invokestatic", "invokevirtual", "invokespecial", "invokeinterface"}:
                operand = _instruction_operand(instruction)
                if ":" not in operand:
                    return JavaTranslation(False, statements, None, method.get("return_type", "void"), f"unresolved invocation at pc {pc}", declarations, evidence)
                target, descriptor = operand.rsplit(":", 1)
                owner, _, method_name = target.rpartition(".")
                args = method_args(descriptor)
                return_type, _ = parse_method_descriptor(descriptor)
                if method_name == "<init>":
                    if mnemonic == "invokespecial":
                        if stack:
                            stack.pop()
                        call = "super" if superclass and owner == superclass else "this"
                        statements.append(f"{call}({', '.join(args)});")
                        continue
                    return JavaTranslation(False, statements, None, method.get("return_type", "void"), f"constructor invocation at pc {pc} is unresolved", declarations, evidence)
                call = f"{owner}.{method_name}({', '.join(args)})"
                if mnemonic in {"invokevirtual", "invokeinterface"}:
                    receiver = pop()[0]
                    call = f"{receiver}.{method_name}({', '.join(args)})"
                if return_type == "void":
                    statements.append(_statement(call))
                else:
                    stack.append((call, return_type))
                continue
            if mnemonic in {"ireturn", "lreturn", "freturn", "dreturn", "areturn", "return"}:
                return_type = method.get("return_type") or "void"
                if mnemonic == "return":
                    return JavaTranslation(True, statements, None, return_type, None, declarations, evidence)
                if not stack:
                    return JavaTranslation(False, statements, None, return_type, f"return at pc {pc} has no operand", declarations, evidence)
                expression = pop()[0]
                return JavaTranslation(True, statements, expression, return_type, None, declarations, evidence)
            if mnemonic in {"pop"}:
                if stack:
                    stack.pop()
                continue
            if mnemonic in {"pop2"}:
                if stack:
                    stack.pop()
                if stack:
                    stack.pop()
                continue
            if mnemonic == "dup":
                if not stack:
                    return JavaTranslation(False, statements, None, method.get("return_type", "void"), f"dup underflow at pc {pc}", declarations, evidence)
                stack.append(stack[-1])
                continue
            if mnemonic in {"swap", "dup_x1", "dup_x2", "dup2", "dup2_x1", "dup2_x2"}:
                return JavaTranslation(False, statements, None, method.get("return_type", "void"), f"stack permutation {mnemonic} at pc {pc} is unresolved", declarations, evidence)
            if mnemonic in {"checkcast"}:
                target = _instruction_operand(instruction)
                value = pop()[0]
                stack.append((f"(({target})({value}))", target))
                continue
            if mnemonic == "instanceof":
                target = _instruction_operand(instruction)
                value = pop()[0]
                stack.append((f"({value} instanceof {target})", "boolean"))
                continue
            if mnemonic == "arraylength":
                value = pop()[0]
                stack.append((f"{value}.length", "int"))
                continue
            if mnemonic in {"iaload", "laload", "faload", "daload", "aaload", "baload", "caload", "saload"}:
                index = pop()[0]
                array = pop()[0]
                result_type = {"iaload": "int", "laload": "long", "faload": "float", "daload": "double", "aaload": "java.lang.Object", "baload": "byte", "caload": "char", "saload": "short"}[mnemonic]
                stack.append((f"{array}[{index}]", result_type))
                continue
            if mnemonic in {"iastore", "lastore", "fastore", "dastore", "aastore", "bastore", "castore", "sastore"}:
                value = pop()[0]
                index = pop()[0]
                array = pop()[0]
                statements.append(f"{array}[{index}] = {value};")
                continue
            if mnemonic in {"if_icmpeq", "if_icmpne", "if_icmplt", "if_icmpge", "if_icmpgt", "if_icmple", "if_acmpeq", "if_acmpne", "ifeq", "ifne", "iflt", "ifge", "ifgt", "ifle"}:
                return JavaTranslation(False, statements, None, method.get("return_type", "void"), f"control flow {mnemonic} at pc {pc} is unresolved", declarations, evidence)
            if mnemonic in {"goto", "goto_w", "jsr", "jsr_w", "ret", "tableswitch", "lookupswitch"}:
                return JavaTranslation(False, statements, None, method.get("return_type", "void"), f"control flow {mnemonic} at pc {pc} is unresolved", declarations, evidence)
            if mnemonic in {"new", "newarray", "anewarray", "multianewarray", "athrow", "monitorenter", "monitorexit", "invokedynamic"}:
                return JavaTranslation(False, statements, None, method.get("return_type", "void"), f"{mnemonic} at pc {pc} is unresolved", declarations, evidence)
            if mnemonic == "wide iinc" or mnemonic.startswith("wide "):
                return JavaTranslation(False, statements, None, method.get("return_type", "void"), f"wide instruction at pc {pc} is unresolved", declarations, evidence)
            return JavaTranslation(False, statements, None, method.get("return_type", "void"), f"unsupported instruction {mnemonic} at pc {pc}", declarations, evidence)
    except (ValueError, IndexError, KeyError) as exc:
        return JavaTranslation(False, statements, None, method.get("return_type", "void"), str(exc), declarations, evidence)
    return_type = method.get("return_type") or "void"
    if return_type == "void":
        return JavaTranslation(True, statements, None, return_type, None, declarations, evidence)
    return JavaTranslation(False, statements, None, return_type, "return value is not proven", declarations, evidence)


def _method_signature(method: dict[str, Any], class_name: str) -> tuple[str, list[str]]:
    modifiers = [item for item in method.get("access", []) if item in {"public", "private", "protected", "static", "final", "synchronized", "native", "abstract"}]
    if "interface" in method.get("access", []) and "public" not in modifiers:
        modifiers.insert(0, "public")
    locals_ = _method_parameter_names(method, class_name)
    slot = 1 if not method.get("static") else 0
    parameters = []
    for index, parameter in enumerate(method.get("parameters") or []):
        local = locals_.get(slot)
        name = local.name if local else f"arg{index}"
        parameters.append(f"{_simple_type(parameter.get('type') or 'Object')} {name}")
        slot += int(parameter.get("slot_width") or 1)
    return " ".join(modifiers), parameters


def _emit_method(
    parsed: dict[str, Any],
    method: dict[str, Any],
    *,
    class_name: str,
    is_interface: bool,
    is_annotation: bool,
) -> tuple[str, dict[str, Any]]:
    name = method.get("name", "unknown")
    return_type = _simple_type(method.get("return_type") or "void")
    if name == "<clinit>":
        translation = translate_method(parsed, method, class_name=class_name, superclass=parsed.get("superclass"))
        if translation.success:
            lines = ["    static {"]
            for local_name, local_type in sorted(translation.declarations.items()):
                lines.append(f"        {_simple_type(local_type)} {local_name} = {_default_value(local_type)};")
            for statement in translation.statements:
                lines.append(f"        {statement}")
            lines.append("    }")
            return "\n".join(lines) + "\n", {
                "name": name,
                "classification": "static_initializer_reconstruction",
                "confidence": 0.4,
                "return_type": "void",
                "bytecode_pcs": translation.evidence,
            }
        return "", {"name": name, "classification": "omitted_class_initializer", "confidence": 0.0, "reason": translation.reason or "static initializer omitted"}
    if name == "<init>":
        modifiers, parameters = _method_signature(method, class_name)
        simple = class_name.rpartition(".")[2]
        translation = translate_method(parsed, method, class_name=class_name, superclass=parsed.get("superclass"))
        if translation.success and any(statement.startswith(("super(", "this(")) for statement in translation.statements):
            lines = [f"    {modifiers + ' ' if modifiers else ''}{simple}({', '.join(parameters)}) {{"]
            for local_name, local_type in sorted(translation.declarations.items()):
                lines.append(f"        {_simple_type(local_type)} {local_name} = {_default_value(local_type)};")
            for statement in translation.statements:
                lines.append(f"        {statement}")
            lines.append("    }")
            return "\n".join(lines) + "\n", {
                "name": name,
                "classification": "constructor_reconstruction",
                "confidence": 0.35,
                "return_type": "void",
                "parameters": parameters,
                "bytecode_pcs": translation.evidence,
            }
        return "", {"name": name, "classification": "omitted_constructor", "confidence": 0.0, "reason": translation.reason or "constructor omitted; default constructor is emitted when possible"}
    if method.get("synthetic") or method.get("bridge"):
        return "", {"name": name, "classification": "omitted_synthetic", "confidence": 0.0, "reason": "synthetic/bridge method omitted"}
    modifiers, parameters = _method_signature(method, class_name)
    signature = f"    {modifiers + ' ' if modifiers else ''}{return_type} {_identifier(name)}({', '.join(parameters)})"
    if is_interface or is_annotation or method.get("abstract"):
        return signature + ";\n", {"name": name, "classification": "declaration", "confidence": 0.8, "reason": None, "return_type": return_type}
    if method.get("native"):
        return signature + ";\n", {"name": name, "classification": "native_declaration", "confidence": 0.9, "reason": None, "return_type": return_type}
    translation = translate_method(parsed, method, class_name=class_name, superclass=parsed.get("superclass"))
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
        classification = "straight_line_reconstruction" if return_type == "void" or translation.return_expr is not None else "partial"
        return "\n".join(lines) + "\n", {
            "name": name,
            "classification": classification,
            "confidence": 0.45,
            "reason": None,
            "return_type": return_type,
            "parameters": parameters,
            "bytecode_pcs": translation.evidence,
        }
    evidence_pcs = ", ".join(str(pc) for pc in translation.evidence[:24])
    body = [
        signature + " {",
        f"        // NEXUS-BI unresolved: {translation.reason}",
        f"        // bytecode evidence PCs: {evidence_pcs}",
    ]
    for instruction in (method.get("code") or {}).get("instructions", [])[:24]:
        body.append(f"        //   {instruction.get('pc')}: {instruction.get('mnemonic')} {' '.join(instruction.get('operands') or [])}")
    body.append('        throw new UnsupportedOperationException("NEXUS-BI: unresolved bytecode reconstruction");')
    body.append("    }")
    return "\n".join(body) + "\n", {
        "name": name,
        "classification": "unresolved",
        "confidence": 0.12,
        "reason": translation.reason,
        "return_type": return_type,
        "parameters": parameters,
        "bytecode_pcs": translation.evidence[:24],
    }


def reconstruct_java_class_source(parsed: dict[str, Any], *, project_name: str = "nexus_java_reconstruction") -> tuple[str, dict[str, Any]]:
    class_name = parsed.get("simple_name") or "Reconstructed"
    dotted = parsed.get("name") or class_name
    package = parsed.get("package") or ""
    is_interface = bool(parsed.get("is_interface"))
    is_annotation = bool(parsed.get("is_annotation"))
    is_enum = bool(parsed.get("is_enum"))
    access = parsed.get("access") or []
    class_modifiers = [item for item in access if item in {"public", "final", "abstract", "static"}]
    if is_interface or is_annotation:
        class_modifiers = [item for item in class_modifiers if item in {"public"}]
    lines = []
    if package:
        lines.append(f"package {package};")
        lines.append("")
    lines.append(f"// NEXUS-BI approximate Java reconstruction of {dotted}.")
    lines.append(f"// Source class file: {parsed.get('source_file') or 'unknown'}; parse confidence: structural.")
    lines.append("// Unresolved methods are explicit; compilation is not behavioral verification.")
    lines.append("")
    declaration = " ".join(class_modifiers + ["@" + "interface" if is_annotation else "interface" if is_interface else "enum" if is_enum else "class", class_name])
    if is_interface or is_annotation:
        declaration = " ".join(class_modifiers + ["@" + "interface" if is_annotation else "interface", class_name])
    extends = parsed.get("superclass")
    if extends and extends not in {"java.lang.Object", "java.lang.Enum", "java.lang.annotation.Annotation"} and not is_interface:
        declaration += f" extends {_simple_type(extends)}"
    interfaces = [_simple_type(item) for item in parsed.get("interfaces") or [] if item]
    if interfaces and not is_annotation:
        declaration += (" extends " if is_interface else " implements ") + ", ".join(interfaces)
    lines.append(declaration + " {")
    if is_enum:
        lines.append("    ;")
        lines.append("    // Enum constants are not recovered from the class-file shape.")
    field_lines = []
    for field_item in parsed.get("fields") or []:
        if field_item.get("synthetic"):
            continue
        modifiers = [item for item in field_item.get("access", []) if item in {"public", "private", "protected", "static", "final", "volatile", "transient"}]
        field_type = _simple_type(field_item.get("type") or "Object")
        initializer = ""
        if field_item.get("constant_value") is not None:
            initializer = f" = {field_item['constant_value']}"
        elif field_item.get("final") or field_item.get("static"):
            initializer = f" = {_default_value(field_type)}"
        field_lines.append(f"    {' '.join(modifiers + [field_type, _identifier(field_item.get('name', 'field'))])}{initializer};")
    if field_lines:
        lines.extend(field_lines)
        lines.append("")
    methods = parsed.get("methods") or []
    needs_default_constructor = not any(method.get("name") == "<init>" for method in methods) and not is_interface and not is_annotation and not is_enum
    if needs_default_constructor and (extends in {None, "java.lang.Object"}):
        lines.append(f"    public {class_name}() {{")
        lines.append("        super();")
        lines.append("    }")
        lines.append("")
    method_evidence = []
    for method in methods:
        emitted, evidence = _emit_method(parsed, method, class_name=dotted, is_interface=is_interface, is_annotation=is_annotation)
        method_evidence.append(evidence)
        if emitted:
            lines.append(emitted.rstrip())
            lines.append("")
    lines.append("}")
    lines.append("")
    evidence = {
        "class_name": dotted,
        "package": package,
        "source_file": parsed.get("source_file"),
        "class_file_sha256": hashlib.sha256(json.dumps(parsed, sort_keys=True, default=str).encode("utf-8")).hexdigest(),
        "methods": method_evidence,
        "fields": [{"name": item.get("name"), "type": item.get("type"), "descriptor": item.get("descriptor")} for item in parsed.get("fields") or []],
        "limitations": [
            "Straight-line bytecode only; branches, exceptions, invokedynamic and object construction are unresolved.",
            "Constructors and class initializers are omitted; the default constructor may differ from the original.",
            "Generic signatures and annotations are not reconstructed.",
            "Compilation is a syntax check, not behavioral verification.",
        ],
        "confidence": 0.45 if any(item.get("classification") == "straight_line_reconstruction" for item in method_evidence) else 0.2,
    }
    return "\n".join(lines), evidence


def reconstruct_java_class_file(
    path: str | Path,
    *,
    out_dir: str | Path | None = None,
    project_name: str = "nexus_java_reconstruction",
) -> dict[str, Any]:
    from nexus.formats.java_class import parse_java_class

    parsed = parse_java_class(path)
    source, evidence = reconstruct_java_class_source(parsed, project_name=project_name)
    evidence["class_file_sha256"] = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    package = parsed.get("package") or ""
    class_name = parsed.get("simple_name") or "Reconstructed"
    source_path = f"src/main/java/{package.replace('.', '/') + '/' if package else ''}{class_name}.java"
    files = {
        source_path: source,
        "pom.xml": (
            '<?xml version="1.0" encoding="UTF-8"?>\n'
            '<project xmlns="http://maven.apache.org/POM/4.0.0" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" '
            'xsi:schemaLocation="http://maven.apache.org/POM/4.0.0 https://maven.apache.org/xsd/maven-4.0.0.xsd">\n'
            "  <modelVersion>4.0.0</modelVersion>\n"
            "  <groupId>local.nexus</groupId>\n"
            f"  <artifactId>{project_name}</artifactId>\n  <version>0.1.0</version>\n"
            "  <properties><maven.compiler.release>17</maven.compiler.release></properties>\n"
            "</project>\n"
        ),
        "build.gradle": (
            "plugins { id 'java' }\n"
            "group = 'local.nexus'\nversion = '0.1.0'\n"
            "java { toolchain { languageVersion = JavaLanguageVersion.of(17) } }\n"
        ),
        "README.md": (
            f"# {project_name}\n\n"
            f"Approximate Java reconstruction of `{parsed.get('name')}` from a class file. "
            "The emitter preserves declarations and translates a straight-line bytecode subset. "
            "Control flow, constructors, exceptions and invokedynamic are explicit unresolved constructs. "
            "Compilation does not establish behavioral equivalence.\n"
        ),
        "evidence_map.json": json.dumps(evidence, indent=2) + "\n",
    }
    if out_dir is not None:
        root = Path(out_dir)
        for relative, content in files.items():
            target = root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
    return {
        "language": "java",
        "source_format": "class",
        "project_name": project_name,
        "classification": "structural_line_reconstruction",
        "class": parsed,
        "files": files,
        "evidence_map": evidence,
        "limitations": evidence["limitations"],
    }
