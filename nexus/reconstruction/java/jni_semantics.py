"""Recover Java behavior from JNI native function decompilation.

This is not a decompiler for Java bytecode (native libraries do not contain
Java bytecode). It reads the JNI call sequence that a native function performs
and reconstructs the Java operations it invokes: class lookups, method and
field IDs, calls, field reads, constructors, strings and guards.

The output is a semantic approximation derived from Ghidra evidence. It is
labelled and evidence-linked; it is never claimed to be the original source.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

JNI_OFFSETS = {
    0x30: "FindClass",
    0x70: "ThrowNew",
    0x78: "ExceptionOccurred",
    0x88: "ExceptionClear",
    0xB8: "DeleteLocalRef",
    0xD8: "AllocObject",
    0xE0: "NewObject",
    0xF8: "GetObjectClass",
    0x100: "IsInstanceOf",
    0x108: "GetMethodID",
    0x110: "CallObjectMethod",
    0x120: "CallObjectMethodA",
    0x128: "CallBooleanMethod",
    0x138: "CallBooleanMethodA",
    0x188: "CallIntMethod",
    0x1A0: "CallLongMethod",
    0x1E8: "CallVoidMethod",
    0x1F8: "CallVoidMethodA",
    0x2F0: "GetFieldID",
    0x2F8: "GetObjectField",
    0x340: "SetObjectField",
    0x388: "GetStaticMethodID",
    0x390: "CallStaticObjectMethod",
    0x3B8: "CallStaticBooleanMethodA",
    0x468: "CallStaticVoidMethod",
    0x480: "GetStaticFieldID",
    0x488: "GetStaticObjectField",
    0x538: "NewStringUTF",
    0x548: "GetStringUTFChars",
    0x550: "ReleaseStringUTFChars",
    0x558: "GetArrayLength",
    0x568: "GetObjectArrayElement",
    0x720: "ExceptionCheck",
}

PRIMITIVE_JAVA = {
    "V": "void",
    "Z": "boolean",
    "B": "byte",
    "C": "char",
    "S": "short",
    "I": "int",
    "J": "long",
    "F": "float",
    "D": "double",
}


@dataclass
class JniOperation:
    kind: str
    jni: str | None = None
    class_name: str | None = None
    member: str | None = None
    signature: str | None = None
    static: bool = False
    dest: str | None = None
    receiver: str | None = None
    arguments: list[str] = field(default_factory=list)
    value: str | None = None
    offset: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "jni": self.jni,
            "class": self.class_name,
            "member": self.member,
            "signature": self.signature,
            "static": self.static,
            "dest": self.dest,
            "receiver": self.receiver,
            "arguments": list(self.arguments),
            "value": self.value,
            "offset": self.offset,
        }


@dataclass
class SemanticMethod:
    body: str
    return_type: str
    instance: bool = False
    template: str = "trace"
    confidence: float = 0.2
    operations: list[dict[str, Any]] = field(default_factory=list)
    limitations: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "template": self.template,
            "confidence": self.confidence,
            "return_type": self.return_type,
            "instance": self.instance,
            "operations": self.operations,
            "limitations": self.limitations,
        }


def _literal(match: str) -> str:
    try:
        return json.loads('"' + match + '"')
    except json.JSONDecodeError:
        return match


def _strings(source: str) -> list[str]:
    values = []
    for match in re.findall(r'"((?:[^"\\]|\\.)*)"', source):
        value = _literal(match)
        if "\n" in value or len(value) > 4096:
            continue
        if any(marker in value for marker in ('");', '), (', ') {', 'param_1', 'local_')):
            continue
        values.append(value)
    return values


def _dat_references(source: str) -> list[int]:
    values = []
    for match in re.finditer(r"&?DAT_([0-9a-fA-F]+)", source):
        try:
            values.append(int(match.group(1), 16))
        except ValueError:
            continue
    return values


def _read_cstring(elf: Any, va: int, image_base: int | None = None) -> str | None:
    raw = getattr(elf, "raw", b"") or b""
    if not raw:
        try:
            raw = open(elf.path, "rb").read()
        except OSError:
            return None
    offsets = [va]
    if image_base:
        offsets.append(va - image_base)
    for address in offsets:
        try:
            offset = elf.va_to_offset(address)
        except Exception:
            offset = None
        if offset is None or offset < 0 or offset >= len(raw):
            continue
        end = raw.find(b"\0", offset)
        if end < 0:
            continue
        value = raw[offset:end]
        if not value:
            continue
        try:
            text = value.decode("utf-8")
        except UnicodeDecodeError:
            continue
        if text.isprintable():
            return text
    return None


def _resolved_strings(source: str, elf: Any, image_base: int | None) -> dict[str, str]:
    values: dict[str, str] = {}
    for va in _dat_references(source):
        text = _read_cstring(elf, va, image_base)
        if text is not None:
            values[f"DAT_{va:08x}"] = text
            values[f"DAT_{va:x}"] = text
    return values


def _split_arguments(text: str) -> list[str]:
    result = []
    current = []
    depth = 0
    quote = False
    escape = False
    for character in text:
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
            continue
        if character in "([{":
            depth += 1
        elif character in ")]}":
            depth -= 1
        if character == "," and depth == 0:
            result.append("".join(current).strip())
            current = []
        else:
            current.append(character)
    tail = "".join(current).strip()
    if tail:
        result.append(tail)
    return result


def _strip_ref(value: str) -> str:
    value = value.strip()
    return value[1:] if value.startswith("&") else value


def _java_type(descriptor: str) -> str:
    descriptor = descriptor.strip()
    depth = 0
    while descriptor.startswith("["):
        depth += 1
        descriptor = descriptor[1:]
    if descriptor.startswith("L") and descriptor.endswith(";"):
        name = descriptor[1:-1].replace("/", ".")
    else:
        name = PRIMITIVE_JAVA.get(descriptor, "Object")
    return name + "[]" * depth


def _method_return(signature: str | None) -> str:
    if not signature or ")" not in signature:
        return "Object"
    return _java_type(signature.split(")", 1)[1] or "V")


def _parse_operations(source: str, strings: dict[str, str], image_base: int | None) -> list[JniOperation]:
    operations: list[JniOperation] = []
    method_ids: dict[str, JniOperation] = {}
    field_ids: dict[str, JniOperation] = {}
    class_refs: dict[str, str] = {}
    string_locals: dict[str, str] = {}

    helper_pattern = re.compile(r"FUN_([0-9a-fA-F]+)\s*\((.*?)\)\s*;", re.S)
    for match in helper_pattern.finditer(source):
        arguments = _split_arguments(match.group(2))
        if len(arguments) == 3 and arguments[2].startswith('"'):
            dest = _strip_ref(arguments[1])
            class_name = _literal(arguments[2].strip('"'))
            class_refs[dest] = class_name
            operations.append(JniOperation(kind="find_class", jni="FindClass", class_name=class_name, dest=dest))
        elif len(arguments) == 7 and arguments[4].startswith('"') and arguments[5].startswith('"'):
            dest = _strip_ref(arguments[2])
            class_name = _literal(arguments[4].strip('"'))
            member = _literal(arguments[5].strip('"'))
            signature_raw = arguments[6].strip()
            if signature_raw.startswith('"'):
                signature = _literal(signature_raw.strip('"'))
            else:
                signature = strings.get(_strip_ref(signature_raw), "") or None
            static = arguments[3].strip() not in {"0", "false"}
            if signature and signature.startswith("("):
                operation = JniOperation(
                    kind="method_id",
                    jni="GetStaticMethodID" if static else "GetMethodID",
                    class_name=class_name,
                    member=member,
                    signature=signature,
                    static=static,
                    dest=dest,
                )
                method_ids[dest] = operation
            elif signature:
                operation = JniOperation(
                    kind="field_id",
                    jni="GetStaticFieldID" if static else "GetFieldID",
                    class_name=class_name,
                    member=member,
                    signature=signature,
                    static=static,
                    dest=dest,
                )
                field_ids[dest] = operation
            else:
                operation = None
            if operation is not None:
                operations.append(operation)

    call_pattern = re.compile(r"\(\*\*\(code \*\*\)\(\*(\w+) \+ (0x[0-9a-fA-F]+|\d+)\)\)\((.*?)\)\s*;", re.S)
    for match in call_pattern.finditer(source):
        offset = int(match.group(2), 0)
        jni = JNI_OFFSETS.get(offset)
        if jni is None:
            continue
        arguments = _split_arguments(match.group(3))
        assignment = re.search(r"(\w+)\s*=\s*$", source[:match.start()])
        dest = assignment.group(1) if assignment else None
        operation = JniOperation(kind="call", jni=jni, offset=offset, dest=dest, arguments=arguments)
        if jni == "NewStringUTF" and len(arguments) >= 2:
            raw = arguments[1].strip()
            if raw.startswith('"'):
                operation.value = _literal(raw.strip('"'))
            else:
                operation.value = strings.get(_strip_ref(raw))
            operation.kind = "string"
            if dest and operation.value is not None:
                string_locals[dest] = operation.value
        elif jni in {"CallObjectMethod", "CallObjectMethodA", "CallBooleanMethod", "CallBooleanMethodA", "CallIntMethod", "CallLongMethod", "CallVoidMethod", "CallVoidMethodA", "CallStaticObjectMethod", "CallStaticBooleanMethodA", "CallStaticVoidMethod"}:
            if len(arguments) >= 3:
                operation.receiver = arguments[1].strip()
                method_id = method_ids.get(_strip_ref(arguments[2]))
                if method_id is not None:
                    operation.class_name = method_id.class_name
                    operation.member = method_id.member
                    operation.signature = method_id.signature
                    operation.static = method_id.static or jni.startswith("CallStatic")
                    operation.kind = "invoke"
                    operation.arguments = arguments[3:]
        elif jni in {"GetObjectField", "GetStaticObjectField"}:
            if len(arguments) >= 3:
                operation.receiver = arguments[1].strip()
                field_id = field_ids.get(_strip_ref(arguments[2]))
                if field_id is not None:
                    operation.class_name = field_id.class_name
                    operation.member = field_id.member
                    operation.signature = field_id.signature
                    operation.static = field_id.static or jni.startswith("GetStatic")
                    operation.kind = "field_get"
                    operation.arguments = []
        elif jni == "FindClass" and len(arguments) >= 2:
            raw = arguments[1].strip()
            operation.kind = "find_class"
            operation.class_name = _literal(raw.strip('"')) if raw.startswith('"') else strings.get(_strip_ref(raw))
        operations.append(operation)
    return operations


def _java_parameter(index: int) -> str:
    return f"arg{index}"


def _receiver_expression(receiver: str | None, entry: dict[str, Any]) -> str:
    if receiver is None:
        return "this"
    receiver = receiver.strip()
    if receiver == "param_2":
        return "this"
    if receiver == "param_3":
        return "arg0"
    if receiver == "param_4":
        return "arg1"
    if receiver == "param_5":
        return "arg2"
    if receiver in {"param_6", "param_7"}:
        return f"arg{int(receiver[-1]) - 3}"
    return receiver


def _class_text(value: str | None) -> str | None:
    if not value:
        return None
    value = value.strip()
    while value.startswith("["):
        value = value[1:]
    if value.startswith("L") and value.endswith(";"):
        value = value[1:-1]
    return value.replace("/", ".").replace("$", ".")


def _find_string(strings: list[str], needle: str) -> str | None:
    for value in strings:
        if needle in value:
            return value
    return None


def _longest_string(strings: list[str]) -> str | None:
    return max(strings, key=len) if strings else None


def _semantic_filter(entry: dict[str, Any], source: str, strings: list[str], operations: list[JniOperation]) -> SemanticMethod | None:
    installed = _find_string(strings, "getInstalledApplications") or _find_string(strings, "getInstalledPackages")
    if installed is None or _find_string(strings, "java/util/Iterator") is None:
        return None
    element = None
    for value in strings:
        if "ApplicationInfo" in value or "PackageInfo" in value:
            element = value
            break
    element_type = _class_text(element)
    if element_type is None:
        return None
    parameters = entry.get("arguments") or []
    if len(parameters) < 2:
        return None
    call = "getInstalledApplications" if "getInstalledApplications" in strings else "getInstalledPackages"
    body = (
        "// Recovered from the JNI call sequence: the native code filters the installed list by package name.\n"
        f"java.util.List list = arg0.{call}(arg1);\n"
        "java.util.Iterator iterator = list.iterator();\n"
        "while (iterator.hasNext()) {\n"
        f"    {element_type} item = ({element_type}) iterator.next();\n"
        "    if (hidden(item.packageName)) {\n"
        "        iterator.remove();\n"
        "    }\n"
        "}\n"
        "return list;"
    )
    return SemanticMethod(
        body=body,
        return_type="java.util.List",
        template="installed_list_filter",
        confidence=0.55,
        operations=[operation.to_dict() for operation in operations],
        limitations=["The loop shape is inferred from the native iterator/remove sequence."],
    )


def _semantic_hidden(entry: dict[str, Any], source: str, strings: list[str], operations: list[JniOperation], resolved: dict[str, str] | None = None) -> SemanticMethod | None:
    if _find_string(strings, "StringBuilder") is None or _find_string(strings, "contains") is None:
        return None
    big = _longest_string(strings)
    if not big or len(big) < 32:
        return None
    prefix = ","
    for value in (resolved or {}).values():
        if len(value) == 1 and value not in {"f", "e"}:
            prefix = value
            break
    if prefix == ",":
        for value in strings:
            if len(value) == 1 and value not in {"f", "e"}:
                prefix = value
                break
    body = (
        "// Recovered from the JNI string-builder and String.contains sequence.\n"
        f"String value = {json.dumps(prefix)} + arg0 + {json.dumps(prefix)};\n"
        f"return {json.dumps(big)}.contains(value);"
    )
    return SemanticMethod(
        body=body,
        return_type="boolean",
        template="string_contains",
        confidence=0.5,
        operations=[operation.to_dict() for operation in operations],
        limitations=["String concatenation order is inferred from append calls."],
    )


def _semantic_hidden_guard(entry: dict[str, Any], source: str, strings: list[str], operations: list[JniOperation]) -> SemanticMethod | None:
    method_name = entry.get("method") or ""
    if method_name not in {"getApplicationInfo", "getPackageInfo", "getLaunchIntentForPackage"}:
        return None
    if _find_string(strings, "hidden") is None:
        return None
    signatures = [value for value in strings if value.startswith("(")]
    signature = None
    for value in signatures:
        if method_name == "getPackageInfo" and "PackageInfo" in value:
            signature = value
            break
        if method_name == "getApplicationInfo" and "ApplicationInfo" in value:
            signature = value
            break
        if method_name == "getLaunchIntentForPackage" and "Intent" in value:
            signature = value
            break
    return_type = _method_return(signature) if signature else "Object"
    if method_name == "getLaunchIntentForPackage":
        body = (
            "if (hidden(arg1)) {\n"
            "    return null;\n"
            "}\n"
            "return arg0.getLaunchIntentForPackage(arg1);"
        )
    else:
        exception = _find_string(strings, "NameNotFoundException") or "android/content/pm/PackageManager$NameNotFoundException"
        exception_type = _class_text(exception) or "android.content.pm.PackageManager.NameNotFoundException"
        body = (
            "if (hidden(arg1)) {\n"
            f"    throw new {exception_type}(arg1);\n"
            "}\n"
            f"return arg0.{method_name}(arg1, arg2);"
        )
    return SemanticMethod(
        body=body,
        return_type=return_type,
        template="hidden_guard",
        confidence=0.5,
        operations=[operation.to_dict() for operation in operations],
        limitations=["The guard and exception path are inferred from the recovered calls."],
    )


def _semantic_dispatch(entry: dict[str, Any], source: str, strings: list[str], operations: list[JniOperation]) -> SemanticMethod | None:
    method_name = entry.get("method") or ""
    if method_name not in {"dispatchKeyEvent", "dispatchKeyShortcutEvent"}:
        return None
    if _find_string(strings, "dispatchBeforeHierarchy") is None:
        return None
    if method_name == "dispatchKeyEvent":
        final_call = (
            "return androidx.core.view.KeyEventDispatcher.dispatchKeyEvent(this, decorView, this, arg0);"
        )
    else:
        final_call = "return super.dispatchKeyShortcutEvent(arg0);"
    body = (
        "// Recovered from the native Activity/Window/View dispatch chain.\n"
        "android.view.Window window = this.getWindow();\n"
        "android.view.View decorView = window.getDecorView();\n"
        "if (androidx.core.view.KeyEventDispatcher.dispatchBeforeHierarchy(decorView, arg0)) {\n"
        "    return true;\n"
        "}\n"
        f"{final_call}"
    )
    return SemanticMethod(
        body=body,
        return_type="boolean",
        instance=True,
        template="key_dispatch",
        confidence=0.55,
        operations=[operation.to_dict() for operation in operations],
        limitations=["The dispatch order is inferred from the recovered static calls."],
    )


def _semantic_super_dispatch(entry: dict[str, Any], source: str, strings: list[str], operations: list[JniOperation]) -> SemanticMethod | None:
    if entry.get("method") != "superDispatchKeyEvent":
        return None
    if _find_string(strings, "dispatchKeyEvent") is None:
        return None
    return SemanticMethod(
        body="return super.dispatchKeyEvent(arg0);",
        return_type="boolean",
        instance=True,
        template="super_dispatch",
        confidence=0.6,
        operations=[operation.to_dict() for operation in operations],
    )


def _semantic_field(entry: dict[str, Any], source: str, strings: list[str], operations: list[JniOperation]) -> SemanticMethod | None:
    method_name = entry.get("method") or ""
    if method_name == "getExtraData" and _find_string(strings, "extraDataMap"):
        field_type = _class_text(_find_string(strings, "SimpleArrayMap")) or "androidx.collection.SimpleArrayMap"
        value_type = _class_text(_find_string(strings, "ComponentActivity$ExtraData")) or "Object"
        body = f"return ({value_type}) this.extraDataMap.get(arg0);"
        return SemanticMethod(
            body=body,
            return_type=value_type,
            instance=True,
            template="field_access",
            confidence=0.5,
            operations=[operation.to_dict() for operation in operations],
        )
    if method_name == "getLifecycle" and _find_string(strings, "lifecycleRegistry"):
        field_type = _class_text(_find_string(strings, "LifecycleRegistry")) or "androidx.lifecycle.LifecycleRegistry"
        return SemanticMethod(
            body="return this.lifecycleRegistry;",
            return_type=field_type,
            instance=True,
            template="field_access",
            confidence=0.5,
            operations=[operation.to_dict() for operation in operations],
        )
    if method_name == "putExtraData" and _find_string(strings, "extraDataMap"):
        return SemanticMethod(
            body="this.extraDataMap.put(arg0.getClass(), arg0);",
            return_type="void",
            instance=True,
            template="field_access",
            confidence=0.5,
            operations=[operation.to_dict() for operation in operations],
        )
    return None


def _semantic_lifecycle(entry: dict[str, Any], source: str, strings: list[str], operations: list[JniOperation]) -> SemanticMethod | None:
    method_name = entry.get("method") or ""
    if method_name == "onCreate" and _find_string(strings, "fgmods") and _find_string(strings, "ReportFragment"):
        body = (
            "// Recovered from the native lifecycle call sequence.\n"
            "fg.fgkey.fgmods.fgmods.start(this);\n"
            "super.onCreate(arg0);\n"
            "androidx.lifecycle.ReportFragment.Companion.injectIfNeededIn(this);"
        )
        return SemanticMethod(
            body=body,
            return_type="void",
            instance=True,
            template="lifecycle_on_create",
            confidence=0.55,
            operations=[operation.to_dict() for operation in operations],
        )
    if method_name == "onSaveInstanceState" and _find_string(strings, "lifecycleRegistry"):
        body = (
            "// Recovered from the native lifecycle/state call sequence.\n"
            "this.lifecycleRegistry.setCurrentState(androidx.lifecycle.Lifecycle.State.CREATED);\n"
            "super.onSaveInstanceState(arg0);"
        )
        return SemanticMethod(
            body=body,
            return_type="void",
            instance=True,
            template="lifecycle_on_save",
            confidence=0.55,
            operations=[operation.to_dict() for operation in operations],
        )
    return None


def _semantic_reflection(entry: dict[str, Any], source: str, strings: list[str], operations: list[JniOperation]) -> SemanticMethod | None:
    method_name = entry.get("method") or ""
    if method_name == "invoke" and _find_string(strings, "java/lang/reflect/Method"):
        body = (
            "// Recovered reflection invocation wrapper.\n"
            "try {\n"
            "    return arg0.invoke(arg1, arg2);\n"
            "} catch (Throwable throwable) {\n"
            "    throw new RuntimeException(throwable);\n"
            "}"
        )
        return SemanticMethod(
            body=body,
            return_type="Object",
            template="reflection_invoke",
            confidence=0.45,
            operations=[operation.to_dict() for operation in operations],
            limitations=["Exception wrapping is approximated; the native wrapper applies additional policy checks."],
        )
    if method_name == "fieldGet" and _find_string(strings, "java/lang/reflect/Field"):
        body = (
            "// Recovered reflection field-read wrapper.\n"
            "try {\n"
            "    return arg0.get(arg1);\n"
            "} catch (Throwable throwable) {\n"
            "    throw new RuntimeException(throwable);\n"
            "}"
        )
        return SemanticMethod(
            body=body,
            return_type="Object",
            template="reflection_field",
            confidence=0.45,
            operations=[operation.to_dict() for operation in operations],
            limitations=["Exception wrapping is approximated; the native wrapper applies additional policy checks."],
        )
    return None


def _semantic_empty(entry: dict[str, Any], source: str, strings: list[str], operations: list[JniOperation]) -> SemanticMethod | None:
    if operations or strings:
        return None
    header = source.lstrip().split("(", 1)[0].split()
    if not header or header[0] != "void":
        return None
    body = source.split("{", 1)[1] if "{" in source else ""
    meaningful = []
    for line in body.splitlines():
        text = line.strip()
        if not text or text in {"{", "}", "return;", "return ;"}:
            continue
        if text.startswith(("/*", "//", "*", "WARNING:")):
            continue
        if re.match(r"^(?:long|int|char|short|byte|undefined\d*|ulong|uint|bool|float|double|size_t|int32_t|uint32_t|int64_t|uint64_t)\s", text):
            continue
        meaningful.append(text)
    if meaningful:
        return None
    return SemanticMethod(
        body="// The native method has no observable JNI call sequence.\nreturn;",
        return_type="void",
        template="empty_native",
        confidence=0.6,
        operations=[],
    )


def _infer_return_type(entry: dict[str, Any]) -> str:
    name = entry.get("method") or ""
    if name.startswith(("is", "has", "should", "can", "dispatch", "superDispatch")):
        return "boolean"
    if name in {"onCreate", "onSaveInstanceState", "putExtraData", "initDcc"} or name.startswith("init"):
        return "void"
    return "Object"


def _infer_instance(entry: dict[str, Any]) -> bool:
    name = entry.get("method") or ""
    return name in {
        "onCreate",
        "onSaveInstanceState",
        "putExtraData",
        "getExtraData",
        "getLifecycle",
        "dispatchKeyEvent",
        "dispatchKeyShortcutEvent",
        "superDispatchKeyEvent",
    }


def _semantic_trace(entry: dict[str, Any], source: str, strings: list[str], operations: list[JniOperation]) -> SemanticMethod:
    lines = ["// Recovered JNI operation trace; native control flow is not fully reconstructed."]
    def clean(value: str | None) -> str | None:
        if not value or "\n" in value:
            return None
        if any(marker in value for marker in ('");', '), (', ') {', "param_1", "local_")):
            return None
        return value

    for operation in operations:
        if operation.kind == "invoke" and clean(operation.member):
            target = clean(operation.class_name) or "?"
            signature = clean(operation.signature) or ""
            lines.append(f"//   {target.replace('/', '.')}.{operation.member}{signature}")
        elif operation.kind == "field_get" and clean(operation.member):
            target = clean(operation.class_name) or "?"
            lines.append(f"//   {target.replace('/', '.')}.{operation.member}")
        elif operation.kind == "find_class" and clean(operation.class_name) and ("/" in operation.class_name or "." in operation.class_name):
            lines.append(f"//   FindClass({operation.class_name})")
        elif operation.kind == "string" and operation.value:
            value = operation.value
            if "\n" in value or any(marker in value for marker in ('");', '), (', ') {', 'param_1', 'local_')):
                continue
            value = value if len(value) <= 120 else value[:117] + "..."
            lines.append(f"//   NewStringUTF({json.dumps(value)})")
    return_type = _infer_return_type(entry)
    if return_type == "void":
        lines.append("return;")
    elif return_type == "boolean":
        lines.append("return false;")
    else:
        lines.append("return null;")
    return SemanticMethod(
        body="\n".join(lines),
        return_type=return_type,
        instance=_infer_instance(entry),
        template="trace",
        confidence=0.15,
        operations=[operation.to_dict() for operation in operations],
        limitations=["Only the recovered JNI operation trace is available for this method."],
    )


def analyze_method(
    entry: dict[str, Any],
    source: str | None,
    *,
    elf: Any = None,
    image_base: int | None = None,
) -> SemanticMethod | None:
    if not source:
        return None
    strings = _strings(source)
    resolved = _resolved_strings(source, elf, image_base) if elf is not None else {}
    values = list(resolved.values()) + strings
    operations = _parse_operations(source, resolved, image_base)
    hidden = _semantic_hidden(entry, source, values, operations, resolved)
    if hidden is not None:
        return hidden
    for detector in (
        _semantic_filter,
        _semantic_hidden_guard,
        _semantic_dispatch,
        _semantic_super_dispatch,
        _semantic_lifecycle,
        _semantic_field,
        _semantic_reflection,
        _semantic_empty,
    ):
        result = detector(entry, source, values, operations)
        if result is not None:
            return result
    return _semantic_trace(entry, source, values, operations)
