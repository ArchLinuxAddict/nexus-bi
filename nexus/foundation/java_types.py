"""Deterministic JNI descriptor to Java declaration canonicalization.

The residual model is a repair layer, not a reliable type synthesizer. When a
JNI method and descriptor are available, the exact Java signature is derived
deterministically and the model output's declaration line is replaced. This
keeps generated code syntactically exact without asking the model to memorize
descriptor-to-type mappings.
"""

from __future__ import annotations

from nexus.jni.analysis import decode_jni_descriptor


def canonical_declaration(method: str, descriptor: str, *, instance: bool = False) -> str:
    arguments, return_type = decode_jni_descriptor(descriptor)
    if return_type == "unknown":
        return_type = "Object"
    params = ", ".join(f"{argument} arg{index}" for index, argument in enumerate(arguments))
    modifier = "public native" if instance else "public static native"
    return f"{modifier} {return_type} {method}({params});"


def canonicalize_generated(text: str, method: str, descriptor: str, *, instance: bool = False) -> str:
    declaration = canonical_declaration(method, descriptor, instance=instance)
    lines = text.splitlines()
    for index, line in enumerate(lines):
        if method in line and "native" in line:
            lines[index] = declaration
            return "\n".join(lines)
    return declaration
