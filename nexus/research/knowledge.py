"""Offline technical knowledge and local documentation lookup.

NEXUS-BI does not require network access to reason about common ABI, ELF, JNI
and compiler facts.  This module exposes a curated, versioned knowledge base and
a bounded search over local system headers.  Every snippet keeps its source path
and line number so that it can be cited as evidence.
"""

from __future__ import annotations

import functools
import os
import re
from pathlib import Path
from typing import Any

KNOWLEDGE_VERSION = "2026-11-04"
HEADER_SUFFIXES = {".h", ".hpp", ".hh", ".hxx"}
MAX_HEADER_FILES = 30000
MAX_HEADER_BYTES = 512 * 1024

KNOWLEDGE: list[dict[str, Any]] = [
    {
        "id": "abi.sysv-amd64",
        "title": "System V AMD64 calling convention",
        "topics": ["x86_64", "amd64", "abi", "calling convention", "argument", "return", "stack"],
        "facts": [
            "Integer/pointer arguments use rdi, rsi, rdx, rcx, r8, r9; further arguments go on the stack.",
            "The return value is in rax (and rdx for some wide returns); floating-point arguments use xmm registers.",
            "The stack is 16-byte aligned before a call; the callee-saved registers are rbx, rbp, r12-r15.",
            "The red zone is 128 bytes below rsp and may be used by leaf functions.",
            "Variadic calls use al to communicate the number of vector registers used.",
        ],
        "sources": ["System V ABI, AMD64 Supplement", "ELF psABI conventions"],
    },
    {
        "id": "abi.aapcs64",
        "title": "AAPCS64 calling convention",
        "topics": ["aarch64", "arm64", "abi", "calling convention", "argument", "return"],
        "facts": [
            "Integer/pointer arguments use x0-x7; the return value is in x0.",
            "x19-x28 are callee-saved; x29 is the frame pointer and x30 is the link register.",
            "The stack pointer must remain 16-byte aligned at public interfaces.",
            "SIMD/floating-point arguments use v0-v7.",
            "The caller may pass a function pointer in x16/x17 for PLT/veneer sequences.",
        ],
        "sources": ["ARM Architecture Reference Manual", "AAPCS64"],
    },
    {
        "id": "abi.aapcs32",
        "title": "AAPCS32 calling convention",
        "topics": ["arm", "armv7", "abi", "calling convention", "argument", "return", "thumb"],
        "facts": [
            "Integer/pointer arguments use r0-r3; the return value is in r0.",
            "r4-r11 are callee-saved; r13 is the stack pointer, r14 the link register and r15 the program counter.",
            "The stack must be 8-byte aligned at public interfaces.",
            "Interworking uses bx/blx and the Thumb bit in the low address bit.",
        ],
        "sources": ["ARM Architecture Reference Manual", "AAPCS32"],
    },
    {
        "id": "abi.riscv-psabi",
        "title": "RISC-V psABI",
        "topics": ["riscv", "riscv64", "riscv32", "abi", "calling convention", "argument", "return"],
        "facts": [
            "Integer/pointer arguments use a0-a7; the return value is in a0 (a1 for wide returns).",
            "s0-s11 are callee-saved; sp is the stack pointer and ra is the return address.",
            "The stack is 16-byte aligned at public interfaces.",
            "Compressed instructions may reduce instruction width; the decoder reports exact bytes.",
            "The syscall ABI uses a7 for the syscall number and a0-a5 for arguments.",
        ],
        "sources": ["RISC-V ELF psABI", "RISC-V Instruction Set Manual"],
    },
    {
        "id": "elf.dynamic-linking",
        "title": "ELF dynamic linking and PLT/GOT",
        "topics": ["elf", "plt", "got", "dynamic", "relocation", "shared library", "so"],
        "facts": [
            "Exported dynamic symbols are in .dynsym; their strings are in .dynstr.",
            "JUMP_SLOT relocations populate the GOT entries used by PLT stubs on lazy binding.",
            "DT_NEEDED records direct shared-library dependencies; DT_RPATH/DT_RUNPATH affect search paths.",
            "GNU symbol versions appear through .gnu.version, .gnu.version_d and .gnu.version_r.",
            "A call to <name@plt> is an internal PLT stub, not proof of the final resolved callee address.",
        ],
        "sources": ["System V ABI", "ELF gABI"],
    },
    {
        "id": "jni.naming",
        "title": "JNI native method naming and overloads",
        "topics": ["jni", "java", "native", "symbol", "overload", "register natives"],
        "facts": [
            "A static JNI symbol is Java_<mangled class>_<mangled method>.",
            "Overloaded native methods add __<encoded parameter descriptors>; the return type is not encoded.",
            "Class names use _ for '/', and underscores are escaped as _1; ';' is _2 and '[' is _3.",
            "RegisterNatives can bind Java methods to native functions with arbitrary symbol names.",
            "JNIEnv* and jclass/jobject are additional leading native parameters not represented in the Java descriptor.",
        ],
        "sources": ["Java Native Interface Specification"],
    },
    {
        "id": "jni.references",
        "title": "JNI reference management",
        "topics": ["jni", "reference", "local", "global", "weak global", "memory"],
        "facts": [
            "Local references are valid for the duration of a native method unless deleted or pushed/popped.",
            "Global references must be released with DeleteGlobalRef; weak globals with DeleteWeakGlobalRef.",
            "GetStringUTFChars/GetByteArrayElements pairs must be released with the matching Release call.",
            "ExceptionCheck/ExceptionOccurred must be handled before making most further JNI calls.",
        ],
        "sources": ["Java Native Interface Specification"],
    },
    {
        "id": "compiler.patterns",
        "title": "Compiler and ABI patterns",
        "topics": ["compiler", "prologue", "optimization", "inlining", "stack", "frame", "pattern"],
        "facts": [
            "Frame-pointer omission makes stack reconstruction heuristic; debug/DWARF evidence is stronger.",
            "Inlining removes call boundaries and can make source-level function recovery ambiguous.",
            "Tail calls may appear as jumps rather than calls.",
            "Position-independent code often uses PC-relative addressing and GOT loads.",
            "Constant folding and strength reduction can erase the source-level arithmetic shape.",
        ],
        "sources": ["Compiler engineering practice", "ABI documentation"],
    },
    {
        "id": "security.static-review",
        "title": "Static security review limits",
        "topics": ["security", "vulnerability", "strcpy", "system", "bounds", "buffer"],
        "facts": [
            "A call to strcpy/system/gets is a review candidate, not a confirmed vulnerability.",
            "Exploitability requires provenance, reachability, bounds and attacker-control evidence.",
            "Integer overflow, use-after-free and race conditions usually require data-flow or runtime evidence.",
            "Findings must state their evidence and confidence and remain separate from hypotheses.",
        ],
        "sources": ["NEXUS-BI security model", "CWE definitions"],
    },
]


def _topic_score(entry: dict[str, Any], query: str) -> int:
    words = [word for word in re.split(r"[^a-z0-9_]+", query.lower()) if word]
    haystack = " ".join([entry["id"], entry["title"], *entry["topics"], *entry["facts"]]).lower()
    return sum(2 if word in entry["topics"] else 1 for word in words if word in haystack)


def search_knowledge(query: str, limit: int = 8) -> list[dict[str, Any]]:
    scored = []
    for entry in KNOWLEDGE:
        score = _topic_score(entry, query)
        if score:
            scored.append((score, entry))
    scored.sort(key=lambda item: (-item[0], item[1]["id"]))
    return [{**entry, "score": score} for score, entry in scored[:limit]]


@functools.lru_cache(maxsize=1)
def _header_paths() -> tuple[Path, ...]:
    roots = [Path("/usr/include"), Path("/usr/local/include")]
    paths = []
    for root in roots:
        if not root.is_dir():
            continue
        for current, directories, files in os.walk(root):
            directories[:] = [name for name in directories if not name.startswith(".")]
            for name in files:
                if Path(name).suffix.lower() in HEADER_SUFFIXES:
                    paths.append(Path(current) / name)
                    if len(paths) >= MAX_HEADER_FILES:
                        return tuple(paths)
    return tuple(paths)


def search_system_headers(query: str, limit: int = 20, *, content: bool = True) -> dict[str, Any]:
    words = [word for word in re.split(r"[^A-Za-z0-9_]+", query) if len(word) > 2]
    if not words:
        return {"query": query, "matches": [], "searched_files": 0, "limitations": ["query has no searchable terms"]}
    paths = _header_paths()
    matches = []
    searched = 0
    for path in paths:
        searched += 1
        name_match = any(word.lower() in path.name.lower() for word in words)
        if name_match:
            matches.append({"source": str(path), "line": None, "text": path.name, "kind": "filename"})
        if content and not name_match:
            try:
                if path.stat().st_size > MAX_HEADER_BYTES:
                    continue
                text = path.read_text(encoding="utf-8", errors="replace")
            except (OSError, UnicodeError):
                continue
            lowered = text.lower()
            if all(word.lower() in lowered for word in words):
                for line_number, line in enumerate(text.splitlines(), start=1):
                    if all(word.lower() in line.lower() for word in words):
                        matches.append({"source": str(path), "line": line_number, "text": line.strip()[:240], "kind": "content"})
                        break
        if len(matches) >= limit:
            break
    return {
        "query": query,
        "matches": matches,
        "searched_files": searched,
        "available_files": len(paths),
        "limitations": [
            "Only local header files are searched; no network documentation is fetched.",
            "A match is textual evidence, not a verified semantic explanation.",
        ],
    }


def research_topic(query: str, *, include_headers: bool = True, limit: int = 8) -> dict[str, Any]:
    knowledge = search_knowledge(query, limit=limit)
    headers = search_system_headers(query, limit=min(limit, 8)) if include_headers else None
    return {
        "schema_version": "nexus.research/v1",
        "knowledge_version": KNOWLEDGE_VERSION,
        "query": query,
        "knowledge": knowledge,
        "system_headers": headers,
        "provenance": {
            "knowledge": "curated local reference set",
            "headers": "local system include paths",
        },
        "limitations": [
            "The knowledge base is a compact reference, not a complete specification.",
            "Documentation matches do not replace binary evidence or behavioral verification.",
        ],
    }
