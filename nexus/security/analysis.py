import re
from typing import Any

from nexus.elf.model import ElfFile
from nexus.program.model import FunctionCandidate

DANGEROUS_CALLS = {
    "gets": ("high", "unbounded input into a fixed buffer"),
    "strcpy": ("high", "unbounded string copy"),
    "strcat": ("high", "unbounded string concatenation"),
    "sprintf": ("high", "unbounded formatted output"),
    "vsprintf": ("high", "unbounded formatted output"),
    "system": ("high", "command execution with caller-controlled data may be possible"),
    "popen": ("high", "shell command execution"),
    "execl": ("high", "process execution"),
    "execve": ("high", "process execution"),
    "scanf": ("medium", "unbounded or weakly bounded input conversion"),
    "sscanf": ("medium", "format-controlled input conversion"),
    "fscanf": ("medium", "format-controlled input conversion"),
    "memcpy": ("low", "size correctness depends on caller and bounds checks"),
    "memmove": ("low", "size correctness depends on caller and bounds checks"),
    "alloca": ("low", "stack allocation size may be attacker-influenced"),
}


def _call_label(instruction) -> str | None:
    for operand in instruction.operands:
        match = re.search(r"<([^>]+)>", operand)
        if match:
            return match.group(1).split("@")[0]
    return None


def scan_security(elf: ElfFile, functions: list[FunctionCandidate], jni: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    seen: set[tuple[str, int]] = set()
    function_by_instruction = {}
    for function in functions:
        for instruction in function.instructions:
            function_by_instruction[instruction.address] = function
    for function in functions:
        for instruction in function.instructions:
            if instruction.control_flow not in {"call", "indirect_call"}:
                continue
            label = _call_label(instruction)
            if not label:
                continue
            label = label.split("@")[0]
            if label not in DANGEROUS_CALLS:
                continue
            key = (label, instruction.address)
            if key in seen:
                continue
            seen.add(key)
            severity, rationale = DANGEROUS_CALLS[label]
            findings.append({
                "finding_id": f"sec_{elf.sha256[:12]}_{instruction.address:x}_{label}",
                "severity": severity,
                "category": "unsafe_api",
                "title": f"Call to {label}",
                "description": f"{label} is called at 0x{instruction.address:x}; {rationale}.",
                "function_id": function.function_id,
                "function_name": function.name,
                "address": f"0x{instruction.address:x}",
                "evidence": [f"ev_{elf.sha256[:12]}_sec_{instruction.address:x}"],
                "analysis_method": "call-site/import scan",
                "confidence": 0.45 if severity == "high" else 0.3,
                "exploitability": "unknown; requires data-flow and runtime validation",
                "status": "needs_review",
                "remediation": "validate bounds and input provenance; prefer bounded APIs and explicit size checks.",
            })
    if jni:
        imports = set(jni.get("imports", []))
        gets = {name for name in imports if name.startswith("Get")}
        releases = {name for name in imports if name.startswith("Release")}
        if gets and not releases:
            findings.append({
                "finding_id": f"sec_{elf.sha256[:12]}_jni_reference",
                "severity": "low",
                "category": "jni_reference_management",
                "title": "JNI Get* calls without observed Release* calls",
                "description": "The binary imports JNI Get* functions but no Release* functions; this may indicate local-reference or native-memory management risk.",
                "function_id": None,
                "function_name": None,
                "address": None,
                "evidence": [f"ev_{elf.sha256[:12]}_jni_imports"],
                "analysis_method": "JNI import scan",
                "confidence": 0.25,
                "exploitability": "unknown; may be a false positive",
                "status": "needs_review",
                "remediation": "verify local/global reference and native buffer release paths.",
            })
    return findings
