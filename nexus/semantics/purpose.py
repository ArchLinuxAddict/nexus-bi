from typing import Any

from nexus.program.model import CallGraph, FunctionCandidate

PURPOSE_RULES = [
    ("memory allocation", {"malloc", "calloc", "realloc", "new"}),
    ("memory release", {"free", "delete"}),
    ("console output", {"printf", "puts", "fprintf", "write"}),
    ("string processing", {"strlen", "strcmp", "strncmp", "strcpy", "strncpy", "strcat", "memcmp", "memcpy", "memset"}),
    ("file input/output", {"fopen", "fclose", "fread", "fwrite", "open", "close", "read", "write"}),
    ("threading", {"pthread_create", "pthread_join", "pthread_mutex_lock", "pthread_mutex_unlock"}),
    ("cryptography-related", {"AES_encrypt", "SHA256_Init", "EVP_EncryptInit", "SSL_CTX_new"}),
    ("JNI boundary", {"JNI_OnLoad", "RegisterNatives"}),
    ("sorting", {"qsort", "sort"}),
]


def infer_purpose(function: FunctionCandidate, callgraph: CallGraph, binary_sha256: str) -> dict[str, Any]:
    callee_names = {(edge.target_name or "").split("@")[0] for edge in function.callees}
    evidence = [f"ev_{binary_sha256[:12]}_purpose_{function.entry:x}"]
    for label, names in PURPOSE_RULES:
        matched = sorted(callee_names & names)
        if matched:
            return {
                "purpose": label,
                "confidence": 0.55,
                "evidence": evidence,
                "matched_calls": matched,
                "method": "callee-name heuristic",
            }
    if any(instruction.mnemonic in {"jmp", "jne", "je", "jl", "jg", "jle", "jge"} for instruction in function.instructions):
        return {
            "purpose": "conditional computation",
            "confidence": 0.3,
            "evidence": evidence,
            "matched_calls": [],
            "method": "control-flow heuristic",
        }
    return {
        "purpose": "unknown",
        "confidence": 0.1,
        "evidence": evidence,
        "matched_calls": [],
        "method": "no matching evidence",
    }
