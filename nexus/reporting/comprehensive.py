"""Evidence-linked human report over native and Java/JNI analysis."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from nexus.program.bundle import analyze_binary
from nexus.program.record import build_program_record
from nexus.reconstruction.c.generator import reconstruct_c
from nexus.research.knowledge import search_knowledge
from nexus.verification.compiler import compile_c_project, compile_java_project
from nexus.verification.static import verify_program_record


def _truncate(value: str, limit: int = 240) -> str:
    return value if len(value) <= limit else value[: limit - 3] + "..."


def _function_purpose(bundle) -> dict[str, int]:
    counts: dict[str, int] = {}
    for purpose in bundle.purposes.values():
        key = str(purpose.get("purpose") or "unknown")
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))


def _imports(bundle) -> list[dict[str, Any]]:
    return [symbol.to_dict() for symbol in bundle.elf.dynamic_symbols if symbol.is_import]


def _exports(bundle) -> list[dict[str, Any]]:
    return [symbol.to_dict() for symbol in bundle.elf.dynamic_symbols if symbol.is_function]


def _library_purpose(bundle) -> dict[str, Any]:
    imports = sorted({item.get("name", "") for item in _imports(bundle) if item.get("name")})
    exports = sorted({item.get("name", "") for item in _exports(bundle) if item.get("name")})
    interesting = [
        name for name in imports
        if any(token in name.lower() for token in ("jni", "java", "ssl", "crypto", "sql", "http", "pthread", "dlopen", "mmap"))
    ]
    purposes = _function_purpose(bundle)
    likely = [item for item in purposes if item not in {"unknown", "conditional computation"}][:6]
    return {
        "classification": "inferred_from_imports_exports_and_callee_evidence",
        "summary": (
            "A native shared library with JNI evidence."
            if bundle.jni_report.get("is_jni")
            else "A native shared library; no single purpose is proven by static evidence."
            if bundle.elf.is_shared_object
            else "A native executable; no single purpose is proven by static evidence."
        ),
        "candidate_purposes": likely,
        "notable_imports": interesting[:24],
        "import_count": len(imports),
        "export_count": len(exports),
        "confidence": 0.35 if likely else 0.15,
    }


def _architecture_section(bundle) -> dict[str, Any]:
    return {
        "machine": bundle.elf.header.machine_name,
        "elf_class": bundle.elf.header.elf_class,
        "endianness": bundle.elf.header.endianness,
        "entry": f"0x{bundle.elf.header.entry:x}",
        "abi": (bundle.architecture or {}).get("abi"),
        "pointer_width": (bundle.architecture or {}).get("pointer_width"),
        "argument_registers": (bundle.architecture or {}).get("argument_registers"),
        "return_register": (bundle.architecture or {}).get("return_register"),
        "callee_saved": (bundle.architecture or {}).get("callee_saved"),
        "decoder_capabilities": (bundle.architecture or {}).get("capabilities"),
        "notes": (bundle.architecture or {}).get("notes") or [],
    }


def _dependencies(bundle) -> list[dict[str, Any]]:
    return list(bundle.elf.dependencies)


def _security(bundle) -> list[dict[str, Any]]:
    return list(bundle.security_findings)


def _performance(bundle) -> dict[str, Any]:
    functions = sorted(bundle.functions, key=lambda item: -item.size)
    instructions = len(bundle.disassembly.instructions) if bundle.disassembly else 0
    return {
        "function_count": len(functions),
        "instruction_count": instructions,
        "largest_functions": [
            {"name": item.name, "entry": f"0x{item.entry:x}", "size": item.size, "instructions": len(item.instructions)}
            for item in functions[:10]
        ],
        "call_edges": len(bundle.callgraph.edges),
        "external_call_edges": sum(1 for edge in bundle.callgraph.edges if edge.kind.startswith("call_external")),
        "average_instructions_per_function": round(instructions / len(functions), 2) if functions else 0.0,
        "limitation": "Static size and call-shape proxies are not runtime performance measurements.",
    }


def _unresolved(bundle) -> list[dict[str, Any]]:
    items = []
    for function in bundle.functions:
        for unknown in function.unknowns[:3]:
            items.append({"function": function.name, "entry": f"0x{function.entry:x}", **unknown})
    items.extend({"kind": "global", "question": limitation, "status": "limitation"} for limitation in bundle.limitations[:10])
    return items[:100]


def _function_rows(bundle) -> list[dict[str, Any]]:
    rows = []
    for function in sorted(bundle.functions, key=lambda item: item.entry):
        signature = (bundle.signatures or {}).get(function.function_id, {})
        rows.append(
            {
                "name": function.name,
                "entry": f"0x{function.entry:x}",
                "size": function.size,
                "confidence": function.confidence,
                "purpose": function.purpose,
                "calling_convention": signature.get("calling_convention"),
                "return_type": signature.get("return_type"),
                "parameters": [
                    {"name": item.get("name"), "type": item.get("type"), "register": item.get("register"), "provenance": item.get("provenance")}
                    for item in signature.get("parameters") or []
                ],
                "callees": [edge.target_name or edge.target for edge in function.callees],
                "unknowns": function.unknowns[:3],
            }
        )
    return rows


def build_comprehensive_report(
    binary_path: str | Path,
    *,
    java_path: str | Path | None = None,
    native_path: str | Path | None = None,
    out_dir: str | Path | None = None,
    max_functions: int = 40,
) -> dict[str, Any]:
    binary = Path(binary_path)
    bundle = analyze_binary(binary)
    record = build_program_record(bundle)
    record_validation = verify_program_record(record)
    report_dir = Path(out_dir) if out_dir else Path("artifacts") / "reports" / f"{bundle.elf.sha256[:12]}"
    report_dir.mkdir(parents=True, exist_ok=True)
    c_result = None
    try:
        c_result = reconstruct_c(bundle, function_ids=None, out_dir=report_dir / "c-reconstruction", project_name=f"reconstructed_{bundle.elf.sha256[:8]}")
    except Exception as exc:
        c_result = {"classification": "failed", "error": str(exc), "language": "c"}
    c_verification = None
    if c_result and c_result.get("classification") != "failed":
        try:
            c_verification = compile_c_project(report_dir / "c-reconstruction")
            c_result["verification"] = c_verification
        except Exception as exc:
            c_verification = {"status": "failed", "error": str(exc)}
            c_result["verification"] = c_verification
    java_result = None
    if java_path is not None:
        from nexus.java.service import reconstruct_java_artifact
        try:
            java_result = reconstruct_java_artifact(
                java_path,
                out_dir=report_dir / "java-reconstruction",
                project_name=f"java_{bundle.elf.sha256[:8]}",
            )
        except Exception as exc:
            java_result = {"classification": "failed", "error": str(exc), "language": "java"}
        if java_result and java_result.get("classification") != "failed":
            try:
                java_result["verification"] = compile_java_project(report_dir / "java-reconstruction")
            except Exception as exc:
                java_result["verification"] = {"status": "failed", "error": str(exc)}
    jni_result = None
    if java_path is not None and native_path is not None:
        from nexus.jni.correlation import correlate_java_native
        try:
            jni_result = correlate_java_native(java_path, native_path)
        except Exception as exc:
            jni_result = {"error": str(exc)}
    elif bundle.jni_report.get("is_jni"):
        jni_result = {
            "java": None,
            "native": {"path": str(binary), "sha256": bundle.elf.sha256},
            "native_symbols": [],
            "matches": [],
            "unmatched_java_methods": [],
            "summary": {
                "java_native_methods": 0,
                "native_jni_symbols": len(bundle.jni_report.get("entries", [])),
                "matched": 0,
                "unmatched_java": 0,
                "unmatched_native": 0,
                "jni_onload": bool(bundle.jni_report.get("onload")),
            },
            "limitations": list(bundle.jni_report.get("limitations", [])),
        }
    research = search_knowledge(
        f"{(bundle.architecture or {}).get('abi') or bundle.elf.header.machine_name} calling convention jni elf",
        limit=5,
    )
    facts = {
        "binary": {
            "path": str(binary),
            "sha256": bundle.elf.sha256,
            "size": bundle.elf.size,
            "format": "ELF",
            "machine": bundle.elf.header.machine_name,
            "type": bundle.elf.header.type_name,
            "entry": f"0x{bundle.elf.header.entry:x}",
            "stripped": bundle.elf.is_stripped,
            "pie": bundle.elf.is_pie,
            "shared_object": bundle.elf.is_shared_object,
        },
        "exports": _exports(bundle),
        "imports": _imports(bundle),
        "dependencies": _dependencies(bundle),
        "sections": [section.to_dict() for section in bundle.elf.sections],
        "architecture": _architecture_section(bundle),
        "functions": _function_rows(bundle),
        "string_count": len(bundle.strings),
        "record_validation": record_validation,
    }
    inferences = {
        "library_purpose": _library_purpose(bundle),
        "purpose_distribution": _function_purpose(bundle),
        "performance_proxies": _performance(bundle),
        "security_candidates": _security(bundle),
    }
    hypotheses = [
        {
            "claim": "The library may serve the dominant imported API families.",
            "status": "hypothesis",
            "evidence": _library_purpose(bundle)["notable_imports"],
            "confidence": 0.3,
        }
    ]
    unresolved = _unresolved(bundle)
    evidence = {
        "generated_at": datetime.now(UTC).isoformat(),
        "pipeline": "nexus.reporting.comprehensive/v1",
        "facts": facts,
        "inferences": inferences,
        "hypotheses": hypotheses,
        "unresolved": unresolved,
        "reconstruction": {
            "c": None if c_result is None else {key: value for key, value in c_result.items() if key != "files"},
            "java": None if java_result is None else {key: value for key, value in java_result.items() if key != "files"},
        },
        "jni": jni_result,
        "research_references": research,
        "confidence_model": {
            "facts": "Observed from deterministic parsers with addresses/hashes.",
            "inferences": "Derived from observed evidence; may be wrong.",
            "hypotheses": "Awaiting independent verification.",
            "limitations": "No dynamic execution or behavioral equivalence verification.",
        },
    }
    (report_dir / "report.json").write_text(json.dumps(evidence, indent=2, default=str) + "\n", encoding="utf-8")
    lines = [
        f"# NEXUS-BI Report: {binary.name}",
        "",
        f"- SHA-256: `{bundle.elf.sha256}`",
        f"- Architecture: {bundle.elf.header.machine_name} ({bundle.elf.header.elf_class}, {bundle.elf.header.endianness})",
        f"- ABI: {(bundle.architecture or {}).get('abi')}",
        f"- Type: {bundle.elf.header.type_name}; stripped={bundle.elf.is_stripped}; PIE={bundle.elf.is_pie}; shared={bundle.elf.is_shared_object}",
        f"- Functions: {len(bundle.functions)}; instructions: {len(bundle.disassembly.instructions) if bundle.disassembly else 0}; imports: {len(_imports(bundle))}; exports: {len(_exports(bundle))}",
        "",
        "## Library purpose",
        "",
        _library_purpose(bundle)["summary"],
        "",
        f"Candidate purpose families: {', '.join(_library_purpose(bundle)['candidate_purposes']) or 'none proven'}",
        f"Notable imports: {', '.join(_library_purpose(bundle)['notable_imports']) or 'none'}",
        "",
        "## Exported APIs",
        "",
    ]
    for export in _exports(bundle)[:80]:
        lines.append(f"- `{export.get('name')}` at {export.get('value')} (size {export.get('size')})")
    lines.extend(["", "## Imported APIs and dependencies", ""])
    for item in _imports(bundle)[:120]:
        lines.append(f"- `{item.get('name')}` from `{item.get('library') or 'unknown'}`")
    for item in _dependencies(bundle)[:80]:
        lines.append(f"- dependency `{item.get('name')}` ({item.get('kind')})")
    lines.extend(["", "## ABI and calling conventions", "", f"- {_architecture_section(bundle).get('abi')}"])
    lines.append(f"- argument registers: {', '.join(_architecture_section(bundle).get('argument_registers') or [])}")
    lines.append(f"- return register: {_architecture_section(bundle).get('return_register')}")
    lines.append(f"- callee-saved: {', '.join(_architecture_section(bundle).get('callee_saved') or [])}")
    lines.extend(["", "## Functions", ""])
    for row in _function_rows(bundle)[:max_functions]:
        params = ", ".join(f"{item['type']} {item['name']}" for item in row["parameters"]) or "void"
        lines.append(f"### `{row['name']}` at {row['entry']} ({row['size']} bytes)")
        lines.append(f"- ABI: {row['calling_convention']}; return: {row['return_type']}; parameters: {params}")
        lines.append(f"- purpose: {row['purpose']}; confidence: {row['confidence']}")
        if row["callees"]:
            lines.append(f"- callees: {', '.join(str(item) for item in row['callees'][:12])}")
        if row["unknowns"]:
            lines.append(f"- unknowns: {'; '.join(str(item.get('question') or item) for item in row['unknowns'])}")
        lines.append("")
    lines.extend(["## Memory and data-flow observations", ""])
    lines.append("Register def-use, stack slots and global candidates are deterministic heuristics; aliasing and indirect flows are not soundly resolved.")
    lines.extend(["", "## Security review candidates", ""])
    findings = _security(bundle)
    if findings:
        for finding in findings[:80]:
            lines.append(f"- **{finding['severity']}** `{finding['title']}` at {finding.get('address')}: {finding['description']}")
    else:
        lines.append("No dangerous-call candidates were found by the limited static scan.")
    lines.extend(["", "## Performance proxies", ""])
    performance = _performance(bundle)
    lines.append(f"Largest functions: {', '.join(item['name'] for item in performance['largest_functions'][:8]) or 'none'}")
    lines.append(f"Average instructions/function: {performance['average_instructions_per_function']}")
    lines.append(performance["limitation"])
    if research:
        lines.extend(["", "## Reference knowledge", ""])
        for entry in research:
            lines.append(f"- **{entry['title']}**: {'; '.join(entry['facts'][:3])}")
            lines.append(f"  - sources: {', '.join(entry['sources'])}")
    if jni_result:
        lines.extend(["", "## Java/JNI relationships", ""])
        if jni_result.get("error"):
            lines.append(f"Correlation failed: {jni_result['error']}")
        else:
            summary = jni_result.get("summary", {})
            lines.append(f"- Java native methods: {summary.get('java_native_methods')}; native JNI symbols: {summary.get('native_jni_symbols')}; matched: {summary.get('matched')}")
            for match in (jni_result.get("matches") or [])[:80]:
                lines.append(f"- `{match['class_name']}.{match['method']}{match['descriptor']}` -> `{match['native_symbol']}` ({match['confidence']})")
            for miss in (jni_result.get("unmatched_java_methods") or [])[:40]:
                lines.append(f"- unresolved Java native: `{miss['class_name']}.{miss['method']}{miss['descriptor']}`")
    lines.extend(["", "## Reconstructed source", ""])
    if c_result:
        lines.append(f"- C reconstruction: {c_result.get('classification')} at `{report_dir / 'c-reconstruction'}`")
        if c_verification:
            lines.append(f"- C compile verification: {c_verification.get('status')} (rate {c_verification.get('compile_rate')})")
        for function in (c_result.get("functions") or [])[:20]:
            lines.append(f"  - `{function['name']}`: {function['classification']} (confidence {function['confidence']})")
    if java_result:
        lines.append(f"- Java reconstruction: {java_result.get('classification')} at `{report_dir / 'java-reconstruction'}`")
        if java_result.get("verification"):
            lines.append(f"- Java compile verification: {java_result['verification'].get('status')} (rate {java_result['verification'].get('compile_rate')})")
    lines.extend(["", "## Evidence, confidence and unresolved questions", ""])
    for item in unresolved[:50]:
        lines.append(f"- {item.get('question') or item}")
    lines.extend(["", "## Limitations", ""])
    for limitation in (bundle.limitations + list((c_result or {}).get("limitations", [])))[:40]:
        lines.append(f"- {limitation}")
    lines.append("")
    (report_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")
    return {
        "status": "completed",
        "report_dir": str(report_dir),
        "report_markdown": str(report_dir / "report.md"),
        "report_json": str(report_dir / "report.json"),
        "binary": facts["binary"],
        "architecture": facts["architecture"],
        "function_count": len(bundle.functions),
        "c_reconstruction": None if c_result is None else c_result.get("classification"),
        "c_compile_verification": None if c_verification is None else c_verification.get("status"),
        "java_reconstruction": None if java_result is None else java_result.get("classification"),
        "java_compile_verification": None if java_result is None else (java_result.get("verification") or {}).get("status"),
        "jni": None if jni_result is None else jni_result.get("summary"),
        "limitations": [
            "Static analysis only; no dynamic execution or behavioral equivalence.",
            "Inferences and hypotheses are labeled separately from facts.",
        ],
    }
