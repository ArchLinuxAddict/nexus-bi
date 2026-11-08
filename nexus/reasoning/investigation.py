"""Deterministic, evidence-grounded investigation baseline.

This is not a trained reasoning model.  It demonstrates the typed tool,
evidence, hypothesis and contradiction interfaces that the learned model is
intended to use.  No analyzed artifact is executed; dynamic verification is
refused by the sandbox gate.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from nexus.agent.broker import StaticAnalysisBroker
from nexus.agent.evidence import InvestigationReport
from nexus.core.errors import BudgetExceeded
from nexus.sandbox import gate as sandbox_gate

ROOT = Path(__file__).resolve().parents[2]
REPORT_ROOT = ROOT / "artifacts" / "investigation"


@dataclass
class Hypothesis:
    hypothesis_id: str
    description: str
    confidence: float
    supporting_evidence: list[str] = field(default_factory=list)
    contradicting_evidence: list[str] = field(default_factory=list)
    missing_evidence: list[str] = field(default_factory=list)
    verification_plan: str = ""
    status: str = "open"

    def to_dict(self) -> dict[str, Any]:
        return {
            "hypothesis_id": self.hypothesis_id,
            "description": self.description,
            "confidence": self.confidence,
            "supporting_evidence": self.supporting_evidence,
            "contradicting_evidence": self.contradicting_evidence,
            "missing_evidence": self.missing_evidence,
            "verification_plan": self.verification_plan,
            "status": self.status,
        }


def _evidence_id(result) -> str:
    return result.evidence.evidence_id if result is not None and result.evidence is not None else ""


def _add_contradiction(
    contradictions: list[dict[str, Any]],
    hypothesis: Hypothesis,
    evidence: str,
    reason: str,
) -> None:
    contradictions.append({"hypothesis_id": hypothesis.hypothesis_id, "evidence": evidence, "reason": reason})
    if evidence:
        hypothesis.contradicting_evidence.append(evidence)
    hypothesis.status = "contradicted"
    hypothesis.confidence = max(0.0, hypothesis.confidence - 0.25)


def investigate(path: str, objective: str, max_steps: int = 12, *, evidence_path: str | Path | None = None) -> dict[str, Any]:
    if max_steps <= 0 or max_steps > 32:
        raise BudgetExceeded("max_steps must be between 1 and 32")
    started = datetime.now(UTC).isoformat()
    investigation_id = f"inv_{uuid.uuid4().hex[:16]}"
    broker = StaticAnalysisBroker(evidence_path=evidence_path or (REPORT_ROOT / investigation_id / "evidence.jsonl"))
    report = InvestigationReport(
        investigation_id=investigation_id,
        artifact={"path": str(Path(path).resolve())},
        objective=objective,
    )
    hypotheses: list[Hypothesis] = []
    contradictions: list[dict[str, Any]] = []
    unknowns: list[dict[str, Any]] = []
    calls: list[dict[str, Any]] = []
    steps = 0

    def call(tool: str, arguments: dict[str, Any]):
        nonlocal steps
        if steps >= max_steps:
            raise BudgetExceeded(f"investigation step budget exhausted at {max_steps}")
        steps += 1
        result = broker.call(tool, arguments)
        calls.append(result.to_dict() | {"arguments": arguments})
        return result

    format_result = call("identify_format", {"path": path})
    fmt = str((format_result.output or {}).get("format") or "unknown")
    report.add_fact("fact_format", str(Path(path).resolve()), "format", fmt, [_evidence_id(format_result)], 0.99 if fmt != "unknown" else 0.2)
    report.artifact.update(
        {
            "format": fmt,
            "architecture": (format_result.output or {}).get("architecture"),
            "size": (format_result.output or {}).get("size"),
            "sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
        }
    )

    if fmt == "ELF":
        analyze = call("analyze_binary", {"path": path})
        summary = analyze.output or {}
        report.add_fact("fact_elf_summary", str(Path(path).resolve()), "analysis_summary", summary, [_evidence_id(analyze)], 0.9)
        if analyze.status != "ok":
            hypothesis = Hypothesis("hyp_elf_parse", "The artifact is an ELF image and its structure is recoverable.", 0.4, missing_evidence=["successful ELF parse"], verification_plan="Inspect parser diagnostics.")
            _add_contradiction(contradictions, hypothesis, _evidence_id(analyze), f"ELF analysis failed: {analyze.error}")
            hypotheses.append(hypothesis)
        else:
            hypothesis = Hypothesis(
                "hyp_elf_structure",
                "The ELF image exposes recoverable functions and a direct call structure.",
                0.7,
                supporting_evidence=[_evidence_id(analyze)],
                missing_evidence=["dynamic call frequency", "source-level intent"],
                verification_plan="Inspect call graph, strings and dataflow, then use authorized dynamic analysis.",
            )
            hypotheses.append(hypothesis)
            functions = call("list_functions", {"path": path, "max_items": 128})
            report.add_fact("fact_functions", str(Path(path).resolve()), "recovered_functions", {"count": len((functions.output or {}).get("functions", []))}, [_evidence_id(functions)], 0.7)
            if summary.get("functions") and not (functions.output or {}).get("functions"):
                _add_contradiction(contradictions, hypothesis, _evidence_id(functions), "summary reports functions but the function tool returned none")
            if max_steps >= 6:
                graph = call("call_graph", {"path": path, "max_items": 512})
                report.add_fact("fact_callgraph", str(Path(path).resolve()), "call_graph", {"edges": len((graph.output or {}).get("edges", []))}, [_evidence_id(graph)], 0.6)
            if max_steps >= 8:
                strings = call("extract_strings", {"path": path, "max_items": 128})
                report.add_fact("fact_strings", str(Path(path).resolve()), "strings", {"count": len((strings.output or {}).get("strings", []))}, [_evidence_id(strings)], 0.8)
            jni = call("jni_analysis", {"path": path})
            jni_output = jni.output or {}
            if jni_output.get("is_jni"):
                jni_hypothesis = Hypothesis(
                    "hyp_jni_boundary",
                    "The native image exposes a JNI boundary and reconstructed native methods.",
                    0.65,
                    supporting_evidence=[_evidence_id(jni)],
                    missing_evidence=["Java-side class files", "runtime registration behavior"],
                    verification_plan="Parse the corresponding DEX/class files and compare JNI signatures; compile the Java skeleton.",
                )
                if not jni_output.get("entries"):
                    _add_contradiction(contradictions, jni_hypothesis, _evidence_id(jni), "JNI flag is true but no entries were recorded")
                hypotheses.append(jni_hypothesis)
            security = call("security_scan", {"path": path})
            findings = (security.output or {}).get("findings", [])
            if findings:
                risk = Hypothesis(
                    "hyp_security_findings",
                    "Deterministic security findings identify potentially risky behavior.",
                    0.5,
                    supporting_evidence=[_evidence_id(security)],
                    missing_evidence=["data-flow proof", "runtime trigger", "exploitability validation"],
                    verification_plan="Trace argument provenance and run authorized isolated tests; do not claim exploitability from static findings alone.",
                )
                hypotheses.append(risk)
            if not summary.get("functions") and summary.get("symbols"):
                _add_contradiction(contradictions, hypothesis, _evidence_id(analyze), "function recovery returned zero despite symbol evidence")
    elif fmt == "PE":
        pe = call("parse_pe", {"path": path})
        output = pe.output or {}
        report.add_fact("fact_pe_header", str(Path(path).resolve()), "pe_header", output.get("header", {}), [_evidence_id(pe)], 0.9)
        report.add_fact("fact_pe_sections", str(Path(path).resolve()), "pe_sections", {"count": len(output.get("sections", []))}, [_evidence_id(pe)], 0.9)
        hypothesis = Hypothesis(
            "hyp_pe_imports",
            "PE import and section facts constrain the dependency surface and likely runtime libraries.",
            0.6,
            supporting_evidence=[_evidence_id(pe)],
            missing_evidence=["x86/x86-64 instruction recovery for PE", "dynamic loading behavior"],
            verification_plan="Add PE code-region disassembly and import-thunk resolution before claiming function semantics.",
        )
        if not output.get("imports") and not output.get("sections"):
            _add_contradiction(contradictions, hypothesis, _evidence_id(pe), "PE parser returned neither imports nor sections")
        hypotheses.append(hypothesis)
        unknowns.append({"unknown_id": "unk_pe_code", "question": "What are the recovered function semantics inside PE code sections?", "blocking": True, "probe": "pe_disassembly", "reason": "PE instruction recovery is not implemented in this build."})
    elif fmt == "Mach-O":
        macho = call("parse_macho", {"path": path})
        output = macho.output or {}
        report.add_fact("fact_macho_header", str(Path(path).resolve()), "macho_header", output.get("header", {}), [_evidence_id(macho)], 0.9)
        report.add_fact("fact_macho_dependencies", str(Path(path).resolve()), "macho_dependencies", output.get("dependencies", []), [_evidence_id(macho)], 0.9)
        hypothesis = Hypothesis(
            "hyp_macho_dependencies",
            "Mach-O segments, sections and dylib dependencies constrain the artifact's runtime structure.",
            0.6,
            supporting_evidence=[_evidence_id(macho)],
            missing_evidence=["instruction recovery for Mach-O", "code-signing verification", "dyld resolution"],
            verification_plan="Add Mach-O disassembly and code-signing checks before claiming function semantics.",
        )
        if not output.get("segments"):
            _add_contradiction(contradictions, hypothesis, _evidence_id(macho), "no Mach-O segments were parsed")
        hypotheses.append(hypothesis)
        unknowns.append({"unknown_id": "unk_macho_code", "question": "What are the recovered function semantics inside Mach-O code sections?", "blocking": True, "probe": "macho_disassembly", "reason": "Mach-O instruction recovery is not implemented in this build."})
    elif fmt == "APK":
        apk = call("parse_apk", {"path": path})
        output = apk.output or {}
        report.add_fact("fact_apk_entries", str(Path(path).resolve()), "apk_entries", {"count": output.get("entry_count")}, [_evidence_id(apk)], 0.9)
        report.add_fact("fact_apk_dex", str(Path(path).resolve()), "apk_dex", output.get("dex", []), [_evidence_id(apk)], 0.9)
        report.add_fact("fact_apk_native", str(Path(path).resolve()), "apk_native_libraries", output.get("native_libraries", []), [_evidence_id(apk)], 0.9)
        hypothesis = Hypothesis(
            "hyp_apk_structure",
            "The APK contains DEX and/or native components whose class and native-library boundaries can be inventoried statically.",
            0.65,
            supporting_evidence=[_evidence_id(apk)],
            missing_evidence=["AndroidManifest semantic decoding", "resources.arsc decoding", "DEX data-flow", "runtime behavior"],
            verification_plan="Add manifest/resource decoding and DEX method-body analysis; keep execution disabled.",
        )
        if output.get("classes_dex_count") and not output.get("dex"):
            _add_contradiction(contradictions, hypothesis, _evidence_id(apk), "classes.dex exists but no DEX inventory was produced")
        hypotheses.append(hypothesis)
        if output.get("native_libraries"):
            hypotheses.append(Hypothesis(
                "hyp_apk_jni",
                "Native libraries in the APK indicate a managed/native boundary that should be analyzed with JNI tooling.",
                0.6,
                supporting_evidence=[_evidence_id(apk)],
                missing_evidence=["JNI symbol mapping", "Java-side call sites"],
                verification_plan="Parse each native library with the ELF tool and map JNI exports to DEX classes/methods.",
            ))
        unknowns.append({"unknown_id": "unk_apk_manifest", "question": "What components, permissions and entry points does AndroidManifest.xml declare?", "blocking": True, "probe": "android_manifest_decode", "reason": "Binary AndroidManifest.xml semantic decoding is not implemented in this build."})
    elif fmt == "DEX":
        dex = call("parse_dex", {"path": path})
        output = dex.output or {}
        report.add_fact("fact_dex_classes", str(Path(path).resolve()), "dex_classes", {"count": len(output.get("classes", []))}, [_evidence_id(dex)], 0.9)
        hypotheses.append(Hypothesis(
            "hyp_dex_inventory",
            "DEX tables expose class and method inventory suitable for structural analysis.",
            0.7,
            supporting_evidence=[_evidence_id(dex)],
            missing_evidence=["method-body dataflow", "runtime behavior"],
            verification_plan="Add DEX bytecode decoding and control-flow recovery; do not claim Java source reconstruction.",
        ))
    elif fmt == "Java class":
        java = call("parse_java_class", {"path": path})
        output = java.output or {}
        report.add_fact("fact_java_class", str(Path(path).resolve()), "java_class", {"name": output.get("name"), "methods": output.get("method_count")}, [_evidence_id(java)], 0.9)
        hypotheses.append(Hypothesis(
            "hyp_java_structure",
            "The class file exposes fields, methods and interface structure.",
            0.7,
            supporting_evidence=[_evidence_id(java)],
            missing_evidence=["method-body semantics", "runtime behavior"],
            verification_plan="Add bytecode dataflow and inter-class call analysis.",
        ))
    else:
        scan = call("scan_magic", {"path": path, "max_findings": 128})
        report.add_fact("fact_unknown_scan", str(Path(path).resolve()), "embedded_signatures", scan.output or {}, [_evidence_id(scan)], 0.4)
        hypotheses.append(Hypothesis(
            "hyp_unknown_format",
            "The artifact format is not identified by the current parser set.",
            0.2,
            supporting_evidence=[_evidence_id(format_result), _evidence_id(scan)],
            missing_evidence=["format specification", "structural parser"],
            verification_plan="Inspect embedded signatures, entropy and region boundaries; add a parser only when the format is established.",
            status="unresolved",
        ))
        unknowns.append({"unknown_id": "unk_unknown_format", "question": "What is the container/format and where are its executable regions?", "blocking": True, "probe": "scan_magic + entropy + read_bytes", "reason": "No parser currently recognizes this format."})

    sandbox = sandbox_gate.status().to_dict()
    report.add_fact("fact_sandbox", str(Path(path).resolve()), "dynamic_sandbox", sandbox, [], 1.0)
    if not sandbox.get("available"):
        unknowns.append({"unknown_id": "unk_dynamic", "question": "What is the artifact's runtime behavior?", "blocking": True, "probe": "verified KVM sandbox", "reason": sandbox.get("reason", "sandbox unavailable")})

    # Resolve every evidence reference and fail closed if a claim points nowhere.
    all_evidence = [item for hypothesis in hypotheses for item in hypothesis.supporting_evidence + hypothesis.contradicting_evidence]
    resolution = broker.evidence.resolve([item for item in all_evidence if item])
    if not resolution["complete"]:
        unknowns.append({"unknown_id": "unk_evidence_refs", "question": "Do all hypothesis references resolve to stored evidence?", "blocking": True, "probe": "evidence_store.resolve", "reason": f"missing evidence: {resolution['missing'][:10]}"})

    for hypothesis in hypotheses:
        report.add_hypothesis(
            hypothesis.hypothesis_id,
            hypothesis.description,
            "contradicted" if hypothesis.status == "contradicted" else ("unresolved" if hypothesis.status == "unresolved" else "proposed"),
            hypothesis.supporting_evidence,
            hypothesis.contradicting_evidence,
            [hypothesis.verification_plan],
        )
    for contradiction in contradictions:
        report.add_claim(
            f"contradiction_{len(report.claims)+1}",
            contradiction["reason"],
            "CONTRADICTED",
            0.0,
            [],
            [contradiction["evidence"]] if contradiction["evidence"] else [],
            ["additional evidence"],
        )
    for unknown in unknowns:
        report.add_unknown(unknown["unknown_id"], unknown["question"], unknown["blocking"], unknown["probe"], unknown["reason"])
    report.tool_calls = calls
    report.limitations = [
        "The investigation policy is deterministic and heuristic; it is not a trained reasoning model.",
        "Static findings are not behavioral proof; no artifact was executed.",
        "Evidence references resolve to the persisted evidence store, but model-generated claims remain unverified until independent tests pass.",
    ]
    report.status = "completed"
    finished = datetime.now(UTC).isoformat()
    payload = report.to_dict()
    payload.update(
        {
            "started_at": started,
            "finished_at": finished,
            "steps": steps,
            "max_steps": max_steps,
            "actions": calls,
            "hypotheses": [hypothesis.to_dict() for hypothesis in hypotheses],
            "contradictions": contradictions,
            "evidence_resolution": {"complete": resolution["complete"], "missing": resolution["missing"], "stored": broker.evidence.count()},
            "sandbox": sandbox,
            "binary_sha256": report.artifact.get("sha256"),
            "format": fmt,
            "architecture": report.artifact.get("architecture"),
            "conclusion": {
                "summary": f"Identified {fmt}; executed {len(calls)} typed tool calls and recorded {len(hypotheses)} hypotheses and {len(contradictions)} contradictions.",
                "confidence": 0.5 if hypotheses else 0.1,
                "verification": "static only; dynamic verification is disabled until a verified KVM sandbox exists",
            },
        }
    )
    target = REPORT_ROOT / investigation_id
    target.mkdir(parents=True, exist_ok=True)
    (target / "report.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload
