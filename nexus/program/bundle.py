from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from nexus.architectures import adapter_for_elf
from nexus.architectures.abi import infer_abi_signature
from nexus.core.errors import CapabilityUnavailable
from nexus.dataflow.analysis import analyze_function_dataflow
from nexus.disassembly.model import Disassembly
from nexus.disassembly.objdump import disassemble_elf
from nexus.elf.model import ElfFile
from nexus.elf.parser import parse_elf
from nexus.jni.analysis import analyze_jni
from nexus.program.analysis import recover_functions
from nexus.program.model import CallGraph, FunctionCandidate
from nexus.program.multiarch import analyze_with_llvm
from nexus.program.strings import extract_strings
from nexus.security.analysis import scan_security
from nexus.semantics.purpose import infer_purpose
from nexus.types.inference import infer_types


@dataclass
class AnalysisBundle:
    path: str
    elf: ElfFile
    disassembly: Disassembly | None
    functions: list[FunctionCandidate]
    callgraph: CallGraph
    strings: list[dict[str, Any]]
    dataflows: dict[str, dict[str, Any]]
    type_candidates: dict[str, list[dict[str, Any]]]
    purposes: dict[str, dict[str, Any]]
    security_findings: list[dict[str, Any]]
    jni_report: dict[str, Any]
    started_at: str
    finished_at: str
    warnings: list[str] = field(default_factory=list)
    limitations: list[str] = field(default_factory=list)
    signatures: dict[str, dict[str, Any]] = field(default_factory=dict)
    architecture: dict[str, Any] | None = None

    @property
    def binary_sha256(self) -> str:
        return self.elf.sha256

    def function_by_id(self, function_id: str) -> FunctionCandidate | None:
        return next((function for function in self.functions if function.function_id == function_id), None)

    def function_by_name(self, name: str) -> FunctionCandidate | None:
        return next((function for function in self.functions if function.name == name), None)

    def summary(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "sha256": self.elf.sha256,
            "size": self.elf.size,
            "machine": self.elf.header.machine_name,
            "elf_class": self.elf.header.elf_class,
            "entry": f"0x{self.elf.header.entry:x}",
            "architecture": self.architecture,
            "functions": len(self.functions),
            "instructions": len(self.disassembly.instructions) if self.disassembly else 0,
            "call_edges": len(self.callgraph.edges),
            "strings": len(self.strings),
            "security_findings": len(self.security_findings),
            "jni": self.jni_report.get("is_jni", False),
            "warnings": self.warnings,
            "limitations": self.limitations,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
        }


def _architecture(elf: ElfFile) -> dict[str, Any] | None:
    try:
        adapter = adapter_for_elf(elf)
    except CapabilityUnavailable:
        return None
    return {
        "name": adapter.name,
        "abi": adapter.abi,
        "pointer_width": adapter.pointer_width,
        "endianness": adapter.endianness,
        "decoder": adapter.decoder,
        "argument_registers": list(adapter.argument_registers),
        "return_register": adapter.return_register,
        "stack_pointer": adapter.stack_pointer,
        "link_register": adapter.link_register,
        "callee_saved": list(adapter.callee_saved),
        "capabilities": {
            "elf_recognition": adapter.elf_recognition,
            "instruction_decoding": adapter.instruction_decoding,
            "cfg": adapter.cfg,
            "semantic_analysis": adapter.semantic_analysis,
            "reconstruction": adapter.reconstruction,
            "verification": adapter.verification,
        },
        "notes": list(adapter.notes),
    }


def _finish(
    *,
    path: str,
    elf: ElfFile,
    disassembly: Disassembly | None,
    functions: list[FunctionCandidate],
    callgraph: CallGraph,
    strings: list[dict[str, Any]],
    dataflows: dict[str, dict[str, Any]],
    type_candidates: dict[str, list[dict[str, Any]]],
    purposes: dict[str, dict[str, Any]],
    signatures: dict[str, dict[str, Any]],
    jni_report: dict[str, Any],
    started: str,
    warnings: list[str],
    limitations: list[str],
) -> AnalysisBundle:
    security_findings = scan_security(elf, functions, jni_report)
    limitations.extend(jni_report.get("limitations", []))
    limitations.append("Dynamic execution and differential verification are disabled until a verified KVM sandbox exists.")
    return AnalysisBundle(
        path=path,
        elf=elf,
        disassembly=disassembly,
        functions=functions,
        callgraph=callgraph,
        strings=strings,
        dataflows=dataflows,
        type_candidates=type_candidates,
        purposes=purposes,
        security_findings=security_findings,
        jni_report=jni_report,
        started_at=started,
        finished_at=datetime.now(UTC).isoformat(),
        warnings=warnings,
        limitations=limitations,
        signatures=signatures,
        architecture=_architecture(elf),
    )


def analyze_binary(path: str | Path) -> AnalysisBundle:
    started = datetime.now(UTC).isoformat()
    elf = parse_elf(path)
    strings = extract_strings(elf)
    warnings: list[str] = []
    limitations: list[str] = []
    try:
        adapter = adapter_for_elf(elf)
    except CapabilityUnavailable:
        adapter = None

    if adapter is None:
        warnings.append(f"No architecture adapter for machine {elf.header.machine_name}; metadata-only analysis.")
        limitations.append("Instruction decoding and call-graph recovery are unavailable for this architecture.")
        functions = []
        callgraph = CallGraph(limitations=["No architecture adapter."])
        jni_report = analyze_jni(elf, functions, callgraph, strings)
        return _finish(
            path=str(path), elf=elf, disassembly=None, functions=functions,
            callgraph=callgraph, strings=strings, dataflows={}, type_candidates={},
            purposes={}, signatures={}, jni_report=jni_report, started=started,
            warnings=warnings, limitations=limitations,
        )

    if adapter.name != "x86_64" or elf.header.e_type == 1:
        if not adapter.instruction_decoding:
            warnings.append(f"Disassembly is not implemented for {elf.header.machine_name}; metadata-only analysis.")
            limitations.append(f"Instruction decoding is unavailable for {adapter.name}.")
            functions: list[FunctionCandidate] = []
            callgraph = CallGraph(limitations=[f"No call graph for {adapter.name}."])
            jni_report = analyze_jni(elf, functions, callgraph, strings)
            return _finish(
                path=str(path), elf=elf, disassembly=None, functions=functions,
                callgraph=callgraph, strings=strings, dataflows={}, type_candidates={},
                purposes={}, signatures={}, jni_report=jni_report, started=started,
                warnings=warnings, limitations=limitations,
            )
        try:
            disassembly, functions, callgraph, llvm_warnings, llvm_limitations = analyze_with_llvm(path, elf)
        except Exception as exc:  # decoder/tool failure is an explicit limitation
            warnings.append(f"LLVM disassembly failed for {adapter.name}: {exc}; metadata-only analysis.")
            limitations.append("Instruction decoding failed; no instruction-level claims are made.")
            functions = []
            callgraph = CallGraph(limitations=["Disassembly backend failed."])
            jni_report = analyze_jni(elf, functions, callgraph, strings)
            return _finish(
                path=str(path), elf=elf, disassembly=None, functions=functions,
                callgraph=callgraph, strings=strings, dataflows={}, type_candidates={},
                purposes={}, signatures={}, jni_report=jni_report, started=started,
                warnings=warnings, limitations=limitations,
            )
        warnings.extend(llvm_warnings)
        limitations.extend(llvm_limitations)
        signatures: dict[str, dict[str, Any]] = {}
        purposes: dict[str, dict[str, Any]] = {}
        dataflows: dict[str, dict[str, Any]] = {}
        for function in functions:
            signatures[function.function_id] = infer_abi_signature(function, adapter)
            dataflows[function.function_id] = analyze_function_dataflow(
                function,
                elf.sha256,
                argument_registers=list(adapter.argument_registers),
                return_register=adapter.return_register or "rax",
                stack_pointer=adapter.stack_pointer or "rsp",
                frame_pointer=adapter.frame_pointer or "rbp",
                instruction_pointer="pc",
                architecture=adapter.name,
            )
            purpose = infer_purpose(function, callgraph, elf.sha256)
            function.purpose = purpose.get("purpose")
            purposes[function.function_id] = purpose
        jni_report = analyze_jni(elf, functions, callgraph, strings)
        return _finish(
            path=str(path), elf=elf, disassembly=disassembly, functions=functions,
            callgraph=callgraph, strings=strings, dataflows=dataflows, type_candidates={},
            purposes=purposes, signatures=signatures, jni_report=jni_report,
            started=started, warnings=warnings, limitations=limitations,
        )

    disassembly = disassemble_elf(elf)
    functions, callgraph = recover_functions(elf, disassembly)
    dataflows: dict[str, dict[str, Any]] = {}
    type_candidates: dict[str, list[dict[str, Any]]] = {}
    purposes: dict[str, dict[str, Any]] = {}
    signatures: dict[str, dict[str, Any]] = {}
    for function in functions:
        dataflow = analyze_function_dataflow(
            function,
            elf.sha256,
            argument_registers=list(adapter.argument_registers),
            return_register=adapter.return_register or "rax",
            stack_pointer=adapter.stack_pointer or "rsp",
            frame_pointer=adapter.frame_pointer or "rbp",
            instruction_pointer="rip",
            architecture=adapter.name,
        )
        dataflows[function.function_id] = dataflow
        type_candidates[function.function_id] = infer_types(function, dataflow, elf.sha256)
        purpose = infer_purpose(function, callgraph, elf.sha256)
        function.purpose = purpose.get("purpose")
        purposes[function.function_id] = purpose
        signatures[function.function_id] = infer_abi_signature(
            function, adapter, existing={"parameters": type_candidates[function.function_id]}
        )
    jni_report = analyze_jni(elf, functions, callgraph, strings)
    limitations.extend(disassembly.limitations)
    return _finish(
        path=str(path), elf=elf, disassembly=disassembly, functions=functions,
        callgraph=callgraph, strings=strings, dataflows=dataflows,
        type_candidates=type_candidates, purposes=purposes, signatures=signatures,
        jni_report=jni_report, started=started, warnings=warnings, limitations=limitations,
    )
