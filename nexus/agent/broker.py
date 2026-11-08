"""Closed, typed tool broker for static and compiler-backed analysis.

No unrestricted shell, no arbitrary execution of the analyzed artifact, no
network.  Every call produces a typed evidence record with input hashes,
limitations and a reproducibility command.  Dynamic execution remains refused
by ``nexus.sandbox.gate`` until a verified isolated worker exists.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from nexus.agent.evidence import EvidenceRecord, EvidenceStore
from nexus.formats.apk import parse_apk
from nexus.formats.dex import parse_dex
from nexus.formats.identify import identify_format
from nexus.formats.java_class import parse_java_class
from nexus.formats.macho import parse_macho
from nexus.formats.pe import parse_pe
from nexus.program.bundle import analyze_binary
from nexus.program.ir import build_program_ir
from nexus.program.record import build_program_record
from nexus.research.knowledge import research_topic
from nexus.sandbox import gate as sandbox_gate

ROOT = Path(__file__).resolve().parents[2]
MAX_ARTIFACT_BYTES = 256 * 1024 * 1024
MAX_SOURCE_BYTES = 4 * 1024 * 1024
MAX_OUTPUT_ITEMS = 2000


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _arguments_hash(arguments: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(arguments, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def _limit(value: Any, limit: int = MAX_OUTPUT_ITEMS) -> Any:
    if isinstance(value, list):
        return value[:limit]
    if isinstance(value, dict):
        return {key: _limit(item, limit) for key, item in value.items()}
    return value


def _summary(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: (
                {"type": type(item).__name__, "count": len(item)}
                if isinstance(item, (list, dict))
                else item
            )
            for key, item in value.items()
            if key
            in {
                "path",
                "format",
                "architecture",
                "class",
                "size",
                "functions",
                "instructions",
                "blocks",
                "sections",
                "segments",
                "symbols",
                "imports",
                "exports",
                "strings",
                "classes",
                "methods",
                "fields",
                "dex",
                "native_libraries",
                "signatures",
                "entry_count",
                "method_count",
                "field_count",
                "class_names",
                "returncode",
                "status",
            }
        }
    if isinstance(value, (list, tuple)):
        return {"type": type(value).__name__, "count": len(value)}
    if isinstance(value, str):
        return value[:500]
    return value


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    required_arguments: tuple[str, ...]
    optional_arguments: tuple[str, ...] = ()
    timeout_seconds: float = 120.0
    max_input_bytes: int = MAX_ARTIFACT_BYTES
    read_only: bool = True
    tier: str = "FACT"
    limitations: tuple[str, ...] = ()

    def validate(self, arguments: dict[str, Any]) -> list[str]:
        problems = []
        for key in self.required_arguments:
            if key not in arguments:
                problems.append(f"missing required argument: {key}")
            elif not isinstance(arguments[key], (str, int, float, bool)) and not isinstance(arguments[key], list):
                problems.append(f"argument has unsupported type: {key}")
        allowed = set(self.required_arguments) | set(self.optional_arguments)
        for key in arguments:
            if key not in allowed:
                problems.append(f"unknown argument: {key}")
        return problems


@dataclass
class ToolResult:
    tool: str
    status: str
    duration_seconds: float
    evidence: EvidenceRecord | None = None
    output: Any = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool": self.tool,
            "status": self.status,
            "duration_seconds": round(self.duration_seconds, 4),
            "evidence": self.evidence.to_dict() if self.evidence else None,
            "output_summary": _summary(self.output),
            "error": self.error,
        }


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, tuple[ToolSpec, Callable[[dict[str, Any]], Any]]] = {}

    def register(self, spec: ToolSpec, handler: Callable[[dict[str, Any]], Any]) -> None:
        if spec.name in self._tools:
            raise ValueError(f"tool already registered: {spec.name}")
        self._tools[spec.name] = (spec, handler)

    def names(self) -> list[str]:
        return sorted(self._tools)

    def spec(self, name: str) -> ToolSpec:
        if name not in self._tools:
            raise KeyError(f"unknown tool: {name}")
        return self._tools[name][0]

    def handler(self, name: str) -> Callable[[dict[str, Any]], Any]:
        if name not in self._tools:
            raise KeyError(f"unknown tool: {name}")
        return self._tools[name][1]

    def catalog(self) -> list[dict[str, Any]]:
        return [
            {
                "name": spec.name,
                "description": spec.description,
                "required_arguments": list(spec.required_arguments),
                "optional_arguments": list(spec.optional_arguments),
                "read_only": spec.read_only,
                "limitations": list(spec.limitations),
            }
            for spec, _handler in sorted(self._tools.values(), key=lambda item: item[0].name)
        ]


def _bundle(args: dict[str, Any]):
    return analyze_binary(Path(args["path"]))


def _record(args: dict[str, Any]):
    return build_program_record(_bundle(args))


def _run_compiler(command: list[str], cwd: Path, timeout: float) -> dict[str, Any]:
    try:
        result = subprocess.run(command, cwd=cwd, capture_output=True, text=True, timeout=timeout)
        return {
            "command": command,
            "returncode": result.returncode,
            "stdout": result.stdout[-4000:],
            "stderr": result.stderr[-4000:],
            "status": "passed" if result.returncode == 0 else "failed",
        }
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"command": command, "returncode": None, "stdout": "", "stderr": str(exc), "status": "unavailable"}


def _compile_source(arguments: dict[str, Any], *, syntax_only: bool, language: str) -> dict[str, Any]:
    source = str(arguments.get("source") or "")
    if len(source.encode("utf-8")) > MAX_SOURCE_BYTES:
        raise ValueError("source exceeds the cooperative source-size budget")
    if language == "c":
        compiler = shutil.which(str(arguments.get("compiler") or "gcc")) or shutil.which("clang")
        if compiler is None:
            return {"status": "unavailable", "reason": "no C compiler"}
        suffix = ".c"
        command = [compiler, "-std=c11"]
        if syntax_only:
            command.append("-fsyntax-only")
        else:
            command.extend(["-c", "candidate.c", "-o", "candidate.o"])
    else:
        compiler = shutil.which("javac")
        if compiler is None:
            return {"status": "unavailable", "reason": "no javac"}
        suffix = ".java"
        command = [compiler, "-proc:none"]
        if syntax_only:
            command.extend(["-d", "classes", "Candidate.java"])
        else:
            command.extend(["-d", "classes", "Candidate.java"])
    with tempfile.TemporaryDirectory(dir=arguments.get("workspace") or None) as directory:
        root = Path(directory)
        (root / f"Candidate{suffix}").write_text(source, encoding="utf-8")
        if language == "c":
            command = [part if part != "candidate.c" else f"Candidate{suffix}" for part in command]
            command = [part if part != "candidate.o" else "Candidate.o" for part in command]
        return _run_compiler(command, root, float(arguments.get("timeout_seconds") or 30))


def _scan_magic(arguments: dict[str, Any]) -> dict[str, Any]:
    path = Path(arguments["path"])
    data = path.read_bytes()
    signatures = [
        (b"\x7fELF", "ELF"),
        (b"MZ", "PE"),
        (b"PK\x03\x04", "ZIP/APK"),
        (b"\x1f\x8b", "gzip"),
        (b"\xfd7zXZ\x00", "xz"),
        (b"(\xb5/\xfd", "zstd"),
        (b"hsqs", "SquashFS"),
        (b"sqsh", "SquashFS"),
        (b"ANDROID!", "Android boot"),
        (b"\x27\x05\x19\x56", "U-Boot"),
        (b"dex\n", "DEX"),
        (b"\xca\xfe\xba\xbe", "Java class"),
    ]
    findings: list[dict[str, Any]] = []
    limit = int(arguments.get("max_findings") or 256)
    for magic, name in signatures:
        start = 0
        while len(findings) < limit:
            offset = data.find(magic, start)
            if offset < 0:
                break
            findings.append({"offset": offset, "format": name, "magic": magic.hex()})
            start = offset + 1
    findings.sort(key=lambda item: (item["offset"], item["format"]))
    return {"path": str(path), "size": len(data), "findings": findings[:limit], "truncated": len(findings) >= limit}


class StaticAnalysisBroker:
    def __init__(self, allowed_roots: list[Path] | None = None, evidence_path: str | Path | None = None):
        self.allowed_roots = [path.resolve() for path in (allowed_roots or [ROOT, Path("/tmp")])]
        self.registry = ToolRegistry()
        self.evidence = EvidenceStore(evidence_path or (ROOT / "artifacts" / "investigation" / "evidence.jsonl"))
        self.audit_path = self.evidence.path.with_name("audit.jsonl")
        self._register_tools()

    def _register_tools(self) -> None:
        register = self.registry.register
        register(
            ToolSpec("identify_format", "Identify an artifact format from magic bytes.", ("path",)),
            lambda args: identify_format(args["path"]),
        )
        register(
            ToolSpec("analyze_binary", "Parse an ELF and return a structural AnalysisBundle summary.", ("path",)),
            lambda args: _bundle(args).summary(),
        )
        register(
            ToolSpec("build_program_record", "Build a schema-validated ProgramRecord from an ELF.", ("path",), ("max_functions",)),
            lambda args: _limit(_record(args), int(args.get("max_functions") or MAX_OUTPUT_ITEMS)),
        )
        register(
            ToolSpec("elf_metadata", "Return ELF header, segments, sections, symbols, relocations and dependencies.", ("path",), ("max_items",)),
            lambda args: self._elf_metadata(args),
        )
        register(
            ToolSpec("list_functions", "List recovered functions with entry, size, confidence and evidence.", ("path",), ("max_items",)),
            lambda args: {
                "functions": _limit([function.to_dict() for function in _bundle(args).functions], int(args.get("max_items") or 512))
            },
        )
        register(
            ToolSpec("disassemble_function", "Return decoded instructions for one recovered function.", ("path", "function_id")),
            lambda args: self._function_view(args),
        )
        register(
            ToolSpec("extract_cfg", "Return basic blocks, successors and instructions for one function.", ("path", "function_id")),
            lambda args: self._cfg_view(args),
        )
        register(
            ToolSpec("extract_dataflow", "Return deterministic dataflow candidates for one function.", ("path", "function_id")),
            lambda args: {"dataflow": _bundle(args).dataflows.get(str(args["function_id"]), {})},
        )
        register(
            ToolSpec("call_graph", "Return the recovered direct call graph.", ("path",), ("max_items",)),
            lambda args: _limit(_bundle(args).callgraph.to_dict(), int(args.get("max_items") or 1024)),
        )
        register(
            ToolSpec("build_ir", "Build the normalized NEXUS-BI IR for the program.", ("path",), ("max_functions",)),
            lambda args: _limit(build_program_ir(_bundle(args)), int(args.get("max_functions") or 128)),
        )
        register(
            ToolSpec("extract_strings", "Extract printable strings with addresses and evidence.", ("path",), ("max_items",)),
            lambda args: {"strings": _limit(_bundle(args).strings, int(args.get("max_items") or 512))},
        )
        register(
            ToolSpec("jni_analysis", "Return JNI boundary candidates from ELF symbols and imports.", ("path",)),
            lambda args: _bundle(args).jni_report,
        )
        register(
            ToolSpec("security_scan", "Return deterministic security findings with evidence.", ("path",)),
            lambda args: {"findings": _limit(_bundle(args).security_findings, 512)},
        )
        register(
            ToolSpec("research_topic", "Search curated ABI/ELF/JNI knowledge and local system headers with citations.", ("query",), ("include_headers", "limit")),
            lambda args: research_topic(
                str(args["query"]),
                include_headers=bool(args.get("include_headers", True)),
                limit=max(1, min(int(args.get("limit") or 8), 32)),
            ),
        )
        register(
            ToolSpec("parse_pe", "Parse PE32/PE32+ headers, sections, imports and exports.", ("path",)),
            lambda args: parse_pe(args["path"]),
        )
        register(
            ToolSpec("parse_macho", "Parse Mach-O headers, segments, sections, dependencies and symbols.", ("path",)),
            lambda args: parse_macho(args["path"]),
        )
        register(
            ToolSpec("parse_apk", "Inventory an APK: ZIP entries, DEX tables, native libraries and signatures.", ("path",)),
            lambda args: parse_apk(args["path"]),
        )
        register(
            ToolSpec("parse_dex", "Parse DEX tables into class and method inventory.", ("path",)),
            lambda args: parse_dex(args["path"]),
        )
        register(
            ToolSpec("parse_java_class", "Parse a Java class file into fields, methods and interfaces.", ("path",)),
            lambda args: parse_java_class(args["path"]),
        )
        register(
            ToolSpec("compile_c_source", "Compile untrusted C source with a bounded compiler invocation.", ("source",), ("compiler", "workspace", "timeout_seconds"), read_only=False, limitations=("Compiler invocation is not a sandbox for malicious compiler plugins.",)),
            lambda args: _compile_source(args, syntax_only=False, language="c"),
        )
        register(
            ToolSpec("syntax_check_c", "Run a C syntax-only compiler check.", ("source",), ("compiler", "workspace", "timeout_seconds"), read_only=False),
            lambda args: _compile_source(args, syntax_only=True, language="c"),
        )
        register(
            ToolSpec("compile_java_source", "Compile a Java source file with javac.", ("source",), ("workspace", "timeout_seconds"), read_only=False),
            lambda args: _compile_source(args, syntax_only=False, language="java"),
        )
        register(
            ToolSpec("scan_magic", "Scan a file for recognizable embedded format signatures.", ("path",), ("max_findings",)),
            lambda args: _scan_magic(args),
        )
        register(
            ToolSpec("sandbox_status", "Report whether a verified dynamic-analysis sandbox is available.", ()),
            lambda _args: sandbox_gate.status().to_dict(),
        )
        register(
            ToolSpec("differential_test", "Request differential execution; refused while the sandbox is unavailable.", ("reference", "candidate"), ("cases",)),
            lambda args: sandbox_gate.differential_test(args),
        )

    def _validate_path(self, path: str) -> Path:
        target = Path(path).expanduser().resolve()
        if not any(target == root or root in target.parents for root in self.allowed_roots):
            raise ValueError(f"artifact path is outside approved roots: {target}")
        if not target.exists() or not target.is_file():
            raise FileNotFoundError(f"artifact is missing or not a file: {target}")
        if target.stat().st_size > MAX_ARTIFACT_BYTES:
            raise ValueError(f"artifact exceeds {MAX_ARTIFACT_BYTES} bytes: {target}")
        return target

    def _elf_metadata(self, args: dict[str, Any]) -> dict[str, Any]:
        bundle = _bundle(args)
        limit = int(args.get("max_items") or 512)
        return {
            "header": bundle.elf.header.to_dict(),
            "segments": _limit([segment.to_dict() for segment in bundle.elf.program_headers], limit),
            "sections": _limit([section.to_dict() for section in bundle.elf.sections], limit),
            "symbols": _limit([symbol.to_dict() for symbol in bundle.elf.symbols], limit),
            "dynamic_symbols": _limit([symbol.to_dict() for symbol in bundle.elf.dynamic_symbols], limit),
            "relocations": _limit([relocation.to_dict() for relocation in bundle.elf.relocations], limit),
            "dependencies": _limit(bundle.elf.dependencies, limit),
            "build_id": bundle.elf.build_id,
            "stripped": bundle.elf.is_stripped if hasattr(bundle.elf, "is_stripped") else None,
        }

    def _function_view(self, args: dict[str, Any]) -> dict[str, Any]:
        bundle = _bundle(args)
        function = bundle.function_by_id(str(args["function_id"]))
        if function is None:
            raise KeyError(f"unknown function: {args['function_id']}")
        return {
            "function": function.to_dict(),
            "instructions": [instruction.to_dict() for instruction in function.instructions[:MAX_OUTPUT_ITEMS]],
        }

    def _cfg_view(self, args: dict[str, Any]) -> dict[str, Any]:
        bundle = _bundle(args)
        function = bundle.function_by_id(str(args["function_id"]))
        if function is None:
            raise KeyError(f"unknown function: {args['function_id']}")
        return {
            "function_id": function.function_id,
            "blocks": [block.to_dict() for block in function.blocks[:MAX_OUTPUT_ITEMS]],
        }

    def _audit(self, payload: dict[str, Any]) -> None:
        self.audit_path.parent.mkdir(parents=True, exist_ok=True)
        with self.audit_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True) + "\n")

    def call(self, tool: str, arguments: dict[str, Any]) -> ToolResult:
        started = time.perf_counter()
        try:
            spec = self.registry.spec(tool)
            problems = spec.validate(arguments)
            if problems:
                result = ToolResult(tool, "rejected", time.perf_counter() - started, error="; ".join(problems))
                self._audit(result.to_dict() | {"arguments": arguments})
                return result
            subject: dict[str, Any] = {"arguments_sha256": _arguments_hash(arguments)}
            if "path" in arguments:
                target = self._validate_path(str(arguments["path"]))
                subject.update({"artifact_sha256": _sha256(target), "path": str(target), "size": target.stat().st_size})
            output = self.registry.handler(tool)(arguments)
            evidence = EvidenceRecord(
                producer=tool,
                producer_version="nexus-agent-v1",
                subject=subject,
                observation={"kind": "tool_result", "summary": _summary(output)},
                tier=spec.tier,
                confidence=1.0 if spec.read_only else 0.95,
                limitations=list(spec.limitations) + (["static analysis only; no execution"] if spec.read_only else []),
                reproducibility={"tool": tool, "arguments": arguments, "command": f"nexus-agent {tool}"},
            )
            self.evidence.append(evidence)
            elapsed = time.perf_counter() - started
            status = "ok" if elapsed <= spec.timeout_seconds else "timeout"
            result = ToolResult(tool, status, elapsed, evidence=evidence, output=output, error=None if status == "ok" else "tool exceeded timeout")
            self._audit(result.to_dict() | {"arguments": arguments})
            return result
        except Exception as exc:
            result = ToolResult(tool, "error", time.perf_counter() - started, error=str(exc))
            self._audit(result.to_dict() | {"arguments": arguments})
            return result

    def catalog(self) -> list[dict[str, Any]]:
        return self.registry.catalog()
