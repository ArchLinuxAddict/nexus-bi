"""Unified C, Java and JNI reconstruction interface.

Deterministic analysis/reconstruction and neural generation are kept separate.
Each candidate carries explicit syntax, compilation, structural, behavioral,
finite-domain and universal-equivalence statuses.  Compilation is never reported
as behavioral correctness, and unresolved operations remain visible.
"""

from __future__ import annotations

import difflib
import json
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import torch

from nexus.foundation.checkpoint import load_foundation_checkpoint
from nexus.foundation.config import FoundationConfig
from nexus.foundation.model import NexusFoundationModel
from nexus.foundation.representation import RepresentationOptions, collate_samples, encode_function_sample
from nexus.foundation.vocab import Vocabulary
from nexus.program.bundle import analyze_binary
from nexus.program.record import build_program_record
from nexus.reconstruction.c.generator import reconstruct_c
from nexus.reconstruction.java.generator import reconstruct_java

STATUS_VALUES = ("passed", "failed", "not_attempted", "unknown", "not_claimed")
JNI_TYPES = [
    "unknown",
    "void",
    "jboolean",
    "jbyte",
    "jchar",
    "jshort",
    "jint",
    "jlong",
    "jfloat",
    "jdouble",
    "jobject",
    "jstring",
]
JNI_ARGUMENT_TYPES = [
    "unknown",
    "jboolean",
    "jbyte",
    "jchar",
    "jshort",
    "jint",
    "jlong",
    "jfloat",
    "jdouble",
    "jobject",
    "jstring",
    "jarray",
]


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[A-Za-z_][A-Za-z0-9_]*|0[xX][0-9a-fA-F]+|\d+|[^\s]", text or "")


def structural_similarity(generated: str, reference: str | None) -> dict[str, Any]:
    if not reference:
        return {"status": "not_attempted", "reason": "no reference source is available", "ratio": None, "jaccard": None}
    generated_tokens = _tokenize(generated)
    reference_tokens = _tokenize(reference)
    ratio = difflib.SequenceMatcher(a=generated_tokens, b=reference_tokens, autojunk=False).ratio()
    generated_set = set(generated_tokens)
    reference_set = set(reference_tokens)
    jaccard = len(generated_set & reference_set) / max(1, len(generated_set | reference_set))
    return {
        "status": "measured" if ratio >= 0 else "failed",
        "ratio": round(ratio, 6),
        "jaccard": round(jaccard, 6),
        "generated_tokens": len(generated_tokens),
        "reference_tokens": len(reference_tokens),
        "note": "Structural similarity is not behavioral equivalence.",
    }


def _run(command: list[str], cwd: Path, timeout: int = 20) -> dict[str, Any]:
    try:
        result = subprocess.run(command, cwd=cwd, capture_output=True, text=True, timeout=timeout)
        return {
            "command": command,
            "returncode": result.returncode,
            "stdout": result.stdout[-2000:],
            "stderr": result.stderr[-2000:],
        }
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"command": command, "returncode": None, "stdout": "", "stderr": str(exc)}


def check_c_source(source: str, workspace: str | Path | None = None, include_files: dict[str, str] | None = None) -> dict[str, Any]:
    compiler = shutil.which("gcc") or shutil.which("clang")
    if compiler is None:
        return {
            "syntax": {"status": "not_attempted", "reason": "no C compiler"},
            "compilation": {"status": "not_attempted", "reason": "no C compiler"},
        }
    with tempfile.TemporaryDirectory(dir=workspace) as directory:
        root = Path(directory)
        path = root / "candidate.c"
        include_flags = []
        if include_files:
            include_root = root / "include"
            include_root.mkdir(parents=True, exist_ok=True)
            for name, content in include_files.items():
                if name.startswith("include/"):
                    target = include_root / name[len("include/") :]
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_text(content, encoding="utf-8")
            include_flags = ["-I", str(include_root)]
            cleaned = source or ""
        else:
            cleaned = re.sub(r'(?m)^#include\s+"[^"]+"\s*$\n?', "", source or "")
        path.write_text(
            "#include <stdint.h>\n#include <stddef.h>\n#include <stdlib.h>\n#include <string.h>\n" + cleaned,
            encoding="utf-8",
        )
        syntax = _run([compiler, "-std=c11", "-fsyntax-only", *include_flags, str(path)], root)
        compile_result = _run([compiler, "-std=c11", "-c", *include_flags, str(path), "-o", str(root / "candidate.o")], root)
    return {
        "syntax": {
            "status": "passed" if syntax["returncode"] == 0 else "failed",
            "returncode": syntax["returncode"],
            "diagnostics": (syntax["stderr"] or syntax["stdout"])[-1500:],
        },
        "compilation": {
            "status": "passed" if compile_result["returncode"] == 0 else "failed",
            "returncode": compile_result["returncode"],
            "diagnostics": (compile_result["stderr"] or compile_result["stdout"])[-1500:],
        },
        "note": "Syntax and compilation are checked; the artifact is not executed.",
    }


def check_java_source(source: str, class_name: str = "ReconstructedModule", workspace: str | Path | None = None) -> dict[str, Any]:
    compiler = shutil.which("javac")
    if compiler is None:
        return {
            "syntax": {"status": "not_attempted", "reason": "no javac"},
            "compilation": {"status": "not_attempted", "reason": "no javac"},
        }
    with tempfile.TemporaryDirectory(dir=workspace) as directory:
        root = Path(directory)
        path = root / f"{class_name}.java"
        path.write_text(source, encoding="utf-8")
        syntax = _run([compiler, "-proc:none", "-d", str(root / "classes"), str(path)], root)
    return {
        "syntax": {
            "status": "passed" if syntax["returncode"] == 0 else "failed",
            "returncode": syntax["returncode"],
            "diagnostics": (syntax["stderr"] or syntax["stdout"])[-1500:],
        },
        "compilation": {
            "status": "passed" if syntax["returncode"] == 0 else "failed",
            "returncode": syntax["returncode"],
            "diagnostics": (syntax["stderr"] or syntax["stdout"])[-1500:],
        },
        "note": "javac compilation is not behavioral verification.",
    }


def _empty_statuses() -> dict[str, Any]:
    return {
        "syntax_valid": "unknown",
        "compilation": "not_attempted",
        "structural_similarity": "not_attempted",
        "behavioral_agreement": "not_attempted",
        "finite_domain_verified": "not_attempted",
        "universal_equivalence": "not_claimed",
    }


def _candidate(
    language: str,
    source: str,
    method: str,
    statuses: dict[str, Any],
    evidence: dict[str, Any],
    unresolved: list[str],
    confidence: float | None,
    verification: dict[str, Any] | None = None,
    reference: str | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    similarity = structural_similarity(source, reference)
    if statuses.get("structural_similarity") == "not_attempted" and similarity["status"] != "not_attempted":
        statuses["structural_similarity"] = similarity["status"]
    return {
        "schema_version": "0.1.0",
        "language": language,
        "method": method,
        "source": source,
        "statuses": statuses,
        "verification": verification or {},
        "structural_metrics": similarity,
        "unresolved_operations": unresolved,
        "confidence": confidence,
        "confidence_note": "Confidence is a model/evidence estimate, not a correctness probability.",
        "evidence": evidence,
        "additional": extra or {},
        "limitations": [
            "Compilation is not behavioral agreement.",
            "Finite-domain tests are not universal equivalence.",
            "Universal equivalence is never claimed without a valid proof.",
            "Untrusted artifacts are not executed on the analysis host.",
        ],
    }


def encode_binary_function(
    record: dict[str, Any],
    function_id: str,
    vocab: Vocabulary,
    config: FoundationConfig,
) -> dict[str, Any] | None:
    function = next((item for item in record.get("functions", []) if item.get("function_id") == function_id), None)
    if function is None:
        return None
    return encode_function_sample(
        record,
        function,
        vocab,
        config,
        identity={
            "sample_id": f"inference:{function_id}",
            "function_id": function_id,
            "project_id": "inference",
            "dataset_id": "inference",
            "split": "inference",
            "binary_sha256": record.get("program", {}).get("sha256", ""),
            "source_family": "",
            "match_group": "",
        },
        options=RepresentationOptions(include_source_target=False, include_purpose_target=False),
    )


class UnifiedReconstructor:
    def __init__(
        self,
        checkpoint_path: str | Path | None = None,
        model: NexusFoundationModel | None = None,
        vocab: Vocabulary | None = None,
        config: FoundationConfig | None = None,
        checkpoint_metadata: dict[str, Any] | None = None,
        checkpoint_sha256: str | None = None,
        workspace: str | Path | None = None,
        allow_compilation: bool = True,
    ):
        self.checkpoint_path = str(checkpoint_path) if checkpoint_path else None
        self.model = model
        self.vocab = vocab
        self.config = config
        self.checkpoint_metadata = checkpoint_metadata or {}
        self.checkpoint_sha256 = checkpoint_sha256
        self.workspace = Path(workspace) if workspace else None
        if self.workspace is not None:
            self.workspace.mkdir(parents=True, exist_ok=True)
        self.allow_compilation = allow_compilation
        if checkpoint_path is not None and model is None:
            loaded = load_foundation_checkpoint(checkpoint_path, load_training_state=False)
            self.model = loaded.model
            self.vocab = loaded.vocab
            self.config = loaded.config
            self.checkpoint_metadata = loaded.metadata
            self.checkpoint_sha256 = loaded.sha256
        if self.model is not None:
            self.model.eval()

    def _model_candidate(self, batch: dict[str, Any], language: str, max_new_tokens: int) -> tuple[str, dict[str, Any]]:
        if self.model is None or self.vocab is None:
            return "", {"status": "unavailable", "reason": "no foundation checkpoint was supplied"}
        result = self.model.generate_reconstruction(batch, language=language, max_new_tokens=max_new_tokens)
        tokens = result["token_ids"][0]
        source = self.vocab.decode_text(tokens)
        with torch.no_grad():
            outputs = self.model(batch)
        confidence = float(outputs["confidence"][0].detach())
        return source, {
            "status": "generated",
            "token_count": len(tokens),
            "token_ids": tokens[:256],
            "checkpoint": self.checkpoint_path,
            "checkpoint_sha256": self.checkpoint_sha256,
            "checkpoint_step": self.checkpoint_metadata.get("step"),
            "checkpoint_status": self.checkpoint_metadata.get("status", "unknown"),
            "confidence": confidence,
        }

    def _verify(self, language: str, source: str, class_name: str = "ReconstructedModule", include_files: dict[str, str] | None = None) -> dict[str, Any]:
        if not self.allow_compilation:
            return {
                "syntax": {"status": "not_attempted", "reason": "compilation disabled by caller"},
                "compilation": {"status": "not_attempted", "reason": "compilation disabled by caller"},
            }
        if language == "c":
            return check_c_source(source, self.workspace, include_files=include_files)
        return check_java_source(source, class_name, self.workspace)

    def reconstruct_function(
        self,
        binary_path: str | Path,
        function_id: str,
        language: str = "c",
        reference_source: str | None = None,
        max_new_tokens: int = 96,
        include_deterministic: bool = True,
        include_neural: bool = True,
    ) -> dict[str, Any]:
        bundle = analyze_binary(binary_path)
        record = build_program_record(bundle)
        function = next(
            (
                item
                for item in record.get("functions", [])
                if item.get("function_id") == function_id
                or any(name.get("value") == function_id for name in item.get("names", []) if isinstance(name, dict))
            ),
            None,
        )
        if function is None:
            return {"status": "not_found", "function_id": function_id, "binary_sha256": record.get("program", {}).get("sha256")}
        return self._reconstruct_function(
            bundle,
            record,
            function,
            language=language,
            reference_source=reference_source,
            max_new_tokens=max_new_tokens,
            include_deterministic=include_deterministic,
            include_neural=include_neural,
        )

    def _reconstruct_function(
        self,
        bundle: Any,
        record: dict[str, Any],
        function: dict[str, Any],
        language: str = "c",
        reference_source: str | None = None,
        max_new_tokens: int = 96,
        include_deterministic: bool = True,
        include_neural: bool = True,
    ) -> dict[str, Any]:
        candidates = []
        evidence = {
            "binary_sha256": record.get("program", {}).get("sha256"),
            "function_id": function.get("function_id"),
            "function_ranges": function.get("ranges", []),
            "deterministic_evidence": [entry for entry in function.get("evidence", [])],
            "names_used_as_model_input": False,
            "source_used_as_model_input": False,
        }
        reference = reference_source
        if reference is None:
            for source in bundle.source_paths if hasattr(bundle, "source_paths") else []:
                reference = Path(source).read_text(encoding="utf-8", errors="replace")
                break
        if include_neural and self.model is not None and self.vocab is not None:
            sample = encode_binary_function(record, function["function_id"], self.vocab, self.config)
            if sample is not None:
                batch = collate_samples([sample], self.config)
                source, generation = self._model_candidate(batch, language, max_new_tokens)
                verification = self._verify(language, source)
                statuses = _empty_statuses()
                statuses["syntax_valid"] = verification.get("syntax", {}).get("status", "unknown")
                statuses["compilation"] = verification.get("compilation", {}).get("status", "not_attempted")
                unresolved = []
                if verification.get("syntax", {}).get("status") != "passed":
                    unresolved.append("generated source did not parse/compile; no semantic claim is made")
                candidates.append(
                    _candidate(
                        language,
                        source,
                        "neural_unified_model",
                        statuses,
                        evidence,
                        unresolved,
                        generation.get("confidence"),
                        verification,
                        reference,
                        extra=generation,
                    )
                )
        if include_deterministic and language == "c":
            try:
                function_ids = [item.function_id for item in getattr(bundle, "functions", [])][:64]
                if function["function_id"] not in function_ids:
                    function_ids.append(function["function_id"])
                deterministic = reconstruct_c(bundle, function_ids, strategy="semantic_ir")
            except Exception as exc:
                deterministic = {"functions": [], "files": {}, "error": f"{type(exc).__name__}: {exc}"}
            function_result = next((item for item in deterministic.get("functions", []) if item.get("function_id") == function["function_id"]), None)
            source = ""
            if function_result:
                path = f"src/{function_result.get('generated_name')}.c"
                source = deterministic.get("files", {}).get(path, "")
            verification = self._verify("c", source, include_files=deterministic.get("files", {}))
            statuses = _empty_statuses()
            statuses["syntax_valid"] = verification.get("syntax", {}).get("status", "unknown")
            statuses["compilation"] = verification.get("compilation", {}).get("status", "not_attempted")
            unresolved = []
            if deterministic.get("error"):
                unresolved.append(f"deterministic reconstruction failed: {deterministic['error']}")
            if function_result:
                if function_result.get("classification") == "unresolved":
                    unresolved.append(function_result.get("reason") or "deterministic reconstruction is unresolved")
                phase_stats = function_result.get("semantic_stats") or {}
                unresolved.extend(str(item) for item in (phase_stats.get("unsupported", []) or []))
            candidates.append(
                _candidate(
                    "c",
                    source,
                    "ghidra_decompiler" if function_result and function_result.get("classification") == "ghidra_decompiler_reconstruction" else "deterministic_semantic_ir",
                    statuses,
                    evidence,
                    unresolved,
                    function_result.get("confidence") if function_result else None,
                    verification,
                    reference,
                    extra={"classification": function_result.get("classification") if function_result else "unresolved", "deterministic": True},
                )
            )
        if include_deterministic and language in {"java", "jni"}:
            try:
                deterministic = reconstruct_java(bundle, [function["function_id"]], project_name="nexus_java_reconstruction")
            except Exception as exc:
                deterministic = {"files": {}, "jni": {}, "classification": "failed", "error": f"{type(exc).__name__}: {exc}"}
            source = ""
            for path, content in deterministic.get("files", {}).items():
                if path.endswith(".java"):
                    source = content
                    break
            verification = self._verify("java", source, "ReconstructedModule")
            statuses = _empty_statuses()
            statuses["syntax_valid"] = verification.get("syntax", {}).get("status", "unknown")
            statuses["compilation"] = verification.get("compilation", {}).get("status", "not_attempted")
            candidates.append(
                _candidate(
                    "java",
                    source,
                    "deterministic_jni",
                    statuses,
                    evidence,
                    ([f"deterministic Java reconstruction failed: {deterministic['error']}"] if deterministic.get("error") else [])
                    if verification.get("compilation", {}).get("status") != "passed"
                    else [],
                    0.5,
                    verification,
                    reference,
                    extra={"jni": deterministic.get("jni"), "classification": deterministic.get("classification")},
                )
            )
        if len(candidates) >= 2:
            neural = next((item for item in candidates if item.get("method") == "neural_unified_model"), None)
            deterministic = next((item for item in candidates if (item.get("additional") or {}).get("deterministic")), None)
            selected = None
            rule = "no candidate compiled"
            if deterministic and deterministic.get("statuses", {}).get("compilation") == "passed":
                selected = deterministic
                rule = "deterministic candidate compiles; selected over uncompiled neural output"
            elif neural and neural.get("statuses", {}).get("compilation") == "passed":
                selected = neural
                rule = "neural candidate compiles; deterministic candidate did not"
            if selected is not None:
                hybrid = dict(selected)
                hybrid["method"] = "constrained_hybrid"
                hybrid["additional"] = {
                    **(selected.get("additional") or {}),
                    "selected_method": selected.get("method"),
                    "selection_rule": rule,
                }
                hybrid["limitations"] = list(selected.get("limitations", [])) + [
                    "Hybrid selection chooses a compiling candidate; it does not establish behavioral correctness."
                ]
                candidates.append(hybrid)
        return {
            "schema_version": "0.1.0",
            "status": "completed" if candidates else "no_candidate",
            "function_id": function.get("function_id"),
            "binary_sha256": record.get("program", {}).get("sha256"),
            "candidates": candidates,
            "model": {
                "checkpoint": self.checkpoint_path,
                "checkpoint_sha256": self.checkpoint_sha256,
                "parameter_count": self.model.count_parameters() if self.model is not None else None,
            },
            "limitations": [
                "Neural candidates are not automatically correct.",
                "Deterministic candidates may be explicit unresolved stubs.",
                "No candidate was dynamically executed.",
                "Universal equivalence is never claimed.",
            ],
        }

    def reconstruct_project(
        self,
        binary_path: str | Path,
        language: str = "c",
        max_functions: int = 12,
        max_new_tokens: int = 96,
        include_neural: bool = True,
    ) -> dict[str, Any]:
        bundle = analyze_binary(binary_path)
        record = build_program_record(bundle)
        functions = [
            function
            for function in record.get("functions", [])
            if function.get("confidence", 0) >= 0.5 and not str(function.get("name", "")).startswith("_")
        ][:max_functions]
        results = [
            self._reconstruct_function(
                bundle,
                record,
                function,
                language=language,
                max_new_tokens=max_new_tokens,
                include_deterministic=True,
                include_neural=include_neural,
            )
            for function in functions
        ]
        statuses: dict[str, int] = {}
        unresolved: list[str] = []
        for result in results:
            for candidate in result.get("candidates", []):
                status = candidate.get("statuses", {}).get("compilation", "unknown")
                statuses[status] = statuses.get(status, 0) + 1
                unresolved.extend(str(item) for item in candidate.get("unresolved_operations", []))
        return {
            "schema_version": "0.1.0",
            "status": "completed",
            "language": language,
            "binary_sha256": record.get("program", {}).get("sha256"),
            "functions_considered": len(functions),
            "functions_with_candidates": sum(1 for result in results if result.get("candidates")),
            "candidate_status_counts": statuses,
            "unresolved_operations": sorted(set(unresolved)),
            "results": results,
            "verification_scope": "compilation and syntax only; no execution or behavioral equivalence",
            "limitations": [
                "Project reconstruction is assembled from per-function candidates.",
                "Unresolved functions and operations remain explicit.",
                "No universal-equivalence claim is made.",
            ],
        }

    def predict_jni_signature(self, binary_path: str | Path, function_id: str) -> dict[str, Any]:
        if self.model is None or self.vocab is None:
            return {"status": "unavailable", "reason": "no foundation checkpoint"}
        bundle = analyze_binary(binary_path)
        record = build_program_record(bundle)
        sample = encode_binary_function(record, function_id, self.vocab, self.config)
        if sample is None:
            return {"status": "not_found", "function_id": function_id}
        batch = collate_samples([sample], self.config)
        with torch.no_grad():
            outputs = self.model(batch)
        logits = outputs["jni_logits"][0]
        predictions = logits.argmax(dim=-1).tolist()
        deterministic = (record.get("jni") or {}).get("entries", [])
        return {
            "status": "predicted_unverified",
            "function_id": function_id,
            "predicted_return_type": JNI_TYPES[min(max(int(predictions[0]), 0), len(JNI_TYPES) - 1)],
            "predicted_argument_types": [
                JNI_ARGUMENT_TYPES[min(max(int(value), 0), len(JNI_ARGUMENT_TYPES) - 1)] for value in predictions[1:]
            ],
            "confidence": float(outputs["confidence"][0].detach()),
            "deterministic_entries": deterministic,
            "note": "Neural JNI signature prediction is not a verified JNI mapping.",
        }


def reconstruction_report(candidate: dict[str, Any]) -> str:
    return json.dumps(candidate, indent=2, default=str)
