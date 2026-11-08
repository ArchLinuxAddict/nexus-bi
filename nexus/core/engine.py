import json
import uuid
from pathlib import Path
from typing import Any

from nexus.core.capabilities import as_dict as capability_dict
from nexus.core.config import NexusPaths, default_paths
from nexus.core.errors import NotFoundError
from nexus.program.bundle import analyze_binary
from nexus.program.record import build_program_record
from nexus.reconstruction.c.generator import reconstruct_c
from nexus.reconstruction.java.generator import reconstruct_java
from nexus.storage.store import Storage
from nexus.verification.compiler import verify_reconstruction
from nexus.verification.static import (
    verify_program_record,
    verify_reconstruction_structure,
)


class Engine:
    def __init__(self, paths: NexusPaths | None = None, storage: Storage | None = None):
        self.paths = (paths or default_paths()).ensure()
        self.storage = storage or Storage(self.paths)

    def analyze_path(self, path: str | Path, persist: bool = True, include_record: bool = False) -> dict[str, Any]:
        binary = self.storage.add_binary(path) if persist else {"id": None, "sha256": None, "path": str(path), "size": Path(path).stat().st_size}
        bundle = analyze_binary(binary["path"])
        record = build_program_record(bundle)
        validation = verify_program_record(record)
        record_dir = self.paths.artifacts / "records"
        record_dir.mkdir(parents=True, exist_ok=True)
        record_path = record_dir / f"{bundle.elf.sha256}.json"
        record_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        summary = {
            **bundle.summary(),
            "binary_id": binary["id"],
            "validation": validation,
            "record_path": str(record_path),
            "function_count": len(bundle.functions),
            "instruction_count": len(bundle.disassembly.instructions) if bundle.disassembly else 0,
            "security_findings_count": len(bundle.security_findings),
            "jni": bundle.jni_report.get("is_jni", False),
            "top_functions": [
                {"function_id": function.function_id, "name": function.name, "entry_va": f"0x{function.entry:x}", "confidence": function.confidence}
                for function in sorted(bundle.functions, key=lambda item: (-len(item.instructions), item.entry))[:12]
            ],
        }
        if persist:
            self.storage.update_binary_machine(binary["id"], bundle.elf.header.machine_name)
            analysis_id = self.storage.create_analysis(binary["id"], summary, record_path, status="completed" if validation["status"] == "passed" else "completed_with_validation_errors")
            self.storage.add_findings(binary["id"], bundle.security_findings)
            summary["analysis_id"] = analysis_id
        result = {"summary": summary, "record_path": str(record_path), "bundle": bundle}
        if include_record:
            result["record"] = record
        return result

    def load_record(self, analysis_id: str) -> dict[str, Any]:
        analysis = self.storage.get_analysis(analysis_id)
        path = analysis.get("result_path")
        if not path or not Path(path).exists():
            raise NotFoundError(f"analysis record not found for {analysis_id}")
        return json.loads(Path(path).read_text(encoding="utf-8"))

    def get_analysis(self, analysis_id: str, include_record: bool = False) -> dict[str, Any]:
        analysis = self.storage.get_analysis(analysis_id)
        analysis["analysis_id"] = analysis.get("id")
        if include_record:
            analysis["record"] = self.load_record(analysis_id)
        return analysis

    def functions(self, analysis_id: str) -> list[dict[str, Any]]:
        record = self.load_record(analysis_id)
        return record.get("functions", [])

    def function(self, analysis_id: str, function_id: str) -> dict[str, Any]:
        record = self.load_record(analysis_id)
        for function in record.get("functions", []):
            if function.get("function_id") == function_id:
                return function
        raise NotFoundError(f"function not found: {function_id}")

    def cfg(self, analysis_id: str, function_id: str | None = None) -> dict[str, Any]:
        record = self.load_record(analysis_id)
        cfg = record.get("cfg", {})
        if function_id is None:
            return cfg
        blocks = {block["block_id"] for function in record.get("functions", []) if function.get("function_id") == function_id for block in function.get("blocks", [])}
        return {
            "nodes": [node for node in cfg.get("nodes", []) if node.get("node_id") in blocks or node.get("function_id") == function_id],
            "edges": [edge for edge in cfg.get("edges", []) if edge.get("source") in blocks or edge.get("target") in blocks],
            "confidence": cfg.get("confidence"),
            "limitations": cfg.get("limitations", []),
        }

    def callgraph(self, analysis_id: str) -> dict[str, Any]:
        return self.load_record(analysis_id).get("callgraph", {})

    def disassembly(self, analysis_id: str, limit: int = 2000) -> list[dict[str, Any]]:
        return self.load_record(analysis_id).get("instructions", [])[:limit]

    def semantic_ir(self, analysis_id: str) -> dict[str, Any]:
        from nexus.program.ir import build_program_ir
        analysis = self.storage.get_analysis(analysis_id)
        binary = self.storage.get_binary(analysis["binary_id"])
        return build_program_ir(analyze_binary(binary["path"]))

    def dataflow(self, analysis_id: str, function_id: str | None = None) -> dict[str, Any]:
        dataflow = self.load_record(analysis_id).get("dataflow", {})
        if function_id:
            return dataflow.get(function_id, {})
        return dataflow

    def types(self, analysis_id: str) -> list[dict[str, Any]]:
        return self.load_record(analysis_id).get("types", [])

    def security(self, analysis_id: str) -> list[dict[str, Any]]:
        return self.load_record(analysis_id).get("security_findings", [])

    def jni(self, analysis_id: str) -> dict[str, Any]:
        return self.load_record(analysis_id).get("jni", {})

    def reconstruct(self, analysis_id: str, language: str, function_ids: list[str] | None = None, out_dir: str | Path | None = None, persist: bool = True) -> dict[str, Any]:
        analysis = self.storage.get_analysis(analysis_id)
        binary = self.storage.get_binary(analysis["binary_id"])
        bundle = analyze_binary(binary["path"])
        root = Path(out_dir) if out_dir else self.paths.artifacts / "reconstructions" / f"rec_{uuid.uuid4().hex[:12]}"
        root.mkdir(parents=True, exist_ok=True)
        project_name = f"reconstructed_{bundle.elf.sha256[:8]}"
        if language == "c":
            result = reconstruct_c(bundle, function_ids=function_ids, out_dir=root, project_name=project_name)
        elif language == "java":
            result = reconstruct_java(bundle, function_ids=function_ids, out_dir=root, project_name=project_name)
        else:
            raise ValueError(f"unsupported reconstruction language: {language}")
        structure = verify_reconstruction_structure(result, root)
        result["structure_verification"] = structure
        result["output_dir"] = str(root)
        result["analysis_id"] = analysis_id
        if persist:
            reconstruction_id = self.storage.create_reconstruction(analysis_id, language, result, root, status="completed" if structure["status"] == "passed" else "needs_review")
            result["reconstruction_id"] = reconstruction_id
        return result

    def get_reconstruction(self, reconstruction_id: str) -> dict[str, Any]:
        return self.storage.get_reconstruction(reconstruction_id)

    def verify_reconstruction(self, reconstruction_id: str) -> dict[str, Any]:
        reconstruction = self.storage.get_reconstruction(reconstruction_id)
        summary = reconstruction["summary"]
        output_dir = reconstruction.get("output_dir")
        result = verify_reconstruction(summary, output_dir)
        verification_id = self.storage.create_verification(reconstruction_id, result)
        result["verification_id"] = verification_id
        return result

    def investigate(self, analysis_id: str, objective: str, max_steps: int = 8) -> dict[str, Any]:
        analysis = self.storage.get_analysis(analysis_id)
        binary = self.storage.get_binary(analysis["binary_id"])
        from nexus.reasoning.investigation import investigate
        report = investigate(binary["path"], objective, max_steps=max_steps)
        report["analysis_id"] = analysis_id
        self.storage.add_investigation(binary["id"], report)
        return report

    def get_investigation(self, investigation_id: str) -> dict[str, Any]:
        return self.storage.get_investigation(investigation_id)

    def investigations(self, binary_id: str | None = None) -> list[dict[str, Any]]:
        return self.storage.list_investigations(binary_id=binary_id)

    def capabilities(self) -> dict[str, Any]:
        return capability_dict()

    def datasets(self) -> list[dict[str, Any]]:
        return self.storage.list_datasets()

    def models(self) -> list[dict[str, Any]]:
        return self.storage.list_models()

    def experiments(self) -> list[dict[str, Any]]:
        return self.storage.list_experiments()

    def read_source_file(self, reconstruction_id: str, relative_path: str) -> str:
        reconstruction = self.storage.get_reconstruction(reconstruction_id)
        root = Path(reconstruction.get("output_dir") or "")
        target = (root / relative_path).resolve()
        if not target.is_relative_to(root.resolve()):
            raise NotFoundError("source path escapes reconstruction directory")
        if not target.exists() or not target.is_file():
            raise NotFoundError(f"source file not found: {relative_path}")
        return target.read_text(encoding="utf-8", errors="replace")
