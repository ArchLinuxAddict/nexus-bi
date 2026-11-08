import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, File, Header, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from nexus.core.capabilities import as_dict as capability_dict
from nexus.core.engine import Engine
from nexus.core.errors import (
    CapabilityUnavailable,
    NexusError,
    NotFoundError,
    ValidationError,
)
from nexus.core.hardware import hardware_report

MAX_UPLOAD_BYTES = 100 * 1024 * 1024

FOUNDATION_ROOT = Path(__file__).resolve().parents[2] / "artifacts" / "final_training_package"
REALWORLD_RUN_ROOT = Path(__file__).resolve().parents[2] / "artifacts" / "realworld_so_runs"


def _foundation_root() -> Path:
    return FOUNDATION_ROOT.resolve()


def _foundation_run_roots() -> list[Path]:
    roots = []
    canonical = FOUNDATION_ROOT.resolve() / "runs"
    if canonical.exists():
        roots.append(canonical)
    if REALWORLD_RUN_ROOT.exists():
        roots.append(REALWORLD_RUN_ROOT.resolve())
    return roots




def _java_target(binary: dict[str, Any], upload_root: Path) -> tuple[Path, Path | None]:
    target = Path(binary["path"])
    suffix = {"JAR": ".jar", "APK": ".apk", "DEX": ".dex", "CLASS": ".class"}.get(str(binary.get("format") or "").upper())
    if not suffix or target.suffix.lower() == suffix:
        return target, None
    upload_root.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(dir=upload_root, prefix="java_"))
    link = temporary / f"artifact{suffix}"
    os.symlink(target, link)
    return link, temporary


def _foundation_safe_path(relative: str) -> Path:
    candidate = Path(relative)
    target = candidate.resolve() if candidate.is_absolute() else (_foundation_root() / candidate).resolve()
    for root in (_foundation_root(), REALWORLD_RUN_ROOT.resolve()):
        try:
            target.relative_to(root)
            return target
        except ValueError:
            continue
    raise ValidationError("foundation path escapes the approved artifact roots")


def _foundation_checkpoint_index() -> list[dict[str, Any]]:
    from nexus.foundation.checkpoint import inspect_checkpoint

    root = _foundation_root()
    candidates = []
    paths = list(root.glob("checkpoints/*.pt")) + list((root / "runs").glob("*/*.pt")) + list(REALWORLD_RUN_ROOT.glob("*/*.pt"))
    for path in sorted(paths):
        try:
            info = inspect_checkpoint(path)
        except Exception as exc:
            info = {"path": str(path), "error": str(exc)}
        try:
            relative_path = str(path.relative_to(root))
        except ValueError:
            relative_path = str(path)
        candidates.append(
            {
                "relative_path": relative_path,
                "path": str(path),
                "sha256": info.get("sha256"),
                "size": info.get("size"),
                "step": (info.get("metadata") or {}).get("step"),
                "status": (info.get("metadata") or {}).get("status"),
                "parameter_count": (info.get("metadata") or {}).get("parameter_count"),
                "initialization": (info.get("metadata") or {}).get("initialization"),
            }
        )
    return candidates


def _foundation_runs() -> list[dict[str, Any]]:
    runs = []
    for root in _foundation_run_roots():
        for run_dir in sorted([path for path in root.iterdir() if path.is_dir()], key=lambda path: path.stat().st_mtime, reverse=True):
            manifest_path = run_dir / "manifest.json"
            if not manifest_path.exists():
                continue
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue
            validation = manifest.get("validation")
            if not validation:
                validation_path = run_dir / "validation.json"
                if validation_path.exists():
                    try:
                        validation = json.loads(validation_path.read_text(encoding="utf-8"))
                    except json.JSONDecodeError:
                        validation = None
            best_score = manifest.get("best_validation_score")
            if best_score is None and validation:
                best_score = validation.get("weighted_validation_score")
            runs.append(
                {
                    "run_name": run_dir.name,
                    "path": str(run_dir),
                    "status": manifest.get("status"),
                    "created_at": manifest.get("created_at"),
                    "completed_at": manifest.get("completed_at"),
                    "parameter_count": manifest.get("parameter_count"),
                    "progress": manifest.get("progress"),
                    "best_validation_score": best_score,
                    "best_step": manifest.get("best_step"),
                    "mission": manifest.get("mission"),
                    "checkpoint": manifest.get("checkpoint"),
                    "validation": validation,
                    "resources": manifest.get("resources"),
                    "limitations": manifest.get("limitations", []),
                }
            )
    return sorted(runs, key=lambda item: Path(item["path"]).stat().st_mtime, reverse=True)



class AnalysisRequest(BaseModel):
    binary_id: str = Field(min_length=1)


class InvestigationRequest(BaseModel):
    analysis_id: str = Field(min_length=1)
    objective: str = Field(min_length=1, max_length=4000)
    max_steps: int = Field(default=8, ge=1, le=32)


class ReconstructionRequest(BaseModel):
    analysis_id: str = Field(min_length=1)
    language: str = Field(pattern="^(c|java)$")
    function_ids: list[str] | None = None

class FoundationAnalysisRequest(BaseModel):
    binary_id: str = Field(min_length=1)
    checkpoint: str = Field(min_length=1)
    max_functions: int = Field(default=8, ge=1, le=64)


class JavaAnalyzeRequest(BaseModel):
    path: str = Field(min_length=1)
    full: bool = False


class JavaReconstructRequest(BaseModel):
    path: str = Field(min_length=1)
    out_dir: str | None = None
    project_name: str = Field(default="nexus_java_reconstruction", min_length=1, max_length=128)
    max_classes: int = Field(default=2000, ge=1, le=50000)
    verify: bool = True


class ResidualInferRequest(BaseModel):
    tool_text: str = Field(min_length=1, max_length=200000)
    target_text: str = Field(default="", max_length=200000)
    language: str = Field(default="java", pattern="^(c|java|jni)$")
    tool: str = Field(default="jadx", max_length=32)
    candidate_text: str = Field(default="", max_length=20000)
    jni_method: str = Field(default="", max_length=256)
    jni_descriptor: str = Field(default="", max_length=2048)
    jni_instance: bool = False
    tool_status: str = Field(default="completed", max_length=32)
    compile_status: str = Field(default="unknown", max_length=32)
    warning_count: int = Field(default=0, ge=0, le=10000)
    cached: bool = False
    max_new_tokens: int = Field(default=128, ge=8, le=512)


class JavaBinaryRequest(BaseModel):
    binary_id: str = Field(min_length=1)
    project_name: str = Field(default="nexus_java_reconstruction", min_length=1, max_length=128)
    max_classes: int = Field(default=2000, ge=1, le=50000)
    verify: bool = True
    full: bool = False


class JNICorrelationRequest(BaseModel):
    java_path: str = Field(min_length=1)
    native_path: str = Field(min_length=1)


class ComprehensiveReportRequest(BaseModel):
    binary_path: str = Field(min_length=1)
    java_path: str | None = None
    native_path: str | None = None
    out_dir: str | None = None
    max_functions: int = Field(default=40, ge=1, le=5000)


class FoundationReconstructionRequest(BaseModel):
    binary_id: str = Field(min_length=1)
    function_id: str = Field(min_length=1)
    language: str = Field(default="c", pattern="^(c|java|jni)$")
    checkpoint: str | None = None
    reference_source: str | None = None


def create_app(engine: Engine | None = None, api_key: str | None = None) -> FastAPI:
    app = FastAPI(title="NEXUS-BI", version="0.7.0", description="Integrated binary-intelligence platform with multi-architecture and Java/JNI static analysis")
    engine = engine or Engine()
    app.state.engine = engine
    dashboard_dir = Path(__file__).resolve().parents[1] / "dashboard"

    def require_key(x_api_key: str | None = Header(default=None)) -> None:
        expected = api_key if api_key is not None else os.environ.get("NEXUS_API_KEY")
        if expected and x_api_key != expected:
            raise HTTPException(status_code=401, detail="invalid or missing API key")

    @app.exception_handler(NexusError)
    async def nexus_error_handler(_request, exc: NexusError):
        if isinstance(exc, NotFoundError):
            status = 404
        elif isinstance(exc, CapabilityUnavailable):
            status = 503
        elif isinstance(exc, ValidationError):
            status = 422
        else:
            status = 400
        return JSONResponse(status_code=status, content={"error": exc.to_dict()})

    @app.get("/healthz")
    def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/readyz")
    def readyz() -> dict[str, Any]:
        return {"status": "ready", "database": str(engine.storage.db_path), "capabilities": len(capability_dict())}

    @app.get("/v1/version")
    def version() -> dict[str, str]:
        return {"name": "NEXUS-BI", "version": "0.7.0", "model_status": "one unified foundation model; trained compact checkpoint available"}

    @app.get("/v1/mission")
    def mission() -> dict[str, Any]:
        from nexus.mission.policy import audit_registry, load_policy, load_registry
        return {
            "audit": audit_registry(),
            "policy": load_policy(),
            "datasets": load_registry()["datasets"],
        }

    @app.get("/v1/tools")
    def tools_matrix() -> dict[str, Any]:
        from nexus.toolchain.registry import ToolRegistry

        return ToolRegistry().matrix()

    @app.get("/v1/capabilities")
    def capabilities() -> dict[str, Any]:
        return capability_dict()

    @app.get("/v1/hardware")
    def hardware() -> dict[str, Any]:
        return hardware_report()

    @app.post("/v1/binaries", dependencies=[Depends(require_key)])
    async def upload_binary(file: UploadFile = File(...)) -> dict[str, Any]:
        upload_dir = engine.paths.artifacts / "uploads"
        upload_dir.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_path = tempfile.mkstemp(dir=upload_dir, prefix="upload_")
        os.close(descriptor)
        size = 0
        try:
            with open(temporary_path, "wb") as handle:
                while True:
                    chunk = await file.read(1024 * 1024)
                    if not chunk:
                        break
                    size += len(chunk)
                    if size > MAX_UPLOAD_BYTES:
                        raise HTTPException(status_code=413, detail=f"upload exceeds {MAX_UPLOAD_BYTES} bytes")
                    handle.write(chunk)
            stored = engine.storage.add_binary(temporary_path, original_name=file.filename)
            return {key: stored[key] for key in ("id", "sha256", "size", "format")}
        finally:
            try:
                os.unlink(temporary_path)
            except OSError:
                pass

    @app.get("/v1/binaries", dependencies=[Depends(require_key)])
    def list_binaries(limit: int = 100) -> list[dict[str, Any]]:
        return [
            {key: item[key] for key in ("id", "sha256", "size", "format", "machine", "created_at")}
            for item in engine.storage.list_binaries(limit=min(max(limit, 1), 500))
        ]

    @app.get("/v1/binaries/{binary_id}", dependencies=[Depends(require_key)])
    def get_binary(binary_id: str) -> dict[str, Any]:
        item = engine.storage.get_binary(binary_id)
        return {key: item[key] for key in ("id", "sha256", "size", "format", "machine", "created_at")}

    @app.post("/v1/analyses", dependencies=[Depends(require_key)])
    def create_analysis(request: AnalysisRequest) -> dict[str, Any]:
        binary = engine.storage.get_binary(request.binary_id)
        job_id = engine.storage.create_job("analysis", {"binary_id": request.binary_id})
        engine.storage.update_job(job_id, "running")
        try:
            result = engine.analyze_path(binary["path"])
        except Exception as exc:
            engine.storage.update_job(job_id, "failed", error=str(exc))
            raise
        result["summary"].pop("bundle", None)
        result["summary"].pop("path", None)
        result["summary"].pop("record_path", None)
        result["summary"]["job_id"] = job_id
        engine.storage.update_job(job_id, "completed", result={"analysis_id": result["summary"].get("analysis_id")})
        return result["summary"]

    @app.get("/v1/analyses", dependencies=[Depends(require_key)])
    def list_analyses(binary_id: str | None = None, limit: int = 100) -> list[dict[str, Any]]:
        results = engine.storage.list_analyses(binary_id=binary_id, limit=min(max(limit, 1), 500))
        for item in results:
            item.pop("result_path", None)
            item["summary"].pop("path", None)
            item["summary"].pop("record_path", None)
        return results

    @app.get("/v1/analyses/{analysis_id}", dependencies=[Depends(require_key)])
    def get_analysis(analysis_id: str) -> dict[str, Any]:
        result = engine.get_analysis(analysis_id)
        result.pop("result_path", None)
        result["summary"].pop("path", None)
        result["summary"].pop("record_path", None)
        return result

    @app.get("/v1/analyses/{analysis_id}/functions", dependencies=[Depends(require_key)])
    def functions(analysis_id: str) -> list[dict[str, Any]]:
        return engine.functions(analysis_id)

    @app.get("/v1/analyses/{analysis_id}/functions/{function_id}", dependencies=[Depends(require_key)])
    def function(analysis_id: str, function_id: str) -> dict[str, Any]:
        return engine.function(analysis_id, function_id)

    @app.get("/v1/analyses/{analysis_id}/cfg", dependencies=[Depends(require_key)])
    def cfg(analysis_id: str, function_id: str | None = None) -> dict[str, Any]:
        return engine.cfg(analysis_id, function_id)

    @app.get("/v1/analyses/{analysis_id}/callgraph", dependencies=[Depends(require_key)])
    def callgraph(analysis_id: str) -> dict[str, Any]:
        return engine.callgraph(analysis_id)

    @app.get("/v1/analyses/{analysis_id}/disassembly", dependencies=[Depends(require_key)])
    def disassembly(analysis_id: str, limit: int = 2000) -> list[dict[str, Any]]:
        return engine.disassembly(analysis_id, limit=min(max(limit, 1), 10000))

    @app.get("/v1/analyses/{analysis_id}/program-record", dependencies=[Depends(require_key)])
    def program_record(analysis_id: str) -> dict[str, Any]:
        return engine.load_record(analysis_id)

    @app.get("/v1/analyses/{analysis_id}/ir", dependencies=[Depends(require_key)])
    def semantic_ir(analysis_id: str) -> dict[str, Any]:
        return engine.semantic_ir(analysis_id)

    @app.get("/v1/analyses/{analysis_id}/dataflow", dependencies=[Depends(require_key)])
    def dataflow(analysis_id: str, function_id: str | None = None) -> dict[str, Any]:
        return engine.dataflow(analysis_id, function_id)

    @app.get("/v1/analyses/{analysis_id}/types", dependencies=[Depends(require_key)])
    def types(analysis_id: str) -> list[dict[str, Any]]:
        return engine.types(analysis_id)

    @app.get("/v1/analyses/{analysis_id}/security", dependencies=[Depends(require_key)])
    def security(analysis_id: str) -> list[dict[str, Any]]:
        return engine.security(analysis_id)

    @app.get("/v1/analyses/{analysis_id}/jni", dependencies=[Depends(require_key)])
    def jni(analysis_id: str) -> dict[str, Any]:
        return engine.jni(analysis_id)

    @app.post("/v1/investigations", dependencies=[Depends(require_key)])
    def create_investigation(request: InvestigationRequest) -> dict[str, Any]:
        return engine.investigate(request.analysis_id, request.objective, request.max_steps)

    @app.get("/v1/investigations", dependencies=[Depends(require_key)])
    def list_investigations(limit: int = 50) -> list[dict[str, Any]]:
        return engine.investigations()[:min(max(limit, 1), 200)]

    @app.get("/v1/investigations/{investigation_id}", dependencies=[Depends(require_key)])
    def get_investigation(investigation_id: str) -> dict[str, Any]:
        return engine.get_investigation(investigation_id)

    @app.post("/v1/reconstructions", dependencies=[Depends(require_key)])
    def create_reconstruction(request: ReconstructionRequest) -> dict[str, Any]:
        job_id = engine.storage.create_job("reconstruction", request.model_dump())
        engine.storage.update_job(job_id, "running")
        try:
            result = engine.reconstruct(request.analysis_id, request.language, request.function_ids)
        except Exception as exc:
            engine.storage.update_job(job_id, "failed", error=str(exc))
            raise
        engine.storage.update_job(job_id, "completed", result={"reconstruction_id": result.get("reconstruction_id")})
        result.pop("output_dir", None)
        result["job_id"] = job_id
        return result

    @app.get("/v1/reconstructions/{reconstruction_id}", dependencies=[Depends(require_key)])
    def get_reconstruction(reconstruction_id: str) -> dict[str, Any]:
        result = engine.get_reconstruction(reconstruction_id)
        result.pop("output_dir", None)
        return {**result, "file_list": sorted(result.get("summary", {}).get("files", {}).keys())}

    @app.get("/v1/reconstructions/{reconstruction_id}/files/{relative_path:path}", dependencies=[Depends(require_key)])
    def get_reconstruction_file(reconstruction_id: str, relative_path: str) -> PlainTextResponse:
        return PlainTextResponse(engine.read_source_file(reconstruction_id, relative_path))

    @app.post("/v1/reconstructions/{reconstruction_id}/verify", dependencies=[Depends(require_key)])
    def verify(reconstruction_id: str) -> dict[str, Any]:
        return engine.verify_reconstruction(reconstruction_id)

    @app.get("/v1/security/findings", dependencies=[Depends(require_key)])
    def findings(binary_id: str | None = None, limit: int = 200) -> list[dict[str, Any]]:
        return engine.storage.list_findings(binary_id=binary_id, limit=min(max(limit, 1), 1000))

    @app.get("/v1/datasets", dependencies=[Depends(require_key)])
    def datasets() -> list[dict[str, Any]]:
        return engine.datasets()

    @app.get("/v1/model", dependencies=[Depends(require_key)])
    def model_info() -> dict[str, Any]:
        models = engine.models()
        return {"status": "tool-first rebuild; no trained residual checkpoint", "checkpoint": None, "models": models, "policy": "Ghidra/JADX own reconstruction; the residual model is not trained yet"}

    @app.post("/v1/model/smoke-test", dependencies=[Depends(require_key)])
    def model_smoke_test(steps: int = 2) -> dict[str, Any]:
        from nexus.foundation.smoke import smoke_test
        result = smoke_test(steps=max(1, min(steps, 5)))
        if Path(result["checkpoint"]["path"]).exists():
            model_id = engine.storage.add_model(result["checkpoint"]["path"], result["checkpoint"]["metadata"])
            result["model_id"] = model_id
        return result

    @app.get("/v1/jobs", dependencies=[Depends(require_key)])
    def list_jobs(limit: int = 100) -> list[dict[str, Any]]:
        return engine.storage.list_jobs(limit=min(max(limit, 1), 500))

    @app.get("/v1/jobs/{job_id}", dependencies=[Depends(require_key)])
    def get_job(job_id: str) -> dict[str, Any]:
        return engine.storage.get_job(job_id)

    @app.get("/v1/tools", dependencies=[Depends(require_key)])
    def tools() -> list[dict[str, Any]]:
        from nexus.reasoning.tools import builtin_registry
        return builtin_registry().list()

    @app.get("/v1/experiments", dependencies=[Depends(require_key)])
    def experiments() -> list[dict[str, Any]]:
        return engine.experiments()



    @app.get("/v1/residual/status")
    def residual_status() -> dict[str, Any]:
        from nexus.foundation.residual_runtime import get_runtime

        return get_runtime().status()

    @app.post("/v1/residual/infer", dependencies=[Depends(require_key)])
    def residual_infer(request: ResidualInferRequest) -> dict[str, Any]:
        from nexus.foundation.residual_runtime import get_runtime

        return get_runtime().infer(**request.model_dump())

    @app.post("/v1/residual/evaluate", dependencies=[Depends(require_key)])
    def residual_evaluate() -> dict[str, Any]:
        from nexus.foundation.residual_runtime import get_runtime

        return get_runtime().evaluate()

    @app.get("/v1/foundation/model", dependencies=[Depends(require_key)])
    def foundation_model() -> dict[str, Any]:
        from nexus.foundation.config import profile_report
        from nexus.foundation.objectives import objective_registry

        runs = _foundation_runs()
        latest = runs[0] if runs else None
        return {
            "status": "tool-first rebuild; no residual checkpoint loaded",
            "architecture_version": "nexus-bi/foundation-1",
            "profiles": profile_report(),
            "modalities": {
                "bytes": "implemented",
                "instructions": "implemented",
                "semantic_ir": "implemented",
                "graph_cfg": "implemented with typed graph attention",
                "hierarchy": "instruction -> block -> function -> module -> project",
                "c_decoder": "implemented",
                "java_decoder": "reserved for JADX/CFR residual tasks; no checkpoint loaded",
                "jni_head": "reserved for JNI residual tasks; no checkpoint loaded",
            },
            "objectives": objective_registry(),
            "training_runs": runs[:10],
            "latest_run": latest,
            "checkpoints": _foundation_checkpoint_index(),
            "honesty": {
                "foundation_model_trained_at_scale": False,
                "residual_model_trained": False,
                "old_controlled_checkpoints_removed": True,
                "pretrained_weights_loaded": False,
            },
        }

    @app.get("/v1/foundation/objectives", dependencies=[Depends(require_key)])
    def foundation_objectives() -> dict[str, Any]:
        from nexus.foundation.objectives import objective_registry
        return objective_registry()

    @app.get("/v1/foundation/checkpoints", dependencies=[Depends(require_key)])
    def foundation_checkpoints() -> list[dict[str, Any]]:
        return _foundation_checkpoint_index()

    @app.get("/v1/foundation/checkpoints/{relative_path:path}", dependencies=[Depends(require_key)])
    def foundation_checkpoint(relative_path: str) -> dict[str, Any]:
        from nexus.foundation.checkpoint import inspect_checkpoint
        path = _foundation_safe_path(relative_path)
        if not path.exists() or not path.is_file():
            raise NotFoundError(f"checkpoint not found: {relative_path}")
        return inspect_checkpoint(path)

    @app.get("/v1/foundation/training", dependencies=[Depends(require_key)])
    def foundation_training() -> list[dict[str, Any]]:
        return _foundation_runs()

    @app.get("/v1/foundation/training/{run_name}", dependencies=[Depends(require_key)])
    def foundation_training_run(run_name: str) -> dict[str, Any]:
        run_dir = next((root / run_name for root in _foundation_run_roots() if (root / run_name).is_dir()), None)
        if run_dir is None:
            raise NotFoundError(f"training run not found: {run_name}")
        manifest = run_dir / "manifest.json"
        result = run_dir / "result.json"
        evaluation = run_dir / "evaluation.json"
        payload: dict[str, Any] = {"run_name": run_name, "path": str(run_dir)}
        if manifest.exists():
            payload["manifest"] = json.loads(manifest.read_text(encoding="utf-8"))
        if result.exists():
            payload["result"] = json.loads(result.read_text(encoding="utf-8"))
        if evaluation.exists():
            payload["evaluation"] = json.loads(evaluation.read_text(encoding="utf-8"))
        return payload

    @app.get("/v1/foundation/evaluation", dependencies=[Depends(require_key)])
    def foundation_evaluation() -> dict[str, Any]:
        root = _foundation_root() / "evaluation"
        reports = sorted(root.glob("*.json"), key=lambda path: path.stat().st_mtime, reverse=True) if root.exists() else []
        if not reports:
            return {"status": "no evaluation report recorded"}
        return {"path": str(reports[0]), "report": json.loads(reports[0].read_text(encoding="utf-8"))}

    @app.get("/v1/research", dependencies=[Depends(require_key)])
    def research(query: str, include_headers: bool = True, limit: int = 8) -> dict[str, Any]:
        from nexus.research.knowledge import research_topic
        return research_topic(query, include_headers=include_headers, limit=max(1, min(limit, 32)))

    @app.post("/v1/java/analyze", dependencies=[Depends(require_key)])
    def java_analyze(request: JavaAnalyzeRequest) -> dict[str, Any]:
        from nexus.java.service import analyze_java_artifact
        result = analyze_java_artifact(request.path)
        return result if request.full else {"format": result.get("format"), "path": result.get("path"), "summary": result.get("summary")}


    @app.post("/v1/java/analyze-binary", dependencies=[Depends(require_key)])
    def java_analyze_binary(request: JavaBinaryRequest) -> dict[str, Any]:
        from nexus.java.service import analyze_java_artifact

        binary = engine.storage.get_binary(request.binary_id)
        target, temporary = _java_target(binary, engine.paths.artifacts / "uploads")
        try:
            result = analyze_java_artifact(target)
        finally:
            if temporary is not None:
                shutil.rmtree(temporary, ignore_errors=True)
        if request.full:
            return result
        return {"format": result.get("format"), "summary": result.get("summary")}

    @app.post("/v1/java/reconstruct-binary", dependencies=[Depends(require_key)])
    def java_reconstruct_binary(request: JavaBinaryRequest) -> dict[str, Any]:
        from nexus.java.service import reconstruct_java_artifact
        from nexus.verification.compiler import compile_java_project

        binary = engine.storage.get_binary(request.binary_id)
        out_dir = engine.paths.artifacts / "java_reconstructions" / request.project_name
        target, temporary = _java_target(binary, engine.paths.artifacts / "uploads")
        try:
            result = reconstruct_java_artifact(
                target,
                out_dir=out_dir,
                project_name=request.project_name,
                max_classes=request.max_classes,
            )
        finally:
            if temporary is not None:
                shutil.rmtree(temporary, ignore_errors=True)
        if request.verify:
            result["verification"] = compile_java_project(out_dir)
        result["binary_id"] = request.binary_id
        result.pop("output_dir", None)
        return result

    @app.post("/v1/java/reconstruct", dependencies=[Depends(require_key)])
    def java_reconstruct(request: JavaReconstructRequest) -> dict[str, Any]:
        from nexus.java.service import reconstruct_java_artifact
        out_dir = Path(request.out_dir) if request.out_dir else engine.paths.artifacts / "java_reconstructions" / request.project_name
        result = reconstruct_java_artifact(
            request.path,
            out_dir=out_dir,
            project_name=request.project_name,
            max_classes=request.max_classes,
        )
        if request.verify:
            from nexus.verification.compiler import compile_java_project
            result["verification"] = compile_java_project(out_dir)
        return result

    @app.post("/v1/report", dependencies=[Depends(require_key)])
    def comprehensive_report(request: ComprehensiveReportRequest) -> dict[str, Any]:
        from nexus.reporting.comprehensive import build_comprehensive_report
        return build_comprehensive_report(
            request.binary_path,
            java_path=request.java_path,
            native_path=request.native_path,
            out_dir=request.out_dir,
            max_functions=request.max_functions,
        )

    @app.post("/v1/jni/correlate", dependencies=[Depends(require_key)])
    def jni_correlate(request: JNICorrelationRequest) -> dict[str, Any]:
        from nexus.jni.correlation import correlate_java_native
        return correlate_java_native(request.java_path, request.native_path)

    @app.post("/v1/foundation/analyze", dependencies=[Depends(require_key)])
    def foundation_analyze(request: FoundationAnalysisRequest) -> dict[str, Any]:
        from nexus.foundation.analysis import analyze_with_foundation
        binary = engine.storage.get_binary(request.binary_id)
        checkpoint = _foundation_safe_path(request.checkpoint)
        if not checkpoint.exists():
            raise NotFoundError(f"checkpoint not found: {request.checkpoint}")
        return analyze_with_foundation(binary["path"], checkpoint, max_functions=request.max_functions)

    @app.post("/v1/foundation/reconstruct", dependencies=[Depends(require_key)])
    def foundation_reconstruct(request: FoundationReconstructionRequest) -> dict[str, Any]:
        from nexus.foundation.reconstruction import UnifiedReconstructor
        binary = engine.storage.get_binary(request.binary_id)
        checkpoint = None
        if request.checkpoint:
            checkpoint = _foundation_safe_path(request.checkpoint)
            if not checkpoint.exists():
                raise NotFoundError(f"checkpoint not found: {request.checkpoint}")
        reconstructor = UnifiedReconstructor(checkpoint_path=checkpoint, workspace=engine.paths.artifacts / "reconstruction")
        result = reconstructor.reconstruct_function(
            binary["path"],
            request.function_id,
            language=request.language,
            reference_source=request.reference_source,
            include_deterministic=True,
            include_neural=checkpoint is not None,
        )
        return result

    if dashboard_dir.exists():
        app.mount("/static", StaticFiles(directory=str(dashboard_dir)), name="static")

        @app.get("/")
        def dashboard() -> FileResponse:
            return FileResponse(dashboard_dir / "index.html")

    return app


app = create_app()
