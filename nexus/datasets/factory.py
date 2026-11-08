import hashlib
import json
import shutil
import subprocess
import tarfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from nexus.core.hashing import sha256_file
from nexus.datasets.fixtures import FIXTURES, fixture_spec
from nexus.mission.policy import manifest_mission
from nexus.program.bundle import analyze_binary
from nexus.program.record import build_program_record


def _run(command: list[str], cwd: Path, timeout: int = 60) -> subprocess.CompletedProcess:
    return subprocess.run(command, cwd=cwd, capture_output=True, text=True, timeout=timeout)


def _version(command: str) -> str | None:
    try:
        result = _run([command, "--version"], Path.cwd(), 10)
    except Exception:
        return None
    text = result.stdout or result.stderr
    return text.splitlines()[0] if text else None


def _target(compiler: str) -> str | None:
    try:
        result = _run([compiler, "-dumpmachine"], Path.cwd(), 10)
    except Exception:
        return None
    return result.stdout.strip() or None


def _split_for(project_id: str) -> str:
    digest = hashlib.sha256(f"nexus-split-v1:{project_id}".encode()).hexdigest()
    bucket = int(digest[:4], 16) % 100
    if bucket < 70:
        return "train"
    if bucket < 85:
        return "validation"
    return "test"


def _tar_source(source: Path, spec_path: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(target, "w") as archive:
        for path in (source, spec_path):
            info = tarfile.TarInfo(name=path.name)
            payload = path.read_bytes()
            info.size = len(payload)
            info.mtime = 0
            info.uid = 0
            info.gid = 0
            info.uname = ""
            info.gname = ""
            info.mode = 0o644
            import io
            archive.addfile(info, io.BytesIO(payload))


def _artifact(root: Path, path: Path, media_type: str) -> dict[str, Any]:
    return {
        "path": str(path.relative_to(root)),
        "sha256": sha256_file(path),
        "size": path.stat().st_size,
        "media_type": media_type,
    }


def _build(
    source: Path,
    output: Path,
    compiler: str,
    optimization: str,
    debug: bool,
    pie: bool,
    workdir: Path,
) -> dict[str, Any]:
    flags = [
        f"-{optimization}",
        "-std=c11",
        "-Wall",
        "-Wextra",
        "-fno-omit-frame-pointer",
        f"-ffile-prefix-map={source.parent}=.",
        f"-fdebug-prefix-map={source.parent}=.",
        f"-ffile-prefix-map={workdir}=.",
        f"-fdebug-prefix-map={workdir}=.",
        f"-frandom-seed={hashlib.sha256(source.read_bytes()).hexdigest()[:16]}",
    ]
    flags.extend(["-g"] if debug else ["-g0"])
    flags.extend(["-fPIE", "-pie"] if pie else ["-fno-pie", "-no-pie"])
    flags.append("-Wl,--build-id=none")
    command = [compiler, *flags, str(source), "-o", str(output)]
    output.parent.mkdir(parents=True, exist_ok=True)
    result = _run(command, workdir, 60)
    return {"command": command, "returncode": result.returncode, "stdout": result.stdout[-2000:], "stderr": result.stderr[-2000:]}


def _compile_aux(source: Path, compiler: str, optimization: str, debug: bool, pie: bool, output: Path, suffix: str, workdir: Path) -> dict[str, Any]:
    flags = [f"-{optimization}", "-std=c11", "-fno-omit-frame-pointer", f"-ffile-prefix-map={source.parent}=.", f"-fdebug-prefix-map={source.parent}=.", f"-ffile-prefix-map={workdir}=.", f"-fdebug-prefix-map={workdir}=."]
    flags.extend(["-g" ] if debug else ["-g0"])
    flags.extend(["-fPIE"] if pie else ["-fno-pie"])
    command = [compiler, *flags, suffix, str(source), "-o", str(output)]
    output.parent.mkdir(parents=True, exist_ok=True)
    result = _run(command, workdir, 60)
    return {"command": command, "returncode": result.returncode, "stdout": result.stdout[-2000:], "stderr": result.stderr[-2000:]}


def _compile_ir(source: Path, compiler: str, optimization: str, debug: bool, pie: bool, output: Path, workdir: Path) -> dict[str, Any]:
    flags = [f"-{optimization}", "-S", "-emit-llvm", "-fno-omit-frame-pointer", f"-ffile-prefix-map={source.parent}=.", f"-fdebug-prefix-map={source.parent}=.", f"-ffile-prefix-map={workdir}=.", f"-fdebug-prefix-map={workdir}=."]
    flags.extend(["-g"] if debug else ["-g0"])
    flags.extend(["-fPIE"] if pie else ["-fno-pie"])
    command = [compiler, *flags, str(source), "-o", str(output)]
    result = _run(command, workdir, 60)
    return {"command": command, "returncode": result.returncode, "stdout": result.stdout[-2000:], "stderr": result.stderr[-2000:]}


def _sanitize_text(path: Path, replacements: list[tuple[str, str]]) -> None:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return
    for old, new in replacements:
        if old:
            text = text.replace(old, new)
    path.write_text(text, encoding="utf-8")


def _dwarf_dump(binary: Path, sample_dir: Path, workdir: Path) -> list[dict[str, str]]:
    tool = shutil.which("llvm-dwarfdump") or shutil.which("readelf")
    if not tool:
        return []
    artifacts = []
    if Path(tool).name.startswith("llvm-dwarfdump"):
        for name, args in (("dwarf_info.txt", ["--debug-info"]), ("dwarf_line.txt", ["--debug-line"])):
            result = _run([tool, *args, str(binary)], workdir, 60)
            if result.returncode == 0 and result.stdout:
                target = sample_dir / name
                target.write_text(result.stdout, encoding="utf-8")
                _sanitize_text(target, [(str(workdir), "."), (str(binary), binary.name)])
                artifacts.append({"path": str(target), "media_type": "text/plain"})
    return artifacts


def _labels(bundle, root: Path, sample_id: str, sample_dir: Path) -> tuple[list[dict], dict[str, Any]]:
    functions = []
    for function in bundle.functions:
        functions.append({
            "sample_id": sample_id,
            "function_id": function.function_id,
            "name": function.name,
            "entry_va": f"0x{function.entry:x}",
            "ranges": [{"start_va": f"0x{start:x}", "end_va": f"0x{end:x}"} for start, end in function.ranges],
            "source": "symbol" if function.source in {"symbol", "disassembly_label"} else "inferred",
            "confidence": function.confidence,
            "instruction_count": len(function.instructions),
            "block_count": len(function.blocks),
            "aliases": function.aliases,
        })
    functions_payload = {"sample_id": sample_id, "functions": functions, "label_source": "symbol", "confidence": 0.7}
    functions_path = sample_dir / "functions.json"
    functions_path.write_text(json.dumps(functions_payload, indent=2) + "\n", encoding="utf-8")
    record = build_program_record(bundle)
    record_path = sample_dir / "program_record.json"
    record_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    cfg_path = sample_dir / "cfg.json"
    cfg_path.write_text(json.dumps(record.get("cfg", {}), indent=2) + "\n", encoding="utf-8")
    return functions, {
        "record": record,
        "record_path": record_path,
        "functions_path": functions_path,
        "cfg_path": cfg_path,
    }


def build_sample(
    root: Path,
    source: Path,
    spec_path: Path,
    project_id: str,
    source_family: str,
    compiler: str,
    optimization: str,
    debug: bool,
    pie: bool,
    split: str | None = None,
) -> dict[str, Any]:
    mode = "debug" if debug else "stripped"
    sample_id = f"sample_{source_family}_{compiler}_{optimization}_{mode}_{'pie' if pie else 'nopie'}"
    sample_dir = root / "samples" / sample_id
    sample_dir.mkdir(parents=True, exist_ok=True)
    binary = sample_dir / source_family
    build = _build(source, binary, compiler, optimization, debug, pie, root)
    if build["returncode"] != 0:
        return {"status": "failed", "sample_id": sample_id, "stage": "compile", "build": build}
    assembly = sample_dir / f"{source_family}.s"
    object_file = sample_dir / f"{source_family}.o"
    aux_assembly = _compile_aux(source, compiler, optimization, debug, pie, assembly, "-S", root)
    aux_object = _compile_aux(source, compiler, optimization, debug, pie, object_file, "-c", root)
    extra_artifacts = []
    if compiler == "clang":
        ir_path = sample_dir / f"{source_family}.ll"
        ir_result = _compile_ir(source, compiler, optimization, debug, pie, ir_path, root)
        if ir_result["returncode"] == 0 and ir_path.exists():
            _sanitize_text(ir_path, [(str(root), "."), (str(source), source.name)])
            extra_artifacts.append({"path": str(ir_path), "media_type": "text/plain"})
    if debug:
        for item in _dwarf_dump(binary, sample_dir, root):
            extra_artifacts.append(item)
    if not debug and shutil.which("strip"):
        _run(["strip", "--strip-all", str(binary)], root, 30)
    build_log = sample_dir / "build.log"
    build_log.write_text(
        "\n".join([
            f"compiler: {_version(compiler)}",
            f"target: {_target(compiler)}",
            f"source_sha256: {sha256_file(source)}",
            f"optimization: {optimization}",
            f"debug: {debug}",
            f"pie: {pie}",
            f"command: {' '.join(build['command'])}",
            f"assembly_command: {' '.join(aux_assembly['command'])}",
            f"object_command: {' '.join(aux_object['command'])}",
            f"llvm_ir: {any(item['path'].endswith('.ll') for item in extra_artifacts)}",
            f"dwarf: {any('dwarf' in item['path'] for item in extra_artifacts)}",
            "",
        ]),
        encoding="utf-8",
    )
    _sanitize_text(build_log, [(str(root), ".")])
    source_archive = sample_dir / "source.tar"
    _tar_source(source, spec_path, source_archive)
    bundle = analyze_binary(binary)
    functions, label_paths = _labels(bundle, root, sample_id, sample_dir)
    record = label_paths["record"]
    rebuild = sample_dir / "rebuild"
    rebuild.mkdir(parents=True, exist_ok=True)
    rebuild_binary = rebuild / source_family
    rebuild_result = _build(source, rebuild_binary, compiler, optimization, debug, pie, root)
    if not debug and shutil.which("strip"):
        _run(["strip", "--strip-all", str(rebuild_binary)], root, 30)
    reproducible = rebuild_result["returncode"] == 0 and sha256_file(binary) == sha256_file(rebuild_binary)
    shutil.rmtree(rebuild, ignore_errors=True)
    manifest = {
        "schema_version": "0.1.0",
        "sample_id": sample_id,
        "project_id": project_id,
        "source_family": source_family,
        "split": split or _split_for(project_id),
        "license": {"spdx": "MIT", "redistributable": True, "notes": "Synthetic fixture authored for NEXUS-BI."},
        "provenance": {
            "repository": None,
            "commit": None,
            "source_archive_sha256": sha256_file(source_archive),
            "source_paths": [str(source.relative_to(root))],
            "upstream_license_file_sha256": None,
        },
        "build": {
            "compiler": compiler if compiler in {"gcc", "clang"} else "other",
            "compiler_version": _version(compiler) or "unknown",
            "target": _target(compiler) or "unknown",
            "arch": record.get("program", {}).get("machine", "x86_64"),
            "abi": "SysV-AMD64" if record.get("program", {}).get("machine") == "x86_64" else "unknown",
            "optimization": optimization,
            "flags": [flag for flag in build["command"][1:-3] if flag.startswith("-")],
            "linker": "system",
            "lto": False,
            "pie": pie,
            "static": False,
            "stripped": not debug,
            "debug_info": debug,
            "sanitizers": [],
        },
        "binary": {
            "sha256": record["program"]["sha256"],
            "size": record["program"]["size"],
            "format": "ELF",
            "class": record["program"]["class"],
            "machine": record["program"]["machine"],
            "endianness": record["program"]["endianness"],
        },
        "artifacts": [
            _artifact(root, source, "text/x-c"),
            _artifact(root, spec_path, "application/json"),
            _artifact(root, binary, "application/x-elf"),
            _artifact(root, object_file, "application/x-object"),
            _artifact(root, assembly, "text/x-asm"),
            _artifact(root, build_log, "text/plain"),
            _artifact(root, source_archive, "application/x-tar"),
            _artifact(root, label_paths["record_path"], "application/json"),
            *[_artifact(root, Path(item["path"]), item["media_type"]) for item in extra_artifacts],
        ],
        "label_manifests": [
            {"kind": "functions", "artifact": _artifact(root, label_paths["functions_path"], "application/json"), "notes": "Function candidates from symbol/disassembly evidence."},
            {"kind": "cfg", "artifact": _artifact(root, label_paths["cfg_path"], "application/json"), "notes": "Conservative CFG from x86-64 disassembly."},
        ],
        "tests": [],
        "quality": {
            "label_source": "compiler",
            "human_reviewed": False,
            "validation_status": "compile_reproduced" if reproducible else "schema_valid",
            "confidence": 0.7 if reproducible else 0.5,
            "known_gaps": [
                "No instruction-to-source mapping.",
                "No behavioral tests were executed.",
                "Function labels are symbol/disassembly candidates, not source-verified boundaries.",
            ] if not reproducible else [
                "No instruction-to-source mapping.",
                "No behavioral tests were executed.",
            ],
        },
        "lineage": {"parent_sample_id": f"sample_{source_family}_{compiler}_{optimization}_{mode}", "variant_kind": "pie" if pie else "nopie"},
        "mission_relevance": manifest_mission(
            "controlled_binary_source",
            ["binary_representation_masking", "function_boundary_prediction", "c_reconstruction"],
            "Controlled source and binary with compiler metadata and function/CFG labels.",
        ),
    }
    return {
        "status": "ok",
        "sample_id": sample_id,
        "manifest": manifest,
        "bundle": bundle,
        "record": record,
        "reproducible": reproducible,
    }


def generate_dataset(
    out_dir: str | Path,
    fixtures: list[str] | None = None,
    compilers: list[str] | None = None,
    optimizations: list[str] | None = None,
    debug_modes: list[bool] | None = None,
    pie_modes: list[bool] | None = None,
    include_records: bool = True,
) -> dict[str, Any]:
    root = Path(out_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    fixtures = fixtures or list(FIXTURES)
    compilers = compilers or [name for name in ("gcc", "clang") if shutil.which(name)]
    optimizations = optimizations or ["O0", "O2"]
    debug_modes = debug_modes if debug_modes is not None else [True, False]
    pie_modes = pie_modes if pie_modes is not None else [True]
    sources_dir = root / "sources"
    sources_dir.mkdir(parents=True, exist_ok=True)
    manifests = []
    quarantined = []
    seen_hashes: set[str] = set()
    for name in fixtures:
        source = sources_dir / f"{name}.c"
        source.write_text(FIXTURES[name], encoding="utf-8")
        spec_path = sources_dir / f"{name}.json"
        spec_path.write_text(json.dumps(fixture_spec(name), indent=2) + "\n", encoding="utf-8")
        project_id = f"project_{name}"
        for compiler in compilers:
            for optimization in optimizations:
                for debug in debug_modes:
                    for pie in pie_modes:
                        result = build_sample(root, source, spec_path, project_id, name, compiler, optimization, debug, pie)
                        if result["status"] != "ok":
                            quarantined.append({"sample_id": result.get("sample_id"), "reason": result.get("stage", "build_failed"), "detail": result.get("build")})
                            continue
                        manifest = result["manifest"]
                        if manifest["binary"]["sha256"] in seen_hashes:
                            quarantined.append({"sample_id": manifest["sample_id"], "reason": "duplicate_binary_hash"})
                            continue
                        seen_hashes.add(manifest["binary"]["sha256"])
                        try:
                            import jsonschema
                            schema_path = Path(__file__).resolve().parents[2] / "schemas" / "dataset_manifest.schema.json"
                            schema = json.loads(schema_path.read_text(encoding="utf-8"))
                            errors = list(jsonschema.Draft202012Validator(schema).iter_errors(manifest))
                        except Exception as exc:
                            errors = [exc]
                        if errors:
                            quarantined.append({"sample_id": manifest["sample_id"], "reason": "schema_invalid", "detail": str(errors[:3])})
                            continue
                        manifests.append(manifest)
    manifests.sort(key=lambda item: item["sample_id"])
    manifest_path = root / "manifests.jsonl"
    with open(manifest_path, "w", encoding="utf-8") as handle:
        for manifest in manifests:
            handle.write(json.dumps(manifest, sort_keys=True) + "\n")
    quarantine_path = root / "quarantine.jsonl"
    with open(quarantine_path, "w", encoding="utf-8") as handle:
        handle.writelines(json.dumps(item, sort_keys=True) + "\n" for item in quarantined)
    by_split: dict[str, int] = {}
    by_compiler: dict[str, int] = {}
    by_optimization: dict[str, int] = {}
    by_stripped: dict[str, int] = {}
    projects = set()
    builds_per_project: dict[str, int] = {}
    source_programs = set()
    llvm_ir_samples = 0
    dwarf_samples = 0
    total_bytes = 0
    total_instructions = 0
    for manifest in manifests:
        by_split[manifest["split"]] = by_split.get(manifest["split"], 0) + 1
        by_compiler[manifest["build"]["compiler"]] = by_compiler.get(manifest["build"]["compiler"], 0) + 1
        by_optimization[manifest["build"]["optimization"]] = by_optimization.get(manifest["build"]["optimization"], 0) + 1
        key = "stripped" if manifest["build"]["stripped"] else "unstripped"
        by_stripped[key] = by_stripped.get(key, 0) + 1
        projects.add(manifest["project_id"])
        builds_per_project[manifest["project_id"]] = builds_per_project.get(manifest["project_id"], 0) + 1
        source_programs.add(manifest["source_family"])
        if any(artifact["path"].endswith(".ll") for artifact in manifest["artifacts"]):
            llvm_ir_samples += 1
        if any("dwarf" in artifact["path"] for artifact in manifest["artifacts"]):
            dwarf_samples += 1
        total_bytes += manifest["binary"]["size"]
        record_path = root / next(item["path"] for item in manifest["artifacts"] if item["path"].endswith("program_record.json"))
        try:
            record = json.loads(record_path.read_text(encoding="utf-8"))
            total_instructions += len(record.get("instructions", []))
        except Exception:
            pass
    split_projects: dict[str, set[str]] = {}
    for manifest in manifests:
        split_projects.setdefault(manifest["split"], set()).add(manifest["project_id"])
    leakage = []
    for split, projects_in_split in split_projects.items():
        for other_split, other_projects in split_projects.items():
            if split >= other_split:
                continue
            overlap = projects_in_split & other_projects
            if overlap:
                leakage.append({"splits": [split, other_split], "projects": sorted(overlap)})
    summary = {
        "schema_version": "0.1.0",
        "created_at": datetime.now(UTC).isoformat(),
        "dataset_id": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "samples": len(manifests),
        "quarantined": len(quarantined),
        "projects": len(projects),
        "unique_source_programs": len(source_programs),
        "builds_per_project": builds_per_project,
        "samples_with_llvm_ir": llvm_ir_samples,
        "samples_with_dwarf_dump": dwarf_samples,
        "by_split": by_split,
        "by_compiler": by_compiler,
        "by_optimization": by_optimization,
        "by_stripped": by_stripped,
        "total_binary_bytes": total_bytes,
        "total_instructions": total_instructions,
        "leakage": leakage,
        "capabilities": {
            "cross_architecture": "disabled: cross toolchains/QEMU unavailable",
            "jni_fixtures": "separate path; not generated in this dataset build",
        },
        "limitations": [
            "Only x86-64 builds are produced on this host.",
            "No behavioral tests are executed.",
            "Function labels are symbol/disassembly candidates, not source-verified boundaries.",
            "Synthetic fixtures do not represent real-world project complexity.",
        ],
    }
    (root / "dataset_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return {
        "status": "completed",
        "path": str(root),
        "manifest_path": str(manifest_path),
        "summary_path": str(root / "dataset_summary.json"),
        "summary": summary,
        "quarantined": len(quarantined),
    }


def inspect_dataset(path: str | Path) -> dict[str, Any]:
    root = Path(path)
    manifest_path = root / "manifests.jsonl" if root.is_dir() else root
    if not manifest_path.exists():
        return {"status": "missing", "path": str(manifest_path)}
    manifests = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    projects = {item["project_id"] for item in manifests}
    splits = sorted({item["split"] for item in manifests})
    split_projects: dict[str, set[str]] = {}
    for item in manifests:
        split_projects.setdefault(item["split"], set()).add(item["project_id"])
    leakage = []
    for split in splits:
        for other in splits:
            if split >= other:
                continue
            overlap = split_projects.get(split, set()) & split_projects.get(other, set())
            if overlap:
                leakage.append({"splits": [split, other], "projects": sorted(overlap)})
    return {
        "status": "ok",
        "path": str(manifest_path),
        "dataset_id": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "samples": len(manifests),
        "projects": len(projects),
        "splits": splits,
        "leakage": leakage,
        "by_compiler": {name: sum(1 for item in manifests if item["build"]["compiler"] == name) for name in sorted({item["build"]["compiler"] for item in manifests})},
        "by_optimization": {name: sum(1 for item in manifests if item["build"]["optimization"] == name) for name in sorted({item["build"]["optimization"] for item in manifests})},
        "stripped": sum(1 for item in manifests if item["build"]["stripped"]),
        "unstripped": sum(1 for item in manifests if not item["build"]["stripped"]),
    }
