import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from nexus.core.hashing import sha256_file
from nexus.elf.parser import parse_elf

DEFAULT_LINK_LIBRARIES = ["-lm", "-ldl", "-pthread"]
EXCLUDED_DIRS = {
    "test", "tests", "testing", "examples", "example", "demo", "demos", "fuzz", "fuzzing",
    "bench", "benchmark", "benchmarks", "contrib", "platforms", "docs", "doc",
    "third_party", "third-party", "externals", "crates", "vendor", "subprojects",
    "misc", "scripts", "testdir", "testes",
}
COMPILER_CANDIDATES = ("gcc", "clang")


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _version(command: str) -> str | None:
    try:
        result = subprocess.run([command, "--version"], capture_output=True, text=True, timeout=20)
    except (OSError, subprocess.TimeoutExpired):
        return None
    text = result.stdout or result.stderr
    return text.splitlines()[0].strip() if text else None


def _target(command: str) -> str | None:
    try:
        result = subprocess.run([command, "-dumpmachine"], capture_output=True, text=True, timeout=20)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return result.stdout.strip() or None


@dataclass(frozen=True)
class BuildVariant:
    variant_id: str
    compiler: str
    optimization: str
    debug: bool
    pie: bool
    link: str = "executable"
    strip: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class BuildRecipe:
    kind: str = "auto"
    sources: list[str] = field(default_factory=list)
    include: list[str] = field(default_factory=list)
    defines: list[str] = field(default_factory=list)
    libraries: list[str] = field(default_factory=list)
    auto_include: bool = True
    main_exclude: bool = True
    extra_flags: list[str] = field(default_factory=list)
    auxiliary: bool = True

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "BuildRecipe":
        payload = payload or {}
        known = {field_name for field_name in cls.__dataclass_fields__}
        return cls(**{key: value for key, value in payload.items() if key in known})


def _variant(compiler: str, optimization: str, debug: bool, pie: bool, link: str = "executable", strip: bool = False) -> BuildVariant:
    mode = "debug" if debug else "stripped"
    return BuildVariant(
        variant_id=f"{compiler}_{optimization}_{mode}_{'pie' if pie else 'nopie'}_{link}",
        compiler=compiler,
        optimization=optimization,
        debug=debug,
        pie=pie,
        link=link,
        strip=strip,
    )


def pilot_matrix(project_kind: str = "auto") -> list[BuildVariant]:
    link = "shared" if project_kind == "shared" else "executable"
    variants = []
    for compiler in ("gcc", "clang"):
        for optimization in ("O0", "O1", "O2", "O3"):
            variants.append(_variant(compiler, optimization, True, True, link))
        variants.append(_variant(compiler, "O2", False, True, link, strip=True))
    return variants


def extended_matrix(project_kind: str = "auto") -> list[BuildVariant]:
    link = "shared" if project_kind == "shared" else "executable"
    variants = pilot_matrix(project_kind)
    for compiler in ("gcc", "clang"):
        variants.append(_variant(compiler, "O2", True, False, link))
        variants.append(_variant(compiler, "O2", False, False, link, strip=True))
    variants.append(_variant("gcc", "O2", True, True, "shared"))
    variants.append(_variant("clang", "O2", True, True, "shared"))
    variants.append(_variant("gcc", "O0", True, False, "static"))
    return variants


def fixture_matrix() -> list[BuildVariant]:
    variants = []
    for compiler in ("gcc", "clang"):
        for optimization in ("O0", "O1", "O2", "O3"):
            for debug in (True, False):
                for pie in (True, False):
                    variants.append(_variant(compiler, optimization, debug, pie, "executable", strip=not debug))
    variants.extend([
        _variant("gcc", "O2", True, True, "shared"),
        _variant("clang", "O2", True, True, "shared"),
        _variant("gcc", "O0", False, False, "static"),
    ])
    return variants


def _contains_main(path: Path) -> bool:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return False
    return bool(re.search(r"\bint\s+main\s*\(", text)) or bool(re.search(r"\bint\s+wmain\s*\(", text))


def _match_sources(root: Path, patterns: list[str], limit: int = 400) -> list[Path]:
    selected: dict[str, Path] = {}
    for pattern in patterns:
        if any(character in pattern for character in "*?["):
            matches = root.glob(pattern)
        else:
            candidate = root / pattern
            matches = [candidate] if candidate.exists() else []
        for path in matches:
            if path.is_file() and path.suffix in {".c", ".cc", ".cpp", ".cxx"}:
                selected[str(path)] = path
            elif path.is_dir():
                for child in path.rglob("*.c"):
                    selected[str(child)] = child
        if len(selected) >= limit:
            break
    return sorted(selected.values())[:limit]


def _auto_sources(root: Path, kind: str, limit: int = 400) -> list[Path]:
    files = []
    for path in root.rglob("*.c"):
        relative = path.relative_to(root)
        if any(part.lower() in EXCLUDED_DIRS for part in relative.parts[:-1]):
            continue
        if path.stat().st_size > 4 * 1024 * 1024:
            continue
        if re.search(r"(test|bench|fuzz|example|demo)", path.stem, re.IGNORECASE):
            continue
        files.append(path)
    files.sort()
    if kind == "shared":
        non_main = [path for path in files if not _contains_main(path)]
        if non_main:
            return non_main[:limit]
    return files[:limit]


def resolve_sources(root: Path, recipe: BuildRecipe) -> tuple[list[Path], list[str]]:
    warnings = []
    if recipe.sources:
        selected = _match_sources(root, recipe.sources)
        missing = [pattern for pattern in recipe.sources if not _match_sources(root, [pattern]) and not any(character in pattern for character in "*?[")]
        if missing:
            warnings.append(f"recipe source patterns missing: {', '.join(missing[:8])}")
    else:
        selected = _auto_sources(root, recipe.kind if recipe.kind in {"shared", "executable", "static"} else "auto")
    if recipe.kind in {"shared", "static"} and recipe.main_exclude:
        without_main = [path for path in selected if not _contains_main(path)]
        if without_main:
            selected = without_main
    if not selected:
        warnings.append("no C sources selected")
    return selected, warnings


def _include_dirs(root: Path, recipe: BuildRecipe, sources: list[Path]) -> list[Path]:
    directories = [root]
    for item in recipe.include:
        candidate = root / item
        if candidate.is_dir():
            directories.append(candidate)
    if recipe.auto_include:
        header_dirs = sorted({path.parent for path in root.rglob("*.h") if path.stat().st_size < 2 * 1024 * 1024})
        for directory in header_dirs:
            if len(directories) >= 80:
                break
            if directory not in directories:
                directories.append(directory)
        for source in sources:
            if source.parent not in directories:
                directories.append(source.parent)
    return directories[:120]


def build_command(
    *,
    root: Path,
    sources: list[Path],
    recipe: BuildRecipe,
    variant: BuildVariant,
    output: Path,
    random_seed: str,
) -> tuple[list[str], list[str]]:
    warnings = []
    includes = _include_dirs(root, recipe, sources)
    common = [
        f"-{variant.optimization}",
        "-std=gnu11",
        "-D_GNU_SOURCE",
        "-fno-omit-frame-pointer",
        f"-ffile-prefix-map={root}=.",
        f"-fdebug-prefix-map={root}=.",
        f"-frandom-seed={random_seed}",
        "-w",
        *[f"-I{directory}" for directory in includes],
        *[f"-D{value}" for value in recipe.defines],
        *recipe.extra_flags,
    ]
    common.extend(["-g3"] if variant.debug else ["-g0"])
    if variant.link == "shared":
        common.extend(["-fPIC", "-shared"])
    elif variant.link == "static":
        common.extend(["-fno-pie", "-static"])
    elif variant.pie:
        common.extend(["-fPIE", "-pie"])
    else:
        common.extend(["-fno-pie", "-no-pie"])
    command = [variant.compiler, *common, *[str(path) for path in sources], *DEFAULT_LINK_LIBRARIES, *recipe.libraries]
    if variant.link != "shared":
        command.extend(["-Wl,--build-id=none"])
    command.extend(["-o", str(output)])
    return command, warnings


def _run(command: list[str], cwd: Path, timeout: int = 240) -> dict[str, Any]:
    env = {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "HOME": tempfile.gettempdir(),
        "LANG": "C",
        "LC_ALL": "C",
        "TMPDIR": tempfile.gettempdir(),
    }
    limited = command
    if shutil.which("prlimit") and command and Path(command[0]).name in {"gcc", "clang", "strip", "objdump", "readelf", "llvm-dwarfdump", "llvm-objdump"}:
        limited = ["prlimit", "--as=4294967296", "--cpu=240", "--nproc=4096", "--fsize=1073741824", "--", *command]
    try:
        result = subprocess.run(limited, cwd=cwd, capture_output=True, text=True, timeout=timeout, env=env)
        return {"command": command, "returncode": result.returncode, "stdout": result.stdout[-8000:], "stderr": result.stderr[-8000:]}
    except subprocess.TimeoutExpired:
        return {"command": command, "returncode": None, "stdout": "", "stderr": "timeout"}
    except OSError as exc:
        return {"command": command, "returncode": None, "stdout": "", "stderr": str(exc)}


def _sanitize(text: str, replacements: list[tuple[str, str]]) -> str:
    for old, new in replacements:
        if old:
            text = text.replace(str(old), new)
    return text


def _elf_metadata(path: Path) -> dict[str, Any]:
    try:
        elf = parse_elf(path)
    except Exception as exc:
        return {"status": "unavailable", "reason": f"{type(exc).__name__}: {exc}"}
    text_section = elf.section_by_name(".text")
    text_sha = None
    if text_section is not None and text_section.offset + text_section.size <= len(elf.raw):
        text_sha = hashlib.sha256(elf.raw[text_section.offset : text_section.offset + text_section.size]).hexdigest()
    return {
        "status": "ok",
        "sha256": elf.sha256,
        "size": elf.size,
        "class": elf.header.elf_class,
        "machine": elf.header.machine_name,
        "endianness": elf.header.endianness,
        "type": elf.header.type_name,
        "entry": f"0x{elf.header.entry:x}",
        "pie": elf.is_pie,
        "shared": elf.is_shared_object,
        "static": elf.is_static,
        "stripped": elf.is_stripped,
        "build_id": elf.build_id,
        "sections": len(elf.sections),
        "symbols": len(elf.symbols),
        "dynamic_symbols": len(elf.dynamic_symbols),
        "relocations": len(elf.relocations),
        "function_symbols": len(elf.function_symbols()),
        "text_sha256": text_sha,
        "text_size": text_section.size if text_section else None,
        "dependencies": elf.dependencies,
    }


def _write_auxiliary(build_dir: Path, binary: Path, root: Path, variant: BuildVariant, recipe: BuildRecipe, sources: list[Path], command: list[str]) -> dict[str, Any]:
    extras: dict[str, Any] = {"assembly": None, "objects": [], "ir": None, "dwarf": [], "disassembly": None, "symbols": None}
    if not recipe.auxiliary:
        return extras
    base = ["objdump", "-d", str(binary)]
    result = _run(base, root, timeout=120)
    if result["returncode"] == 0:
        path = build_dir / "disassembly.txt"
        path.write_text(_sanitize(result["stdout"], [(str(root), "."), (str(binary), binary.name)]), encoding="utf-8")
        extras["disassembly"] = path.name
    result = _run(["readelf", "-sW", str(binary)], root, timeout=120)
    if result["returncode"] == 0:
        path = build_dir / "symbols.txt"
        path.write_text(_sanitize(result["stdout"], [(str(root), "."), (str(binary), binary.name)]), encoding="utf-8")
        extras["symbols"] = path.name
    if variant.debug and variant.compiler in {"gcc", "clang"}:
        source_strings = {str(source) for source in sources}
        prefix = []
        for token in command:
            if token in source_strings:
                break
            prefix.append(token)
        assembly_dir = build_dir / "assembly"
        assembly_dir.mkdir(exist_ok=True)
        assembly_files = []
        for source in sources:
            relative = source.relative_to(root)
            assembly = assembly_dir / (str(relative).replace("/", "__") + ".s")
            assembly_command = [*prefix, "-S", str(source), "-o", str(assembly)]
            result = _run(assembly_command, root, timeout=240)
            if result["returncode"] == 0 and assembly.exists():
                assembly.write_text(_sanitize(assembly.read_text(encoding="utf-8", errors="replace"), [(str(root), ".")]), encoding="utf-8")
                assembly_files.append(str(assembly.relative_to(build_dir)))
        extras["assembly"] = assembly_files
        if variant.compiler == "clang":
            ir_dir = build_dir / "ir"
            ir_dir.mkdir(exist_ok=True)
            ir_files = []
            for source in sources:
                relative = source.relative_to(root)
                ir_path = ir_dir / (str(relative).replace("/", "__") + ".ll")
                ir_command = [*prefix, "-S", "-emit-llvm", str(source), "-o", str(ir_path)]
                result = _run(ir_command, root, timeout=240)
                if result["returncode"] == 0 and ir_path.exists():
                    ir_path.write_text(_sanitize(ir_path.read_text(encoding="utf-8", errors="replace"), [(str(root), ".")]), encoding="utf-8")
                    ir_files.append(str(ir_path.relative_to(build_dir)))
            extras["ir"] = ir_files
    if variant.debug and shutil.which("llvm-dwarfdump"):
        for name, args in (("dwarf_info.txt", ["--debug-info"]), ("dwarf_line.txt", ["--debug-line"])):
            result = _run(["llvm-dwarfdump", *args, str(binary)], root, timeout=180)
            if result["returncode"] == 0 and result["stdout"]:
                path = build_dir / name
                path.write_text(_sanitize(result["stdout"], [(str(root), "."), (str(binary), binary.name)]), encoding="utf-8")
                extras["dwarf"].append(path.name)
    if variant.debug:
        objects_dir = build_dir / "objects"
        objects_dir.mkdir(exist_ok=True)
        source_strings = {str(source) for source in sources}
        prefix = []
        for token in command:
            if token in source_strings:
                break
            prefix.append(token)
        for source in sources:
            relative = source.relative_to(root)
            object_path = objects_dir / (str(relative).replace("/", "__") + ".o")
            object_command = [*prefix, "-c", str(source), "-o", str(object_path)]
            result = _run(object_command, root, timeout=180)
            if result["returncode"] == 0 and object_path.exists():
                extras["objects"].append({"source": str(relative), "object": str(object_path.relative_to(build_dir)), "sha256": sha256_file(object_path), "size": object_path.stat().st_size})
    return extras


def _build_one(dataset_root: Path, project: dict[str, Any], variant: BuildVariant, recipe: BuildRecipe, source_root: Path, timeout: int = 240, build_root: Path | None = None) -> dict[str, Any]:
    project_slug = project["project_id"]
    output_root = (build_root or dataset_root).resolve()
    build_dir = output_root / "builds" / project_slug / variant.variant_id
    build_dir.mkdir(parents=True, exist_ok=True)
    sources, warnings = resolve_sources(source_root, recipe)
    artifact_name = project_slug + (".so" if variant.link == "shared" else "")
    output = build_dir / artifact_name
    if not sources:
        return {"status": "unavailable", "reason": "no selected sources", "variant": variant.to_dict(), "warnings": warnings, "sources": []}
    random_seed = hashlib.sha256(f"{project_slug}:{recipe.kind}".encode()).hexdigest()[:16]
    command, command_warnings = build_command(root=source_root, sources=sources, recipe=recipe, variant=variant, output=output, random_seed=random_seed)
    warnings.extend(command_warnings)
    result = _run(command, source_root, timeout=timeout)
    build_log = build_dir / "build.log"
    build_log.write_text(
        "\n".join(
            [
                f"project: {project.get('name')}",
                f"project_id: {project_slug}",
                f"commit: {project.get('commit')}",
                f"license: {project.get('license')}",
                f"compiler: {_version(variant.compiler)}",
                f"target: {_target(variant.compiler)}",
                f"optimization: {variant.optimization}",
                f"debug: {variant.debug}",
                f"pie: {variant.pie}",
                f"link: {variant.link}",
                f"strip: {variant.strip}",
                f"source_root: {source_root}",
                f"command: {' '.join(command)}",
                f"returncode: {result['returncode']}",
                "",
                "stdout:",
                result["stdout"],
                "stderr:",
                result["stderr"],
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    _sanitize_in_place(build_log, [(str(source_root), "."), (str(dataset_root), ".")])
    if result["returncode"] != 0 or not output.exists():
        return {
            "status": "failed",
            "reason": "compile or link failed",
            "variant": variant.to_dict(),
            "command": command,
            "returncode": result["returncode"],
            "stderr": _sanitize(result["stderr"], [(str(source_root), "."), (str(dataset_root), ".")])[-4000:],
            "stdout": _sanitize(result["stdout"], [(str(source_root), "."), (str(dataset_root), ".")])[-2000:],
            "sources": [str(path.relative_to(source_root)) for path in sources],
            "warnings": warnings,
        }
    strip_result = None
    if variant.strip:
        strip_result = _run(["strip", "--strip-all", str(output)], source_root, timeout=60)
    metadata = _elf_metadata(output)
    extras = _write_auxiliary(build_dir, output, source_root, variant, recipe, sources, command)
    source_hashes = {str(path.relative_to(source_root)): sha256_file(path) for path in sources}
    payload = {
        "schema_version": "0.1.0",
        "project_id": project_slug,
        "variant": variant.to_dict(),
        "status": "built",
        "artifact": output.name,
        "artifact_path": str(output.relative_to(output_root)),
        "artifact_sha256": sha256_file(output),
        "artifact_size": output.stat().st_size,
        "elf": metadata,
        "compiler": variant.compiler,
        "compiler_version": _version(variant.compiler),
        "target": _target(variant.compiler),
        "command": command,
        "command_sanitized": _sanitize(" ".join(command), [(str(source_root), "."), (str(dataset_root), ".")]),
        "sources": [str(path.relative_to(source_root)) for path in sources],
        "source_hashes": source_hashes,
        "warnings": warnings,
        "strip": {"command": strip_result["command"] if strip_result else None, "returncode": strip_result["returncode"] if strip_result else None},
        "extras": extras,
        "built_at": _now(),
    }
    (build_dir / "build_metadata.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def _sanitize_in_place(path: Path, replacements: list[tuple[str, str]]) -> None:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return
    path.write_text(_sanitize(text, replacements), encoding="utf-8")


def _project_recipe(project: dict[str, Any]) -> BuildRecipe:
    recipe = BuildRecipe.from_dict(project.get("build_recipe"))
    if recipe.kind == "auto":
        recipe.kind = "shared"
    return recipe


def _validate_variant(variant: BuildVariant, available: set[str]) -> str | None:
    if variant.compiler not in available:
        return f"compiler not installed: {variant.compiler}"
    return None


def run_build_matrix(
    dataset_root: str | Path,
    *,
    matrix: str = "pilot",
    max_projects: int | None = None,
    jobs: int = 2,
    timeout: int = 240,
    reproduce: bool = True,
    project_ids: list[str] | None = None,
) -> dict[str, Any]:
    root = Path(dataset_root).resolve()
    catalog = json.loads((root / "catalog.json").read_text(encoding="utf-8"))
    if project_ids:
        catalog = [project for project in catalog if project["project_id"] in set(project_ids)]
    accepted = [project for project in catalog if project.get("status") == "accepted"]
    if max_projects is not None:
        accepted = accepted[:max_projects]
    available = {compiler for compiler in COMPILER_CANDIDATES if shutil.which(compiler)}
    state_path = root / ".state" / "build_state.json"
    state = {}
    if state_path.exists():
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            state = {}
    results = []
    for project in accepted:
        recipe = _project_recipe(project)
        source_root = (root / project["source_root"]).resolve()
        if not source_root.exists():
            results.append({"project_id": project["project_id"], "status": "unavailable", "reason": "source root missing", "builds": []})
            continue
        if matrix == "fixture":
            variants = fixture_matrix()
        elif matrix == "extended":
            variants = extended_matrix(recipe.kind)
        else:
            variants = pilot_matrix(recipe.kind)
        builds = []
        pending = []
        for variant in variants:
            unavailable = _validate_variant(variant, available)
            key = f"{project['project_id']}:{variant.variant_id}"
            if unavailable:
                builds.append({"status": "unavailable", "reason": unavailable, "variant": variant.to_dict()})
                continue
            metadata_path = root / "builds" / project["project_id"] / variant.variant_id / "build_metadata.json"
            if metadata_path.exists() and state.get(key):
                try:
                    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
                    artifact = root / payload["artifact_path"]
                    if artifact.exists() and sha256_file(artifact) == payload["artifact_sha256"]:
                        builds.append(payload)
                        continue
                except (OSError, json.JSONDecodeError, KeyError):
                    pass
            pending.append((variant, key))
        if pending:
            workers = max(1, min(jobs, len(pending)))
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = {
                    executor.submit(_build_one, root, project, variant, recipe, source_root, timeout): (variant, key)
                    for variant, key in pending
                }
                for future in as_completed(futures):
                    variant, key = futures[future]
                    try:
                        payload = future.result()
                    except Exception as exc:
                        payload = {"status": "failed", "reason": f"{type(exc).__name__}: {exc}", "variant": variant.to_dict()}
                    builds.append(payload)
                    state[key] = {"status": payload.get("status"), "artifact_sha256": payload.get("artifact_sha256"), "at": _now()}
                    state_path.parent.mkdir(parents=True, exist_ok=True)
                    state_path.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
        builds.sort(key=lambda item: (item.get("variant") or {}).get("variant_id", ""))
        reproduction = None
        if reproduce:
            reference = next((item for item in builds if item.get("status") == "built" and item["variant"]["variant_id"] == "gcc_O0_debug_pie_executable"), None)
            if reference is None:
                reference = next((item for item in builds if item.get("status") == "built" and item["variant"]["variant_id"].startswith("gcc_O0_debug")), None)
            if reference:
                with tempfile.TemporaryDirectory(prefix="nexus-repro-", dir=root) as temporary:
                    repeat = _build_one(root, project, BuildVariant(**reference["variant"]), recipe, source_root, timeout, build_root=Path(temporary))
                    reproduction = {
                        "variant": reference["variant"]["variant_id"],
                        "first_sha256": reference["artifact_sha256"],
                        "repeat_sha256": repeat.get("artifact_sha256"),
                        "reproducible": repeat.get("status") == "built" and repeat.get("artifact_sha256") == reference["artifact_sha256"],
                        "repeat_status": repeat.get("status"),
                    }
        results.append({"project_id": project["project_id"], "name": project.get("name"), "category": project.get("category"), "status": "completed", "builds": builds, "reproduction": reproduction})
    built = [build for project in results for build in project.get("builds", []) if build.get("status") == "built"]
    failed = [build for project in results for build in project.get("builds", []) if build.get("status") == "failed"]
    unavailable = [build for project in results for build in project.get("builds", []) if build.get("status") == "unavailable"]
    by_compiler: dict[str, int] = {}
    by_optimization: dict[str, int] = {}
    by_debug: dict[str, int] = {}
    by_variant: dict[str, int] = {}
    for build in built:
        variant = build.get("variant", {})
        by_compiler[variant.get("compiler", "unknown")] = by_compiler.get(variant.get("compiler", "unknown"), 0) + 1
        by_optimization[variant.get("optimization", "unknown")] = by_optimization.get(variant.get("optimization", "unknown"), 0) + 1
        by_debug["debug" if variant.get("debug") else "stripped"] = by_debug.get("debug" if variant.get("debug") else "stripped", 0) + 1
        by_variant[variant.get("variant_id", "unknown")] = by_variant.get(variant.get("variant_id", "unknown"), 0) + 1
    report = {
        "schema_version": "0.1.0",
        "created_at": _now(),
        "dataset_root": str(root),
        "matrix": matrix,
        "projects_requested": len(accepted),
        "projects_completed": len([project for project in results if project.get("builds")]),
        "builds_built": len(built),
        "builds_failed": len(failed),
        "builds_unavailable": len(unavailable),
        "by_compiler": dict(sorted(by_compiler.items())),
        "by_optimization": dict(sorted(by_optimization.items())),
        "by_debug_mode": dict(sorted(by_debug.items())),
        "by_variant": dict(sorted(by_variant.items())),
        "compiler_versions": {compiler: _version(compiler) for compiler in sorted(available)},
        "compiler_targets": {compiler: _target(compiler) for compiler in sorted(available)},
        "reproducibility": [
            {"project_id": project["project_id"], **(project.get("reproduction") or {})}
            for project in results
            if project.get("reproduction") is not None
        ],
        "projects": results,
        "warnings": [
            "The pilot matrix exercises PIE executable and shared-library builds. Non-PIE executables are exercised by controlled alignment fixtures. Static linking is implemented but was not exercised across the real-world pilot; unavailable and failed configurations are reported separately.",
            "Builds use controlled single-command compiles, not the upstream project build systems.",
            "Stripped variants are compiled with -g0 and then stripped; debug/stripped text identity is checked before alignment transfer.",
        ],
    }
    (root / "build_matrix_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report
