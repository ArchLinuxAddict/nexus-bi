import hashlib
import json
import os
import re
import shutil
import subprocess
import tarfile
import tempfile
import urllib.request
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from nexus.core.hashing import sha256_file
from nexus.datasets.dwarf_source import extract_source_labels
from nexus.datasets.factory import _artifact, _labels, _split_for, _target, _version
from nexus.mission.policy import manifest_mission
from nexus.program.bundle import analyze_binary

MAX_ARCHIVE_BYTES = 20 * 1024 * 1024
MAX_SOURCE_FILES = 400


@dataclass
class BuildVariant:
    compiler: str
    optimization: str
    debug: bool
    link: str = "auto"


DEFAULT_VARIANTS = (
    BuildVariant("gcc", "O0", True),
    BuildVariant("gcc", "O2", True),
    BuildVariant("gcc", "O2", False),
    BuildVariant("clang", "O0", True),
    BuildVariant("clang", "O2", True),
    BuildVariant("clang", "O2", False),
)


def _slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]+", "_", value).strip("_").lower()


def safe_extract_tar(archive: Path, destination: Path) -> dict[str, Any]:
    destination.mkdir(parents=True, exist_ok=True)
    extracted = 0
    total_bytes = 0
    with tarfile.open(archive, "r:*") as handle:
        for member in handle:
            name = member.name
            if name.startswith("/") or ".." in Path(name).parts:
                raise ValueError(f"unsafe archive member: {name}")
            if member.issym() or member.islnk() or member.isdev():
                raise ValueError(f"unsupported archive member type: {name}")
            if not member.isfile() and not member.isdir():
                continue
            if extracted >= 5000 or total_bytes >= 200 * 1024 * 1024:
                raise ValueError("archive extraction budget exceeded")
            target = destination / name
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            source = handle.extractfile(member)
            if source is None:
                continue
            with open(target, "wb") as output:
                shutil.copyfileobj(source, output)
            extracted += 1
            total_bytes += member.size
    return {"files": extracted, "bytes": total_bytes}


def download_archive(project: dict[str, Any], target: Path, timeout: int = 60) -> dict[str, Any]:
    target.parent.mkdir(parents=True, exist_ok=True)
    url = f"https://codeload.github.com/{project['full_name']}/tar.gz/{project['commit']}"
    request = urllib.request.Request(url, headers={"User-Agent": "nexus-bi-dataset"})
    with urllib.request.urlopen(request, timeout=timeout) as response, open(target, "wb") as output:
        total = 0
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            total += len(chunk)
            if total > MAX_ARCHIVE_BYTES:
                raise ValueError("source archive exceeds size limit")
            output.write(chunk)
    return {"url": url, "path": str(target), "sha256": sha256_file(target), "size": target.stat().st_size}


def find_license(root: Path) -> dict[str, Any]:
    candidates = []
    for path in root.rglob("*"):
        if path.is_file() and path.name.lower() in {"license", "license.txt", "license.md", "copying", "copying.txt", "license-mit", "license-bsd"}:
            candidates.append(path)
    if not candidates:
        return {"found": False}
    path = sorted(candidates, key=lambda item: len(item.name))[0]
    return {"found": True, "path": str(path.relative_to(root)), "sha256": sha256_file(path), "text_preview": path.read_text(encoding="utf-8", errors="replace")[:500]}


def discover_c_files(root: Path) -> list[Path]:
    excluded_dirs = {"test", "tests", "examples", "example", "third_party", "vendor", "build", "obj", ".git", "node_modules", "fuzz", "fuzzers", "bench", "benchmark"}
    excluded_names = re.compile(r"(^|_)(win|windows|darwin|macos|android|ios|fuzz|bench|example|demo|sample)(_|\.|$)", re.IGNORECASE)
    files = []
    for path in root.rglob("*.c"):
        relative = path.relative_to(root)
        if any(part.lower() in excluded_dirs for part in relative.parts[:-1]):
            continue
        if excluded_names.search(path.stem):
            continue
        if path.stat().st_size > 2 * 1024 * 1024:
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="replace")[:20000]
        except OSError:
            continue
        if re.search(r"#\s*include\s*[<\"](windows\.h|objidl\.h|Python\.h|android/|TargetConditionals\.h)", text):
            continue
        files.append(path)
    return sorted(files)[:MAX_SOURCE_FILES]


def discover_include_dirs(root: Path) -> list[Path]:
    directories = {root}
    for path in root.rglob("*.h"):
        if len(directories) >= 80:
            break
        directories.add(path.parent)
    return sorted(directories)


def _contains_main(path: Path) -> bool:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return False
    return bool(re.search(r"\bint\s+main\s*\(", text))


def _sanitize_text(path: Path, replacements: list[tuple[str, str]]) -> None:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return
    for old, new in replacements:
        if old:
            text = text.replace(old, new)
    path.write_text(text, encoding="utf-8")


def _compile(command: list[str], cwd: Path, timeout: int = 60) -> dict[str, Any]:
    env = {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "HOME": tempfile.gettempdir(),
        "LANG": "C",
        "LC_ALL": "C",
    }
    limited = command
    if shutil.which("prlimit") and command and command[0] in {"gcc", "clang", "strip"}:
        limited = [
            "prlimit", "--as=2147483648", "--cpu=60", "--nproc=4096",
            "--fsize=536870912", "--", *command,
        ]
    try:
        result = subprocess.run(limited, cwd=cwd, capture_output=True, text=True, timeout=timeout, env=env)
        return {"command": command, "returncode": result.returncode, "stdout": result.stdout[-4000:], "stderr": result.stderr[-4000:]}
    except subprocess.TimeoutExpired:
        return {"command": command, "returncode": None, "stdout": "", "stderr": "timeout"}


def build_project(project_root: Path, compiler: str, optimization: str, debug: bool, output: Path) -> dict[str, Any]:
    files = discover_c_files(project_root)
    if not files:
        return {"status": "failed", "reason": "no C source files"}
    includes = [f"-I{directory}" for directory in discover_include_dirs(project_root)]
    common = [
        f"-{optimization}", "-std=gnu11", "-D_GNU_SOURCE", "-pthread", "-fno-omit-frame-pointer",
        f"-ffile-prefix-map={project_root}=.", f"-fdebug-prefix-map={project_root}=.",
        *includes, "-w",
    ]
    common.extend(["-g"] if debug else ["-g0"])
    main_files = [path for path in files if _contains_main(path)]
    non_main = [path for path in files if path not in main_files]
    attempts = []
    link_libraries = ["-lm", "-ldl", "-pthread"]
    executable_sets = []
    if main_files:
        executable_sets.append(files)
        for main_file in sorted(main_files, key=lambda item: item.stat().st_size)[:3]:
            executable_sets.append([main_file, *non_main])
            executable_sets.append([main_file])
    seen_sets = set()
    for candidate_files in executable_sets:
        key = tuple(str(path) for path in candidate_files)
        if key in seen_sets:
            continue
        seen_sets.add(key)
        output.parent.mkdir(parents=True, exist_ok=True)
        command = [compiler, *common, "-fPIE", "-pie", *[str(path) for path in candidate_files], *link_libraries, "-Wl,--build-id=none", "-o", str(output)]
        result = _compile(command, project_root)
        attempts.append({**result, "kind": "executable", "file_count": len(candidate_files)})
        if result["returncode"] == 0:
            return {"status": "ok", "kind": "executable", "files": [str(path.relative_to(project_root)) for path in candidate_files], "build": result, "attempts": attempts}
    library = output.with_suffix(".so")
    library_files = non_main or list(files)
    library_sets = [library_files]
    for path in sorted(library_files, key=lambda item: item.stat().st_size, reverse=True)[:3]:
        library_sets.append([path])
    seen_library_sets = set()
    for candidate_files in library_sets:
        key = tuple(str(path) for path in candidate_files)
        if key in seen_library_sets:
            continue
        seen_library_sets.add(key)
        library.parent.mkdir(parents=True, exist_ok=True)
        command = [compiler, *common, "-shared", "-fPIC", *[str(path) for path in candidate_files], *link_libraries, "-Wl,--build-id=none", "-o", str(library)]
        result = _compile(command, project_root)
        attempts.append({**result, "kind": "shared_library", "file_count": len(candidate_files)})
        if result["returncode"] == 0:
            return {"status": "ok", "kind": "shared_library", "files": [str(path.relative_to(project_root)) for path in candidate_files], "build": result, "attempts": attempts}
    return {"status": "failed", "reason": "no controlled link succeeded", "files": [str(path.relative_to(project_root)) for path in files], "attempts": attempts}


def build_sample(
    root: Path,
    source_root: Path,
    project: dict[str, Any],
    variant: BuildVariant,
    archive: dict[str, Any],
    license_info: dict[str, Any],
) -> dict[str, Any]:
    root = Path(root).resolve()
    slug = _slug(project["full_name"])
    sample_id = f"real_{slug}_{variant.compiler}_{variant.optimization}_{'debug' if variant.debug else 'stripped'}"
    sample_dir = root / "samples" / sample_id
    sample_dir.mkdir(parents=True, exist_ok=True)
    output = sample_dir / slug
    build = build_project(source_root, variant.compiler, variant.optimization, variant.debug, output)
    if build["status"] != "ok":
        return {"status": "failed", "sample_id": sample_id, "reason": build.get("reason"), "build": build}
    binary = output if build["kind"] == "executable" else output.with_suffix(".so")
    if not variant.debug and shutil.which("strip"):
        _compile(["strip", "--strip-all", str(binary)], root, 30)
    build_log = sample_dir / "build.log"
    build_log.write_text(
        "\n".join([
            f"project: {project['full_name']}",
            f"commit: {project['commit']}",
            f"license: {project.get('license')}",
            f"compiler: {variant.compiler}",
            f"optimization: {variant.optimization}",
            f"debug: {variant.debug}",
            f"kind: {build['kind']}",
            f"command: {' '.join(build['build']['command'])}",
            "",
        ]),
        encoding="utf-8",
    )
    _sanitize_text(build_log, [(str(source_root), "."), (str(root), ".")])
    bundle = analyze_binary(binary)
    functions, label_paths = _labels(bundle, root, sample_id, sample_dir)
    record = label_paths["record"]
    disassembly_path = sample_dir / "disassembly.txt"
    disassembly_result = subprocess.run(["objdump", "-d", str(binary)], capture_output=True, text=True, timeout=60)
    disassembly_path.write_text(disassembly_result.stdout, encoding="utf-8")
    _sanitize_text(disassembly_path, [(str(source_root), "."), (str(root), ".")])
    symbols_path = sample_dir / "symbols.txt"
    symbols_result = subprocess.run(["readelf", "-sW", str(binary)], capture_output=True, text=True, timeout=60)
    symbols_path.write_text(symbols_result.stdout, encoding="utf-8")
    dwarf_paths = []
    if variant.debug and shutil.which("llvm-dwarfdump"):
        for name, args in (("dwarf_info.txt", ["--debug-info"]), ("dwarf_line.txt", ["--debug-line"])):
            result = subprocess.run(["llvm-dwarfdump", *args, str(binary)], capture_output=True, text=True, timeout=60)
            if result.returncode == 0 and result.stdout:
                target = sample_dir / name
                target.write_text(result.stdout, encoding="utf-8")
                _sanitize_text(target, [(str(source_root), "."), (str(root), "."), (str(binary), binary.name)])
                dwarf_paths.append(target)
    if variant.debug and (sample_dir / "dwarf_info.txt").exists():
        labels = extract_source_labels(record, source_root, (sample_dir / "dwarf_info.txt").read_text(encoding="utf-8", errors="replace"))
        if labels:
            by_id = {str(item.get("function_id")): item for item in record.get("functions", [])}
            for function_id, label in labels.items():
                function = by_id.get(function_id)
                if function is not None:
                    function["source_label_text"] = label["text"]
                    function["source_path"] = label["source_path"]
                    function["source_line"] = label["source_line"]
                    function["source_sha256"] = label["source_sha256"]
            for function in functions:
                label = labels.get(str(function.get("function_id")))
                if label:
                    function["source_text"] = label["text"]
                    function["source_path"] = label["source_path"]
                    function["source_line"] = label["source_line"]
            label_paths["record_path"].write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
            label_paths["functions_path"].write_text(json.dumps({"sample_id": sample_id, "functions": functions, "label_source": "dwarf", "confidence": 1.0}, indent=2) + "\n", encoding="utf-8")
    source_archive = root / "sources" / f"{slug}.tar.gz"
    artifacts = [
        _artifact(root, binary, "application/x-elf"),
        _artifact(root, build_log, "text/plain"),
        _artifact(root, label_paths["record_path"], "application/json"),
        _artifact(root, disassembly_path, "text/plain"),
        _artifact(root, symbols_path, "text/plain"),
        _artifact(root, source_archive, "application/gzip"),
    ]
    for path in dwarf_paths:
        artifacts.append(_artifact(root, path, "text/plain"))
    manifest = {
        "schema_version": "0.1.0",
        "sample_id": sample_id,
        "project_id": f"project_{slug}",
        "source_family": slug,
        "split": project.get("split") or _split_for(f"project_{slug}"),
        "license": {"spdx": project.get("license") or "NOASSERTION", "redistributable": True, "notes": f"Source: {project['full_name']} @ {project['commit']}; license evidence: {license_info.get('path')}"},
        "provenance": {
            "repository": project["full_name"],
            "commit": project["commit"],
            "source_archive_sha256": archive["sha256"],
            "source_paths": build["files"],
            "upstream_license_file_sha256": license_info.get("sha256"),
        },
        "build": {
            "compiler": variant.compiler if variant.compiler in {"gcc", "clang"} else "other",
            "compiler_version": _version(variant.compiler) or "unknown",
            "target": _target(variant.compiler) or "unknown",
            "arch": record.get("program", {}).get("machine", "x86_64"),
            "abi": "SysV-AMD64" if record.get("program", {}).get("machine") == "x86_64" else "unknown",
            "optimization": variant.optimization,
            "flags": [flag.replace(str(source_root), ".").replace(str(root), ".") for flag in build["build"]["command"][1:-3] if flag.startswith("-")],
            "linker": "system",
            "lto": False,
            "pie": build["kind"] == "executable",
            "static": False,
            "stripped": not variant.debug,
            "debug_info": variant.debug,
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
        "artifacts": artifacts,
        "label_manifests": [
            {"kind": "functions", "artifact": _artifact(root, label_paths["functions_path"], "application/json")},
            {"kind": "cfg", "artifact": _artifact(root, label_paths["cfg_path"], "application/json")},
        ],
        "tests": [],
        "quality": {
            "label_source": "dwarf" if variant.debug else "compiler",
            "human_reviewed": False,
            "validation_status": "schema_valid",
            "confidence": 0.55 if variant.debug else 0.35,
            "known_gaps": [
                "No behavioral tests executed.",
                "Function labels are symbol/disassembly candidates; source alignment is not verified.",
                "Build uses a controlled single-command compile, not the project's original build system.",
            ],
        },
        "lineage": {"parent_sample_id": None, "variant_kind": f"{variant.compiler}_{variant.optimization}_{'debug' if variant.debug else 'stripped'}"},
        "mission_relevance": manifest_mission(
            "real_world_binary_source",
            ["binary_representation_masking", "function_boundary_prediction", "cross_compiler_function_retrieval", "c_reconstruction"],
            "Licensed real-world source with exact commit and controlled binaries for binary-intelligence tasks.",
        ),
    }
    return {"status": "ok", "sample_id": sample_id, "manifest": manifest, "bundle": bundle, "record": record, "binary": binary}


def ingest_candidates(
    candidates_path: str | Path,
    out_dir: str | Path,
    max_projects: int = 120,
    variants: list[BuildVariant] | None = None,
    timeout: int = 60,
) -> dict[str, Any]:
    root = Path(out_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    candidates = json.loads(Path(candidates_path).read_text(encoding="utf-8"))
    variants = variants or list(DEFAULT_VARIANTS)
    archives = root / "sources"
    archives.mkdir(parents=True, exist_ok=True)
    existing_catalog = json.loads((root / "catalog.json").read_text(encoding="utf-8")) if (root / "catalog.json").exists() else []
    existing_manifests = [json.loads(line) for line in (root / "manifests.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()] if (root / "manifests.jsonl").exists() else []
    existing_quarantine = [json.loads(line) for line in (root / "quarantine.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()] if (root / "quarantine.jsonl").exists() else []
    catalog = list(existing_catalog)
    manifests = list(existing_manifests)
    quarantine = list(existing_quarantine)
    seen_repositories = {str(item.get("name") or "") for item in catalog}
    seen_archives = {str(item.get("source_archive_sha256") or "") for item in catalog}
    successful_projects = len(catalog)

    def _write_snapshot() -> None:
        manifests.sort(key=lambda item: item["sample_id"])
        with open(root / "manifests.jsonl", "w", encoding="utf-8") as handle:
            handle.writelines(json.dumps(manifest, sort_keys=True) + "\n" for manifest in manifests)
        with open(root / "quarantine.jsonl", "w", encoding="utf-8") as handle:
            handle.writelines(json.dumps(item, sort_keys=True) + "\n" for item in quarantine)
        (root / "catalog.json").write_text(json.dumps(catalog, indent=2) + "\n", encoding="utf-8")
        (root / "dataset_summary.json").write_text(json.dumps(summarize_realworld(root, catalog, manifests, quarantine), indent=2) + "\n", encoding="utf-8")
    for project in candidates:
        if successful_projects >= max_projects:
            break
        full_name = project.get("full_name")
        if not full_name or full_name in seen_repositories:
            continue
        seen_repositories.add(full_name)
        slug = _slug(full_name)
        archive = archives / f"{slug}.tar.gz"
        if not archive.exists():
            try:
                archive_info = download_archive(project, archive, timeout=60)
            except Exception as exc:
                quarantine.append({"project": full_name, "stage": "download", "reason": str(exc)})
                continue
        else:
            archive_info = {"url": None, "path": str(archive), "sha256": sha256_file(archive), "size": archive.stat().st_size}
        if archive_info["sha256"] in seen_archives:
            quarantine.append({"project": full_name, "stage": "ingest", "reason": "duplicate source archive"})
            continue
        seen_archives.add(archive_info["sha256"])
        with tempfile.TemporaryDirectory(prefix="nexus-realworld-") as temporary:
            extracted = Path(temporary)
            try:
                safe_extract_tar(archive, extracted)
            except Exception as exc:
                quarantine.append({"project": full_name, "stage": "extract", "reason": str(exc)})
                continue
            roots = [path for path in extracted.iterdir() if path.is_dir()]
            source_root = roots[0] if len(roots) == 1 else extracted
            license_info = find_license(source_root)
            if not license_info.get("found"):
                quarantine.append({"project": full_name, "stage": "license", "reason": "no license file"})
                continue
            built_samples = []
            failed_builds = []
            for variant in variants:
                result = build_sample(root, source_root, project, variant, archive_info, license_info)
                if result["status"] == "ok":
                    built_samples.append(result)
                else:
                    failed_builds.append(result)
            if not built_samples:
                quarantine.append({"project": full_name, "stage": "build", "reason": "no build succeeded", "attempts": failed_builds[:2]})
                continue
            for sample in built_samples:
                try:
                    import jsonschema
                    schema = json.loads((Path(__file__).resolve().parents[2] / "schemas" / "dataset_manifest.schema.json").read_text(encoding="utf-8"))
                    errors = list(jsonschema.Draft202012Validator(schema).iter_errors(sample["manifest"]))
                except Exception as exc:
                    errors = [exc]
                if errors:
                    quarantine.append({"project": full_name, "stage": "schema", "reason": str(errors[:2]), "sample_id": sample["sample_id"]})
                    continue
                manifests.append(sample["manifest"])
            source_files = discover_c_files(source_root)
            catalog.append({
                "project_id": f"project_{slug}",
                "name": full_name,
                "description": project.get("description"),
                "repository": f"https://github.com/{full_name}",
                "commit": project.get("commit"),
                "license": project.get("license"),
                "license_file": license_info.get("path"),
                "license_file_sha256": license_info.get("sha256"),
                "source_archive_sha256": archive_info["sha256"],
                "source_archive_path": str(archive.relative_to(root)),
                "source_files": len(source_files),
                "source_bytes": sum(path.stat().st_size for path in source_files),
                "category": category_for(project),
                "split": _split_for(f"project_{slug}"),
                "built_variants": len(built_samples),
                "failed_variants": len(failed_builds),
                "source_paths": [str(path.relative_to(source_root)) for path in source_files],
            })
            successful_projects += 1
            _write_snapshot()
    _write_snapshot()
    summary = summarize_realworld(root, catalog, manifests, quarantine)
    return {"status": "completed", "path": str(root), "catalog": str(root / "catalog.json"), "summary": summary, "quarantined": len(quarantine)}


def category_for(project: dict[str, Any]) -> str:
    text = " ".join([str(project.get("description") or ""), " ".join(project.get("topics") or [])]).lower()
    rules = [
        ("compression", ["compress", "zlib", "gzip", "lz", "brotli", "xz", "qoi", "image"]),
        ("networking", ["network", "http", "dns", "tcp", "socket", "vpn", "proxy", "server", "protocol"]),
        ("cryptography", ["crypto", "aes", "sha", "hash", "cipher", "tls", "signature"]),
        ("parsing", ["parser", "json", "yaml", "markdown", "config", "lexer", "compiler", "interpreter"]),
        ("data_structures", ["data structure", "list", "tree", "radix", "hash table", "array", "queue"]),
        ("utilities", ["tool", "utility", "cli", "terminal", "editor", "clock", "file manager"]),
        ("systems", ["kernel", "driver", "memory", "allocator", "thread", "concurrency", "os"]),
        ("algorithms", ["algorithm", "search", "sort", "math", "bignum"]),
    ]
    for category, words in rules:
        if any(word in text for word in words):
            return category
    return "other"


def summarize_realworld(root: Path, catalog: list[dict[str, Any]], manifests: list[dict[str, Any]], quarantine: list[dict[str, Any]]) -> dict[str, Any]:
    by_split: dict[str, int] = {}
    by_compiler: dict[str, int] = {}
    by_optimization: dict[str, int] = {}
    by_license: dict[str, int] = {}
    by_category: dict[str, int] = {}
    projects = set()
    source_files = 0
    source_bytes = 0
    for item in catalog:
        projects.add(item["project_id"])
        by_license[item.get("license") or "unknown"] = by_license.get(item.get("license") or "unknown", 0) + 1
        by_category[item.get("category") or "other"] = by_category.get(item.get("category") or "other", 0) + 1
        source_files += item["source_files"]
        source_bytes += item["source_bytes"]
    for manifest in manifests:
        by_split[manifest["split"]] = by_split.get(manifest["split"], 0) + 1
        by_compiler[manifest["build"]["compiler"]] = by_compiler.get(manifest["build"]["compiler"], 0) + 1
        by_optimization[manifest["build"]["optimization"]] = by_optimization.get(manifest["build"]["optimization"], 0) + 1
    project_splits: dict[str, set[str]] = {}
    for manifest in manifests:
        project_splits.setdefault(manifest["project_id"], set()).add(manifest["split"])
    leakage = [{"project": project, "splits": sorted(splits)} for project, splits in project_splits.items() if len(splits) > 1]
    duplicate_binaries = {}
    seen = {}
    for manifest in manifests:
        digest = manifest["binary"]["sha256"]
        if digest in seen:
            duplicate_binaries[digest] = duplicate_binaries.get(digest, 1) + 1
        seen[digest] = True
    return {
        "schema_version": "0.1.0",
        "created_at": datetime.now(UTC).isoformat(),
        "dataset_id": hashlib.sha256((root / "manifests.jsonl").read_bytes()).hexdigest(),
        "projects": len(projects),
        "samples": len(manifests),
        "source_files": source_files,
        "source_bytes": source_bytes,
        "by_split": by_split,
        "by_compiler": by_compiler,
        "by_optimization": by_optimization,
        "by_license": by_license,
        "by_category": by_category,
        "quarantined": len(quarantine),
        "leakage": leakage,
        "duplicate_binary_hashes": duplicate_binaries,
        "limitations": [
            "Controlled single-command builds, not original project build systems.",
            "Only projects with a detected license file are included.",
            "No behavioral tests are executed.",
            "Source-function alignment is not yet verified.",
        ],
    }


def _source_file_hashes(archive: Path) -> dict[str, str]:
    result = {}
    with tarfile.open(archive, "r:*") as handle:
        for member in handle:
            if not member.isfile() or not member.name.endswith(".c"):
                continue
            if member.size > 2 * 1024 * 1024:
                continue
            source = handle.extractfile(member)
            if source is None:
                continue
            payload = source.read()
            result[member.name] = hashlib.sha256(payload).hexdigest()
    return result


def _project_signature(archive: Path, dimensions: int = 32) -> list[int]:
    tokens = []
    with tarfile.open(archive, "r:*") as handle:
        for member in handle:
            if not member.isfile() or not member.name.endswith(".c") or member.size > 1024 * 1024:
                continue
            source = handle.extractfile(member)
            if source is None:
                continue
            text = source.read().decode("utf-8", errors="replace").lower()
            tokens.extend(re.findall(r"[a-z_][a-z0-9_]*|\d+", text))
            if len(tokens) > 200_000:
                break
    if len(tokens) < 5:
        return [0] * dimensions
    shingles = {" ".join(tokens[index:index + 5]) for index in range(0, len(tokens) - 4, 2)}
    signature = []
    for dimension in range(dimensions):
        signature.append(min(int(hashlib.blake2b(shingle.encode(), digest_size=8, salt=dimension.to_bytes(8, "little")).hexdigest(), 16) for shingle in shingles))
    return signature


def _minhash_jaccard(first: list[int], second: list[int]) -> float:
    if not first or not second:
        return 0.0
    return sum(1 for a, b in zip(first, second) if a == b) / len(first)


def audit_realworld_dataset(path: str | Path) -> dict[str, Any]:
    root = Path(path)
    catalog = json.loads((root / "catalog.json").read_text(encoding="utf-8"))
    manifests = [json.loads(line) for line in (root / "manifests.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    source_hashes: dict[str, set[str]] = {}
    project_signatures = {}
    duplicate_source_files = {}
    file_owners = {}
    for project in catalog:
        archive = root / project["source_archive_path"]
        hashes = _source_file_hashes(archive)
        for relative, digest in hashes.items():
            file_owners.setdefault(digest, []).append((project["project_id"], relative))
            source_hashes.setdefault(project["project_id"], set()).add(digest)
        project_signatures[project["project_id"]] = _project_signature(archive)
    for digest, owners in file_owners.items():
        projects = sorted({owner[0] for owner in owners})
        if len(projects) > 1:
            duplicate_source_files[digest] = owners
    near_duplicates = []
    projects = sorted(project_signatures)
    for index, first in enumerate(projects):
        for second in projects[index + 1:]:
            similarity = _minhash_jaccard(project_signatures[first], project_signatures[second])
            if similarity >= 0.60:
                near_duplicates.append({"a": first, "b": second, "estimated_jaccard": round(similarity, 3)})
    project_splits: dict[str, set[str]] = {}
    binary_hashes = {}
    for manifest in manifests:
        project_splits.setdefault(manifest["project_id"], set()).add(manifest["split"])
        binary_hashes.setdefault(manifest["binary"]["sha256"], []).append(manifest["sample_id"])
    split_leakage = [{"project": project, "splits": sorted(splits)} for project, splits in project_splits.items() if len(splits) > 1]
    debug_functions = set()
    for manifest in manifests:
        if not manifest["build"]["debug_info"]:
            continue
        functions_artifact = next((item["artifact"] for item in manifest["label_manifests"] if item["kind"] == "functions"), None)
        if not functions_artifact:
            continue
        functions = json.loads((root / functions_artifact["path"]).read_text(encoding="utf-8")).get("functions", [])
        for function in functions:
            debug_functions.add((manifest["project_id"], function.get("name")))
    licenses_with_evidence = sum(1 for project in catalog if project.get("license") and project.get("license_file_sha256"))
    artifact_failures = []
    for manifest in manifests:
        for artifact in manifest.get("artifacts", []):
            target = root / artifact["path"]
            if not target.exists() or sha256_file(target) != artifact["sha256"]:
                artifact_failures.append({"sample_id": manifest["sample_id"], "path": artifact["path"]})
    reproducibility = reproducibility_audit(root, sample_limit=20)
    report = {
        "dataset_id": json.loads((root / "dataset_summary.json").read_text(encoding="utf-8")).get("dataset_id"),
        "projects": len(catalog),
        "samples": len(manifests),
        "source_files": sum(project["source_files"] for project in catalog),
        "source_bytes": sum(project["source_bytes"] for project in catalog),
        "unique_debug_function_names": len(debug_functions),
        "licenses_with_evidence": licenses_with_evidence,
        "license_coverage": round(licenses_with_evidence / max(1, len(catalog)), 4),
        "exact_duplicate_source_files_across_projects": len(duplicate_source_files),
        "near_duplicate_projects": near_duplicates,
        "split_leakage": split_leakage,
        "duplicate_binary_hashes": {digest: samples for digest, samples in binary_hashes.items() if len(samples) > 1},
        "artifact_failures": artifact_failures,
        "reproducibility": reproducibility,
        "status": "passed" if not split_leakage and not artifact_failures and licenses_with_evidence == len(catalog) and reproducibility["rate"] == 1.0 else "needs_review",
        "limitations": [
            "Near-duplicate detection uses a 5-token MinHash approximation over source identifiers; it is not a formal clone detector.",
            "Shared standard headers or vendored files can create exact source-file duplicates across projects.",
            "Debug function names are observations from symbols, not verified source-level unique functions.",
        ],
    }
    (root / "quality_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def reproducibility_audit(path: str | Path, sample_limit: int = 20) -> dict[str, Any]:
    root = Path(path)
    catalog = json.loads((root / "catalog.json").read_text(encoding="utf-8"))
    manifests = [json.loads(line) for line in (root / "manifests.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    checks = []
    passed = 0
    for project in catalog[:sample_limit]:
        manifest = next((item for item in manifests if item["project_id"] == project["project_id"] and item["build"]["compiler"] == "gcc" and item["build"]["optimization"] == "O0" and item["build"]["debug_info"]), None)
        if manifest is None:
            continue
        archive = root / project["source_archive_path"]
        with tempfile.TemporaryDirectory(prefix="nexus-repro-") as directory:
            extracted = Path(directory)
            try:
                safe_extract_tar(archive, extracted)
                roots = [item for item in extracted.iterdir() if item.is_dir()]
                source_root = roots[0] if len(roots) == 1 else extracted
                rebuilt = Path(directory) / "rebuilt"
                build = build_project(source_root, "gcc", "O0", True, rebuilt)
                if build["status"] != "ok":
                    checks.append({"project": project["project_id"], "passed": False, "reason": build.get("reason")})
                    continue
                binary = rebuilt if build["kind"] == "executable" else rebuilt.with_suffix(".so")
                expected = manifest["binary"]["sha256"]
                actual = sha256_file(binary)
                ok = expected == actual
                passed += 1 if ok else 0
                checks.append({"project": project["project_id"], "passed": ok, "expected": expected, "actual": actual})
            except Exception as exc:
                checks.append({"project": project["project_id"], "passed": False, "reason": str(exc)})
    return {"checked": len(checks), "passed": passed, "rate": round(passed / max(1, len(checks)), 4), "checks": checks}
