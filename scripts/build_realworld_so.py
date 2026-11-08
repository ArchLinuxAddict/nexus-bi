#!/usr/bin/env python3
"""Build a large real-world shared-object corpus and a frozen function corpus."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import struct
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

V4 = ROOT / "artifacts" / "training_runtime" / "frozen_dataset" / "v4"
V5 = ROOT / "artifacts" / "training_runtime" / "frozen_dataset" / "v5_realworld_so"
DATASET = ROOT / "artifacts" / "dataset" / "realworld_so"
CATALOG = DATASET / "catalog.jsonl"
SUMMARY = DATASET / "summary.json"
SAMPLE_ROOT = DATASET / "samples"
ANALYSIS_ROOT = DATASET / "analysis"

DEFAULT_ROOTS = (
    "/usr/lib",
    "/lib",
    "/usr/local/lib",
    ".venv",
    "artifacts/dataset/realworld",
    "artifacts/dataset/phase2b_pilot",
)

SKIP_NAME_PARTS = {"cmake", "pkgconfig", "node_modules", ".git", "__pycache__"}

_WORKER_VOCAB = None
_WORKER_CONFIG = None
_WORKER_MAX_FUNCTIONS = 80


def _json_hash(payload: dict) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _marker(text: str, name: str) -> str | None:
    marker = f"%{name}%"
    if marker not in text:
        return None
    rest = text.split(marker, 1)[1].split("\n", 1)[1]
    return rest.split("%", 1)[0].strip()


def _pacman_packages() -> tuple[dict[str, dict], dict[str, dict]]:
    by_path: dict[str, dict] = {}
    by_name: dict[str, dict] = {}
    base = Path("/var/lib/pacman/local")
    if not base.exists():
        return by_path, by_name
    for package_dir in base.iterdir():
        if not package_dir.is_dir():
            continue
        desc_path = package_dir / "desc"
        files_path = package_dir / "files"
        if not desc_path.exists() or not files_path.exists():
            continue
        desc = desc_path.read_text(encoding="utf-8", errors="replace")
        name = _marker(desc, "NAME")
        version = _marker(desc, "VERSION")
        if not name or not version:
            continue
        license_text = _marker(desc, "LICENSE") or "NOASSERTION"
        record = {"package": name, "version": version, "license": license_text, "source_kind": "system_package"}
        by_name[name] = record
        files_text = files_path.read_text(encoding="utf-8", errors="replace")
        for line in files_text.splitlines():
            if not line.startswith("usr/") or line.endswith("/"):
                continue
            by_path["/" + line] = record
    return by_path, by_name


def _wheel_packages() -> dict[str, dict]:
    by_path: dict[str, dict] = {}
    site = ROOT / ".venv" / "lib" / "python3.13" / "site-packages"
    if not site.exists():
        return by_path
    for metadata in site.glob("*.dist-info/METADATA"):
        text = metadata.read_text(encoding="utf-8", errors="replace")
        name = ""
        version = ""
        license_value = ""
        for line in text.splitlines():
            if line.startswith("Name: "):
                name = line[6:].strip()
            elif line.startswith("Version: "):
                version = line[9:].strip()
            elif line.startswith("License: ") and not license_value:
                license_value = line[9:].strip()
        record = {"package": name or metadata.parent.stem, "version": version, "license": license_value or "NOASSERTION", "source_kind": "python_wheel"}
        record_path = metadata.parent / "RECORD"
        if record_path.exists():
            for line in record_path.read_text(encoding="utf-8", errors="replace").splitlines():
                relative = line.split(",", 1)[0]
                if not relative or relative.startswith(".."):
                    continue
                target = site / relative
                if target.exists():
                    by_path[str(target.resolve())] = record
        for candidate in site.glob(f"{name.replace('-', '_')}*"):
            if candidate.is_dir():
                for path in candidate.rglob("*.so*"):
                    if path.is_file():
                        by_path[str(path.resolve())] = record
    return by_path


def _dataset_packages() -> dict[str, dict]:
    by_sha: dict[str, dict] = {}
    for root in (ROOT / "artifacts" / "dataset" / "realworld", ROOT / "artifacts" / "dataset" / "phase2b_pilot", ROOT / "artifacts" / "dataset" / "expanded"):
        manifest_path = root / "manifests.jsonl"
        if not manifest_path.exists():
            continue
        for line in manifest_path.read_text(encoding="utf-8", errors="replace").splitlines():
            if not line.strip():
                continue
            try:
                manifest = json.loads(line)
            except json.JSONDecodeError:
                continue
            sha = str((manifest.get("binary") or {}).get("sha256") or "")
            if not sha:
                continue
            by_sha[sha] = {
                "package": manifest.get("source_family") or manifest.get("project_id") or "dataset",
                "version": (manifest.get("build") or {}).get("variant_id") or "",
                "license": (manifest.get("license") or {}).get("spdx") or "NOASSERTION",
                "source_kind": "licensed_dataset",
                "project_id": manifest.get("project_id"),
                "source_family": manifest.get("source_family"),
                "repository": (manifest.get("provenance") or {}).get("repository"),
                "build": manifest.get("build") or {},
            }
    return by_sha


def _elf_header(path: Path) -> dict | None:
    try:
        with path.open("rb") as handle:
            header = handle.read(64)
        if len(header) < 20 or header[:4] != b"\x7fELF":
            return None
        elf_class = {1: "ELF32", 2: "ELF64"}.get(header[4], "unknown")
        endian = {1: "little", 2: "big"}.get(header[5], "unknown")
        e_type, e_machine = struct.unpack_from("<HH" if endian == "little" else ">HH", header, 16)
        return {"elf_class": elf_class, "endianness": endian, "e_type": e_type, "e_machine": e_machine}
    except OSError:
        return None


def _candidate_files(roots: list[str]) -> list[Path]:
    seen_inodes: dict[tuple[int, int], Path] = {}
    for raw_root in roots:
        root = Path(raw_root)
        if not root.is_absolute():
            root = ROOT / root
        if not root.exists():
            continue
        for current, directories, files in os.walk(root, followlinks=False):
            directories[:] = [name for name in directories if name not in SKIP_NAME_PARTS or "site-packages" in current]
            for name in files:
                if ".so" not in name:
                    continue
                path = Path(current) / name
                try:
                    stat = path.stat()
                except OSError:
                    continue
                if stat.st_size < 1024:
                    continue
                key = (stat.st_dev, stat.st_ino)
                previous = seen_inodes.get(key)
                if previous is None or len(str(path)) < len(str(previous)):
                    seen_inodes[key] = path
    return sorted(seen_inodes.values())


def _package_for(path: Path, pacman_paths: dict[str, dict], wheels: dict[str, dict], datasets: dict[str, dict], sha: str) -> dict:
    if sha in datasets:
        return datasets[sha]
    resolved = str(path.resolve())
    if resolved in wheels:
        return wheels[resolved]
    if str(path) in pacman_paths:
        return pacman_paths[str(path)]
    if resolved in pacman_paths:
        return pacman_paths[resolved]
    try:
        owner = __import__("subprocess").run(["pacman", "-Qo", str(path)], capture_output=True, text=True, timeout=5)
        if owner.returncode == 0 and " is owned by " in owner.stdout:
            package = owner.stdout.split(" is owned by ", 1)[1].split(" ", 1)[0]
            return {"package": package.rsplit("-", 2)[0], "version": package.rsplit("-", 2)[1] if package.count("-") >= 2 else "", "license": "NOASSERTION", "source_kind": "system_package"}
    except Exception:
        pass
    return {"package": path.parent.name or "unknown", "version": "", "license": "NOASSERTION", "source_kind": "local_file"}


def _split_for(key: str) -> str:
    bucket = int(hashlib.sha256(key.encode("utf-8")).hexdigest()[:8], 16) % 10
    if bucket == 0:
        return "extended_test"
    if bucket == 1:
        return "validation"
    return "train"


def catalog(args) -> dict:
    from nexus.elf.parser import parse_elf

    pacman_paths, _ = _pacman_packages()
    wheels = _wheel_packages()
    datasets = _dataset_packages()
    paths = _candidate_files(args.root)
    print(f"candidates={len(paths)}", flush=True)
    DATASET.mkdir(parents=True, exist_ok=True)
    by_sha: dict[str, dict] = {}
    skipped = Counter()
    started = time.time()
    for index, path in enumerate(paths, 1):
        header = _elf_header(path)
        if header is None or header["e_type"] != 3:
            skipped["not_shared_object"] += 1
            continue
        try:
            size = path.stat().st_size
            if size > args.max_catalog_size_mb * 1024 * 1024:
                skipped["over_catalog_size"] += 1
                continue
            elf = parse_elf(path)
            sha = elf.sha256
        except Exception:
            skipped["elf_parse_failed"] += 1
            continue
        if sha in by_sha:
            by_sha[sha]["duplicate_paths"] += 1
            continue
        package = _package_for(path, pacman_paths, wheels, datasets, sha)
        package_key = str(package.get("project_id") or package.get("package") or sha)
        entry = {
            "schema_version": "nexus.realworld-so-catalog/v1",
            "binary_id": f"rwso_{sha[:16]}",
            "sha256": sha,
            "size": size,
            "path": str(path),
            "resolved_path": str(path.resolve()),
            "source_kind": package.get("source_kind", "local_file"),
            "package": package.get("package", "unknown"),
            "package_version": package.get("version", ""),
            "package_key": package_key,
            "license": package.get("license", "NOASSERTION"),
            "repository": package.get("repository"),
            "arch": elf.header.machine_name,
            "elf_class": elf.header.elf_class,
            "endianness": elf.header.endianness,
            "stripped": elf.is_stripped,
            "shared_object": elf.is_shared_object,
            "function_symbols": len(elf.function_symbols()),
            "dependencies": len(elf.dependencies),
            "build_id": elf.build_id,
            "split": _split_for(package_key),
            "duplicate_paths": 0,
            "extraction_status": "not_attempted",
        }
        by_sha[sha] = entry
        if index % 100 == 0:
            print(f"cataloged={len(by_sha)} scanned={index}/{len(paths)} elapsed={time.time()-started:.1f}s", flush=True)
    entries = sorted(by_sha.values(), key=lambda item: item["sha256"])
    CATALOG.write_text("\n".join(json.dumps(item, sort_keys=True) for item in entries) + "\n", encoding="utf-8")
    summary = {
        "schema_version": "nexus.realworld-so-summary/v1",
        "created_at": datetime.now(UTC).isoformat(),
        "roots": [str(root) for root in args.root],
        "candidates": len(paths),
        "unique_shared_objects": len(entries),
        "total_bytes": sum(item["size"] for item in entries),
        "by_source_kind": dict(Counter(item["source_kind"] for item in entries)),
        "by_arch": dict(Counter(item["arch"] for item in entries)),
        "by_split": dict(Counter(item["split"] for item in entries)),
        "by_license": dict(Counter(item["license"] for item in entries).most_common(30)),
        "stripped": sum(1 for item in entries if item["stripped"]),
        "unstripped": sum(1 for item in entries if not item["stripped"]),
        "duplicate_paths": sum(item["duplicate_paths"] for item in entries),
        "skipped": dict(skipped),
        "catalog_sha256": _file_hash(CATALOG),
        "limitations": [
            "System-package licenses are read from pacman metadata; redistribution and legal compatibility are not independently verified.",
            "A shared object is a build artifact, not source-level ground truth.",
            "The catalog records binary identity; function-level samples are extracted separately.",
        ],
    }
    SUMMARY.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: summary[key] for key in ("unique_shared_objects", "total_bytes", "by_source_kind", "by_arch", "by_split", "stripped", "unstripped", "skipped")}, indent=2), flush=True)
    return summary


def _fast_record(path: Path):
    from datetime import UTC, datetime

    from nexus.architectures import adapter_for_elf
    from nexus.core.errors import CapabilityUnavailable
    from nexus.disassembly.objdump import disassemble_elf
    from nexus.elf.parser import parse_elf
    from nexus.program.analysis import recover_functions
    from nexus.program.bundle import AnalysisBundle
    from nexus.program.multiarch import analyze_with_llvm
    from nexus.program.record import build_program_record

    started = datetime.now(UTC).isoformat()
    elf = parse_elf(path)
    try:
        adapter = adapter_for_elf(elf)
    except CapabilityUnavailable:
        adapter = None
    warnings: list[str] = []
    limitations: list[str] = []
    if adapter is not None and adapter.name == "x86_64" and elf.header.e_type != 1:
        disassembly = disassemble_elf(elf)
        functions, callgraph = recover_functions(elf, disassembly)
    elif adapter is not None and adapter.instruction_decoding:
        disassembly, functions, callgraph, warnings, limitations = analyze_with_llvm(path, elf)
    else:
        disassembly = None
        functions = []
        callgraph = __import__("nexus.program.model", fromlist=["CallGraph"]).CallGraph(limitations=["No architecture adapter."])
    bundle = AnalysisBundle(
        path=str(path),
        elf=elf,
        disassembly=disassembly,
        functions=functions,
        callgraph=callgraph,
        strings=[],
        dataflows={},
        type_candidates={},
        purposes={},
        security_findings=[],
        jni_report={},
        started_at=started,
        finished_at=datetime.now(UTC).isoformat(),
        warnings=warnings,
        limitations=limitations,
        signatures={},
    )
    return build_program_record(bundle), bundle


def _worker_init(max_functions: int) -> None:
    global _WORKER_VOCAB, _WORKER_CONFIG, _WORKER_MAX_FUNCTIONS
    from nexus.foundation.config import config_for_vocabulary, profile_config
    from nexus.foundation.vocab import Vocabulary

    _WORKER_VOCAB = Vocabulary.load(V4 / "vocab.json")
    _WORKER_CONFIG = config_for_vocabulary(profile_config("compact"), _WORKER_VOCAB)
    _WORKER_MAX_FUNCTIONS = max_functions


def _worker_extract(item: dict, output_path: str, root: str) -> dict:
    from nexus.foundation.representation import RepresentationOptions, encode_function_sample

    started = time.time()
    path = Path(item["path"])
    try:
        record, bundle = _fast_record(path)
        functions = [function for function in record.get("functions", []) if function.get("blocks")]
        def _start(function: dict) -> int:
            try:
                return int(((function.get("blocks") or [{}])[0].get("range") or {}).get("start_va", "0x0"), 16)
            except (TypeError, ValueError):
                return 0

        functions.sort(key=_start)
        if len(functions) > _WORKER_MAX_FUNCTIONS:
            indexes = sorted({round(position * (len(functions) - 1) / max(1, _WORKER_MAX_FUNCTIONS - 1)) for position in range(_WORKER_MAX_FUNCTIONS)})
            functions = [functions[position] for position in indexes]
        options = RepresentationOptions(include_source_target=False)
        samples = []
        for function in functions:
            function_id = str(function.get("function_id") or "")
            identity = {
                "sample_id": f"rwso_{item['sha256'][:16]}_{function_id or len(samples)}",
                "function_id": function_id,
                "project_id": f"rwso:{item['package_key']}",
                "dataset_id": "realworld_so_v1",
                "split": item["split"],
                "binary_sha256": item["sha256"],
                "source_family": item["package_key"],
                "match_group": None,
                "binary_path": str(path),
                "record_path": None,
                "language": "c",
                "build": {"arch": item["arch"], "stripped": item["stripped"], "debug_info": not item["stripped"]},
            }
            sample = encode_function_sample(record, function, _WORKER_VOCAB, _WORKER_CONFIG, identity=identity, options=options)
            if sample is None:
                continue
            sample["labels"].update({
                "dataset_id": "realworld_so_v1",
                "sample_id": item["binary_id"],
                "project_id": f"rwso:{item['package_key']}",
                "split": item["split"],
                "license": item["license"],
                "package": item["package"],
                "package_version": item["package_version"],
                "source_kind": item["source_kind"],
                "arch": item["arch"],
                "stripped": item["stripped"],
                "source_available": False,
                "binary_path": str(path),
            })
            samples.append(sample)
        target = Path(output_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(".tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            for sample in samples:
                handle.write(json.dumps(sample, sort_keys=True, separators=(",", ":"), default=str) + "\n")
        os.replace(temporary, target)
        result = {
            "status": "ok" if samples else "no_samples",
            "sha256": item["sha256"],
            "samples": len(samples),
            "functions": len(functions),
            "instructions": len(record.get("instructions", [])),
            "warnings": len(bundle.warnings),
            "limitations": len(bundle.limitations),
            "elapsed": round(time.time() - started, 3),
            "output": str(target),
        }
    except Exception as exc:
        result = {
            "status": "error",
            "sha256": item["sha256"],
            "samples": 0,
            "functions": 0,
            "instructions": 0,
            "error": f"{type(exc).__name__}: {exc}",
            "elapsed": round(time.time() - started, 3),
        }
    analysis_path = Path(root) / f"{item['sha256']}.json"
    analysis_path.parent.mkdir(parents=True, exist_ok=True)
    analysis_path.write_text(json.dumps({**item, "extraction": result}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def extract(args) -> dict:
    if not CATALOG.exists():
        raise SystemExit("catalog is missing; run the catalog step first")
    entries = [json.loads(line) for line in CATALOG.read_text(encoding="utf-8").splitlines() if line.strip()]
    selected = [
        item for item in entries
        if item["size"] <= args.max_size_mb * 1024 * 1024 and item.get("shared_object", True)
    ]
    if args.order == "size":
        selected.sort(key=lambda item: (item["size"], item["sha256"]))
    if args.max_binaries:
        selected = selected[: args.max_binaries]
    SAMPLE_ROOT.mkdir(parents=True, exist_ok=True)
    ANALYSIS_ROOT.mkdir(parents=True, exist_ok=True)
    pending = []
    for item in selected:
        target = SAMPLE_ROOT / item["sha256"][:2] / f"{item['sha256']}.jsonl"
        if target.exists() and target.stat().st_size > 0 and not args.force:
            continue
        pending.append((item, target))
    print(f"selected={len(selected)} pending={len(pending)} workers={args.workers}", flush=True)
    results = []
    started = time.time()
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_worker_init, initargs=(args.max_functions,)) as pool:
        futures = {
            pool.submit(_worker_extract, item, str(target), str(ANALYSIS_ROOT)): item
            for item, target in pending
        }
        for index, future in enumerate(as_completed(futures), 1):
            result = future.result()
            results.append(result)
            if index % 25 == 0 or result["status"] != "ok":
                done = sum(entry["samples"] for entry in results)
                print(f"processed={index}/{len(pending)} samples={done} errors={sum(1 for entry in results if entry['status']=='error')} elapsed={time.time()-started:.1f}s", flush=True)
    existing = []
    for item in selected:
        target = SAMPLE_ROOT / item["sha256"][:2] / f"{item['sha256']}.jsonl"
        if target.exists() and target.stat().st_size > 0:
            existing.append(target)
    total_samples = 0
    total_bytes = 0
    for path in existing:
        total_bytes += path.stat().st_size
        with path.open("rb") as handle:
            total_samples += sum(1 for _ in handle)
    summary = {
        "schema_version": "nexus.realworld-so-extraction/v1",
        "created_at": datetime.now(UTC).isoformat(),
        "selected_binaries": len(selected),
        "processed_now": len(results),
        "sample_files": len(existing),
        "samples": total_samples,
        "sample_bytes": total_bytes,
        "errors": sum(1 for entry in results if entry["status"] == "error"),
        "no_samples": sum(1 for entry in results if entry["status"] == "no_samples"),
        "elapsed_seconds": round(time.time() - started, 3),
        "max_functions_per_binary": args.max_functions,
        "max_size_mb": args.max_size_mb,
        "results": results,
    }
    (DATASET / "extraction_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: summary[key] for key in ("selected_binaries", "processed_now", "sample_files", "samples", "sample_bytes", "errors", "no_samples", "elapsed_seconds")}, indent=2), flush=True)
    return summary


def _minimal_row(sample: dict) -> str:
    return json.dumps({
        "sample_id": sample.get("sample_id"),
        "dataset_id": sample.get("dataset_id"),
        "project_id": sample.get("project_id"),
        "split": sample.get("split"),
        "binary_sha256": sample.get("binary_sha256"),
        "arch": (sample.get("labels") or {}).get("arch"),
        "stripped": (sample.get("labels") or {}).get("stripped"),
    }, sort_keys=True, separators=(",", ":"))


def freeze(args) -> dict:
    from nexus.data.sample_store import SampleIndexRecord, SampleStore, SampleStoreWriter
    from nexus.data.store_assembly import _index_record
    from nexus.foundation.vocab import Vocabulary

    if V5.exists() and not args.force:
        raise SystemExit(f"output already exists: {V5}; pass --force to rebuild")
    if V5.exists():
        shutil.rmtree(V5)
    v4 = SampleStore.load(V4 / "samples")
    vocab = Vocabulary.load(V4 / "vocab.json")
    v4_binary_shas = {record.binary_sha256 for record in v4.index.values() if record.binary_sha256}
    entries = [json.loads(line) for line in CATALOG.read_text(encoding="utf-8").splitlines() if line.strip()]
    new_entries = [item for item in entries if item["sha256"] not in v4_binary_shas and item["size"] <= args.max_size_mb * 1024 * 1024]
    split_rows: dict[str, list[str]] = {"train": [], "validation": [], "extended_test": [], "quarantine": []}
    store_root = V5 / "samples"
    writer = SampleStoreWriter(store_root)
    counts = Counter()
    for split in ("train", "validation", "test", "extended_test", "quarantine"):
        for sample_id in v4.by_split.get(split, []):
            sample = v4.get(sample_id)
            writer.append(sample, split, SampleIndexRecord.from_dict(v4.index[sample_id].to_dict()))
            counts[split] += 1
            if split in split_rows:
                split_rows[split].append(_minimal_row(sample))
    new_counts = Counter()
    new_project_ids: set[str] = set()
    new_licenses = Counter()
    new_source_kinds = Counter()
    new_arch = Counter()
    new_language = Counter()
    sample_files = 0
    for item in new_entries:
        path = SAMPLE_ROOT / item["sha256"][:2] / f"{item['sha256']}.jsonl"
        if not path.exists() or path.stat().st_size == 0:
            continue
        sample_files += 1
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                sample = json.loads(line)
                split = str(sample.get("split") or item["split"])
                if split == "test":
                    split = "extended_test"
                if split not in split_rows:
                    split = "train"
                writer.append(sample, split, _index_record(sample, split))
                counts[split] += 1
                new_counts[split] += 1
                split_rows[split].append(_minimal_row(sample))
                new_project_ids.add(str(sample.get("project_id") or ""))
                new_licenses[str((sample.get("labels") or {}).get("license") or "NOASSERTION")] += 1
                new_source_kinds[str((sample.get("labels") or {}).get("source_kind") or "unknown")] += 1
                new_arch[str((sample.get("labels") or {}).get("arch") or "unknown")] += 1
                new_language[str((sample.get("labels") or {}).get("language") or "c")] += 1
    created = datetime.now(UTC).isoformat()
    test_file = V5 / "test.jsonl"
    test_file.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(V4 / "test.jsonl", test_file)
    shutil.copyfile(V4 / "protected_test.json", V5 / "protected_test.json")
    shutil.copyfile(V4 / "vocab.json", V5 / "vocab.json")
    for split, rows in split_rows.items():
        (V5 / f"{split}.jsonl").write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")
    store_manifest = writer.finalize({
        "created_at": created,
        "frozen_id": f"v5-realworld-so-{created[:10]}",
        "vocab_sha256": vocab.digest(),
        "protected_test_file_sha256": _file_hash(test_file),
        "statistics": {
            "base_corpus": "v4",
            "realworld_so_binaries": len(new_entries),
            "realworld_so_sample_files": sample_files,
            "realworld_so_projects": len(new_project_ids),
            "realworld_so_licenses": dict(new_licenses),
            "realworld_so_source_kinds": dict(new_source_kinds),
        },
        "limitations": [
            "Real-world shared objects are unlabeled for source reconstruction.",
            "System-package license metadata is recorded but not legally re-verified.",
            "Real-world samples are self-supervised binary-only samples.",
        ],
    })
    v4_manifest = json.loads((V4 / "manifest.json").read_text(encoding="utf-8"))
    manifest = dict(v4_manifest)
    manifest.update({
        "created_at": created,
        "frozen_id": store_manifest["frozen_id"],
        "counts": dict(counts),
        "store": {
            "index_sha256": store_manifest["index_sha256"],
            "manifest_sha256": _file_hash(store_root / "manifest.json"),
            "store_sha256": store_manifest["store_sha256"],
        },
        "vocab_sha256": vocab.digest(),
        "protected_test_file_sha256": _file_hash(test_file),
        "statistics": {
            **(v4_manifest.get("statistics") or {}),
            "by_architecture": {
                **((v4_manifest.get("statistics") or {}).get("by_architecture") or {}),
                **{key: int(((v4_manifest.get("statistics") or {}).get("by_architecture") or {}).get(key, 0)) + value for key, value in new_arch.items()},
            },
            "by_binary_format": {
                **((v4_manifest.get("statistics") or {}).get("by_binary_format") or {}),
                "ELF": int(((v4_manifest.get("statistics") or {}).get("by_binary_format") or {}).get("ELF", 0)) + sum(new_counts.values()),
            },
            "by_dataset": {
                **((v4_manifest.get("statistics") or {}).get("by_dataset") or {}),
                "realworld_so_v1": sum(new_counts.values()),
            },
            "by_language": {
                **((v4_manifest.get("statistics") or {}).get("by_language") or {}),
                **{key: int(((v4_manifest.get("statistics") or {}).get("by_language") or {}).get(key, 0)) + value for key, value in new_language.items()},
            },
            "realworld_so": {
                "binaries": sample_files,
                "sample_files": sample_files,
                "samples": dict(new_counts),
                "projects": len(new_project_ids),
                "licenses": dict(new_licenses),
                "source_kinds": dict(new_source_kinds),
            },
        },
        "limitations": [
            *list(v4_manifest.get("limitations") or []),
            "Real-world shared-object samples are binary-only and unlabeled for source reconstruction.",
            "System-package license metadata is recorded but not legally re-verified.",
        ],
    })
    (V5 / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary = {
        "schema_version": "nexus.realworld-so-freeze/v1",
        "created_at": created,
        "output": str(V5),
        "counts": dict(counts),
        "base_counts": dict(v4.manifest.get("counts") or {}),
        "new_counts": dict(new_counts),
        "realworld_binaries": sample_files,
        "realworld_sample_files": sample_files,
        "realworld_projects": len(new_project_ids),
        "realworld_licenses": dict(new_licenses),
        "realworld_source_kinds": dict(new_source_kinds),
        "store_sha256": store_manifest["store_sha256"],
        "manifest_sha256": _file_hash(V5 / "manifest.json"),
        "vocab_sha256": vocab.digest(),
        "protected_test_sha256": _file_hash(test_file),
    }
    (DATASET / "freeze_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    dataset_summary = {
        "schema_version": "nexus.realworld-so-dataset/v1",
        "dataset_id": "realworld-so-v1",
        "canonical_id": "realworld-so-v1",
        "created_at": created,
        "path": str(DATASET),
        "catalog_sha256": _file_hash(CATALOG),
        "catalog_binaries": len(entries),
        "extracted_binaries": sample_files,
        "frozen_corpus": str(V5),
        "counts": dict(new_counts),
        "total_samples": sum(new_counts.values()),
        "licenses": dict(new_licenses),
        "source_kinds": dict(new_source_kinds),
        "limitations": [
            "Real-world shared objects are binary-only and unlabeled for source reconstruction.",
            "System-package license metadata is recorded but not legally re-verified.",
        ],
    }
    (DATASET / "dataset_summary.json").write_text(json.dumps(dataset_summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: summary[key] for key in ("output", "counts", "new_counts", "realworld_binaries", "realworld_sample_files", "store_sha256")}, indent=2), flush=True)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a large real-world .so corpus")
    sub = parser.add_subparsers(dest="command", required=True)
    catalog_parser = sub.add_parser("catalog")
    catalog_parser.add_argument("--root", action="append", default=list(DEFAULT_ROOTS))
    catalog_parser.add_argument("--max-catalog-size-mb", type=int, default=512)
    catalog_parser.set_defaults(command="catalog")
    extract_parser = sub.add_parser("extract")
    extract_parser.add_argument("--workers", type=int, default=6)
    extract_parser.add_argument("--max-size-mb", type=int, default=25)
    extract_parser.add_argument("--max-functions", type=int, default=80)
    extract_parser.add_argument("--max-binaries", type=int, default=0)
    extract_parser.add_argument("--order", choices=["size", "sha"], default="size")
    extract_parser.add_argument("--force", action="store_true")
    extract_parser.set_defaults(command="extract")
    args = parser.parse_args()
    if args.command == "catalog":
        catalog(args)
    elif args.command == "extract":
        extract(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
