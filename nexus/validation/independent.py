import json
import random
import re
import shutil
import subprocess
import tempfile
import time
from collections import Counter
from pathlib import Path
from typing import Any

from nexus.core.hashing import sha256_file
from nexus.elf.parser import parse_elf
from nexus.program.bundle import analyze_binary
from nexus.reconstruction.c.generator import reconstruct_c
from nexus.verification.compiler import compile_c_project


def _run(command: list[str], cwd: str | Path | None = None, timeout: int = 120) -> subprocess.CompletedProcess:
    return subprocess.run(command, cwd=cwd, capture_output=True, text=True, timeout=timeout)


def _readelf_text(path: Path, *args: str) -> str:
    if shutil.which("readelf") is None:
        return ""
    result = _run(["readelf", *args, str(path)], timeout=60)
    return result.stdout


def _parse_readelf_file_header(path: Path) -> dict[str, Any]:
    text = _readelf_text(path, "-h")
    values: dict[str, Any] = {}
    for line in text.splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        values[key.strip()] = value.strip()
    def number(key: str) -> int | None:
        raw = values.get(key, "")
        match = re.search(r"0x[0-9a-fA-F]+|\d+", raw)
        return int(match.group(0), 0) if match else None
    return {
        "class": values.get("Class"),
        "data": values.get("Data"),
        "type": values.get("Type"),
        "machine": values.get("Machine"),
        "entry": number("Entry point address"),
        "phoff": number("Start of program headers"),
        "shoff": number("Start of section headers"),
        "phnum": number("Number of program headers"),
        "shnum": number("Number of section headers"),
        "build_id": None,
    }


def _parse_readelf_sections(path: Path) -> list[str]:
    text = _readelf_text(path, "-SW")
    names = []
    for line in text.splitlines():
        match = re.match(r"^\s*\[\s*\d+\]\s+(\S*)", line)
        if match:
            name = match.group(1)
            if name.startswith(".") or name == "":
                names.append(name)
    return names


def _parse_readelf_functions(path: Path) -> dict[int, dict[str, Any]]:
    text = _readelf_text(path, "-sW")
    functions: dict[int, dict[str, Any]] = {}
    for line in text.splitlines():
        parts = line.split()
        if len(parts) < 8 or parts[3] != "FUNC":
            continue
        try:
            value = int(parts[1], 16)
            size = int(parts[2], 0)
        except ValueError:
            continue
        if value == 0:
            continue
        name = parts[7].split("@")[0]
        functions[value] = {"name": name, "size": size}
    return functions


def _parse_readelf_relocations(path: Path) -> int:
    text = _readelf_text(path, "-rW")
    return sum(1 for line in text.splitlines() if re.match(r"^[0-9a-fA-F]+", line.strip()) and "R_" in line)


def _parse_readelf_dependencies(path: Path) -> list[str]:
    text = _readelf_text(path, "-dW")
    dependencies = []
    for line in text.splitlines():
        match = re.search(r"\(NEEDED\).*Shared library: \[([^\]]+)\]", line)
        if match:
            dependencies.append(match.group(1))
    return dependencies


def _parse_readelf_build_id(path: Path) -> str | None:
    text = _readelf_text(path, "-n")
    match = re.search(r"Build ID:\s*([0-9a-fA-F]+)", text)
    return match.group(1).lower() if match else None


def _llvm_disassembly(path: Path, section: str = ".text") -> dict[int, dict[str, str]]:
    if shutil.which("llvm-objdump") is None:
        return {}
    result = _run(["llvm-objdump", "-d", f"--section={section}", str(path)], timeout=120)
    parsed: dict[int, dict[str, str]] = {}
    for line in result.stdout.splitlines():
        match = re.match(r"^\s*([0-9a-fA-F]+):\s+((?:[0-9a-fA-F]{2}\s+)+)\s*(.*)$", line)
        if not match:
            continue
        address = int(match.group(1), 16)
        raw = "".join(match.group(2).split())
        text = match.group(3).split("#", 1)[0].strip()
        mnemonic = text.split()[0].lower() if text else ""
        parsed[address] = {"bytes": raw.lower(), "mnemonic": mnemonic, "text": text}
    return parsed


def _normalize_mnemonic(mnemonic: str) -> str:
    value = mnemonic.lower()
    if value.endswith("q") and value[:-1] in {"mov", "add", "sub", "xor", "and", "or", "cmp", "test", "call", "ret", "push", "pop", "lea", "inc", "dec", "imul", "shl", "shr", "sar"}:
        return value[:-1]
    if value in {"retq", "retl"}:
        return "ret"
    return value


def compare_elf(path: str | Path) -> dict[str, Any]:
    file_path = Path(path)
    elf = parse_elf(file_path)
    readelf = _parse_readelf_file_header(file_path)
    sections = _parse_readelf_sections(file_path)
    functions = _parse_readelf_functions(file_path)
    checks = {
        "class": readelf["class"] == elf.header.elf_class,
        "entry": readelf["entry"] == elf.header.entry,
        "phoff": readelf["phoff"] == elf.header.phoff,
        "shoff": readelf["shoff"] == elf.header.shoff,
        "phnum": readelf["phnum"] == len(elf.program_headers),
        "shnum": readelf["shnum"] == len(elf.sections),
        "section_names": sorted(name for name in sections if name) == sorted(section.name for section in elf.sections if section.name),
        "function_symbols": sorted(address for address, item in functions.items() if item["size"] > 0) == sorted(symbol.value for symbol in elf.function_symbols()),
        "relocation_count": _parse_readelf_relocations(file_path) == len(elf.relocations),
        "dependencies": sorted(_parse_readelf_dependencies(file_path)) == sorted(item["name"] for item in elf.dependencies if item.get("kind") == "needed"),
        "build_id": (_parse_readelf_build_id(file_path) or None) == (elf.build_id.lower() if elf.build_id else None),
    }
    return {
        "path": str(file_path),
        "sha256": elf.sha256,
        "checks": checks,
        "passed": sum(1 for value in checks.values() if value),
        "total": len(checks),
        "mismatches": [name for name, value in checks.items() if not value],
        "readelf": readelf,
        "parser": {
            "class": elf.header.elf_class,
            "entry": elf.header.entry,
            "phoff": elf.header.phoff,
            "shoff": elf.header.shoff,
            "phnum": len(elf.program_headers),
            "shnum": len(elf.sections),
            "function_symbols": len(functions),
            "relocations": len(elf.relocations),
            "dependencies": len(elf.dependencies),
        },
    }


def compare_disassembly(path: str | Path) -> dict[str, Any]:
    file_path = Path(path)
    bundle = analyze_binary(file_path)
    if bundle.disassembly is None:
        return {"path": str(file_path), "status": "unsupported"}
    llvm = _llvm_disassembly(file_path)
    ours = {instruction.address: instruction for instruction in bundle.disassembly.instructions if instruction.section == ".text"}
    if not llvm and ours:
        return {
            "path": str(file_path),
            "status": "llvm_unavailable",
            "reason": "llvm-objdump produced no decodable instructions for this stripped binary",
            "ours": len(ours),
            "llvm": 0,
        }
    common = sorted(set(ours) & set(llvm))
    byte_matches = sum(1 for address in common if ours[address].bytes_hex.lower() == llvm[address]["bytes"])
    mnemonic_matches = sum(1 for address in common if _normalize_mnemonic(ours[address].mnemonic) == _normalize_mnemonic(llvm[address]["mnemonic"]))
    return {
        "path": str(file_path),
        "sha256": bundle.elf.sha256,
        "ours": len(ours),
        "llvm": len(llvm),
        "common": len(common),
        "address_jaccard": round(len(common) / max(1, len(set(ours) | set(llvm))), 4),
        "byte_agreement": round(byte_matches / max(1, len(common)), 4),
        "mnemonic_agreement": round(mnemonic_matches / max(1, len(common)), 4),
        "only_ours": len(set(ours) - set(llvm)),
        "only_llvm": len(set(llvm) - set(ours)),
    }


def function_accuracy(path: str | Path) -> dict[str, Any]:
    file_path = Path(path)
    bundle = analyze_binary(file_path)
    ground_truth = _parse_readelf_functions(file_path)
    predicted = {function.entry: function for function in bundle.functions}
    predicted_conf50 = {entry: function for entry, function in predicted.items() if function.confidence >= 0.5}
    matched = sorted(set(ground_truth) & set(predicted))
    matched_conf50 = sorted(set(ground_truth) & set(predicted_conf50))
    size_checks = []
    for address in matched:
        expected = ground_truth[address]["size"]
        actual = predicted[address].size
        if expected:
            size_checks.append({"address": f"0x{address:x}", "expected": expected, "actual": actual, "within_16": abs(actual - expected) <= 16, "within_10pct": abs(actual - expected) <= max(16, 0.10 * expected)})
    return {
        "path": str(file_path),
        "sha256": bundle.elf.sha256,
        "stripped": bundle.elf.is_stripped,
        "ground_truth_complete": not bundle.elf.is_stripped,
        "ground_truth_functions": len(ground_truth),
        "predicted_functions": len(predicted),
        "predicted_functions_confidence_ge_0.5": len(predicted_conf50),
        "matched_entries": len(matched),
        "matched_entries_confidence_ge_0.5": len(matched_conf50),
        "entry_precision": round(len(matched) / max(1, len(predicted)), 4),
        "entry_recall": round(len(matched) / max(1, len(ground_truth)), 4),
        "entry_precision_confidence_ge_0.5": round(len(matched_conf50) / max(1, len(predicted_conf50)), 4),
        "entry_recall_confidence_ge_0.5": round(len(matched_conf50) / max(1, len(ground_truth)), 4),
        "size_within_16": round(sum(1 for item in size_checks if item["within_16"]) / max(1, len(size_checks)), 4),
        "size_within_10pct": round(sum(1 for item in size_checks if item["within_10pct"]) / max(1, len(size_checks)), 4),
        "size_checked": len(size_checks),
    }


def stripped_function_accuracy(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    with tempfile.TemporaryDirectory(prefix="nexus-validation-strip-") as directory:
        stripped = Path(directory) / source.name
        shutil.copy2(source, stripped)
        if shutil.which("strip"):
            _run(["strip", "--strip-all", str(stripped)], timeout=30)
        else:
            return {"status": "unavailable", "reason": "strip not installed"}
        ground_truth = _parse_readelf_functions(source)
        bundle = analyze_binary(stripped)
        predicted = {function.entry for function in bundle.functions}
        matched = sorted(set(ground_truth) & predicted)
        return {
            "path": str(source),
            "stripped_sha256": sha256_file(stripped),
            "ground_truth_functions": len(ground_truth),
            "predicted_functions": len(predicted),
            "matched_entries": len(matched),
            "entry_precision": round(len(matched) / max(1, len(predicted)), 4),
            "entry_recall": round(len(matched) / max(1, len(ground_truth)), 4),
        }


def cfg_integrity(bundle) -> dict[str, Any]:
    total_blocks = 0
    total_edges = 0
    invalid_edges = 0
    duplicate_blocks = 0
    branch_targets = 0
    resolved_targets = 0
    for function in bundle.functions:
        block_starts = [block.start for block in function.blocks]
        total_blocks += len(block_starts)
        duplicate_blocks += len(block_starts) - len(set(block_starts))
        for block in function.blocks:
            total_edges += len(block.successors)
            for successor in block.successors:
                if successor.startswith("unknown_"):
                    continue
                if successor not in {item.block_id for item in function.blocks}:
                    invalid_edges += 1
            last = block.instructions[-1] if block.instructions else None
            if last and last.control_flow in {"conditional_branch", "unconditional_branch"}:
                branch_targets += 1
                if any(target.va in set(block_starts) for target in last.targets if target.va is not None):
                    resolved_targets += 1
    return {
        "functions": len(bundle.functions),
        "blocks": total_blocks,
        "edges": total_edges,
        "duplicate_block_starts": duplicate_blocks,
        "invalid_edges": invalid_edges,
        "branch_instructions": branch_targets,
        "resolved_branch_targets": resolved_targets,
        "branch_target_resolution": round(resolved_targets / max(1, branch_targets), 4),
    }


def dataflow_integrity(bundle) -> dict[str, Any]:
    chains = 0
    invalid_chains = 0
    memory_accesses = 0
    stack_slots = 0
    for function in bundle.functions:
        dataflow = bundle.dataflows.get(function.function_id, {})
        instruction_ids = {f"ins_{instruction.address:x}" for instruction in function.instructions}
        for chain in dataflow.get("def_use_chains", []):
            chains += 1
            if chain.get("def_instruction") not in instruction_ids or chain.get("use_instruction") not in instruction_ids:
                invalid_chains += 1
        memory_accesses += len(dataflow.get("memory_accesses", []))
        stack_slots += len(dataflow.get("stack_slots", []))
    return {
        "functions": len(bundle.functions),
        "def_use_chains": chains,
        "invalid_def_use_chains": invalid_chains,
        "memory_accesses": memory_accesses,
        "stack_slots": stack_slots,
        "scope": "linear def-use heuristic; not a sound data-flow proof",
    }


def reconstruction_metrics(binary: Path, function_names: list[str] | None = None) -> dict[str, Any]:
    bundle = analyze_binary(binary)
    result = reconstruct_c(bundle, function_ids=function_names, out_dir=None)
    project_dir = Path(tempfile.mkdtemp(prefix="nexus-validation-c-"))
    try:
        for relative, content in result["files"].items():
            target = project_dir / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
        compile_result = compile_c_project(project_dir)
    finally:
        shutil.rmtree(project_dir, ignore_errors=True)
    counts = Counter(item["classification"] for item in result["functions"])
    return {
        "path": str(binary),
        "sha256": bundle.elf.sha256,
        "functions": len(result["functions"]),
        "classifications": dict(counts),
        "reconstructed_fraction": round((counts.get("partial_behavioral_reconstruction", 0) + counts.get("structural_approximation", 0)) / max(1, len(result["functions"])), 4),
        "compile_status": compile_result.get("status"),
        "compile_rate": compile_result.get("compile_rate"),
        "warnings": compile_result.get("warnings"),
        "dynamic_verification": compile_result.get("dynamic", {}).get("status"),
    }


def audit_dataset(path: str | Path) -> dict[str, Any]:
    root = Path(path)
    manifest_path = root / "manifests.jsonl"
    manifests = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    artifact_failures = []
    source_hashes: dict[str, set[str]] = {}
    project_splits: dict[str, set[str]] = {}
    binary_hashes = Counter()
    build_counts = Counter()
    unique_sources = set()
    labels_present = 0
    for manifest in manifests:
        for artifact in manifest.get("artifacts", []):
            target = root / artifact["path"]
            if not target.exists() or sha256_file(target) != artifact["sha256"]:
                artifact_failures.append({"sample_id": manifest["sample_id"], "path": artifact["path"]})
        source_path = manifest["provenance"]["source_paths"][0]
        source_hash = sha256_file(root / source_path)
        source_hashes.setdefault(source_hash, set()).add(manifest["split"])
        unique_sources.add(source_hash)
        project_splits.setdefault(manifest["project_id"], set()).add(manifest["split"])
        binary_hashes[manifest["binary"]["sha256"]] += 1
        build_counts[manifest["project_id"]] += 1
        labels_present += 1 if manifest.get("label_manifests") else 0
    leakage = []
    for source_hash, splits in source_hashes.items():
        if len(splits) > 1:
            leakage.append({"source_sha256": source_hash, "splits": sorted(splits)})
    project_leakage = [{"project": project, "splits": sorted(splits)} for project, splits in project_splits.items() if len(splits) > 1]
    duplicate_binaries = {digest: count for digest, count in binary_hashes.items() if count > 1}
    return {
        "path": str(root),
        "samples": len(manifests),
        "unique_source_programs": len(unique_sources),
        "builds_per_project": dict(build_counts),
        "artifact_failures": artifact_failures,
        "source_hash_leakage": leakage,
        "project_split_leakage": project_leakage,
        "duplicate_binary_hashes": duplicate_binaries,
        "samples_with_label_manifests": labels_present,
        "status": "passed" if not artifact_failures and not leakage and not project_leakage and not duplicate_binaries else "failed",
    }


def audit_checkpoint(path: str | Path) -> dict[str, Any]:
    import torch

    from nexus.foundation.checkpoint import inspect_checkpoint, load_foundation_checkpoint
    from nexus.foundation.smoke import synthetic_batch

    checkpoint_path = Path(path)
    info = inspect_checkpoint(checkpoint_path)
    loaded = load_foundation_checkpoint(checkpoint_path, load_training_state=False)
    model = loaded.model
    metadata = loaded.metadata or {}
    model.train()
    batch = synthetic_batch(loaded.config, batch_size=2, seed=loaded.config.seed)
    outputs = model(batch)
    loss = outputs["pooled"].float().pow(2).mean()
    loss.backward()
    finite = bool(torch.isfinite(loss))
    parameter_count = model.count_parameters()
    return {
        "path": str(checkpoint_path),
        "sha256": info["sha256"],
        "size": checkpoint_path.stat().st_size,
        "config": loaded.config.to_dict(),
        "parameter_count": parameter_count,
        "metadata": metadata,
        "pretrained_weights_loaded": metadata.get("pretrained_weights_loaded"),
        "initialization": metadata.get("initialization"),
        "training_started": metadata.get("training_started"),
        "optimizer_steps": metadata.get("optimizer_steps"),
        "forward_backward_finite": finite,
        "status": "passed" if not metadata.get("pretrained_weights_loaded") and finite else "failed",
    }


def jni_metrics(path: str | Path) -> dict[str, Any]:
    bundle = analyze_binary(path)
    report = bundle.jni_report
    if not report.get("is_jni"):
        return {"status": "not_jni", "path": str(path)}
    return {
        "status": "ok",
        "path": str(path),
        "entries": len(report.get("entries", [])),
        "unique_method_descriptors": len({(entry.get("class_name"), entry.get("method"), entry.get("descriptor")) for entry in report.get("entries", [])}),
        "onload": len(report.get("onload", [])),
        "register_natives": len(report.get("register_natives", [])),
        "imports": report.get("imports", []),
        "entries_detail": [
            {"class_name": entry.get("class_name"), "method": entry.get("method"), "arguments": entry.get("arguments"), "return_type": entry.get("return_type"), "descriptor": entry.get("descriptor")}
            for entry in report.get("entries", [])
        ],
    }


def dataset_reconstruction_metrics(path: str | Path) -> dict[str, Any]:
    root = Path(path)
    manifests = [json.loads(line) for line in (root / "manifests.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    selected = {}
    for manifest in manifests:
        if manifest["build"]["debug_info"] and manifest["build"]["optimization"] == "O0" and manifest["project_id"] not in selected:
            selected[manifest["project_id"]] = manifest
    totals = Counter()
    compile_status = Counter()
    rows = []
    for project, manifest in sorted(selected.items()):
        binary_path = root / next(artifact["path"] for artifact in manifest["artifacts"] if artifact["media_type"] == "application/x-elf")
        metrics = reconstruction_metrics(binary_path)
        rows.append({"project": project, **metrics})
        totals.update(metrics["classifications"])
        compile_status[metrics["compile_status"]] += 1
    total_functions = sum(totals.values())
    reconstructed = totals.get("partial_behavioral_reconstruction", 0) + totals.get("structural_approximation", 0)
    return {
        "projects": len(selected),
        "functions": total_functions,
        "classifications": dict(totals),
        "reconstructed_fraction": round(reconstructed / max(1, total_functions), 4),
        "compile_status": dict(compile_status),
        "compile_pass_rate": round(compile_status.get("passed", 0) / max(1, len(selected)), 4),
        "rows": rows,
    }


def api_security_audit(binary: str | Path) -> dict[str, Any]:
    import tempfile

    from fastapi.testclient import TestClient

    import nexus.api.app as api_module
    from nexus.core.config import NexusPaths
    from nexus.core.engine import Engine
    from nexus.storage.store import Storage
    root = Path(tempfile.mkdtemp(prefix="nexus-validation-api-"))
    paths = NexusPaths(root=root).ensure()
    engine = Engine(paths, Storage(paths))
    client = TestClient(api_module.create_app(engine, api_key="secret"))
    checks = {}
    checks["unauthorized"] = client.get("/v1/binaries").status_code == 401
    checks["authorized"] = client.get("/v1/binaries", headers={"X-API-Key": "secret"}).status_code == 200
    with open(binary, "rb") as handle:
        upload = client.post("/v1/binaries", files={"file": ("validation-binary", handle, "application/octet-stream")}, headers={"X-API-Key": "secret"})
    checks["upload"] = upload.status_code == 200
    binary_id = upload.json().get("id") if upload.status_code == 200 else None
    analysis = client.post("/v1/analyses", json={"binary_id": binary_id}, headers={"X-API-Key": "secret"}) if binary_id else None
    checks["analysis"] = bool(analysis and analysis.status_code == 200)
    if analysis and analysis.status_code == 200:
        summary = analysis.json()
        checks["no_path_in_analysis"] = "path" not in summary and "record_path" not in summary and "result_path" not in summary
        reconstruction = client.post("/v1/reconstructions", json={"analysis_id": summary["analysis_id"], "language": "c", "function_ids": ["add"]}, headers={"X-API-Key": "secret"})
        checks["reconstruction"] = reconstruction.status_code == 200
        if reconstruction.status_code == 200:
            body = reconstruction.json()
            checks["no_output_dir_in_reconstruction"] = "output_dir" not in body
            reconstruction_id = body.get("reconstruction_id")
            if reconstruction_id:
                fetched = client.get(f"/v1/reconstructions/{reconstruction_id}", headers={"X-API-Key": "secret"})
                checks["no_output_dir_in_get"] = fetched.status_code == 200 and "output_dir" not in fetched.json()
                try:
                    engine.read_source_file(reconstruction_id, "../../etc/passwd")
                    checks["path_traversal_rejected"] = False
                except Exception:
                    checks["path_traversal_rejected"] = True
    old_limit = api_module.MAX_UPLOAD_BYTES
    api_module.MAX_UPLOAD_BYTES = 1024
    try:
        oversized = client.post("/v1/binaries", files={"file": ("too-large", b"x" * 2048, "application/octet-stream")}, headers={"X-API-Key": "secret"})
        checks["upload_limit"] = oversized.status_code == 413
    finally:
        api_module.MAX_UPLOAD_BYTES = old_limit
    checks["not_found_error"] = client.get("/v1/binaries/bin_missing", headers={"X-API-Key": "secret"}).status_code == 404
    capabilities = client.get("/v1/capabilities").json()
    checks["dynamic_disabled"] = capabilities["dynamic_binary_execution"]["status"] == "disabled"
    shutil.rmtree(root, ignore_errors=True)
    return {"checks": checks, "status": "passed" if all(checks.values()) else "failed", "failed": [name for name, value in checks.items() if not value]}


def fuzz_parser(path: str | Path, iterations: int = 200, seed: int = 123) -> dict[str, Any]:
    from nexus.core.errors import ElfError
    from nexus.elf.parser import parse_elf
    source = Path(path).read_bytes()
    rng = random.Random(seed)
    unexpected = Counter()
    slow = []
    parsed = 0
    rejected = 0
    with tempfile.TemporaryDirectory(prefix="nexus-validation-fuzz-") as directory:
        for index in range(iterations):
            data = bytearray(source)
            mode = index % 4
            if mode == 0:
                for _ in range(rng.randint(1, 20)):
                    data[rng.randrange(len(data))] = rng.randrange(256)
            elif mode == 1:
                data = data[:rng.randrange(0, len(data))]
            elif mode == 2:
                data = data + rng.randbytes(rng.randrange(1, 100))
            else:
                start = rng.randrange(len(data))
                data[start:start + 16] = rng.randbytes(16)
            target = Path(directory) / f"mutated-{index}.bin"
            target.write_bytes(data)
            started = time.perf_counter()
            try:
                parse_elf(target)
                parsed += 1
            except ElfError:
                rejected += 1
            except Exception as exc:
                unexpected[type(exc).__name__] += 1
            elapsed = time.perf_counter() - started
            if elapsed > 1.0:
                slow.append({"iteration": index, "seconds": round(elapsed, 3)})
    return {
        "iterations": iterations,
        "parsed": parsed,
        "rejected": rejected,
        "unexpected_exceptions": dict(unexpected),
        "slow": slow,
        "status": "passed" if not unexpected and not slow else "failed",
    }


def run_validation(
    binaries: list[str | Path],
    dataset: str | Path | None = None,
    checkpoint: str | Path | None = None,
    output: str | Path = "artifacts/validation/phase1_5_report.json",
    fuzz_iterations: int = 200,
    reconstruction_limit: int | None = 4,
) -> dict[str, Any]:
    started = time.time()
    elf_results = []
    disassembly_results = []
    function_results = []
    stripped_results = []
    cfg_results = []
    dataflow_results = []
    reconstruction_results = []
    jni_results = []
    for binary in binaries:
        path = Path(binary)
        if not path.exists():
            continue
        elf_results.append(compare_elf(path))
        disassembly_results.append(compare_disassembly(path))
        function_results.append(function_accuracy(path))
        stripped_results.append(stripped_function_accuracy(path))
        bundle = analyze_binary(path)
        cfg_results.append({"path": str(path), **cfg_integrity(bundle)})
        dataflow_results.append({"path": str(path), **dataflow_integrity(bundle)})
        jni = jni_metrics(path)
        if jni.get("status") == "ok":
            jni_results.append(jni)
        if reconstruction_limit is None or len(reconstruction_results) < reconstruction_limit:
            reconstruction_results.append(reconstruction_metrics(path))
    dataset_result = audit_dataset(dataset) if dataset and Path(dataset).exists() else {"status": "skipped"}
    dataset_reconstruction = dataset_reconstruction_metrics(dataset) if dataset and Path(dataset).exists() else {"status": "skipped"}
    checkpoint_result = audit_checkpoint(checkpoint) if checkpoint and Path(checkpoint).exists() else {"status": "skipped"}
    fuzz_result = fuzz_parser(binaries[0], iterations=fuzz_iterations) if binaries and Path(binaries[0]).exists() else {"status": "skipped"}
    api_result = api_security_audit(binaries[0]) if binaries and Path(binaries[0]).exists() else {"status": "skipped"}
    elf_passed = sum(item["passed"] for item in elf_results)
    elf_total = sum(item["total"] for item in elf_results)
    available_disassembly = [item for item in disassembly_results if item.get("status") != "llvm_unavailable"]
    controlled_functions = [item for item in function_results if item.get("ground_truth_complete")]
    reconstruction_compiled = sum(1 for item in reconstruction_results if item.get("compile_status") == "passed")
    summary = {
        "elf_field_checks": {"passed": elf_passed, "total": elf_total, "rate": round(elf_passed / max(1, elf_total), 4)},
        "disassembly_byte_agreement": round(sum(item.get("byte_agreement", 0) for item in available_disassembly) / max(1, len(available_disassembly)), 4),
        "disassembly_mnemonic_agreement": round(sum(item.get("mnemonic_agreement", 0) for item in available_disassembly) / max(1, len(available_disassembly)), 4),
        "controlled_function_precision": round(sum(item.get("entry_precision_confidence_ge_0.5", 0) for item in controlled_functions) / max(1, len(controlled_functions)), 4),
        "controlled_function_recall": round(sum(item.get("entry_recall_confidence_ge_0.5", 0) for item in controlled_functions) / max(1, len(controlled_functions)), 4),
        "cfg_invalid_edges": sum(item.get("invalid_edges", 0) for item in cfg_results),
        "dataflow_invalid_chains": sum(item.get("invalid_def_use_chains", 0) for item in dataflow_results),
        "reconstruction_projects_compiled": reconstruction_compiled,
        "reconstruction_projects": len(reconstruction_results),
        "dataset_status": dataset_result.get("status"),
        "checkpoint_status": checkpoint_result.get("status"),
        "api_security_status": api_result.get("status"),
        "parser_fuzz_status": fuzz_result.get("status"),
    }
    payload = {
        "status": "completed",
        "started_at": started,
        "duration_seconds": round(time.time() - started, 3),
        "summary": summary,
        "binaries": [str(item) for item in binaries],
        "elf": elf_results,
        "disassembly": disassembly_results,
        "function_recovery": function_results,
        "stripped_function_recovery": stripped_results,
        "cfg_integrity": cfg_results,
        "dataflow_integrity": dataflow_results,
        "reconstruction": reconstruction_results,
        "jni": jni_results,
        "dataset": dataset_result,
        "dataset_reconstruction": dataset_reconstruction,
        "checkpoint": checkpoint_result,
        "parser_fuzz": fuzz_result,
        "api_security": api_result,
        "limitations": [
            "ELF field comparison uses readelf as a textual reference, not an independent formal oracle.",
            "Disassembly comparison measures decoder agreement, not semantic correctness.",
            "Function boundaries use the unstripped symbol table as ground truth; optimized/inlined code may differ.",
            "CFG validation checks structural integrity, not accuracy against a formal ground-truth CFG.",
            "Data-flow validation checks reference integrity, not soundness or completeness.",
            "C reconstruction metrics are compile rates and classification counts, not behavioral equivalence.",
            "No target binary is executed.",
        ],
    }
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
    return payload
