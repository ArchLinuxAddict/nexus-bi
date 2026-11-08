"""Authored, controlled cross-architecture object corpus.

Only locally available deterministic toolchains are used.  Nothing is linked
or executed.  The source is authored by this project and released as
CC0-1.0, so the samples are clearly distinct from real-world third-party data.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from nexus.data.identity import extract_function_by_name
from nexus.disassembly.llvm_objdump import parse_llvm_object
from nexus.foundation.config import FoundationConfig
from nexus.foundation.representation import RepresentationOptions, encode_function_sample
from nexus.foundation.vocab import Vocabulary

ROOT = Path(__file__).resolve().parents[2]
CORPUS = ROOT / "artifacts" / "dataset" / "cross_arch_controlled"
SOURCE = CORPUS / "source" / "cross_arch.c"
TARGETS = {
    "x86_64": "x86_64-linux-gnu",
    "i386": "i386-linux-gnu",
    "aarch64": "aarch64-linux-gnu",
    "arm": "armv7-linux-gnueabihf",
    "riscv": "riscv64-linux-gnu",
    "mips": "mipsel-linux-gnu",
}
OPTIMIZATIONS = ("O0", "O2")


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _split_for(source_function_id: str) -> str:
    bucket = int(source_function_id[:8], 16) % 10
    if bucket == 0:
        return "extended_test"
    if bucket == 1:
        return "validation"
    return "train"


def _source_function_id(name: str, text: str) -> str:
    return _sha256_bytes((name + "\x00" + text).encode("utf-8"))


def compiler_version() -> str:
    try:
        result = subprocess.run(["clang", "--version"], capture_output=True, text=True, timeout=10)
        return result.stdout.splitlines()[0] if result.stdout else "unknown"
    except (OSError, subprocess.TimeoutExpired):
        return "unavailable"


def compile_cross_arch(root: Path = CORPUS, *, timeout: float = 120) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not SOURCE.exists():
        raise FileNotFoundError(f"cross-arch source is missing: {SOURCE}")
    if shutil.which("clang") is None or shutil.which("llvm-objdump") is None:
        return [], {"status": "unavailable", "reason": "clang or llvm-objdump is missing"}
    builds_dir = root / "builds"
    builds_dir.mkdir(parents=True, exist_ok=True)
    source_sha = _sha256_file(SOURCE)
    version = compiler_version()
    records = []
    for arch, triple in TARGETS.items():
        for optimization in OPTIMIZATIONS:
            target_dir = builds_dir / arch / optimization
            target_dir.mkdir(parents=True, exist_ok=True)
            output = target_dir / "cross_arch.o"
            command = [
                "clang",
                f"--target={triple}",
                f"-{optimization}",
                "-g",
                "-fno-omit-frame-pointer",
                "-c",
                str(SOURCE),
                "-o",
                str(output),
            ]
            try:
                result = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
            except subprocess.TimeoutExpired:
                records.append({"arch": arch, "optimization": optimization, "status": "timeout", "command": command})
                continue
            if result.returncode != 0:
                records.append(
                    {
                        "arch": arch,
                        "optimization": optimization,
                        "status": "failed",
                        "command": command,
                        "stderr": result.stderr[-2000:],
                    }
                )
                continue
            records.append(
                {
                    "arch": arch,
                    "optimization": optimization,
                    "status": "ok",
                    "command": command,
                    "compiler": version,
                    "source_sha256": source_sha,
                    "output_path": str(output.relative_to(ROOT)),
                    "output_sha256": _sha256_file(output),
                    "output_size": output.stat().st_size,
                    "license": "CC0-1.0",
                    "provenance": "authored_controlled",
                }
            )
    summary = {
        "schema_version": "nexus.cross-arch-controlled/v1",
        "created_at": datetime.now(UTC).isoformat(),
        "source": str(SOURCE.relative_to(ROOT)),
        "source_sha256": source_sha,
        "compiler": version,
        "targets": TARGETS,
        "optimizations": list(OPTIMIZATIONS),
        "builds": records,
        "status": "ok" if all(item.get("status") == "ok" for item in records) else "partial",
        "license": "CC0-1.0",
    }
    (root / "builds.jsonl").write_text("".join(json.dumps(item, sort_keys=True) + "\n" for item in records), encoding="utf-8")
    (root / "dataset_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return records, summary


def build_cross_arch_samples(
    vocab: Vocabulary,
    config: FoundationConfig,
    *,
    root: Path = CORPUS,
    max_functions: int | None = None,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    records, summary = compile_cross_arch(root)
    source_text = SOURCE.read_text(encoding="utf-8")
    source_sha = _sha256_file(SOURCE)
    samples: dict[str, list[dict[str, Any]]] = {"train": [], "validation": [], "test": [], "extended_test": [], "quarantine": []}
    skipped: Counter[str] = Counter()
    for record in records:
        if record.get("status") != "ok":
            skipped[f"build_{record.get('status')}"] += 1
            continue
        object_path = ROOT / record["output_path"]
        try:
            parsed = parse_llvm_object(object_path)
        except Exception:
            skipped["parse_failed"] += 1
            continue
        for function in parsed.get("functions", []):
            if max_functions is not None and sum(len(values) for values in samples.values()) >= max_functions:
                break
            name = str(function.get("name") or "")
            text = extract_function_by_name(source_text, name)
            if not text:
                skipped["source_function_not_found"] += 1
                continue
            source_function_id = _source_function_id(name, text)
            split = _split_for(source_function_id)
            c_ids = vocab.encode_text(text, 1_000_000)
            truncated = len(c_ids) > config.max_text_tokens
            c_ids = [] if truncated else c_ids[: config.max_text_tokens]
            identity = {
                "sample_id": f"cross:{record['arch']}:{record['optimization']}:{name}",
                "project_id": f"cross_arch_{name}",
                "dataset_id": "cross_arch_controlled",
                "split": split,
                "binary_sha256": record["output_sha256"],
                "source_family": f"cross_arch_{name}",
                "match_group": source_function_id,
                "binary_path": str(object_path),
                "function_id": function["function_id"],
                "build": {
                    "compiler": "clang",
                    "optimization": record["optimization"],
                    "arch": record["arch"],
                    "debug_info": True,
                },
            }
            sample = encode_function_sample(
                parsed,
                function,
                vocab,
                config,
                identity=identity,
                options=RepresentationOptions(include_purpose_target=False),
                source_label={
                    "text": text,
                    "source_path": "source/cross_arch.c",
                    "source_sha256": source_sha,
                    "alignment": "authored_exact",
                    "confidence": 1.0,
                },
            )
            if sample is None:
                skipped["encoder_rejected"] += 1
                continue
            sample["c_ids"] = c_ids
            sample["c_mask"] = [1] * len(c_ids)
            sample["source_ids"] = list(c_ids)
            sample["source_target_truncated"] = truncated
            sample["source_extraction"] = "authored_exact"
            sample["split"] = split
            sample["labels"] = {
                **(sample.get("labels") or {}),
                "dataset_id": "cross_arch_controlled",
                "project_id": f"cross_arch_{name}",
                "repository_id": f"cross_arch_{name}",
                "compiler": "clang",
                "optimization": record["optimization"],
                "arch": record["arch"],
                "binary_format": "ELF",
                "debug_info": True,
                "license": "CC0-1.0",
                "language": "c",
                "source_alignment": "authored_exact",
                "provenance": {"source": "cross_arch_controlled", "build": record["command"]},
            }
            scoped_identity = _sha256_bytes(("cross_arch.c:" + name).encode("utf-8"))
            sample["source_identity"] = {
                "source_function_id": source_function_id,
                "source_file_id": scoped_identity,
                "source_archive_id": scoped_identity,
                "raw_source_hash": _sha256_bytes(text.encode("utf-8")),
                "normalized_source_hash": _sha256_bytes("".join(text.split()).encode("utf-8")),
                "normalized_function_hash": _sha256_bytes("".join(text.split()).encode("utf-8")),
            }
            sample["build"] = {
                "compiler": "clang",
                "optimization": record["optimization"],
                "arch": record["arch"],
                "debug_info": True,
                "variant_id": f"{record['arch']}:{record['optimization']}",
            }
            samples[split].append(sample)
    audit = {
        "builds": len(records),
        "samples": {split: len(values) for split, values in samples.items()},
        "skipped": dict(skipped),
        "source_sha256": source_sha,
        "status": summary.get("status"),
        "license": "CC0-1.0",
    }
    (root / "corpus_report.json").write_text(json.dumps({"summary": summary, "audit": audit}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return samples, audit
