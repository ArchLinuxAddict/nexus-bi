"""Foundation-model samples built from Java class/DEX bytecode.

This bridges static Java analysis to the existing unified representation:
methods become functions, bytecode instructions become instruction records and
reconstructed method source becomes the reconstruction target.  It is a
research-ready pilot builder, not a claim that Java data is already part of the
default frozen corpus.
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from nexus.data.store_assembly import assemble_store
from nexus.formats.java_class import parse_java_class
from nexus.foundation.config import config_for_vocabulary, profile_config
from nexus.foundation.representation import RepresentationOptions, encode_function_sample
from nexus.foundation.vocab import build_vocabulary
from nexus.reconstruction.java.classfile import _emit_method


def _hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _split(value: str) -> str:
    bucket = int(_hash(value)[:8], 16) % 10
    if bucket == 0:
        return "test"
    if bucket == 1:
        return "validation"
    return "train"


def _operand(value: str, kind: str = "immediate", register: str | None = None) -> dict[str, Any]:
    payload: dict[str, Any] = {"kind": kind, "value": value, "access": "read", "width": None}
    if register:
        payload["register"] = register
    return payload


def _local_register(mnemonic: str, operands: list[str]) -> str | None:
    if mnemonic.startswith(("iload", "lload", "fload", "dload", "aload", "istore", "lstore", "fstore", "dstore", "astore")):
        if "_" in mnemonic and mnemonic.rsplit("_", 1)[1].isdigit():
            return f"jvm_v{mnemonic.rsplit('_', 1)[1]}"
        if operands and operands[0].isdigit():
            return f"jvm_v{operands[0]}"
    if mnemonic in {"iinc"} and operands:
        return f"jvm_v{operands[0]}"
    return None


def class_to_record(parsed: dict[str, Any], class_path: str | Path) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    path = Path(class_path)
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    functions = []
    instructions = []
    blocks = []
    source_labels: dict[str, dict[str, Any]] = {}
    for method_index, method in enumerate(parsed.get("methods") or []):
        code = method.get("code") or {}
        bytecode = code.get("instructions") or []
        function_id = f"jvm_{sha[:12]}_{method_index}"
        method_instructions = []
        for instruction_index, item in enumerate(bytecode):
            instruction_id = f"{function_id}_i{instruction_index}"
            mnemonic = str(item.get("mnemonic") or "unknown")
            operand_texts = [str(value) for value in item.get("operands") or []]
            local = _local_register(mnemonic, operand_texts)
            operands = []
            for operand_index, text in enumerate(operand_texts):
                if local and operand_index == 0:
                    operands.append(_operand(text, "register", local))
                elif text.startswith('"'):
                    operands.append(_operand(text, "address"))
                else:
                    operands.append(_operand(text, "immediate"))
            flow = {"kind": "fallthrough", "targets": []}
            if mnemonic in {"ireturn", "lreturn", "freturn", "dreturn", "areturn", "return"}:
                flow = {"kind": "return", "targets": []}
            elif mnemonic.startswith("invoke"):
                flow = {"kind": "call", "targets": []}
            elif mnemonic in {"goto", "if_icmpeq", "if_icmpne", "ifeq", "ifne", "goto_w"}:
                flow = {"kind": "conditional", "targets": [{"va": f"0x{target:x}", "kind": "direct", "confidence": 0.5} for target in item.get("branch_targets") or []]}
            record = {
                "instruction_id": instruction_id,
                "va": f"0x{int(item.get('pc') or 0):x}",
                "file_offset": int(item.get("pc") or 0),
                "bytes": "",
                "length": int(item.get("length") or 1),
                "arch": "jvm",
                "mode": "bytecode",
                "decoded": True,
                "mnemonic": mnemonic,
                "operands": operands,
                "reads": [local] if local and mnemonic.startswith(("i", "l", "f", "d", "a")) and "load" in mnemonic else [],
                "writes": [local] if local and "store" in mnemonic else [],
                "memory": [],
                "control_flow": flow,
                "flags_read": [],
                "flags_written": [],
                "confidence": 0.7,
            }
            instructions.append(record)
            method_instructions.append(instruction_id)
        if len(method_instructions) < 2:
            continue
        block_id = f"{function_id}_bb0"
        block = {
            "block_id": block_id,
            "function_id": function_id,
            "start": 0,
            "end": max(int(item.get("pc") or 0) + int(item.get("length") or 1) for item in bytecode),
            "instruction_ids": method_instructions,
            "successors": [],
            "predecessors": [],
            "confidence": 0.6,
            "evidence": [f"ev_{sha[:12]}_method_{method_index}"],
        }
        blocks.append(block)
        functions.append(
            {
                "function_id": function_id,
                "names": [{"value": str(method.get("name") or "unknown"), "source": "classfile"}],
                "name": str(method.get("name") or "unknown"),
                "entry_va": "0x0",
                "ranges": [{"start_va": "0x0", "end_va": f"0x{max(int(item.get('pc') or 0) + int(item.get('length') or 1) for item in bytecode):x}"}],
                "size": max(int(item.get("pc") or 0) + int(item.get("length") or 1) for item in bytecode),
                "source": "java_class_method",
                "confidence": 0.75,
                "blocks": [block],
                "callees": [],
                "callers": [],
                "unknowns": [],
                "evidence": [f"ev_{sha[:12]}_method_{method_index}"],
                "data_references": [],
                "purpose": None,
                "method_descriptor": method.get("descriptor"),
                "native": method.get("native", False),
            }
        )
        emitted, evidence = _emit_method(parsed, method, class_name=parsed.get("name", ""), is_interface=bool(parsed.get("is_interface")), is_annotation=bool(parsed.get("is_annotation")))
        if emitted:
            source_labels[function_id] = {
                "text": emitted,
                "source_path": f"{parsed.get('name')}#{method.get('name')}{method.get('descriptor')}",
                "source_sha256": _hash(emitted),
                "alignment": "reconstructed_method",
                "confidence": 0.4,
            }
    record = {
        "schema_version": "0.1.0",
        "program": {
            "sha256": sha,
            "size": path.stat().st_size,
            "format": "JVM",
            "class": "class",
            "endianness": "none",
            "machine": "jvm",
            "entry_va": "0x0",
            "stripped": False,
            "pie": False,
            "static": False,
        },
        "segments": [],
        "sections": [],
        "symbols": [],
        "relocations": [],
        "imports": [],
        "exports": [],
        "dependencies": [],
        "regions": [],
        "functions": functions,
        "blocks": blocks,
        "instructions": instructions,
        "cfg": {"nodes": [], "edges": []},
        "dataflow": [],
        "strings": [],
        "evidence": [],
        "validation": {"status": "java_static", "limitations": parsed.get("limitations", [])},
    }
    return record, source_labels


def build_java_class_records(classes_dir: str | Path) -> list[tuple[dict[str, Any], dict[str, dict[str, Any]], Path]]:
    root = Path(classes_dir)
    records = []
    for class_path in sorted(root.rglob("*.class")):
        parsed = parse_java_class(class_path)
        record, sources = class_to_record(parsed, class_path)
        records.append((record, sources, class_path))
    return records


def build_java_samples(
    classes_dir: str | Path,
    *,
    out_dir: str | Path | None = None,
    text_vocab_size: int = 4096,
) -> dict[str, Any]:
    records = build_java_class_records(classes_dir)
    if not records:
        return {"status": "unavailable", "reason": "no class files found"}
    all_records = [item[0] for item in records]
    source_texts = [label["text"] for _, sources, _ in records for label in sources.values()]
    vocab = build_vocabulary(all_records, source_texts, text_vocab_size=text_vocab_size, source_fingerprint=_hash(str(classes_dir))[:16])
    config = config_for_vocabulary(profile_config("small"), vocab)
    samples: dict[str, list[dict[str, Any]]] = {"train": [], "validation": [], "test": [], "extended_test": [], "quarantine": []}
    sample_number = 0
    for record, source_labels, class_path in records:
        for function in record["functions"]:
            split = "test" if sample_number % 10 == 0 else "validation" if sample_number % 10 == 1 else "train"
            sample_number += 1
            identity = {
                "sample_id": f"java:{function['name']}:{_hash(function['function_id'])[:12]}",
                "project_id": f"java:{Path(class_path).stem}",
                "dataset_id": "java_jni_pilot",
                "split": split,
                "binary_sha256": record["program"]["sha256"],
                "source_family": "java_class",
                "match_group": record["program"]["sha256"],
                "binary_path": str(class_path),
                "function_id": function["function_id"],
                "build": {"compiler": "javac", "optimization": None, "arch": "jvm", "debug_info": False},
            }
            sample = encode_function_sample(
                record,
                function,
                vocab,
                config,
                identity=identity,
                options=RepresentationOptions(include_purpose_target=False),
                source_label=source_labels.get(function["function_id"]),
            )
            if sample is None:
                continue
            sample["labels"] = {
                **(sample.get("labels") or {}),
                "dataset_id": "java_jni_pilot",
                "language": "java",
                "arch": "jvm",
                "license": "CC0-1.0",
                "source_alignment": "reconstructed_method",
            }
            samples[split].append(sample)
    if out_dir is None:
        return {"status": "completed", "samples": {key: len(value) for key, value in samples.items()}, "vocab_sha256": vocab.digest()}
    output = Path(out_dir)
    output.mkdir(parents=True, exist_ok=True)
    vocab.save(output / "vocab.json")
    store_manifest = assemble_store(
        output / "samples",
        samples,
        vocab,
        metadata={
            "created_at": datetime.now(UTC).isoformat(),
            "dataset_id": "java_jni_pilot",
            "license": "CC0-1.0",
            "source": str(classes_dir),
            "limitations": [
                "Methods with fewer than two decoded bytecode instructions are skipped.",
                "Source targets are reconstructed methods, not original source.",
                "No behavioral verification.",
            ],
        },
    )
    manifest = {
        "schema_version": "nexus.dataset.java-foundation/v1",
        "dataset_id": "java_jni_pilot",
        "created_at": datetime.now(UTC).isoformat(),
        "vocab_sha256": vocab.digest(),
        "counts": {key: len(value) for key, value in samples.items()},
        "store": store_manifest,
        "limitations": [
            "Pilot data only; merge with the frozen corpus only after vocabulary and split review.",
            "Reconstructed method source is a target, not recovered original source.",
        ],
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return {"status": "completed", "dataset": str(output), **manifest["counts"], "vocab_sha256": vocab.digest()}
