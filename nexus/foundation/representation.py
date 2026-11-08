"""ProgramRecord -> unified foundation-model tensors.

The input representation is deliberately binary-first: instruction bytes,
typed operands, registers, semantic-IR classes and CFG structure.  Symbol
names, DWARF, source text and debug metadata never enter the input tensors.
Source text is carried only as a reconstruction/alignment label with its own
provenance record.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

from nexus.foundation.config import FoundationConfig
from nexus.foundation.vocab import (
    ACCESS_KINDS,
    BYTE_BASE,
    MEMORY_WIDTHS,
    OPERAND_KINDS,
    PAD,
    UNK,
    Vocabulary,
    canonical_instruction,
    canonical_register_name,
    instruction_shape,
    width_bucket,
)

PURPOSE_CLASSES = [
    "other",
    "algorithms",
    "arithmetic",
    "strings",
    "memory",
    "data_structures",
    "parsing",
    "serialization",
    "cryptography",
    "compression",
    "networking",
    "systems",
    "utilities",
    "io",
    "concurrency",
    "security",
]
PURPOSE_INDEX = {name: index for index, name in enumerate(PURPOSE_CLASSES)}

IR_OPS = [
    "other",
    "move",
    "arithmetic",
    "compare",
    "branch",
    "call",
    "return",
    "trap",
    "stack",
    "load",
    "store",
    "simd",
    "system",
    "nop",
]
IR_INDEX = {name: index for index, name in enumerate(IR_OPS)}

FUNCTION_FEATURES = 18


def _safe(index: dict[str, int], value: str, limit: int) -> int:
    identifier = index.get(value, UNK)
    return identifier if identifier < limit else UNK


def _kind_id(value: str) -> int:
    try:
        return OPERAND_KINDS.index(value)
    except ValueError:
        return OPERAND_KINDS.index("unknown")


def _access_id(value: str) -> int:
    try:
        return ACCESS_KINDS.index(value)
    except ValueError:
        return ACCESS_KINDS.index("unknown")


def _memory_width(value: int | None) -> int:
    bucket = width_bucket(value)
    return min(bucket, len(MEMORY_WIDTHS) - 1)


def _float_feature(value: float, scale: float = 1.0) -> float:
    return float(np.tanh(value / max(1e-6, scale)))


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


def _instruction_mnemonic(instruction: dict[str, Any]) -> str:
    return str(instruction.get("mnemonic") or "unknown").lower()


def _ir_op(instruction: dict[str, Any]) -> str:
    mnemonic = _instruction_mnemonic(instruction)
    flow = (instruction.get("control_flow") or {}).get("kind")
    if mnemonic in {"mov", "movabs", "movzx", "movsx", "movsxd", "lea", "movbe", "movd", "movq", "movaps", "movups"}:
        return "load" if "mov" in mnemonic else "move"
    if mnemonic in {"cmp", "test", "ucomiss", "comiss"}:
        return "compare"
    if mnemonic in {"add", "sub", "imul", "mul", "div", "idiv", "and", "or", "xor", "shl", "shr", "sar", "inc", "dec", "neg", "not", "rol", "ror", "adc", "sbb"}:
        return "arithmetic"
    if mnemonic.startswith(("padd", "psub", "pand", "por", "pxor", "pcmp", "pshuf", "punpck", "pmul", "pack", "v")):
        return "simd"
    if flow in {"conditional_branch", "unconditional_branch", "indirect_branch"}:
        return "branch"
    if flow in {"call", "indirect_call"}:
        return "call"
    if flow == "return":
        return "return"
    if flow == "trap":
        return "trap"
    if mnemonic in {"push", "pop", "leave", "enter"}:
        return "stack"
    if mnemonic in {"syscall", "int", "cpuid", "rdtsc"}:
        return "system"
    if mnemonic in {"nop", "endbr64", "endbr32"}:
        return "nop"
    return "other"


def _target_va(target: Any) -> int | None:
    if isinstance(target, dict):
        value = target.get("va") or target.get("target_va")
        if isinstance(value, str):
            try:
                return int(value, 16)
            except ValueError:
                return None
        if isinstance(value, int):
            return value
    return None


def _instruction_bytes(instruction: dict[str, Any]) -> bytes:
    text = str(instruction.get("bytes") or "")
    try:
        return bytes.fromhex(text)
    except ValueError:
        return b""


def extract_c_function(source: str, name: str) -> str | None:
    """Extract one C function definition by name using deterministic brace matching."""
    if not source or not name:
        return None
    pattern = re.compile(r"(?<![A-Za-z0-9_])" + re.escape(name) + r"\s*\(")
    for match in pattern.finditer(source):
        start = source.rfind("\n", 0, match.start()) + 1
        prefix = source[start:match.start()]
        if "#" in prefix:
            continue
        brace = source.find("{", match.end())
        semicolon = source.find(";", match.end())
        if brace < 0 or (semicolon >= 0 and semicolon < brace):
            continue
        depth = 0
        for index in range(brace, len(source)):
            character = source[index]
            if character == "{":
                depth += 1
            elif character == "}":
                depth -= 1
                if depth == 0:
                    return source[start : index + 1].strip()
    return None


def _read_artifacts(manifest: dict[str, Any], root: Path) -> dict[str, str]:
    result = {}
    for artifact in manifest.get("artifacts", []):
        path = root / artifact["path"]
        if path.suffix in {".c", ".h"} and path.exists():
            result[artifact["path"]] = path.read_text(encoding="utf-8", errors="replace")
        elif path.name == "spec.json" and path.exists():
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue
            if isinstance(payload.get("source"), str):
                result[artifact["path"]] = payload["source"]
    return result


def source_label_for_function(
    manifest: dict[str, Any],
    root: str | Path,
    function: dict[str, Any],
    source_cache: dict[str, str] | None = None,
) -> dict[str, Any] | None:
    """Return a source label only when a deterministic source/binary name match exists."""
    if not manifest.get("build", {}).get("debug_info"):
        return None
    inline_source = function.get("source_label_text")
    if inline_source:
        return {
            "text": inline_source,
            "source_path": function.get("source_path"),
            "source_line": function.get("source_line"),
            "source_sha256": function.get("source_sha256") or _hash(inline_source),
            "alignment": "dwarf_inline",
            "confidence": 1.0,
        }
    names = [entry.get("value") for entry in function.get("names", []) if entry.get("value")]
    if not names:
        return None
    sources = source_cache if source_cache is not None else _read_artifacts(manifest, Path(root))
    for name in names:
        for path, text in sources.items():
            candidate = extract_c_function(text, name)
            if candidate:
                return {
                    "text": candidate,
                    "source_path": path,
                    "source_sha256": _hash(candidate),
                    "function_name_hash": _hash(name),
                    "alignment": "exact_source_function_name",
                    "confidence": 0.95,
                }
    return None


@dataclass
class RepresentationOptions:
    include_bytes: bool = True
    include_instructions: bool = True
    include_ir: bool = True
    include_graph: bool = True
    include_metadata: bool = True
    include_source_target: bool = True
    include_purpose_target: bool = True
    max_instructions: int | None = None
    max_bytes: int | None = None
    max_blocks: int | None = None


def _empty_sample(config: FoundationConfig, identity: dict[str, Any], options: RepresentationOptions) -> dict[str, Any]:
    return {
        "sample_id": identity.get("sample_id", ""),
        "function_id": identity.get("function_id", ""),
        "project_id": identity.get("project_id", ""),
        "dataset_id": identity.get("dataset_id", ""),
        "split": identity.get("split", ""),
        "binary_sha256": identity.get("binary_sha256", ""),
        "source_family": identity.get("source_family", ""),
        "match_group": identity.get("match_group", ""),
        "binary_path": identity.get("binary_path"),
        "record_path": identity.get("record_path"),
        "instr_mnemonic": [],
        "instr_shape": [],
        "instr_id": [],
        "instr_ir": [],
        "instr_regs": [],
        "instr_kinds": [],
        "instr_access": [],
        "instr_width": [],
        "instr_block": [],
        "instr_mem_read": [],
        "instr_mem_write": [],
        "instr_mem_width": [],
        "instr_flags_read": [],
        "instr_flags_written": [],
        "instr_mask": [],
        "byte_ids": [],
        "byte_mask": [],
        "block_mask": [],
        "instruction_edges": [],
        "instruction_edge_types": [],
        "block_edges": [],
        "block_edge_types": [],
        "cfg_target": [],
        "read_targets": [],
        "write_targets": [],
        "function_features": [0.0] * FUNCTION_FEATURES,
        "purpose_id": -100,
        "purpose_label": "unlabeled",
        "source_text": None,
        "source_sha256": None,
        "source_path": None,
        "c_ids": [],
        "c_mask": [],
        "java_ids": [],
        "java_mask": [],
        "source_ids": [],
        "function_start_target": [],
        "jni_targets": [-100] * 5,
        "tool_target": -100,
        "confidence_target": 0.5,
        "confidence_mask": 0,
        "security_target": -100,
        "availability": {
            "bytes": bool(options.include_bytes),
            "instructions": bool(options.include_instructions),
            "ir": bool(options.include_ir),
            "graph": bool(options.include_graph),
            "metadata": bool(options.include_metadata),
            "names": False,
            "source": False,
            "symbols": False,
            "dwarf": False,
        },
        "input_provenance": {
            "binary_only": True,
            "names_used": False,
            "source_used": False,
            "symbols_used": False,
            "dwarf_used": False,
            "notes": "input tensors contain only binary/instruction/IR/CFG features",
        },
        "labels": {},
    }


def encode_function_sample(
    record: dict[str, Any],
    function: dict[str, Any],
    vocab: Vocabulary,
    config: FoundationConfig,
    identity: dict[str, Any] | None = None,
    options: RepresentationOptions | None = None,
    source_label: dict[str, Any] | None = None,
    purpose_label: str | None = None,
    register_targets: list[str] | None = None,
) -> dict[str, Any] | None:
    options = options or RepresentationOptions()
    identity = identity or {}
    sample = _empty_sample(config, identity, options)
    max_instructions = min(options.max_instructions or config.max_instructions, config.max_instructions)
    max_bytes = min(options.max_bytes or config.max_bytes, config.max_bytes)
    max_blocks = min(options.max_blocks or config.max_blocks, config.max_blocks)
    register_targets = register_targets or vocab.register_targets(config.max_register_targets)
    mnemonic_index = vocab.mnemonic_index
    shape_index = vocab.shape_index
    instruction_index = vocab.instruction_index
    register_index = vocab.register_index
    blocks = [block for block in function.get("blocks", []) if isinstance(block, dict)][:max_blocks]
    if not blocks:
        return None
    instructions_by_id = {
        instruction.get("instruction_id"): instruction
        for instruction in record.get("instructions", [])
        if isinstance(instruction, dict)
    }
    ordered = []
    block_for_instruction = {}
    for block_index, block in enumerate(blocks):
        for instruction_id in block.get("instruction_ids", []):
            instruction = instructions_by_id.get(instruction_id)
            if instruction is None:
                continue
            block_for_instruction[instruction_id] = block_index
            ordered.append(instruction)
    ordered = ordered[:max_instructions]
    if len(ordered) < 2:
        return None
    address_to_index = {int(instruction.get("va", "0x0"), 16): index for index, instruction in enumerate(ordered) if instruction.get("va")}
    read_targets = []
    write_targets = []
    raw_byte_ids: list[int] = []
    instruction_edges: list[list[int]] = []
    instruction_edge_types: list[int] = []
    block_edges: list[list[int]] = []
    block_edge_types: list[int] = []
    block_ids = {block.get("block_id"): index for index, block in enumerate(blocks)}
    cfg_target = np.zeros((len(blocks), len(blocks)), dtype=np.float32)
    for index, instruction in enumerate(ordered):
        mnemonic = _instruction_mnemonic(instruction)
        operands = [operand for operand in instruction.get("operands", []) if isinstance(operand, dict)]
        shape = instruction_shape(operands)
        sample["instr_mnemonic"].append(_safe(mnemonic_index, mnemonic, config.mnemonic_vocab))
        sample["instr_shape"].append(_safe(shape_index, shape, config.shape_vocab))
        sample["instr_id"].append(
            _safe(instruction_index, canonical_instruction(mnemonic, operands), config.instruction_vocab)
        )
        ir = _ir_op(instruction)
        sample["instr_ir"].append(min(IR_INDEX.get(ir, 0), config.ir_vocab - 1))
        regs = [PAD] * config.max_operands
        kinds = [0] * config.max_operands
        access = [0] * config.max_operands
        widths = [0] * config.max_operands
        memory = instruction.get("memory") or []
        for operand_index, operand in enumerate(operands[: config.max_operands]):
            kind = str(operand.get("kind", "unknown"))
            kinds[operand_index] = _kind_id(kind)
            access[operand_index] = _access_id(str(operand.get("access", "unknown")))
            width = operand.get("width")
            if kind == "memory":
                memory_info = operand.get("memory") or {}
                register = canonical_register_name(memory_info.get("base") or memory_info.get("index") or "")
                width = memory_info.get("width") or width
            else:
                register = canonical_register_name(operand.get("register") or operand.get("value") or "")
            regs[operand_index] = _safe(register_index, register, config.register_vocab) if register else PAD
            widths[operand_index] = width_bucket(width)
        sample["instr_regs"].append(regs)
        sample["instr_kinds"].append(kinds)
        sample["instr_access"].append(access)
        sample["instr_width"].append(widths)
        sample["instr_block"].append(block_for_instruction.get(instruction.get("instruction_id"), 0))
        sample["instr_mask"].append(1)
        reads = []
        writes = []
        for name in instruction.get("reads", []) or []:
            canonical = canonical_register_name(name)
            if canonical and canonical not in reads:
                reads.append(canonical)
        for name in instruction.get("writes", []) or []:
            canonical = canonical_register_name(name)
            if canonical and canonical not in writes:
                writes.append(canonical)
        read_vector = [0.0] * len(register_targets)
        write_vector = [0.0] * len(register_targets)
        for name in reads:
            if name in register_targets:
                read_vector[register_targets.index(name)] = 1.0
        for name in writes:
            if name in register_targets:
                write_vector[register_targets.index(name)] = 1.0
        read_targets.append(read_vector)
        write_targets.append(write_vector)
        memory_reads = [effect for effect in memory if str(effect.get("kind")) == "read"]
        memory_writes = [effect for effect in memory if str(effect.get("kind")) == "write"]
        sample["instr_mem_read"].append(1 if memory_reads else 0)
        sample["instr_mem_write"].append(1 if memory_writes else 0)
        width_values = [effect.get("width") for effect in memory if effect.get("width")]
        sample["instr_mem_width"].append(_memory_width(max(width_values)) if width_values else 0)
        sample["instr_flags_read"].append(min(2, len(instruction.get("flags_read", []) or []) // 2))
        sample["instr_flags_written"].append(min(2, len(instruction.get("flags_written", []) or []) // 2))
        payload = _instruction_bytes(instruction)
        raw_byte_ids.extend([BYTE_BASE + value for value in payload])
        flow = instruction.get("control_flow") or {}
        for target in flow.get("targets", []) or []:
            if flow.get("kind") in {"call", "indirect_call"}:
                continue
            address = _target_va(target)
            if address in address_to_index:
                target_index = address_to_index[address]
                if target_index < len(ordered):
                    instruction_edges.append([index, target_index])
                    instruction_edge_types.append(1)
        if index + 1 < len(ordered):
            current_block = block_for_instruction.get(instruction.get("instruction_id"))
            next_block = block_for_instruction.get(ordered[index + 1].get("instruction_id"))
            if current_block == next_block:
                instruction_edges.append([index, index + 1])
                instruction_edge_types.append(0)
    for block_index, block in enumerate(blocks):
        cfg_target[block_index, block_index] = 0.0
        for successor in block.get("successors", []) or []:
            target_index = block_ids.get(successor)
            if target_index is None or target_index >= len(blocks):
                continue
            block_edges.append([block_index, target_index])
            block_edge_types.append(0)
            cfg_target[block_index, target_index] = 1.0
            source_instruction = next(
                (
                    index
                    for index, instruction in enumerate(ordered)
                    if sample["instr_block"][index] == block_index
                ),
                None,
            )
            target_instruction = next(
                (
                    index
                    for index, instruction in enumerate(ordered)
                    if sample["instr_block"][index] == target_index
                ),
                None,
            )
            if source_instruction is not None and target_instruction is not None:
                instruction_edges.append([source_instruction, target_instruction])
                instruction_edge_types.append(2)
    sample["instr_mnemonic"] = sample["instr_mnemonic"][:max_instructions]
    sample["instr_shape"] = sample["instr_shape"][:max_instructions]
    sample["instr_id"] = sample["instr_id"][:max_instructions]
    sample["instr_ir"] = sample["instr_ir"][:max_instructions]
    sample["instr_regs"] = sample["instr_regs"][:max_instructions]
    sample["instr_kinds"] = sample["instr_kinds"][:max_instructions]
    sample["instr_access"] = sample["instr_access"][:max_instructions]
    sample["instr_width"] = sample["instr_width"][:max_instructions]
    sample["instr_block"] = sample["instr_block"][:max_instructions]
    sample["instr_mask"] = sample["instr_mask"][:max_instructions]
    sample["instr_mem_read"] = sample["instr_mem_read"][:max_instructions]
    sample["instr_mem_write"] = sample["instr_mem_write"][:max_instructions]
    sample["instr_mem_width"] = sample["instr_mem_width"][:max_instructions]
    sample["instr_flags_read"] = sample["instr_flags_read"][:max_instructions]
    sample["instr_flags_written"] = sample["instr_flags_written"][:max_instructions]
    read_targets = read_targets[:max_instructions]
    write_targets = write_targets[:max_instructions]
    length = len(sample["instr_id"])
    limit = min(len(instruction_edges), max(8, length * 3))
    instruction_edges = [edge for edge in instruction_edges[:limit] if edge[0] < length and edge[1] < length]
    instruction_edge_types = instruction_edge_types[: len(instruction_edges)]
    for block_index in range(len(blocks)):
        block_edges.append([block_index, block_index])
        block_edge_types.append(3)
    block_edges = [edge for edge in block_edges if edge[0] < len(blocks) and edge[1] < len(blocks)]
    block_edge_types = block_edge_types[: len(block_edges)]
    sample["instruction_edges"] = instruction_edges
    sample["instruction_edge_types"] = instruction_edge_types
    sample["block_edges"] = block_edges
    sample["block_edge_types"] = block_edge_types
    sample["block_mask"] = [1] * len(blocks)
    sample["cfg_target"] = cfg_target.tolist()
    sample["read_targets"] = read_targets
    sample["write_targets"] = write_targets
    sample["byte_ids"] = raw_byte_ids[:max_bytes]
    sample["byte_mask"] = [1] * len(sample["byte_ids"])
    sample["function_start_target"] = ([1] + [0] * (len(sample["byte_ids"]) - 1)) if sample["byte_ids"] else []
    if not options.include_bytes:
        sample["byte_ids"] = []
        sample["byte_mask"] = []
    if not options.include_ir:
        sample["instr_ir"] = [0] * length
    if not options.include_graph:
        sample["instruction_edges"] = []
        sample["instruction_edge_types"] = []
        sample["block_edges"] = []
        sample["block_edge_types"] = []
        sample["cfg_target"] = [[0.0] * len(blocks) for _ in range(len(blocks))]
    fingerprint = hashlib.sha256()
    fingerprint.update("|".join(str(value) for value in sample["instr_id"]).encode())
    fingerprint.update("|".join(str(value) for value in sample["instr_ir"]).encode())
    if sample["byte_ids"]:
        fingerprint.update(bytes(value - BYTE_BASE for value in sample["byte_ids"]))
    sample["input_fingerprint"] = fingerprint.hexdigest()
    program = record.get("program", {})
    build = (identity.get("build") or {}) if isinstance(identity.get("build"), dict) else {}
    features = [
        _float_feature(np.log1p(length), 6.0),
        _float_feature(np.log1p(len(blocks)), 4.0),
        _float_feature(np.log1p(len(instruction_edges)), 5.0),
        _float_feature(len(instruction_edges) - length + 2, 6.0),
        _float_feature(np.log1p(len(function.get("callees", []) or [])), 3.0),
        _float_feature(np.log1p(len(function.get("callers", []) or [])), 3.0),
        1.0 if program.get("stripped") else 0.0,
        1.0 if program.get("pie") else 0.0,
        1.0 if program.get("static") else 0.0,
        1.0 if build.get("debug_info") else 0.0,
        1.0 if program.get("machine") == "x86_64" else 0.0,
        1.0 if program.get("machine") == "aarch64" else 0.0,
        1.0 if program.get("machine") == "arm" else 0.0,
        1.0 if program.get("machine") in {"mips", "riscv", "powerpc"} else 0.0,
        1.0 if build.get("optimization") == "O0" else 0.0,
        1.0 if build.get("optimization") in {"O2", "O3"} else 0.0,
        _float_feature(len(function.get("data_references", []) or []), 3.0),
        _float_feature(len(record.get("imports", []) or []), 5.0),
    ]
    sample["function_features"] = features
    if purpose_label and purpose_label in PURPOSE_INDEX:
        sample["purpose_id"] = PURPOSE_INDEX[purpose_label]
        sample["purpose_label"] = purpose_label
    language = str(identity.get("language") or "c")
    if source_label and options.include_source_target:
        sample["source_text"] = source_label["text"]
        sample["source_sha256"] = source_label["source_sha256"]
        sample["source_path"] = source_label.get("source_path")
        encoded = vocab.encode_text(source_label["text"], config.max_text_tokens)
        if language == "java":
            sample["java_ids"] = encoded
            sample["java_mask"] = [1] * len(encoded)
        else:
            sample["c_ids"] = encoded
            sample["c_mask"] = [1] * len(encoded)
        sample["source_ids"] = encoded[: config.max_text_tokens]
        sample["availability"]["source"] = True
        sample["labels"]["source_alignment"] = {
            "sha256": source_label["source_sha256"],
            "alignment": source_label.get("alignment"),
            "confidence": source_label.get("confidence"),
        }
    if identity.get("jni_targets") is not None:
        sample["jni_targets"] = list(identity["jni_targets"])[:5]
        while len(sample["jni_targets"]) < 5:
            sample["jni_targets"].append(-100)
    sample["labels"]["language"] = language
    sample["labels"]["role"] = "function"
    return sample


def encode_record_samples(
    record: dict[str, Any],
    vocab: Vocabulary,
    config: FoundationConfig,
    identity: dict[str, Any],
    options: RepresentationOptions | None = None,
    source_labels: dict[str, dict[str, Any]] | None = None,
    max_functions: int | None = None,
    purpose_labels: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    options = options or RepresentationOptions()
    source_labels = source_labels or {}
    purpose_labels = purpose_labels or {}
    register_targets = vocab.register_targets(config.max_register_targets)
    samples = []
    functions = [function for function in record.get("functions", []) if isinstance(function, dict)]
    for function in functions[: max_functions or len(functions)]:
        function_id = function.get("function_id")
        sample = encode_function_sample(
            record,
            function,
            vocab,
            config,
            identity={**identity, "function_id": function_id},
            options=options,
            source_label=source_labels.get(function_id),
            purpose_label=purpose_labels.get(function_id),
            register_targets=register_targets,
        )
        if sample is not None:
            samples.append(sample)
    return samples


def _pad_1d(values: list[int], length: int, fill: int = 0) -> list[int]:
    return (values + [fill] * length)[:length]


def _pad_2d(values: list[list[int]], rows: int, columns: int, fill: int = 0) -> list[list[int]]:
    result = []
    for row in values[:rows]:
        result.append((list(row) + [fill] * columns)[:columns])
    while len(result) < rows:
        result.append([fill] * columns)
    return result


def _pad_edges(edges: Iterable[list[int]], count: int) -> tuple[torch.Tensor, torch.Tensor]:
    edges = [edge for edge in edges if len(edge) == 2][:count]
    index = torch.zeros((count, 2), dtype=torch.long)
    mask = torch.zeros((count,), dtype=torch.bool)
    for position, edge in enumerate(edges):
        index[position, 0] = int(edge[0])
        index[position, 1] = int(edge[1])
        mask[position] = True
    return index, mask


def collate_samples(samples: list[dict[str, Any]], config: FoundationConfig) -> dict[str, Any]:
    if not samples:
        raise ValueError("cannot collate an empty sample list")
    batch_size = len(samples)
    length = max(2, min(config.max_instructions, max(len(sample["instr_id"]) for sample in samples)))
    bytes_length = min(config.max_bytes, max(1, max(len(sample["byte_ids"]) for sample in samples)))
    blocks = min(config.max_blocks, max(1, max(len(sample["block_mask"]) for sample in samples)))
    operand_slots = config.max_operands
    register_count = config.max_register_targets
    instruction_edge_count = max(1, max(len(sample["instruction_edges"]) for sample in samples))
    block_edge_count = max(1, max(len(sample["block_edges"]) for sample in samples))
    result: dict[str, Any] = {}
    result["instr_mnemonic"] = torch.tensor(
        [_pad_1d(sample["instr_mnemonic"], length, PAD) for sample in samples], dtype=torch.long
    )
    result["instr_shape"] = torch.tensor(
        [_pad_1d(sample["instr_shape"], length, PAD) for sample in samples], dtype=torch.long
    )
    result["instr_id"] = torch.tensor([_pad_1d(sample["instr_id"], length, PAD) for sample in samples], dtype=torch.long)
    result["instr_ir"] = torch.tensor([_pad_1d(sample["instr_ir"], length, 0) for sample in samples], dtype=torch.long)
    result["instr_regs"] = torch.tensor(
        [_pad_2d(sample["instr_regs"], length, operand_slots, PAD) for sample in samples], dtype=torch.long
    )
    result["instr_kinds"] = torch.tensor(
        [_pad_2d(sample["instr_kinds"], length, operand_slots, 0) for sample in samples], dtype=torch.long
    )
    result["instr_access"] = torch.tensor(
        [_pad_2d(sample["instr_access"], length, operand_slots, 0) for sample in samples], dtype=torch.long
    )
    result["instr_width"] = torch.tensor(
        [_pad_2d(sample["instr_width"], length, operand_slots, 0) for sample in samples], dtype=torch.long
    )
    result["instr_block"] = torch.tensor(
        [_pad_1d(sample["instr_block"], length, 0) for sample in samples], dtype=torch.long
    )
    result["instr_mask"] = torch.tensor(
        [_pad_1d(sample["instr_mask"], length, 0) for sample in samples], dtype=torch.bool
    )
    result["instr_mem_read"] = torch.tensor(
        [_pad_1d(sample["instr_mem_read"], length, 0) for sample in samples], dtype=torch.float32
    )
    result["instr_mem_write"] = torch.tensor(
        [_pad_1d(sample["instr_mem_write"], length, 0) for sample in samples], dtype=torch.float32
    )
    result["instr_mem_width"] = torch.tensor(
        [_pad_1d(sample["instr_mem_width"], length, 0) for sample in samples], dtype=torch.long
    )
    result["instr_flags_read"] = torch.tensor(
        [_pad_1d(sample["instr_flags_read"], length, 0) for sample in samples], dtype=torch.float32
    )
    result["instr_flags_written"] = torch.tensor(
        [_pad_1d(sample["instr_flags_written"], length, 0) for sample in samples], dtype=torch.float32
    )
    result["byte_ids"] = torch.tensor(
        [_pad_1d(sample["byte_ids"], bytes_length, PAD) for sample in samples], dtype=torch.long
    )
    result["byte_mask"] = torch.tensor(
        [_pad_1d(sample["byte_mask"], bytes_length, 0) for sample in samples], dtype=torch.bool
    )
    result["block_mask"] = torch.tensor(
        [_pad_1d(sample["block_mask"], blocks, 0) for sample in samples], dtype=torch.bool
    )
    instruction_edges = torch.zeros((batch_size, instruction_edge_count, 2), dtype=torch.long)
    instruction_edge_mask = torch.zeros((batch_size, instruction_edge_count), dtype=torch.bool)
    instruction_edge_types = torch.zeros((batch_size, instruction_edge_count), dtype=torch.long)
    block_edges = torch.zeros((batch_size, block_edge_count, 2), dtype=torch.long)
    block_edge_mask = torch.zeros((batch_size, block_edge_count), dtype=torch.bool)
    block_edge_types = torch.zeros((batch_size, block_edge_count), dtype=torch.long)
    for index, sample in enumerate(samples):
        edges, mask = _pad_edges(sample["instruction_edges"], instruction_edge_count)
        instruction_edges[index] = edges
        instruction_edge_mask[index] = mask
        types = (sample["instruction_edge_types"] + [0] * instruction_edge_count)[:instruction_edge_count]
        instruction_edge_types[index] = torch.tensor(types, dtype=torch.long)
        b_edges, b_mask = _pad_edges(sample["block_edges"], block_edge_count)
        block_edges[index] = b_edges
        block_edge_mask[index] = b_mask
        b_types = (sample["block_edge_types"] + [0] * block_edge_count)[:block_edge_count]
        block_edge_types[index] = torch.tensor(b_types, dtype=torch.long)
    result["instruction_edges"] = instruction_edges
    result["instruction_edge_mask"] = instruction_edge_mask
    result["instruction_edge_types"] = instruction_edge_types
    result["block_edges"] = block_edges
    result["block_edge_mask"] = block_edge_mask
    result["block_edge_types"] = block_edge_types
    cfg = torch.zeros((batch_size, blocks, blocks), dtype=torch.float32)
    for index, sample in enumerate(samples):
        matrix = np.asarray(sample["cfg_target"], dtype=np.float32)
        rows = min(blocks, matrix.shape[0])
        cols = min(blocks, matrix.shape[1])
        cfg[index, :rows, :cols] = torch.tensor(matrix[:rows, :cols], dtype=torch.float32)
    result["cfg_target"] = cfg
    result["cfg_mask"] = torch.ones((batch_size, blocks, blocks), dtype=torch.bool)
    result["read_targets"] = torch.tensor(
        [_pad_2d(sample["read_targets"], length, register_count, 0) for sample in samples], dtype=torch.float32
    )
    result["write_targets"] = torch.tensor(
        [_pad_2d(sample["write_targets"], length, register_count, 0) for sample in samples], dtype=torch.float32
    )
    result["function_features"] = torch.tensor(
        [sample["function_features"] for sample in samples], dtype=torch.float32
    )
    result["purpose_target"] = torch.tensor([sample["purpose_id"] for sample in samples], dtype=torch.long)
    result["tool_target"] = torch.tensor([sample["tool_target"] for sample in samples], dtype=torch.long)
    result["confidence_target"] = torch.tensor(
        [sample["confidence_target"] for sample in samples], dtype=torch.float32
    )
    text_length = min(
        config.max_text_tokens,
        max(
            [len(sample["c_ids"]) for sample in samples]
            + [len(sample["java_ids"]) for sample in samples]
            + [len(sample["source_ids"]) for sample in samples]
            + [1]
        ),
    )
    result["c_ids"] = torch.tensor(
        [_pad_1d(sample["c_ids"], text_length, PAD) for sample in samples], dtype=torch.long
    )
    result["c_mask"] = torch.tensor(
        [_pad_1d(sample["c_mask"], text_length, 0) for sample in samples], dtype=torch.bool
    )
    result["source_ids"] = torch.tensor(
        [_pad_1d(sample["source_ids"], text_length, PAD) for sample in samples], dtype=torch.long
    )
    result["java_ids"] = torch.tensor(
        [_pad_1d(sample["java_ids"], text_length, PAD) for sample in samples], dtype=torch.long
    )
    result["java_mask"] = torch.tensor(
        [_pad_1d(sample["java_mask"], text_length, 0) for sample in samples], dtype=torch.bool
    )
    result["function_start_target"] = torch.tensor(
        [_pad_1d(sample["function_start_target"], bytes_length, 0) for sample in samples], dtype=torch.float32
    )
    result["confidence_mask"] = torch.tensor(
        [sample["confidence_mask"] for sample in samples], dtype=torch.bool
    )
    result["security_target"] = torch.tensor([sample["security_target"] for sample in samples], dtype=torch.long)
    result["jni_targets"] = torch.tensor([sample["jni_targets"] for sample in samples], dtype=torch.long)
    result["sample_meta"] = [
        {
            "sample_id": sample["sample_id"],
            "function_id": sample["function_id"],
            "project_id": sample["project_id"],
            "dataset_id": sample["dataset_id"],
            "split": sample["split"],
            "binary_sha256": sample["binary_sha256"],
            "source_family": sample["source_family"],
            "match_group": sample["match_group"],
            "purpose_label": sample["purpose_label"],
            "input_fingerprint": sample.get("input_fingerprint"),
            "source_sha256": sample.get("source_sha256"),
            "availability": sample.get("availability"),
            "input_provenance": sample.get("input_provenance"),
            "labels": sample.get("labels"),
        }
        for sample in samples
    ]
    return result


def mask_instruction_batch(
    batch: dict[str, Any],
    probability: float = 0.15,
    seed: int = 0,
    mask_id: int = 2,
    *,
    mask_identity_fields: bool = True,
) -> dict[str, Any]:
    """Mask instruction identity and, by default, every field that copies it.

    The historical implementation masked only ``instr_id`` while leaving the
    mnemonic, shape and IR visible, so the target was reconstructible by
    concatenation.  The default now masks the identity-bearing fields jointly.
    """
    ids = batch["instr_id"].clone()
    mask = batch["instr_mask"].clone()
    generator = torch.Generator().manual_seed(seed)
    selected = (torch.rand(ids.shape, generator=generator) < probability) & mask
    for row in range(ids.size(0)):
        if not selected[row].any() and mask[row].any():
            selected[row, int(torch.nonzero(mask[row], as_tuple=False)[0, 0])] = True
    targets = ids.clone()
    ids[selected] = mask_id
    result = dict(batch)
    result["instr_id"] = ids
    result["instr_targets"] = targets
    result["instr_mlm_mask"] = selected
    result["instruction_identity_mask"] = selected
    if mask_identity_fields:
        for field in ("instr_mnemonic", "instr_shape", "instr_ir"):
            values = batch[field].clone()
            values[selected] = mask_id
            result[field] = values
        for field in ("instr_regs", "instr_kinds"):
            values = batch[field].clone()
            values[selected] = mask_id
            result[field] = values
        result["instr_access"] = torch.zeros_like(batch["instr_access"])
        result["instr_width"] = torch.zeros_like(batch["instr_width"])
    return result


def autoregressive_instruction_batch(batch: dict[str, Any]) -> dict[str, Any]:
    ids = batch["instr_id"]
    mask = batch["instr_mask"]
    result = dict(batch)
    result["instr_ar_input"] = ids[:, :-1]
    result["instr_ar_targets"] = ids[:, 1:]
    result["instr_ar_mask"] = mask[:, 1:]
    return result


def mask_byte_batch(batch: dict[str, Any], probability: float = 0.10, seed: int = 0, mask_id: int = 2) -> dict[str, Any]:
    ids = batch["byte_ids"].clone()
    mask = batch["byte_mask"].clone()
    generator = torch.Generator().manual_seed(seed + 7919)
    selected = (torch.rand(ids.shape, generator=generator) < probability) & mask
    for row in range(ids.size(0)):
        if not selected[row].any() and mask[row].any():
            selected[row, int(torch.nonzero(mask[row], as_tuple=False)[0, 0])] = True
    targets = ids.clone()
    ids[selected] = mask_id
    result = dict(batch)
    result["byte_ids"] = ids
    result["byte_targets"] = targets
    result["byte_mlm_mask"] = selected
    return result


def collate_function_module(samples: list[dict[str, Any]], config: FoundationConfig) -> dict[str, Any]:
    """Collate a group of functions from one module for hierarchical encoding."""
    batch = collate_samples(samples, config)
    count = len(samples)
    entry_flags = torch.zeros((count,), dtype=torch.float32)
    entry_flags[0] = 1.0
    batch["function_entry_flags"] = entry_flags
    batch["function_slot"] = torch.arange(count, dtype=torch.long)
    return batch


def mask_evidence_batch(batch: dict[str, Any], probability: float = 0.35, seed: int = 0, mask_id: int = 2) -> dict[str, Any]:
    """Create a label-isolated view for IR, register, memory and CFG objectives.

    The IR target is masked in the input, access/width fields that directly
    encode memory/register effects are removed, and graph edges are hidden.
    This prevents the model from copying its targets while leaving the
    instruction/byte evidence available.
    """
    result = dict(batch)
    source_ir = batch.get("ir_target_source", batch["instr_ir"])
    ir = batch["instr_ir"].clone()
    instruction_mask = batch["instr_mask"]
    generator = torch.Generator().manual_seed(seed + 31337)
    selected = (torch.rand(ir.shape, generator=generator) < probability) & instruction_mask
    for row in range(ir.size(0)):
        if not selected[row].any() and instruction_mask[row].any():
            selected[row, int(torch.nonzero(instruction_mask[row], as_tuple=False)[0, 0])] = True
    targets = source_ir.clone()
    ir[selected] = mask_id
    result["instr_ir"] = ir
    result["ir_targets"] = targets
    result["ir_mlm_mask"] = selected
    for field in ("instr_mnemonic", "instr_shape"):
        values = batch[field].clone()
        values[selected] = mask_id
        result[field] = values
    for field in ("instr_regs", "instr_kinds"):
        values = batch[field].clone()
        values[selected] = mask_id
        result[field] = values
    result["instr_access"] = torch.zeros_like(batch["instr_access"])
    result["instr_width"] = torch.zeros_like(batch["instr_width"])
    result["instruction_edge_mask"] = torch.zeros_like(batch["instruction_edge_mask"])
    result["block_edge_mask"] = torch.zeros_like(batch["block_edge_mask"])
    return result
