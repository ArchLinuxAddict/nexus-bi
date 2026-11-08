"""Deterministic, provenance-tracked vocabularies and tokenizers.

The tokenizer is trained from registered datasets (counts, deterministic
ordering) and saved with a digest.  Unknown symbols fall back to bytes.  The
tokenizer never uses symbol names, source code or debug information for the
binary-only input path; those are target-side only for reconstruction.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

SPECIALS = [
    "<pad>",
    "<unk>",
    "<mask>",
    "<bos>",
    "<eos>",
    "<sep>",
    "<missing>",
    "<num>",
    "<mem>",
    "<reg>",
    "<imm>",
    "<addr>",
]
PAD, UNK, MASK, BOS, EOS, SEP, MISSING, NUM, MEM, REG, IMM, ADDR = range(len(SPECIALS))
BYTE_BASE = len(SPECIALS)
BYTE_COUNT = 256

OPERAND_KINDS = ["pad", "register", "memory", "immediate", "address", "unknown"]
ACCESS_KINDS = ["none", "read", "write", "read_write", "unknown"]
MEMORY_WIDTHS = ["none", "8", "16", "32", "64", "128", "other"]

TEXT_TOKEN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|0[xX][0-9a-fA-F]+|\d+|\s+|[^\s]")
_CANONICAL_REGISTER = re.compile(r"^%?([a-z][a-z0-9]*)$")


def _digest(payload: Any) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def canonical_register_name(value: str) -> str:
    text = str(value or "").strip().lower().lstrip("%$")
    if not text:
        return ""
    if re.fullmatch(r"(?:a64|arm|rv|mips|jvm)_[a-z0-9_]+", text):
        return text
    aliases = {
        "al": "rax", "ah": "rax", "ax": "rax", "eax": "rax", "rax": "rax",
        "bl": "rbx", "bh": "rbx", "bx": "rbx", "ebx": "rbx", "rbx": "rbx",
        "cl": "rcx", "ch": "rcx", "cx": "rcx", "ecx": "rcx", "rcx": "rcx",
        "dl": "rdx", "dh": "rdx", "dx": "rdx", "edx": "rdx", "rdx": "rdx",
        "sil": "rsi", "si": "rsi", "esi": "rsi", "rsi": "rsi",
        "dil": "rdi", "di": "rdi", "edi": "rdi", "rdi": "rdi",
        "spl": "rsp", "esp": "rsp", "rsp": "rsp",
        "bpl": "rbp", "bp": "rbp", "ebp": "rbp", "rbp": "rbp",
        "sp": "rsp", "rip": "rip", "eip": "eip",
    }
    if text in aliases:
        return aliases[text]
    if re.fullmatch(r"[re]?[abcd]x|r(?:[89]|1[0-5])[bwd]?|[xyz]mm\d+|st\d*|mm\d+|k\d+", text):
        return text
    return ""


def width_bucket(width: int | None) -> int:
    if not width:
        return 0
    if width <= 8:
        return 1
    if width <= 16:
        return 2
    if width <= 32:
        return 3
    if width <= 64:
        return 4
    if width <= 128:
        return 5
    return 6


def immediate_bucket(value: str) -> str:
    text = str(value or "").strip().lower()
    negative = text.startswith("-")
    text = text.lstrip("-").lstrip("$").lstrip("+")
    try:
        number = int(text, 0)
    except ValueError:
        return "nonnumeric"
    magnitude = abs(number)
    if magnitude in {0, 1, 2, 4, 8, -1}:
        return f"small_{'neg' if negative else 'pos'}_{magnitude}"
    bits = magnitude.bit_length()
    if bits <= 8:
        return "u8"
    if bits <= 16:
        return "u16"
    if bits <= 32:
        return "u32"
    if bits <= 64:
        return "u64"
    return "wide"


def instruction_shape(operands: list[dict[str, Any]]) -> str:
    parts = []
    for operand in operands:
        kind = str(operand.get("kind", "unknown"))
        if kind == "register":
            parts.append(f"reg{operand.get('width') or width_bucket(None)}")
        elif kind == "memory":
            memory = operand.get("memory") or {}
            parts.append(f"mem{width_bucket(memory.get('width') or operand.get('width'))}")
        elif kind == "immediate":
            parts.append(f"imm:{immediate_bucket(operand.get('value', ''))}")
        elif kind == "address":
            parts.append("addr")
        else:
            parts.append("unk")
    return ",".join(parts) if parts else "none"


def canonical_instruction(mnemonic: str, operands: list[dict[str, Any]]) -> str:
    return f"{mnemonic or 'unknown'}|{instruction_shape(operands)}"


@dataclass
class Vocabulary:
    text_vocab_size: int = 4096
    mnemonic: list[str] = field(default_factory=list)
    shape: list[str] = field(default_factory=list)
    instruction: list[str] = field(default_factory=list)
    register: list[str] = field(default_factory=list)
    ir_op: list[str] = field(default_factory=list)
    text: list[str] = field(default_factory=list)
    counts: dict[str, dict[str, int]] = field(default_factory=dict)
    source_fingerprint: str = ""
    schema_version: str = "0.1.0"

    def __post_init__(self) -> None:
        if not self.mnemonic:
            self.mnemonic = list(SPECIALS) + ["<other>"]
        if not self.shape:
            self.shape = list(SPECIALS) + ["none"]
        if not self.instruction:
            self.instruction = list(SPECIALS) + ["<other>"]
        if not self.register:
            self.register = list(SPECIALS) + ["<other>"]
        if not self.ir_op:
            self.ir_op = list(SPECIALS) + ["other"]
        if not self.text:
            self.text = self._default_text_vocab()

    def _default_text_vocab(self) -> list[str]:
        tokens = list(SPECIALS)
        tokens.extend(f"<byte:{value:02x}>" for value in range(BYTE_COUNT))
        return tokens

    def _index(self, name: str, values: list[str]) -> dict[str, int]:
        return {value: index for index, value in enumerate(values)}

    @property
    def mnemonic_index(self) -> dict[str, int]:
        return self._index("mnemonic", self.mnemonic)

    @property
    def shape_index(self) -> dict[str, int]:
        return self._index("shape", self.shape)

    @property
    def instruction_index(self) -> dict[str, int]:
        return self._index("instruction", self.instruction)

    @property
    def register_index(self) -> dict[str, int]:
        return self._index("register", self.register)

    @property
    def ir_index(self) -> dict[str, int]:
        return self._index("ir_op", self.ir_op)

    @property
    def text_index(self) -> dict[str, int]:
        return self._index("text", self.text)

    def encode(self, name: str, value: str) -> int:
        index = getattr(self, f"{name}_index")
        return index.get(value, UNK)

    def register_targets(self, limit: int) -> list[str]:
        common = [
            "rax", "rbx", "rcx", "rdx", "rsi", "rdi", "rsp", "rbp",
            "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15",
            "rip", "xmm0", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5", "xmm6",
        ]
        present = [name for name in common if name in set(self.register)]
        for name in self.register:
            if name not in present and name not in SPECIALS and name != "<other>":
                present.append(name)
            if len(present) >= limit:
                break
        return present[:limit]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "text_vocab_size": self.text_vocab_size,
            "mnemonic": self.mnemonic,
            "shape": self.shape,
            "instruction": self.instruction,
            "register": self.register,
            "ir_op": self.ir_op,
            "text": self.text,
            "counts": self.counts,
            "source_fingerprint": self.source_fingerprint,
        }

    def digest(self) -> str:
        return _digest(self.to_dict())

    def save(self, path: str | Path) -> dict[str, Any]:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = self.to_dict()
        payload["vocab_sha256"] = self.digest()
        target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return {"path": str(target), "sha256": payload["vocab_sha256"], "size": target.stat().st_size}

    @classmethod
    def load(cls, path: str | Path) -> "Vocabulary":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        digest = payload.pop("vocab_sha256", None)
        vocab = cls(**{key: value for key, value in payload.items() if key in cls.__dataclass_fields__})
        if digest is not None and vocab.digest() != digest:
            raise ValueError("vocabulary digest mismatch")
        return vocab

    def encode_text(self, text: str, max_tokens: int) -> list[int]:
        index = self.text_index
        result = [BOS]
        for token in TEXT_TOKEN.findall(str(text or "")):
            identifier = index.get(token)
            if identifier is not None:
                result.append(identifier)
            else:
                for byte in token.encode("utf-8", errors="replace"):
                    result.append(index.get(f"<byte:{byte:02x}>", UNK))
            if len(result) >= max_tokens - 1:
                break
        result.append(EOS)
        return result[:max_tokens]

    def decode_text(self, ids: Iterable[int]) -> str:
        pieces = []
        for identifier in ids:
            if identifier in {PAD, BOS, EOS, SEP, MASK, MISSING}:
                continue
            token = self.text[identifier] if 0 <= identifier < len(self.text) else "<unk>"
            match = re.fullmatch(r"<byte:([0-9a-f]{2})>", token)
            pieces.append(bytes([int(match.group(1), 16)]).decode("latin-1") if match else token)
        return "".join(pieces)


def _top(counter: Counter[str], limit: int, fallback: list[str]) -> list[str]:
    ordered = [token for token, _ in sorted(counter.items(), key=lambda item: (-item[1], item[0])) if token]
    values = list(fallback)
    for token in ordered:
        if token not in values:
            values.append(token)
        if len(values) >= limit:
            break
    return values[:limit]


def build_vocabulary(
    records: Iterable[dict[str, Any]],
    sources: Iterable[str] = (),
    text_vocab_size: int = 4096,
    source_fingerprint: str = "",
) -> Vocabulary:
    mnemonic = Counter()
    shape = Counter()
    instruction = Counter()
    register = Counter()
    ir_ops = Counter()
    text = Counter()
    record_count = 0
    for record in records:
        record_count += 1
        instructions = record.get("instructions", [])
        for item in instructions:
            name = str(item.get("mnemonic", "unknown"))
            mnemonic[name] += 1
            operands = [operand for operand in item.get("operands", []) if isinstance(operand, dict)]
            shape_value = instruction_shape(operands)
            shape[shape_value] += 1
            instruction[canonical_instruction(name, operands)] += 1
            register_id = canonical_register_name(item.get("register", ""))
            if register_id:
                register[register_id] += 1
            for operand in operands:
                kind = operand.get("kind")
                if kind == "register":
                    value = canonical_register_name(operand.get("register") or operand.get("value", ""))
                    if value:
                        register[value] += 1
                elif kind == "memory":
                    memory = operand.get("memory") or {}
                    for key in ("base", "index"):
                        value = canonical_register_name(memory.get(key) or "")
                        if value:
                            register[value] += 1
            for key in ("reads", "writes"):
                for value in item.get(key, []) or []:
                    canonical = canonical_register_name(value)
                    if canonical:
                        register[canonical] += 1
        for flow in record.get("dataflow", []):
            if isinstance(flow, dict):
                ir_ops[str(flow.get("kind") or flow.get("op") or "other")] += 1
        for block in record.get("blocks", []) or []:
            if isinstance(block, dict) and block.get("loop_header"):
                ir_ops["loop_header"] += 1
    for source in sources:
        text.update(TEXT_TOKEN.findall(source))
    minimum_text = len(SPECIALS) + BYTE_COUNT + 8
    text_limit = max(minimum_text, text_vocab_size)
    text_tokens = list(SPECIALS)
    text_tokens.extend(f"<byte:{value:02x}>" for value in range(BYTE_COUNT))
    for token, _ in sorted(text.items(), key=lambda item: (-item[1], item[0])):
        if len(text_tokens) >= text_limit:
            break
        if token not in text_tokens:
            text_tokens.append(token)
    return Vocabulary(
        text_vocab_size=text_limit,
        mnemonic=_top(mnemonic, 512, list(SPECIALS) + ["<other>"]),
        shape=_top(shape, 256, list(SPECIALS) + ["none"]),
        instruction=_top(instruction, 1024, list(SPECIALS) + ["<other>"]),
        register=_top(register, 128, list(SPECIALS) + ["<other>"]),
        ir_op=_top(ir_ops, 64, list(SPECIALS) + ["other"]),
        text=text_tokens,
        counts={
            "records": record_count,
            "instruction_symbols": len(instruction),
            "mnemonics": len(mnemonic),
            "shapes": len(shape),
            "registers": len(register),
            "text_symbols": len(text),
        },
        source_fingerprint=source_fingerprint,
    )


def build_vocabulary_from_records(
    record_paths: Iterable[str | Path],
    source_texts: Iterable[str] = (),
    text_vocab_size: int = 4096,
    source_fingerprint: str = "",
) -> Vocabulary:
    def iterator():
        for path in record_paths:
            yield json.loads(Path(path).read_text(encoding="utf-8"))

    return build_vocabulary(iterator(), list(source_texts), text_vocab_size=text_vocab_size, source_fingerprint=source_fingerprint)


def extend_vocabulary(
    base: Vocabulary,
    records: Iterable[dict[str, Any]],
    sources: Iterable[str] = (),
    *,
    mnemonic_cap: int = 2048,
    shape_cap: int = 1024,
    instruction_cap: int = 4096,
    register_cap: int = 512,
    ir_cap: int = 128,
    text_cap: int = 65536,
    source_fingerprint: str = "",
) -> Vocabulary:
    """Append new training-only symbols while preserving every existing ID.

    Existing sample stores can therefore keep their token IDs; only the
    embedding tables grow.  New symbols are appended in frequency order so the
    extension is deterministic.
    """
    mnemonic = Counter()
    shape = Counter()
    instruction = Counter()
    register = Counter()
    ir_ops = Counter()
    text = Counter()
    record_count = 0
    for record in records:
        record_count += 1
        for item in record.get("instructions", []):
            name = str(item.get("mnemonic", "unknown"))
            mnemonic[name] += 1
            operands = [operand for operand in item.get("operands", []) if isinstance(operand, dict)]
            shape[instruction_shape(operands)] += 1
            instruction[canonical_instruction(name, operands)] += 1
            register_id = canonical_register_name(item.get("register", ""))
            if register_id:
                register[register_id] += 1
            for operand in operands:
                kind = operand.get("kind")
                if kind == "register":
                    value = canonical_register_name(operand.get("register") or operand.get("value", ""))
                    if value:
                        register[value] += 1
                elif kind == "memory":
                    memory = operand.get("memory") or {}
                    for key in ("base", "index"):
                        value = canonical_register_name(memory.get(key) or "")
                        if value:
                            register[value] += 1
            for key in ("reads", "writes"):
                for value in item.get(key, []) or []:
                    canonical = canonical_register_name(value)
                    if canonical:
                        register[canonical] += 1
        for flow in record.get("dataflow", []):
            if isinstance(flow, dict):
                ir_ops[str(flow.get("kind") or flow.get("op") or "other")] += 1
        for block in record.get("blocks", []) or []:
            if isinstance(block, dict) and block.get("loop_header"):
                ir_ops["loop_header"] += 1
    for source in sources:
        text.update(TEXT_TOKEN.findall(source))

    def append_values(existing: list[str], counter: Counter[str], cap: int) -> list[str]:
        values = list(existing)
        limit = max(cap, len(values))
        seen = set(values)
        for token, _ in sorted(counter.items(), key=lambda item: (-item[1], item[0])):
            if token and token not in seen:
                values.append(token)
                seen.add(token)
            if len(values) >= limit:
                break
        return values

    new_text = list(base.text)
    text_limit = max(text_cap, len(new_text))
    seen_text = set(new_text)
    for token, _ in sorted(text.items(), key=lambda item: (-item[1], item[0])):
        if token not in seen_text:
            new_text.append(token)
            seen_text.add(token)
        if len(new_text) >= text_limit:
            break
    counts = dict(base.counts)
    counts["records"] = int(counts.get("records", 0)) + record_count
    counts["v4_new_mnemonics"] = len(mnemonic)
    counts["v4_new_shapes"] = len(shape)
    counts["v4_new_instructions"] = len(instruction)
    counts["v4_new_registers"] = len(register)
    counts["v4_new_text_symbols"] = len(text)
    return Vocabulary(
        text_vocab_size=text_limit,
        mnemonic=append_values(base.mnemonic, mnemonic, mnemonic_cap),
        shape=append_values(base.shape, shape, shape_cap),
        instruction=append_values(base.instruction, instruction, instruction_cap),
        register=append_values(base.register, register, register_cap),
        ir_op=append_values(base.ir_op, ir_ops, ir_cap),
        text=new_text,
        counts=counts,
        source_fingerprint=(base.source_fingerprint + "|" + source_fingerprint) if source_fingerprint else base.source_fingerprint,
        schema_version=base.schema_version,
    )
