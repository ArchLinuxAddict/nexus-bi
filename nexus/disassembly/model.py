from dataclasses import dataclass, field
from typing import Any


@dataclass
class MemoryEffect:
    kind: str
    address: str | None = None
    width: int | None = None
    base: str | None = None
    index: str | None = None
    displacement: int | None = None
    segment: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "address": self.address,
            "width": self.width,
            "base": self.base,
            "index": self.index,
            "displacement": self.displacement,
        }


@dataclass
class Target:
    va: int | None
    kind: str = "direct"
    confidence: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "va": f"0x{self.va:x}" if self.va is not None else None,
            "kind": self.kind,
            "confidence": self.confidence,
        }


@dataclass
class Instruction:
    address: int
    bytes_hex: str
    length: int
    text: str
    mnemonic: str
    operands: list[str]
    function_label: str | None
    section: str | None
    comment: str | None = None
    reads: list[str] = field(default_factory=list)
    writes: list[str] = field(default_factory=list)
    memory: list[MemoryEffect] = field(default_factory=list)
    control_flow: str = "fallthrough"
    targets: list[Target] = field(default_factory=list)
    flags_read: list[str] = field(default_factory=list)
    flags_written: list[str] = field(default_factory=list)
    decoded: bool = True
    confidence: float = 1.0

    @property
    def end(self) -> int:
        return self.address + self.length

    def to_dict(self) -> dict[str, Any]:
        return {
            "address": self.address,
            "bytes": self.bytes_hex,
            "length": self.length,
            "text": self.text,
            "mnemonic": self.mnemonic,
            "operands": self.operands,
            "function_label": self.function_label,
            "section": self.section,
            "comment": self.comment,
            "reads": self.reads,
            "writes": self.writes,
            "memory": [item.to_dict() for item in self.memory],
            "control_flow": self.control_flow,
            "targets": [item.to_dict() for item in self.targets],
            "flags_read": self.flags_read,
            "flags_written": self.flags_written,
            "decoded": self.decoded,
            "confidence": self.confidence,
        }


@dataclass
class Disassembly:
    backend: str
    version: str | None
    instructions: list[Instruction]
    functions: dict[str, int]
    sections: list[str]
    raw_output_sha256: str | None = None
    limitations: list[str] = field(default_factory=list)

    def by_address(self) -> dict[int, Instruction]:
        return {instruction.address: instruction for instruction in self.instructions}
