from dataclasses import dataclass, field
from typing import Any

from nexus.disassembly.model import Instruction


@dataclass
class BasicBlock:
    block_id: str
    function_id: str
    start: int
    end: int
    instructions: list[Instruction] = field(default_factory=list)
    successors: list[str] = field(default_factory=list)
    predecessors: list[str] = field(default_factory=list)
    confidence: float = 0.8
    evidence: list[str] = field(default_factory=list)

    @property
    def instruction_ids(self) -> list[str]:
        return [f"ins_{instruction.address:x}" for instruction in self.instructions]

    def to_dict(self) -> dict[str, Any]:
        return {
            "block_id": self.block_id,
            "range": {"start_va": f"0x{self.start:x}", "end_va": f"0x{self.end:x}", "size": self.end - self.start},
            "instruction_ids": self.instruction_ids,
            "successors": self.successors,
            "predecessors": self.predecessors,
            "confidence": self.confidence,
            "evidence": self.evidence,
        }


@dataclass
class CallEdge:
    edge_id: str
    source: str
    target: str
    kind: str
    call_site: int
    confidence: float
    target_name: str | None = None
    evidence: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "edge_id": self.edge_id,
            "source": self.source,
            "target": self.target,
            "kind": self.kind,
            "call_site_va": f"0x{self.call_site:x}",
            "confidence": self.confidence,
            "target_name": self.target_name,
            "evidence": self.evidence,
        }


@dataclass
class CallGraph:
    nodes: dict[str, dict[str, Any]] = field(default_factory=dict)
    edges: list[CallEdge] = field(default_factory=list)
    limitations: list[str] = field(default_factory=list)

    def add_node(self, node_id: str, **attributes: Any) -> None:
        self.nodes.setdefault(node_id, {"node_id": node_id, **attributes})

    def add_edge(self, edge: CallEdge) -> None:
        self.edges.append(edge)

    def to_dict(self) -> dict[str, Any]:
        return {
            "nodes": list(self.nodes.values()),
            "edges": [edge.to_dict() for edge in self.edges],
            "confidence": 0.6,
            "limitations": self.limitations,
        }

    def callees_of(self, function_id: str) -> list[CallEdge]:
        return [edge for edge in self.edges if edge.source == function_id]

    def callers_of(self, function_id: str) -> list[CallEdge]:
        return [edge for edge in self.edges if edge.target == function_id]


@dataclass
class FunctionCandidate:
    function_id: str
    name: str
    entry: int
    ranges: list[tuple[int, int]]
    size: int
    source: str
    confidence: float
    instructions: list[Instruction] = field(default_factory=list)
    blocks: list[BasicBlock] = field(default_factory=list)
    callees: list[CallEdge] = field(default_factory=list)
    callers: list[CallEdge] = field(default_factory=list)
    aliases: list[str] = field(default_factory=list)
    evidence: list[str] = field(default_factory=list)
    unknowns: list[dict[str, Any]] = field(default_factory=list)
    purpose: str | None = None

    @property
    def start(self) -> int:
        return self.entry

    @property
    def end(self) -> int:
        return max((end for _, end in self.ranges), default=self.entry)

    def to_dict(self) -> dict[str, Any]:
        return {
            "function_id": self.function_id,
            "name": self.name,
            "aliases": self.aliases,
            "entry_va": f"0x{self.entry:x}",
            "ranges": [{"start_va": f"0x{start:x}", "end_va": f"0x{end:x}", "size": end - start} for start, end in self.ranges],
            "size": self.size,
            "source": self.source,
            "confidence": self.confidence,
            "instruction_count": len(self.instructions),
            "block_count": len(self.blocks),
            "callees": [edge.to_dict() for edge in self.callees],
            "callers": [edge.to_dict() for edge in self.callers],
            "evidence": self.evidence,
            "unknowns": self.unknowns,
            "purpose": self.purpose,
        }
