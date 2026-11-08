"""Machine-readable evidence and investigation reports."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable

EVIDENCE_SCHEMA_VERSION = "nexus.evidence/v1"
REPORT_SCHEMA_VERSION = "nexus.investigation/v1"
TIERS = {"FACT", "STRONG_INFERENCE", "WEAK_HYPOTHESIS", "UNRECOVERABLE", "VERIFIED", "CONTRADICTED", "UNKNOWN"}


def _hash(payload: Any) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


@dataclass
class EvidenceRecord:
    producer: str
    producer_version: str
    subject: dict[str, Any]
    observation: dict[str, Any]
    tier: str = "FACT"
    confidence: float = 1.0
    inputs: list[str] = field(default_factory=list)
    limitations: list[str] = field(default_factory=list)
    reproducibility: dict[str, Any] = field(default_factory=dict)
    evidence_id: str = ""

    def __post_init__(self) -> None:
        if self.tier not in TIERS:
            raise ValueError(f"invalid evidence tier: {self.tier}")
        if not 0.0 <= float(self.confidence) <= 1.0:
            raise ValueError("confidence must be in [0, 1]")
        if not self.evidence_id:
            self.evidence_id = "ev_" + _hash(
                {
                    "producer": self.producer,
                    "subject": self.subject,
                    "observation": self.observation,
                    "tier": self.tier,
                }
            )[:32]

    def to_dict(self) -> dict[str, Any]:
        return {
            "evidence_id": self.evidence_id,
            "schema_version": EVIDENCE_SCHEMA_VERSION,
            "producer": {"name": self.producer, "version": self.producer_version},
            "subject": self.subject,
            "observation": self.observation,
            "tier": self.tier,
            "confidence": self.confidence,
            "inputs": self.inputs,
            "limitations": self.limitations,
            "reproducibility": self.reproducibility,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "EvidenceRecord":
        producer = payload.get("producer") or {}
        return cls(
            producer=str(producer.get("name") or ""),
            producer_version=str(producer.get("version") or ""),
            subject=payload.get("subject") or {},
            observation=payload.get("observation") or {},
            tier=str(payload.get("tier") or "FACT"),
            confidence=float(payload.get("confidence") or 0.0),
            inputs=list(payload.get("inputs") or []),
            limitations=list(payload.get("limitations") or []),
            reproducibility=payload.get("reproducibility") or {},
            evidence_id=str(payload.get("evidence_id") or ""),
        )


class EvidenceStore:
    """Append-only JSONL evidence store with reference resolution."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._records: dict[str, dict[str, Any]] = {}
        if self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                payload = json.loads(line)
                self._records[str(payload.get("evidence_id"))] = payload

    def append(self, record: EvidenceRecord) -> str:
        payload = record.to_dict()
        if payload["evidence_id"] in self._records:
            return payload["evidence_id"]
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True) + "\n")
        self._records[payload["evidence_id"]] = payload
        return payload["evidence_id"]

    def get(self, evidence_id: str) -> dict[str, Any] | None:
        return self._records.get(evidence_id)

    def resolve(self, evidence_ids: Iterable[str]) -> dict[str, Any]:
        found = {}
        missing = []
        for evidence_id in evidence_ids:
            payload = self.get(str(evidence_id))
            if payload is None:
                missing.append(str(evidence_id))
            else:
                found[str(evidence_id)] = payload
        return {"resolved": found, "missing": missing, "complete": not missing}

    def count(self) -> int:
        return len(self._records)


@dataclass
class InvestigationReport:
    investigation_id: str
    artifact: dict[str, Any]
    objective: str
    status: str = "in_progress"
    facts: list[dict[str, Any]] = field(default_factory=list)
    claims: list[dict[str, Any]] = field(default_factory=list)
    hypotheses: list[dict[str, Any]] = field(default_factory=list)
    unknowns: list[dict[str, Any]] = field(default_factory=list)
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    limitations: list[str] = field(default_factory=list)

    def add_fact(self, fact_id: str, subject: str, predicate: str, value: Any, evidence: list[str], confidence: float = 1.0) -> None:
        self.facts.append(
            {"fact_id": fact_id, "subject": subject, "predicate": predicate, "value": value, "tier": "FACT", "evidence": evidence, "confidence": confidence}
        )

    def add_claim(self, claim_id: str, text: str, tier: str, confidence: float, support: list[str], contradict: list[str] | None = None, missing: list[str] | None = None) -> None:
        if tier not in TIERS:
            raise ValueError(f"invalid claim tier: {tier}")
        self.claims.append(
            {"claim_id": claim_id, "text": text, "tier": tier, "confidence": confidence, "support": support, "contradict": contradict or [], "missing": missing or []}
        )

    def add_hypothesis(self, hypothesis_id: str, text: str, status: str, support: list[str], contradict: list[str] | None = None, tests: list[str] | None = None) -> None:
        if status not in {"proposed", "under_investigation", "supported", "contradicted", "unresolved", "verified"}:
            raise ValueError(f"invalid hypothesis status: {status}")
        self.hypotheses.append(
            {"hypothesis_id": hypothesis_id, "text": text, "status": status, "support": support, "contradict": contradict or [], "tests": tests or []}
        )

    def add_unknown(self, unknown_id: str, question: str, blocking: bool, probe: str, reason: str) -> None:
        self.unknowns.append({"unknown_id": unknown_id, "question": question, "blocking": blocking, "probe": probe, "reason": reason})

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": REPORT_SCHEMA_VERSION,
            "investigation_id": self.investigation_id,
            "created_at": datetime.now(UTC).isoformat(),
            "artifact": self.artifact,
            "objective": self.objective,
            "status": self.status,
            "facts": self.facts,
            "claims": self.claims,
            "hypotheses": self.hypotheses,
            "unknowns": self.unknowns,
            "tool_calls": self.tool_calls,
            "limitations": self.limitations,
            "summary": {
                "facts": len(self.facts),
                "claims": len(self.claims),
                "hypotheses": len(self.hypotheses),
                "unknowns": len(self.unknowns),
                "tool_calls": len(self.tool_calls),
            },
        }
