"""Inventory of supported training, evaluation and inference entry points."""

from __future__ import annotations

from typing import Any

ENTRYPOINTS: list[dict[str, Any]] = [
    {
        "entrypoint": "nexus.foundation.launch.launch_training",
        "kind": "canonical_training_facade",
        "path": "nexus/foundation/launch.py",
        "validates_mission": "frozen corpus, init/resume mode, objective readiness, dataset IDs, run mode",
        "status": "PASS",
        "gap": "Mission metadata is registry-derived and is not cryptographically signed.",
    },
    {
        "entrypoint": "nexus.foundation.training.FoundationTrainer",
        "kind": "direct_python",
        "path": "nexus/foundation/training.py",
        "validates_mission": "objectives, categories, dataset IDs, run mode, batch provenance",
        "status": "PASS",
        "gap": "A caller with process access can bypass application-level policy.",
    },
    {
        "entrypoint": "nexus_train.trainer.StandaloneRun",
        "kind": "training_engine",
        "path": "nexus_train/trainer.py",
        "validates_mission": "canonical launch configuration and frozen gate",
        "status": "PASS",
        "gap": "The engine is an implementation detail; use the foundation facade for policy checks.",
    },
    {
        "entrypoint": "nexus foundation train",
        "kind": "cli",
        "path": "nexus/cli/main.py",
        "validates_mission": "canonical launch and dataset/objective readiness",
        "status": "PASS",
        "gap": "None for the supported path.",
    },
    {
        "entrypoint": "nexus foundation resume",
        "kind": "cli",
        "path": "nexus/cli/main.py",
        "validates_mission": "complete resumable checkpoint, config, vocabulary and dataset identity",
        "status": "PASS",
        "gap": "Resume requires a checkpoint produced by the canonical engine.",
    },
    {
        "entrypoint": "nexus foundation evaluate / analyze / reconstruct",
        "kind": "cli_inference",
        "path": "nexus/cli/main.py; nexus/foundation",
        "validates_mission": "evaluation and inference only; never trains",
        "status": "PASS",
        "gap": "No behavioral equivalence is claimed.",
    },
    {
        "entrypoint": "POST /v1/model/smoke-test",
        "kind": "api",
        "path": "nexus/api/app.py",
        "validates_mission": "synthetic infrastructure smoke test only",
        "status": "PASS",
        "gap": "No API training endpoint exists.",
    },
    {
        "entrypoint": "dataset generation and ingestion",
        "kind": "ingestion",
        "path": "nexus/datasets",
        "validates_mission": "mission relevance written to new manifests; training rejects unregistered data",
        "status": "PASS",
        "gap": "Direct library calls can write unregistered manifests.",
    },
    {
        "entrypoint": "custom Python optimizer loop",
        "kind": "uncontrolled_python",
        "path": "any caller with process access",
        "validates_mission": "none",
        "status": "UNVERIFIED",
        "gap": "Application-level policy cannot govern arbitrary Python code.",
    },
]


def inventory() -> dict[str, Any]:
    return {
        "schema_version": "0.2.0",
        "entrypoints": ENTRYPOINTS,
        "counts": {
            "total": len(ENTRYPOINTS),
            "pass": sum(1 for item in ENTRYPOINTS if item["status"] == "PASS"),
            "partial": sum(1 for item in ENTRYPOINTS if item["status"] == "PARTIAL"),
            "unverified": sum(1 for item in ENTRYPOINTS if item["status"] == "UNVERIFIED"),
            "blocked": sum(1 for item in ENTRYPOINTS if item["status"] == "BLOCKED"),
        },
        "limitations": [
            "The inventory covers repository-provided entry points and supported CLI/API paths.",
            "Arbitrary Python code with process access is outside application-level policy enforcement.",
            "Provenance and mission metadata are not cryptographically signed.",
        ],
    }
