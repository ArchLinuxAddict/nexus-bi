
import pytest

from nexus.core.errors import MissionPolicyError
from nexus.mission.policy import (
    audit_registry,
    manifest_mission,
    mission_for_dataset,
    validate_training_run,
)


def test_mission_registry_audit():
    result = audit_registry()
    assert result["status"] == "passed", result["issues"]
    assert result["datasets"] >= 1
    assert result["categories"] >= 6


def test_manifest_mission_fields():
    mission = manifest_mission("real_world_binary_source", ["binary_representation_masking"], "test")
    assert mission["classification"] == "CORE"
    assert mission["allowed_for_training"] is True
    assert "native_binary_understanding" in mission["capabilities"]


def test_training_rejects_unclassified_run():
    with pytest.raises(MissionPolicyError):
        validate_training_run([], [])
    with pytest.raises(MissionPolicyError):
        validate_training_run(["binary_representation_masking"], ["unclassified"])
    with pytest.raises(MissionPolicyError):
        validate_training_run(["binary_representation_masking"], ["generic_internet_text"])


def test_training_accepts_core_binary_objective():
    mission = validate_training_run(["binary_representation_masking"], ["controlled_binary_source"], run_mode="pilot")
    assert mission.dataset_classification == "CORE"
    assert mission.objectives == ("binary_representation_masking",)


def test_registered_dataset_lookup():
    entry = mission_for_dataset(dataset_id="realworld-so-v1")
    assert entry["dataset_id"] == "realworld-so-v1"
    assert entry["classification"] == "CORE"
    assert entry["license_verified"] is False
