"""Random-initialized model construction for the standalone runtime."""

from __future__ import annotations

from typing import Any

from nexus.foundation.config import FoundationConfig, config_for_vocabulary
from nexus.foundation.model import NexusFoundationModel, build_foundation_model, parameter_report
from nexus.foundation.vocab import Vocabulary


def model_config_from_dict(payload: dict[str, Any], vocab: Vocabulary) -> FoundationConfig:
    config = FoundationConfig.from_dict(payload)
    return config_for_vocabulary(config, vocab)


def build_model(payload: dict[str, Any], vocab: Vocabulary, seed: int = 20261103) -> NexusFoundationModel:
    config = model_config_from_dict(payload, vocab)
    return build_foundation_model(config, seed=seed)


def inspect_model(payload: dict[str, Any], vocab: Vocabulary) -> dict[str, Any]:
    model = build_model(payload, vocab)
    return {"parameter_count": model.count_parameters(), "parameter_report": parameter_report(model), "config": model.config.to_dict(), "initialization": "random", "pretrained_weights_loaded": False}
