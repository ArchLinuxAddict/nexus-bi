"""Configuration and model profiles for the NEXUS-BI unified foundation model.

The profiles are configurations of one architecture family, not separate models.
No CUDA is required to construct, import or test the model.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, replace
from typing import Any

ARCHITECTURE_VERSION = "nexus-bi/foundation-1"
CHECKPOINT_FORMAT = "nexus-foundation-checkpoint-v1"
SUPPORTED_PROFILES = ("cpu_dev", "compact", "small", "medium", "large")


@dataclass
class FoundationConfig:
    name: str = "nexus-bi-foundation"
    profile: str = "cpu_dev"
    architecture_version: str = ARCHITECTURE_VERSION
    d_model: int = 96
    n_heads: int = 4
    n_instruction_layers: int = 2
    n_block_layers: int = 1
    n_fusion_layers: int = 1
    n_decoder_layers: int = 2
    d_ff: int = 256
    dropout: float = 0.0
    max_instructions: int = 128
    max_blocks: int = 48
    max_bytes: int = 1024
    max_text_tokens: int = 192
    max_functions: int = 32
    max_operands: int = 4
    max_register_targets: int = 24
    byte_vocab: int = 272
    instruction_vocab: int = 1024
    mnemonic_vocab: int = 512
    shape_vocab: int = 256
    register_vocab: int = 128
    ir_vocab: int = 64
    text_vocab: int = 4096
    num_purpose_classes: int = 16
    num_tool_classes: int = 12
    jni_type_classes: int = 12
    num_security_classes: int = 6
    num_memory_widths: int = 6
    pair_head: bool = True
    gradient_checkpointing: bool = False
    seed: int = 20261103
    notes: str = "randomly initialized; no pretrained weights"

    def __post_init__(self) -> None:
        if self.profile not in SUPPORTED_PROFILES:
            raise ValueError(f"unknown profile: {self.profile}")
        if self.d_model % self.n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        if self.d_model < 16:
            raise ValueError("d_model must be at least 16")
        if self.max_instructions < 2 or self.max_text_tokens < 2:
            raise ValueError("sequence limits are too small")
        if self.architecture_version != ARCHITECTURE_VERSION:
            raise ValueError(f"unsupported architecture version: {self.architecture_version}")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "FoundationConfig":
        known = {field.name for field in cls.__dataclass_fields__.values()}
        return cls(**{key: value for key, value in payload.items() if key in known})

    def config_hash(self) -> str:
        """Hash the effective architecture, excluding display/run-only fields.

        Vocabulary-dependent dimensions are part of the effective architecture
        and therefore remain in the hash.  ``name``, ``profile``, ``notes`` and
        ``seed`` do not change parameter shapes and are excluded so the same
        architecture built from a registry profile and a checkpoint resolves to
        the same identity.
        """
        payload = {key: value for key, value in self.to_dict().items() if key not in CONFIG_HASH_EXCLUDED}
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()

    def legacy_config_hash(self) -> str:
        """Pre-v0.5 hash over every field, retained for checkpoint compatibility."""
        return hashlib.sha256(json.dumps(self.to_dict(), sort_keys=True).encode("utf-8")).hexdigest()

    def estimate_parameters(self) -> int:
        """Analytic lower-bound estimate used for capability gates and reports."""
        d = self.d_model
        embeddings = (
            self.byte_vocab * d
            + self.instruction_vocab * d
            + self.mnemonic_vocab * d
            + self.shape_vocab * d
            + self.register_vocab * d
            + self.ir_vocab * d
            + self.text_vocab * d
            + 256 * d
        )
        attention = 4 * d * d
        ffn = 2 * d * self.d_ff
        encoder_layer = 2 * (attention + ffn) + 4 * d
        decoder_layer = 3 * (attention + ffn) + 6 * d
        trunk = self.n_instruction_layers * encoder_layer + (self.n_block_layers + self.n_fusion_layers) * encoder_layer
        decoder = self.n_decoder_layers * decoder_layer
        heads = d * (self.instruction_vocab + self.ir_vocab + self.num_purpose_classes + self.num_tool_classes)
        heads += d * (self.text_vocab * 2 + self.register_vocab * 2 + self.num_memory_widths)
        return int(embeddings + trunk + decoder + heads)

    def with_overrides(self, **overrides: Any) -> "FoundationConfig":
        return replace(self, **overrides)


CPU_DEV = FoundationConfig(
    name="nexus-bi-cpu-dev",
    profile="cpu_dev",
    d_model=96,
    n_heads=4,
    n_instruction_layers=2,
    n_block_layers=1,
    n_fusion_layers=1,
    n_decoder_layers=2,
    d_ff=256,
    max_instructions=96,
    max_blocks=32,
    max_bytes=768,
    max_text_tokens=160,
    max_functions=16,
)

COMPACT = FoundationConfig(
    name="nexus-bi-compact",
    profile="compact",
    d_model=160,
    n_heads=4,
    n_instruction_layers=5,
    n_block_layers=2,
    n_fusion_layers=2,
    n_decoder_layers=4,
    d_ff=640,
    max_instructions=256,
    max_blocks=96,
    max_bytes=2048,
    max_text_tokens=1024,
    max_functions=64,
    dropout=0.1,
)

SMALL = FoundationConfig(
    name="nexus-bi-small",
    profile="small",
    d_model=256,
    n_heads=8,
    n_instruction_layers=6,
    n_block_layers=2,
    n_fusion_layers=2,
    n_decoder_layers=6,
    d_ff=1024,
    max_instructions=512,
    max_blocks=128,
    max_bytes=4096,
    max_text_tokens=512,
    max_functions=64,
    dropout=0.1,
)

MEDIUM = FoundationConfig(
    name="nexus-bi-medium",
    profile="medium",
    d_model=512,
    n_heads=8,
    n_instruction_layers=12,
    n_block_layers=4,
    n_fusion_layers=3,
    n_decoder_layers=12,
    d_ff=2048,
    max_instructions=1024,
    max_blocks=256,
    max_bytes=8192,
    max_text_tokens=1024,
    max_functions=128,
    dropout=0.1,
)

LARGE = FoundationConfig(
    name="nexus-bi-large",
    profile="large",
    d_model=1024,
    n_heads=16,
    n_instruction_layers=24,
    n_block_layers=6,
    n_fusion_layers=4,
    n_decoder_layers=24,
    d_ff=4096,
    max_instructions=4096,
    max_blocks=1024,
    max_bytes=32768,
    max_text_tokens=2048,
    max_functions=512,
    dropout=0.1,
)

CONFIG_HASH_EXCLUDED = {"name", "profile", "notes", "seed"}
CONFIG_HASH_SCHEME = "nexus.effective-architecture/v1"

MODEL_PROFILES: dict[str, FoundationConfig] = {
    "cpu_dev": CPU_DEV,
    "compact": COMPACT,
    "small": SMALL,
    "medium": MEDIUM,
    "large": LARGE,
}


def config_for_vocabulary(config: FoundationConfig, vocab: Any) -> FoundationConfig:
    """Make embedding sizes match the trained vocabulary exactly."""
    return config.with_overrides(
        byte_vocab=272,
        instruction_vocab=max(16, len(vocab.instruction)),
        mnemonic_vocab=max(16, len(vocab.mnemonic)),
        shape_vocab=max(16, len(vocab.shape)),
        register_vocab=max(16, len(vocab.register)),
        ir_vocab=max(16, len(vocab.ir_op)),
        text_vocab=max(len(vocab.text), 268),
    )


def profile_config(profile: str, **overrides: Any) -> FoundationConfig:
    if profile not in MODEL_PROFILES:
        raise ValueError(f"unknown model profile: {profile}")
    return MODEL_PROFILES[profile].with_overrides(**overrides)


def profile_report() -> dict[str, Any]:
    return {
        "architecture_version": ARCHITECTURE_VERSION,
        "profiles": {
            name: {
                "config": config.to_dict(),
                "estimated_parameters": config.estimate_parameters(),
                "config_hash": config.config_hash(),
            }
            for name, config in MODEL_PROFILES.items()
        },
        "note": "Profiles are one architecture family at different scales; no profile has been trained at scale.",
    }
