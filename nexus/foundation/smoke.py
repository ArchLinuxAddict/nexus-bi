"""Infrastructure smoke test for the single unified foundation model.

This test uses synthetic tensors and one optimizer step only.  It is not
evidence of binary understanding and is not a training run.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import torch

from nexus.foundation.checkpoint import load_foundation_checkpoint, save_foundation_checkpoint
from nexus.foundation.config import config_for_vocabulary, profile_config
from nexus.foundation.model import build_foundation_model
from nexus.foundation.vocab import Vocabulary


def synthetic_batch(config, batch_size: int = 2, instruction_length: int = 8, byte_length: int = 16, seed: int = 1234) -> dict[str, Any]:
    generator = torch.Generator().manual_seed(seed)
    instruction_mask = torch.ones((batch_size, instruction_length), dtype=torch.bool)
    if instruction_length > 2:
        instruction_mask[:, -1] = False
    block_mask = torch.ones((batch_size, 2), dtype=torch.bool)
    block_mask[:, 1] = instruction_length > 4
    return {
        "instr_id": torch.randint(3, max(4, config.instruction_vocab - 1), (batch_size, instruction_length), generator=generator),
        "instr_mask": instruction_mask,
        "instr_mnemonic": torch.randint(3, max(4, config.mnemonic_vocab - 1), (batch_size, instruction_length), generator=generator),
        "instr_shape": torch.randint(3, max(4, config.shape_vocab - 1), (batch_size, instruction_length), generator=generator),
        "instr_ir": torch.randint(1, max(2, config.ir_vocab - 1), (batch_size, instruction_length), generator=generator),
        "instr_regs": torch.randint(1, max(2, config.register_vocab - 1), (batch_size, instruction_length, 3), generator=generator),
        "instr_kinds": torch.randint(1, 7, (batch_size, instruction_length, 3), generator=generator),
        "instr_access": torch.randint(0, 4, (batch_size, instruction_length, 3), generator=generator),
        "instr_width": torch.randint(0, 5, (batch_size, instruction_length, 3), generator=generator),
        "instr_block": torch.zeros((batch_size, instruction_length), dtype=torch.long),
        "block_mask": block_mask,
        "byte_ids": torch.randint(3, max(4, config.byte_vocab - 1), (batch_size, byte_length), generator=generator),
        "byte_mask": torch.ones((batch_size, byte_length), dtype=torch.bool),
        "function_features": torch.randn((batch_size, 18), generator=generator) * 0.1,
    }


def smoke_test(steps: int = 1, seed: int = 1234, checkpoint_path: str | Path | None = None) -> dict[str, Any]:
    torch.set_num_threads(1)
    start = time.perf_counter()
    vocab_path = Path(__file__).resolve().parents[2] / "artifacts" / "training_runtime" / "frozen_dataset" / "v4" / "vocab.json"
    vocab = Vocabulary.load(vocab_path) if vocab_path.exists() else Vocabulary()
    config = config_for_vocabulary(profile_config("compact"), vocab)
    model = build_foundation_model(config, seed=seed)
    model.train()
    batch = synthetic_batch(config, batch_size=2, seed=seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    losses: list[float] = []
    for _ in range(max(1, steps)):
        optimizer.zero_grad()
        outputs = model(batch)
        loss = outputs["pooled"].float().pow(2).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(float(loss.detach()))
    target = Path(checkpoint_path) if checkpoint_path else Path("models") / "foundation-smoke.pt"
    saved = save_foundation_checkpoint(
        target,
        model,
        config,
        vocab,
        metadata={
            "purpose": "unified foundation infrastructure smoke test",
            "trained": False,
            "smoke_steps": steps,
            "seed": seed,
            "data": "synthetic random tensors",
        },
        step=steps,
        include_training_state=False,
    )
    model.eval()
    with torch.no_grad():
        before = model(batch)["pooled"]
    loaded = load_foundation_checkpoint(target, load_training_state=False)
    loaded.model.eval()
    with torch.no_grad():
        after = loaded.model(batch)["pooled"]
    reload_ok = torch.allclose(before, after, atol=1e-6)
    elapsed = time.perf_counter() - start
    return {
        "status": "passed" if reload_ok and all(torch.isfinite(torch.tensor(losses))) else "failed",
        "purpose": "unified foundation infrastructure smoke test; not evidence of binary intelligence",
        "architecture_version": config.architecture_version,
        "profile": config.profile,
        "parameter_count": model.count_parameters(),
        "steps": steps,
        "seed": seed,
        "losses": losses,
        "checkpoint": {"path": saved["path"], "sha256": saved["sha256"], "size": saved["size"], "metadata": saved["metadata"]},
        "reload_ok": reload_ok,
        "runtime_seconds": round(elapsed, 4),
        "large_scale_training": False,
        "limitations": [
            "This is a random-initialization infrastructure test on synthetic tensors.",
            "It does not demonstrate binary understanding, decompilation, reconstruction quality or semantic capability.",
        ],
    }
