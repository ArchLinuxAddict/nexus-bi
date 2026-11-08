"""Tool-aware residual model for the tool-first NEXUS-BI architecture."""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from nexus.foundation.config import FoundationConfig
from nexus.foundation.model import NexusFoundationModel, SourceTextEncoder
from nexus.foundation.vocab import BOS, EOS


class ResidualNexusModel(nn.Module):
    def __init__(self, config: FoundationConfig):
        super().__init__()
        self.config = config
        d = config.d_model
        self.program = NexusFoundationModel(config)
        self.tool_encoder = SourceTextEncoder(config, share_embedding=self.program.text_embedding)
        self.candidate_encoder = SourceTextEncoder(config, share_embedding=self.program.text_embedding)
        self.tool_embedding = nn.Embedding(16, d)
        self.language_embedding = nn.Embedding(8, d)
        self.status_embedding = nn.Embedding(6, d)
        self.compile_embedding = nn.Embedding(4, d)
        self.program_missing = nn.Parameter(torch.zeros(d))
        self.meta_projection = nn.Sequential(nn.Linear(4 * d + 2, d), nn.GELU(), nn.Linear(d, d), nn.LayerNorm(d))
        self.fusion = nn.Sequential(nn.Linear(3 * d, d), nn.GELU(), nn.Dropout(config.dropout), nn.Linear(d, d), nn.LayerNorm(d))
        self.head_fusion = nn.Sequential(nn.Linear(2 * d, d), nn.GELU(), nn.Dropout(config.dropout), nn.Linear(d, d), nn.LayerNorm(d))
        self.candidate_head = nn.Linear(d, 1)
        self.failure_head = nn.Linear(d, 2)
        self.confidence_head = nn.Linear(d, 1)
        self.name_head = nn.Linear(d, config.text_vocab)
        self.copy_query = nn.Linear(d, d, bias=False)
        self.copy_key = nn.Linear(d, d, bias=False)
        self.copy_gate = nn.Linear(2 * d, 1)
        self.use_copy = True

    def encode_context(self, batch: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        tool_sequence = self.tool_encoder.encode_sequence(batch["tool_ids"], batch["tool_mask"])
        weights = batch["tool_mask"].unsqueeze(-1).to(tool_sequence.dtype)
        tool_pooled = (tool_sequence * weights).sum(1) / weights.sum(1).clamp(min=1.0)
        program = batch.get("program")
        if program is not None:
            program_pool = self.program.encode_function(program)["pooled"]
        else:
            program_pool = self.program_missing.unsqueeze(0).expand(tool_sequence.size(0), -1)
        meta = self.meta_projection(
            torch.cat(
                [
                    self.tool_embedding(batch["tool_id"].clamp(max=self.tool_embedding.num_embeddings - 1)),
                    self.language_embedding(batch["language_id"].clamp(max=self.language_embedding.num_embeddings - 1)),
                    self.status_embedding(batch["status_id"].clamp(max=self.status_embedding.num_embeddings - 1)),
                    self.compile_embedding(batch["compile_id"].clamp(max=self.compile_embedding.num_embeddings - 1)),
                    batch["warning_count"].float().unsqueeze(-1) / 10.0,
                    batch["cached"].float().unsqueeze(-1),
                ],
                dim=-1,
            )
        )
        fused = self.fusion(torch.cat([program_pool, tool_pooled, meta], dim=-1))
        candidate_ids = batch.get("candidate_ids")
        candidate_mask = batch.get("candidate_mask")
        if candidate_ids is not None and candidate_mask is not None:
            candidate_pooled = self.candidate_encoder(candidate_ids, candidate_mask)
        else:
            candidate_pooled = torch.zeros_like(fused)
        head_hidden = self.head_fusion(torch.cat([fused, candidate_pooled], dim=-1))
        return fused, tool_sequence, head_hidden

    def forward(
        self,
        batch: dict[str, Any],
        repair_input_ids: torch.Tensor | None = None,
        repair_input_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        fused, tool_sequence, head_hidden = self.encode_context(batch)
        outputs = {
            "pooled": head_hidden,
            "candidate_logit": self.candidate_head(head_hidden).squeeze(-1),
            "failure_logits": self.failure_head(head_hidden),
            "confidence": torch.sigmoid(self.confidence_head(head_hidden)).squeeze(-1),
            "name_logits": self.name_head(head_hidden),
            "tool_sequence": tool_sequence,
        }
        if repair_input_ids is not None:
            memory = torch.cat([fused.unsqueeze(1), tool_sequence], dim=1)
            memory_mask = torch.cat(
                [
                    torch.ones(fused.size(0), 1, dtype=torch.bool, device=fused.device),
                    batch["tool_mask"],
                ],
                dim=1,
            )
            language = batch.get("language_id", torch.zeros(fused.size(0), dtype=torch.long, device=fused.device))
            repair_hidden = self.program.decoder(
                repair_input_ids,
                memory,
                memory_mask,
                language,
                input_mask=repair_input_mask if repair_input_mask is not None else batch.get("repair_mask"),
            )
            outputs["repair_hidden"] = repair_hidden
            outputs["repair_logits"] = F.linear(repair_hidden, self.program.text_embedding.weight)
            if self.use_copy:
                copy_log_probs, copy_gate, copy_scores = self._copy_log_probs(repair_hidden, batch, tool_sequence)
                outputs["repair_log_probs"] = copy_log_probs
                outputs["copy_gate"] = copy_gate
                if copy_scores is not None:
                    outputs["copy_logits"] = copy_scores
        return outputs

    def _copy_log_probs(
        self,
        hidden: torch.Tensor,
        batch: dict[str, Any],
        tool_hidden: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Pointer-generator distribution over vocabulary and tool-text tokens."""
        vocab_log_probs = F.log_softmax(
            F.linear(hidden, self.program.text_embedding.weight),
            dim=-1,
        )
        source_ids = batch.get("tool_ids")
        source_mask = batch.get("tool_mask")
        if source_ids is None or source_mask is None:
            return vocab_log_probs, torch.zeros(hidden.shape[:2], device=hidden.device), None
        query = self.copy_query(hidden)
        key = self.copy_key(tool_hidden)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.config.d_model)
        scores = scores.masked_fill(~source_mask[:, None, :], torch.finfo(scores.dtype).min)
        copy_weights = torch.softmax(scores, dim=-1)
        context = torch.matmul(copy_weights, tool_hidden)
        gate = torch.sigmoid(self.copy_gate(torch.cat([hidden, context], dim=-1))).squeeze(-1)
        batch_size, target_length, source_length = copy_weights.shape
        vocab_size = vocab_log_probs.size(-1)
        copy_probs = torch.zeros(
            batch_size,
            target_length,
            vocab_size,
            device=hidden.device,
            dtype=vocab_log_probs.dtype,
        )
        copy_probs.scatter_add_(
            2,
            source_ids[:, None, :].expand(batch_size, target_length, source_length),
            copy_weights,
        )
        probs = (1.0 - gate).unsqueeze(-1) * vocab_log_probs.exp() + gate.unsqueeze(-1) * copy_probs
        return torch.log(probs.clamp_min(1e-9)), gate, scores

    @torch.no_grad()
    def generate_repair(self, batch: dict[str, Any], max_new_tokens: int = 256) -> torch.Tensor:
        fused, tool_sequence, _ = self.encode_context(batch)
        memory = torch.cat([fused.unsqueeze(1), tool_sequence], dim=1)
        memory_mask = torch.cat(
            [
                torch.ones(fused.size(0), 1, dtype=torch.bool, device=fused.device),
                batch["tool_mask"],
            ],
            dim=1,
        )
        language = batch.get("language_id", torch.zeros(fused.size(0), dtype=torch.long, device=fused.device))
        ids = torch.full((fused.size(0), 1), BOS, dtype=torch.long, device=fused.device)
        for _ in range(max_new_tokens):
            mask = torch.ones(ids.shape, dtype=torch.bool, device=ids.device)
            hidden = self.program.decoder(ids, memory, memory_mask, language, input_mask=mask)
            if self.use_copy:
                log_probs, _, _ = self._copy_log_probs(hidden[:, -1:], batch, tool_sequence)
                next_id = log_probs[:, -1].argmax(-1, keepdim=True)
            else:
                logits = F.linear(hidden[:, -1], self.program.text_embedding.weight)
                next_id = logits.argmax(-1, keepdim=True)
            ids = torch.cat([ids, next_id], dim=1)
            if bool((next_id == EOS).all()):
                break
        return ids

    def count_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)


def build_residual_model(config: FoundationConfig, seed: int | None = None) -> ResidualNexusModel:
    torch.manual_seed(config.seed if seed is None else seed)
    return ResidualNexusModel(config)


def residual_parameter_report(model: ResidualNexusModel) -> dict[str, Any]:
    groups: dict[str, int] = {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        group = name.split(".")[0]
        groups[group] = groups.get(group, 0) + parameter.numel()
    return {"total": model.count_parameters(), "groups": dict(sorted(groups.items(), key=lambda item: -item[1]))}
