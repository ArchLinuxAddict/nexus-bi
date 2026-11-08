"""NEXUS-BI unified foundation model.

One randomly initialized model joins byte, instruction, semantic-IR and graph
modalities through a shared trunk with hierarchical block/function/module/project
aggregation.  Task heads and a conditional reconstruction decoder are integrated
into the same trainable model and one checkpoint.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from nexus.foundation.config import FoundationConfig
from nexus.foundation.vocab import PAD

EDGE_TYPES = 8


class GraphAttention(nn.Module):
    """Multi-head self-attention with an additive typed graph bias."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        if d_model % n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        hidden: torch.Tensor,
        key_mask: torch.Tensor | None = None,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch, length, _ = hidden.shape
        qkv = self.qkv(hidden).reshape(batch, length, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        query, key, value = qkv[0], qkv[1], qkv[2]
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if bias is not None:
            if bias.dim() == 3:
                bias = bias.unsqueeze(1)
            scores = scores + bias[:, :, :length, :length]
        if key_mask is not None:
            scores = scores.masked_fill(~key_mask[:, None, None, :length], torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=-1)
        weights = self.dropout(weights)
        attended = torch.matmul(weights, value)
        attended = attended.transpose(1, 2).reshape(batch, length, self.d_model)
        return self.out(attended)


class GraphTransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.0):
        super().__init__()
        self.norm_attention = nn.LayerNorm(d_model)
        self.attention = GraphAttention(d_model, n_heads, dropout)
        self.norm_ffn = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        hidden: torch.Tensor,
        key_mask: torch.Tensor | None = None,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        hidden = hidden + self.dropout(self.attention(self.norm_attention(hidden), key_mask, bias))
        hidden = hidden + self.dropout(self.ffn(self.norm_ffn(hidden)))
        return hidden


class CrossModalBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.0):
        super().__init__()
        self.norm_query = nn.LayerNorm(d_model)
        self.norm_memory = nn.LayerNorm(d_model)
        self.attention = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.norm_ffn = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )

    def forward(
        self,
        query: torch.Tensor,
        memory: torch.Tensor,
        memory_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        attended, _ = self.attention(
            self.norm_query(query),
            self.norm_memory(memory),
            self.norm_memory(memory),
            key_padding_mask=~memory_mask if memory_mask is not None else None,
            need_weights=False,
        )
        hidden = query + self.dropout(attended)
        return hidden + self.dropout(self.ffn(self.norm_ffn(hidden)))


class ByteEncoder(nn.Module):
    def __init__(self, config: FoundationConfig):
        super().__init__()
        self.token = nn.Embedding(config.byte_vocab, config.d_model, padding_idx=PAD)
        self.position = nn.Embedding(config.max_bytes + 1, config.d_model)
        self.norm = nn.LayerNorm(config.d_model)
        self.conv1 = nn.Conv1d(config.d_model, config.d_model, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(config.d_model, config.d_model, kernel_size=5, padding=2)
        self.conv_norm1 = nn.LayerNorm(config.d_model)
        self.conv_norm2 = nn.LayerNorm(config.d_model)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(ids.size(1), device=ids.device).clamp(max=self.position.num_embeddings - 1)
        hidden = self.token(ids) + self.position(positions).unsqueeze(0)
        hidden = self.norm(hidden)
        sequence = hidden.transpose(1, 2)
        sequence = self.conv_norm1((self.conv1(sequence) + sequence).transpose(1, 2)).transpose(1, 2)
        sequence = F.gelu(sequence)
        sequence = self.conv_norm2((self.conv2(sequence) + sequence).transpose(1, 2)).transpose(1, 2)
        sequence = F.gelu(sequence)
        hidden = sequence.transpose(1, 2)
        return self.dropout(hidden * mask.unsqueeze(-1).to(hidden.dtype))


class SourceTextEncoder(nn.Module):
    def __init__(self, config: FoundationConfig, share_embedding: nn.Embedding | None = None):
        super().__init__()
        self.token = share_embedding or nn.Embedding(config.text_vocab, config.d_model, padding_idx=PAD)
        self.position = nn.Embedding(config.max_text_tokens + 1, config.d_model)
        self.layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    config.d_model,
                    config.n_heads,
                    2 * config.d_model,
                    dropout=config.dropout,
                    batch_first=True,
                    norm_first=True,
                    activation="gelu",
                )
                for _ in range(max(1, config.n_fusion_layers))
            ]
        )
        self.norm = nn.LayerNorm(config.d_model)

    def encode_sequence(self, ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(ids.size(1), device=ids.device).clamp(max=self.position.num_embeddings - 1)
        hidden = self.token(ids) + self.position(positions).unsqueeze(0)
        padding = ~mask
        for layer in self.layers:
            hidden = layer(hidden, src_key_padding_mask=padding)
        return self.norm(hidden * mask.unsqueeze(-1).to(hidden.dtype))

    def forward(self, ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        hidden = self.encode_sequence(ids, mask)
        weights = mask.unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * weights).sum(1) / weights.sum(1).clamp(min=1.0)
        return pooled


class ReconstructionDecoder(nn.Module):
    def __init__(self, config: FoundationConfig, embedding: nn.Embedding):
        super().__init__()
        self.token = embedding
        self.position = nn.Embedding(config.max_text_tokens + 1, config.d_model)
        self.task = nn.Embedding(2, config.d_model)
        self.layers = nn.ModuleList(
            [
                nn.TransformerDecoderLayer(
                    config.d_model,
                    config.n_heads,
                    2 * config.d_model,
                    dropout=config.dropout,
                    batch_first=True,
                    norm_first=True,
                    activation="gelu",
                )
                for _ in range(max(1, config.n_decoder_layers))
            ]
        )
        self.norm = nn.LayerNorm(config.d_model)

    def forward(
        self,
        input_ids: torch.Tensor,
        memory: torch.Tensor,
        memory_mask: torch.Tensor,
        language: int | torch.Tensor,
        input_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch, length = input_ids.shape
        positions = torch.arange(length, device=input_ids.device).clamp(max=self.position.num_embeddings - 1)
        if isinstance(language, int):
            language_tensor = torch.full((batch,), language, dtype=torch.long, device=input_ids.device)
        else:
            language_tensor = language.to(input_ids.device)
        hidden = self.token(input_ids) + self.position(positions).unsqueeze(0) + self.task(language_tensor).unsqueeze(1)
        causal = torch.triu(torch.ones((length, length), dtype=torch.bool, device=input_ids.device), diagonal=1)
        padding = ~input_mask if input_mask is not None else None
        for layer in self.layers:
            hidden = layer(hidden, memory, tgt_mask=causal, tgt_key_padding_mask=padding, memory_key_padding_mask=~memory_mask)
        return self.norm(hidden)


class NexusFoundationModel(nn.Module):
    def __init__(self, config: FoundationConfig):
        super().__init__()
        self.config = config
        d = config.d_model
        self.instruction_token = nn.Embedding(config.instruction_vocab, d, padding_idx=PAD)
        self.mnemonic = nn.Embedding(config.mnemonic_vocab, d, padding_idx=PAD)
        self.shape = nn.Embedding(config.shape_vocab, d, padding_idx=PAD)
        self.ir = nn.Embedding(config.ir_vocab, d, padding_idx=PAD)
        self.register = nn.Embedding(config.register_vocab, d, padding_idx=PAD)
        self.operand_kind = nn.Embedding(8, d, padding_idx=0)
        self.access = nn.Embedding(8, d, padding_idx=0)
        self.width = nn.Embedding(8, d, padding_idx=0)
        self.instruction_position = nn.Embedding(config.max_instructions + 1, d)
        self.block_position = nn.Embedding(config.max_blocks + 1, d)
        self.modality = nn.Embedding(6, d)
        self.instruction_modality = nn.Parameter(torch.zeros(d))
        self.missing = nn.Parameter(torch.zeros(d))
        self.feature_projection = nn.Sequential(nn.Linear(18, d), nn.GELU(), nn.Linear(d, d))
        self.instruction_norm = nn.LayerNorm(d)
        self.instruction_layers = nn.ModuleList(
            [
                GraphTransformerBlock(d, config.n_heads, d * 2, config.dropout)
                for _ in range(config.n_instruction_layers)
            ]
        )
        self.ir_projection = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, d))
        self.block_layers = nn.ModuleList(
            [GraphTransformerBlock(d, config.n_heads, d * 2, config.dropout) for _ in range(config.n_block_layers)]
        )
        self.concat_norm = nn.LayerNorm(5 * d)
        self.fusion = nn.Sequential(nn.Linear(6 * d, d), nn.GELU(), nn.Linear(d, d), nn.LayerNorm(d))
        self.fusion_refine = nn.Sequential(nn.Linear(5 * d, d), nn.GELU(), nn.Linear(d, d), nn.LayerNorm(d))
        self.fusion_gate = nn.Linear(5 * d, 4)
        self.fusion_layers = nn.ModuleList(
            [CrossModalBlock(d, config.n_heads, d * 2, config.dropout) for _ in range(config.n_fusion_layers)]
        )
        self.byte_encoder = ByteEncoder(config)
        self.text_embedding = nn.Embedding(config.text_vocab, d, padding_idx=PAD)
        self.source_encoder = SourceTextEncoder(config, share_embedding=self.text_embedding)
        self.source_projection = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, d))
        self.decoder = ReconstructionDecoder(config, self.text_embedding)
        self.module_layers = nn.ModuleList(
            [GraphTransformerBlock(d, config.n_heads, d * 2, config.dropout) for _ in range(2)]
        )
        self.project_layer = GraphTransformerBlock(d, config.n_heads, d * 2, config.dropout)
        self.module_gate = nn.Linear(d, 1)
        self.instruction_head = nn.Linear(d, config.instruction_vocab)
        self.byte_head = nn.Linear(d, config.byte_vocab)
        self.ir_head = nn.Linear(d, config.ir_vocab)
        self.memory_read_head = nn.Linear(d, 1)
        self.memory_write_head = nn.Linear(d, 1)
        self.memory_width_head = nn.Linear(d, config.num_memory_widths)
        self.flags_read_head = nn.Linear(d, 3)
        self.flags_written_head = nn.Linear(d, 3)
        self.register_read_head = nn.Linear(d, config.max_register_targets)
        self.register_write_head = nn.Linear(d, config.max_register_targets)
        self.cfg_query = nn.Linear(d, d)
        self.cfg_key = nn.Linear(d, d)
        self.cfg_bias = nn.Parameter(torch.zeros(1))
        self.purpose_head = nn.Linear(d, config.num_purpose_classes)
        self.tool_head = nn.Linear(d, config.num_tool_classes)
        self.security_head = nn.Linear(d, config.num_security_classes)
        self.jni_head = nn.Linear(d, config.jni_type_classes * 5)
        self.confidence_head = nn.Linear(d, 1)
        self.retrieval_projection = nn.Linear(d, d)
        self.c_head = nn.Linear(d, config.text_vocab)
        self.java_head = nn.Linear(d, config.text_vocab)
        self.function_entry_head = nn.Linear(d, 1)
        self.evidence_projection = nn.Sequential(nn.Linear(8, d), nn.GELU(), nn.Linear(d, d))
        self.pair_scorer = (
            nn.Sequential(nn.Linear(4 * d, d), nn.GELU(), nn.Linear(d, 1)) if config.pair_head else None
        )
        self.edge_bias = nn.Embedding(EDGE_TYPES, config.n_heads)
        self.ablate_fusion = False
        torch.nn.init.normal_(self.edge_bias.weight, std=0.02)
        torch.nn.init.normal_(self.missing, std=0.02)

    def _graph_bias(
        self,
        edge_index: torch.Tensor | None,
        edge_type: torch.Tensor | None,
        edge_mask: torch.Tensor | None,
        length: int,
    ) -> torch.Tensor | None:
        if edge_index is None or edge_type is None or edge_mask is None or length == 0:
            return None
        batch = edge_index.size(0)
        bias = torch.zeros((batch, self.config.n_heads, length, length), device=edge_index.device)
        for batch_index in range(batch):
            valid = torch.nonzero(edge_mask[batch_index], as_tuple=False).flatten().tolist()
            for position in valid:
                source = int(edge_index[batch_index, position, 0])
                target = int(edge_index[batch_index, position, 1])
                if source >= length or target >= length:
                    continue
                bias[batch_index, :, source, target] = bias[batch_index, :, source, target] + self.edge_bias(
                    edge_type[batch_index, position]
                )
        return bias

    def _segment_pool(
        self,
        hidden: torch.Tensor,
        block_index: torch.Tensor,
        instruction_mask: torch.Tensor,
        block_count: int,
    ) -> torch.Tensor:
        batch, length, d_model = hidden.shape
        block_count = max(1, min(block_count, self.config.max_blocks))
        one_hot = F.one_hot(block_index.clamp(min=0, max=block_count - 1), num_classes=block_count).to(hidden.dtype)
        membership = one_hot * instruction_mask.unsqueeze(-1).to(hidden.dtype)
        scores = self.module_gate(hidden).squeeze(-1)
        scores = scores.masked_fill(~instruction_mask, -1e9)
        scores = scores.unsqueeze(-1).masked_fill(membership <= 0, -1e9)
        weights = torch.softmax(scores, dim=1)
        pooled = torch.einsum("blm,bld->bmd", weights * membership, hidden)
        has_members = membership.sum(dim=1) > 0
        return pooled * has_members.unsqueeze(-1).to(pooled.dtype)

    def _pool(self, hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        weights = mask.unsqueeze(-1).to(hidden.dtype)
        return (hidden * weights).sum(1) / weights.sum(1).clamp(min=1.0)

    def encode_function(self, batch: dict[str, torch.Tensor], causal: bool = False, ablate_fusion: bool | None = None) -> dict[str, torch.Tensor]:
        instruction_ids = batch["instr_id"]
        instruction_mask = batch["instr_mask"]
        batch_size, length = instruction_ids.shape
        positions = torch.arange(length, device=instruction_ids.device).clamp(max=self.config.max_instructions)
        hidden = (
            self.instruction_token(instruction_ids.clamp(max=self.config.instruction_vocab - 1))
            + self.mnemonic(batch["instr_mnemonic"].clamp(max=self.config.mnemonic_vocab - 1))
            + self.shape(batch["instr_shape"].clamp(max=self.config.shape_vocab - 1))
            + self.ir(batch["instr_ir"].clamp(max=self.config.ir_vocab - 1))
            + self.instruction_position(positions).unsqueeze(0)
            + self.instruction_modality.view(1, 1, -1)
        )
        register_hidden = self.register(batch["instr_regs"].clamp(max=self.config.register_vocab - 1))
        kind_hidden = self.operand_kind(batch["instr_kinds"].clamp(max=7))
        access_hidden = self.access(batch["instr_access"].clamp(max=7))
        width_hidden = self.width(batch["instr_width"].clamp(max=7))
        hidden = hidden + (register_hidden + kind_hidden + access_hidden + width_hidden).mean(dim=2)
        hidden = self.instruction_norm(hidden)
        graph_bias = self._graph_bias(
            batch.get("instruction_edges"),
            batch.get("instruction_edge_types"),
            batch.get("instruction_edge_mask"),
            length,
        )
        if causal:
            causal_bias = torch.full((length, length), -1e9, device=hidden.device)
            causal_bias = torch.triu(causal_bias, diagonal=1)
            graph_bias = causal_bias.view(1, 1, length, length) if graph_bias is None else graph_bias + causal_bias.view(1, 1, length, length)
        for layer in self.instruction_layers:
            hidden = layer(hidden, instruction_mask, graph_bias)
        safe_instruction_mask = instruction_mask.clone()
        safe_instruction_mask[:, 0] = True
        instruction_pool = self._pool(hidden, safe_instruction_mask)
        byte_hidden = self.byte_encoder(batch["byte_ids"], batch["byte_mask"])
        byte_mask = batch["byte_mask"].clone()
        byte_mask[:, 0] = True
        byte_pool = self._pool(byte_hidden, byte_mask)
        ir_hidden = self.ir_projection(hidden)
        ir_pool = self._pool(ir_hidden, safe_instruction_mask)
        block_count = int(batch["block_mask"].sum(dim=1).max().item()) if batch["block_mask"].numel() else 1
        block_count = max(1, min(self.config.max_blocks, block_count))
        block_hidden = self._segment_pool(hidden, batch["instr_block"], instruction_mask, block_count)
        block_mask = batch["block_mask"][:, :block_count].clone()
        block_mask[:, 0] = True
        block_positions = torch.arange(block_count, device=hidden.device).clamp(max=self.config.max_blocks)
        block_hidden = block_hidden + self.block_position(block_positions).unsqueeze(0)
        block_bias = self._graph_bias(
            batch.get("block_edges"),
            batch.get("block_edge_types"),
            batch.get("block_edge_mask"),
            block_count,
        )
        for layer in self.block_layers:
            block_hidden = layer(block_hidden, block_mask, block_bias)
        block_pool = self._pool(block_hidden, block_mask)
        feature_embedding = self.feature_projection(batch["function_features"])
        modality_parts = [instruction_pool, block_pool, byte_pool, ir_pool]
        fusion_disabled = self.ablate_fusion if ablate_fusion is None else ablate_fusion
        if fusion_disabled:
            fused = instruction_pool + feature_embedding
        else:
            concat = torch.cat([*modality_parts, feature_embedding], dim=-1)
            gate = torch.sigmoid(self.fusion_gate(concat))
            gated = sum(gate[:, index : index + 1] * modality_parts[index] for index in range(4))
            fused = self.fusion(torch.cat([gated, feature_embedding, *modality_parts], dim=-1))
            # Cross-modal refinement: instruction states attend to byte and IR evidence.
            byte_memory = torch.cat([self.missing.view(1, 1, -1).expand(batch_size, 1, -1), byte_hidden], dim=1)
            byte_memory_mask = torch.cat([torch.ones((batch_size, 1), dtype=torch.bool, device=hidden.device), byte_mask], dim=1)
            for layer in self.fusion_layers:
                hidden = layer(hidden, byte_memory, byte_memory_mask)
            fused = fused + self._pool(hidden, safe_instruction_mask)
            fused = self.fusion_refine(torch.cat([fused, instruction_pool, block_pool, byte_pool, ir_pool], dim=-1))
        return {
            "hidden": hidden,
            "block_hidden": block_hidden,
            "byte_hidden": byte_hidden,
            "ir_hidden": ir_hidden,
            "instruction_pooled": instruction_pool,
            "block_pooled": block_pool,
            "byte_pooled": byte_pool,
            "ir_pooled": ir_pool,
            "pooled": fused,
            "block_mask": block_mask,
            "memory_mask": torch.cat(
                [
                    torch.ones((batch_size, 1), dtype=torch.bool, device=hidden.device),
                    block_mask,
                    safe_instruction_mask,
                ],
                dim=1,
            ),
            "decoder_memory": torch.cat([fused.unsqueeze(1), block_hidden, hidden], dim=1),
            "function_start_logits": self.function_entry_head(byte_hidden).squeeze(-1),
            "function_start_mask": batch["byte_mask"],
        }

    def forward(
        self,
        batch: dict[str, Any],
        run_decoder: bool = False,
        decoder_language: int | str = "c",
        decoder_input_ids: torch.Tensor | None = None,
        causal_instructions: bool = False,
        ablate_fusion: bool | None = None,
    ) -> dict[str, torch.Tensor]:
        states = self.encode_function(batch, causal=causal_instructions, ablate_fusion=ablate_fusion)
        hidden = states["hidden"]
        pooled = states["pooled"]
        logits = {
            "hidden": hidden,
            "block_hidden": states["block_hidden"],
            "byte_hidden": states["byte_hidden"],
            "pooled": pooled,
            "function_embedding": F.normalize(self.retrieval_projection(pooled), dim=-1),
            "instruction_embedding": F.normalize(self.retrieval_projection(states["instruction_pooled"]), dim=-1),
            "byte_embedding": F.normalize(self.retrieval_projection(states["byte_pooled"]), dim=-1),
            "ir_embedding": F.normalize(self.retrieval_projection(states["ir_pooled"]), dim=-1),
            "instruction_logits": self.instruction_head(hidden),
            "byte_logits": self.byte_head(states["byte_hidden"]),
            "ir_logits": self.ir_head(hidden),
            "memory_read_logits": self.memory_read_head(hidden).squeeze(-1),
            "memory_write_logits": self.memory_write_head(hidden).squeeze(-1),
            "memory_width_logits": self.memory_width_head(hidden),
            "flags_read_logits": self.flags_read_head(hidden),
            "flags_written_logits": self.flags_written_head(hidden),
            "register_read_logits": self.register_read_head(hidden),
            "register_write_logits": self.register_write_head(hidden),
            "cfg_logits": self._cfg_logits(states["block_hidden"]),
            "purpose_logits": self.purpose_head(pooled),
            "tool_logits": self.tool_head(pooled),
            "security_logits": self.security_head(pooled),
            "jni_logits": self.jni_head(pooled).view(-1, 5, self.config.jni_type_classes),
            "confidence": torch.sigmoid(self.confidence_head(pooled)).squeeze(-1),
            "function_entry_logits": self.function_entry_head(hidden).squeeze(-1),
            "function_start_logits": states.get("function_start_logits"),
            "function_start_mask": states.get("function_start_mask"),
            "source_embedding": self.source_embedding(batch),
        }
        if run_decoder:
            language = 0 if decoder_language == "c" else 1
            if decoder_input_ids is None:
                key = "c_ids" if decoder_language == "c" else "java_ids"
                ids = batch.get(key)
                if ids is None:
                    ids = batch.get("c_ids")
                if ids is None:
                    raise ValueError(f"decoder requested but no {key} targets are present")
                decoder_input_ids = ids[:, :-1]
                target_mask = batch.get("c_mask" if decoder_language == "c" else "java_mask")
                if target_mask is not None:
                    target_mask = target_mask[:, :-1]
            else:
                target_mask = batch.get("decoder_mask")
            if decoder_input_ids is not None and decoder_input_ids.size(1) == 0:
                logits["decoder_skipped"] = True
                return logits
            memory = states["decoder_memory"]
            memory_mask = states["memory_mask"]
            decoder_hidden = self.decoder(decoder_input_ids, memory, memory_mask, language, target_mask)
            logits["c_logits"] = self.c_head(decoder_hidden)
            logits["java_logits"] = self.java_head(decoder_hidden)
            logits["decoder_hidden"] = decoder_hidden
        return logits

    def _cfg_logits(self, block_hidden: torch.Tensor) -> torch.Tensor:
        query = self.cfg_query(block_hidden)
        key = self.cfg_key(block_hidden)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.config.d_model)
        return scores + self.cfg_bias

    def source_embedding(self, batch: dict[str, torch.Tensor]) -> torch.Tensor | None:
        ids = batch.get("source_ids")
        if ids is None or not ids.numel() or not ids.any():
            return None
        mask = ids.ne(PAD)
        mask[:, 0] = True
        return F.normalize(self.source_projection(self.source_encoder(ids, mask)), dim=-1)

    def score_pair(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        if self.pair_scorer is None:
            raise RuntimeError("model was built without a pair head")
        return self.pair_scorer(torch.cat([left, right, (left - right).abs(), left * right], dim=-1)).squeeze(-1)

    def encode_module(
        self,
        function_batch: dict[str, Any],
        module_index: torch.Tensor,
        module_count: int,
        module_edges: torch.Tensor | None = None,
        module_edge_types: torch.Tensor | None = None,
        module_edge_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        states = self.encode_function(function_batch)
        function_vectors = states["pooled"]
        d_model = function_vectors.size(-1)
        max_functions = min(self.config.max_functions, int(module_count))
        module_vectors = function_vectors.new_zeros((max_functions, self.config.max_functions, d_model))
        module_mask = torch.zeros((max_functions, self.config.max_functions), dtype=torch.bool, device=function_vectors.device)
        for index in range(function_vectors.size(0)):
            module = int(module_index[index])
            if module >= max_functions:
                continue
            slot = int(module_mask[module].sum().item())
            if slot >= self.config.max_functions:
                continue
            module_vectors[module, slot] = function_vectors[index]
            module_mask[module, slot] = True
        module_mask[:, 0] |= module_mask.sum(dim=1) == 0
        block_count = module_vectors.size(1)
        bias = self._graph_bias(module_edges, module_edge_types, module_edge_mask, block_count) if module_edges is not None else None
        for layer in self.module_layers:
            module_vectors = layer(module_vectors, module_mask, bias)
        weights = module_mask.unsqueeze(-1).to(module_vectors.dtype)
        module_pooled = (module_vectors * weights).sum(1) / weights.sum(1).clamp(min=1.0)
        return {
            "module_vectors": module_vectors,
            "module_mask": module_mask,
            "module_pooled": module_pooled,
            "function_vectors": function_vectors,
            "function_context": module_vectors[module_index.clamp(max=max_functions - 1)],
        }

    def encode_project(self, module_vectors: torch.Tensor, module_mask: torch.Tensor | None = None) -> torch.Tensor:
        if module_mask is None:
            module_mask = torch.ones(module_vectors.shape[:2], dtype=torch.bool, device=module_vectors.device)
        hidden = module_vectors
        for layer in [self.project_layer]:
            hidden = layer(hidden, module_mask, None)
        weights = module_mask.unsqueeze(-1).to(hidden.dtype)
        return (hidden * weights).sum(1) / weights.sum(1).clamp(min=1.0)

    @torch.no_grad()
    def generate_reconstruction(
        self,
        batch: dict[str, Any],
        language: str = "c",
        max_new_tokens: int = 64,
        temperature: float = 0.0,
    ) -> dict[str, Any]:
        self.eval()
        states = self.encode_function(batch)
        batch_size = states["pooled"].size(0)
        language_id = 0 if language == "c" else 1
        ids = torch.full((batch_size, 1), 3, dtype=torch.long, device=states["pooled"].device)
        finished = torch.zeros((batch_size,), dtype=torch.bool, device=ids.device)
        for _ in range(max_new_tokens):
            hidden = self.decoder(ids, states["decoder_memory"], states["memory_mask"], language_id)
            logits = self.c_head(hidden[:, -1]) if language == "c" else self.java_head(hidden[:, -1])
            if temperature and temperature > 0:
                probabilities = torch.softmax(logits / temperature, dim=-1)
                next_ids = torch.multinomial(probabilities, 1).squeeze(-1)
            else:
                next_ids = logits.argmax(dim=-1)
            next_ids = torch.where(finished, torch.full_like(next_ids, 4), next_ids)
            ids = torch.cat([ids, next_ids.unsqueeze(-1)], dim=1)
            finished = finished | next_ids.eq(4)
            if bool(finished.all()):
                break
        return {"token_ids": ids[:, 1:].tolist(), "language": language, "max_new_tokens": max_new_tokens}

    def count_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)


def build_foundation_model(config: FoundationConfig, seed: int | None = None) -> NexusFoundationModel:
    torch.manual_seed(config.seed if seed is None else seed)
    model = NexusFoundationModel(config)
    model.eval()
    return model


def parameter_report(model: NexusFoundationModel) -> dict[str, Any]:
    groups: dict[str, int] = {}
    for name, parameter in model.named_parameters():
        prefix = name.split(".")[0]
        groups[prefix] = groups.get(prefix, 0) + parameter.numel()
    return {
        "total": model.count_parameters(),
        "trainable": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
        "groups": dict(sorted(groups.items(), key=lambda item: -item[1])),
        "config_hash": model.config.config_hash(),
        "architecture_version": model.config.architecture_version,
    }
