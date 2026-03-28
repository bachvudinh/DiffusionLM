"""Tensor-parallel embedding and LM head layers.

Derived from JetEngine by Yihan Bian et al.
Reference: https://github.com/Labman42/JetEngine

Architecture Overview
====================

    ┌────────────────────────────────────────────────────────────────┐
    │              VocabParallelEmbedding                            │
    │                                                                │
    │  Full vocab: [0 ─────────────── vocab_size-1]                  │
    │                                                                │
    │  TP=2 sharding:                                                │
    │    Rank 0: [0 ──── vocab/2-1]     Rank 1: [vocab/2 ── vocab-1] │
    │                                                                │
    │  Forward:                                                      │
    │    1. Mask out-of-range token IDs                               │
    │    2. Local embedding lookup                                    │
    │    3. Zero masked positions                                     │
    │    4. all_reduce across TP group                                │
    │                                                                │
    ├────────────────────────────────────────────────────────────────┤
    │              ParallelLMHead                                    │
    │                                                                │
    │  Extends VocabParallelEmbedding for output projection:         │
    │                                                                │
    │  PREFILL mode:                                                  │
    │    Extract last token per sequence using cu_seqlens_q           │
    │    (only predict next block from final hidden state)            │
    │                                                                │
    │  Forward:                                                      │
    │    1. (Prefill only) x = x[cu_seqlens_q[1:] - 1]              │
    │    2. logits = x @ weight.T + bias                              │
    │    3. gather logit shards to rank 0                              │
    │    4. Rank 0 returns full (B, vocab_size) logits               │
    └────────────────────────────────────────────────────────────────┘


VocabParallelEmbedding — Tensor Flow
======================================

    Example: vocab_size=151936, embedding_dim=4096, tp_size=2

    weight per rank: (75968, 4096) bfloat16   ← 151936/2 = 75968

    Forward (tp_size=2, rank=0):
        x: (20,) int64                        ← token IDs, e.g. [42, 100000, ...]

        mask = (x >= 0) & (x < 75968)        ← which tokens are in rank 0's shard
        mask: (20,) bool                        e.g. [True, False, ...]

        x_local = mask * (x - 0)              ← offset to local indices
        x_local: (20,) int64                    e.g. [42, 0, ...]

        y = F.embedding(x_local, weight)
        y: (20, 4096) bfloat16                ← local lookup (wrong for out-of-range)

        y = mask.unsqueeze(1) * y             ← zero out out-of-range entries
        y: (20, 4096) bfloat16                  e.g. [embed(42), zeros, ...]

        dist.all_reduce(y)                    ← sum across ranks
        y: (20, 4096) bfloat16                ← correct embeddings for all tokens


ParallelLMHead — Tensor Flow
==============================

    Example: vocab_size=151936, hidden_size=4096, tp_size=2,
             batch of 3 seqs with cu_seqlens_q=[0, 12, 20, 28]

    PREFILL mode:
        x: (28, 4096) bfloat16               ← all hidden states
                │
                ▼  last_indices = cu_seqlens_q[1:] - 1 = [11, 19, 27]
        x: (3, 4096) bfloat16                ← last token per sequence

    logits = F.linear(x, weight)
        weight: (75968, 4096) bfloat16        ← local vocab shard
        logits: (3, 75968) bfloat16           ← local logit shard
                │
                ▼  dist.gather to rank 0
    Rank 0:
        all_logits = [rank0_logits, rank1_logits]  ← list of (3, 75968)
                │
                ▼  torch.cat(all_logits, dim=-1)
        logits: (3, 151936) bfloat16          ← full vocab logits

    DENOISE mode:
        x: (12, 4096) bfloat16               ← 3 seqs × 4 block_length
                │
                ▼  F.linear → gather → cat
        logits: (12, 151936) bfloat16         ← logits for all block tokens
"""

import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist

from vdllm.engine.sequence import RunType
from vdllm.utils.context import get_context


class VocabParallelEmbedding(nn.Module):
    """Vocabulary embedding sharded across TP ranks.

    Input:
        num_embeddings: int — total vocabulary size
        embedding_dim:  int — embedding dimension
        process_group:  ProcessGroup — TP communication group

    Forward Input:
        x: (num_tokens,) int64 — token IDs

    Forward Output:
        (num_tokens, embedding_dim) — embedding vectors
    """

    def __init__(self, num_embeddings: int, embedding_dim: int,
                 process_group: dist.ProcessGroup):
        super().__init__()
        self.process_group = process_group
        self.tp_rank = dist.get_rank(group=self.process_group)
        self.tp_size = dist.get_world_size(group=self.process_group)
        assert num_embeddings % self.tp_size == 0
        self.num_embeddings = num_embeddings
        self.num_embeddings_per_partition = self.num_embeddings // self.tp_size
        self.vocab_start_idx = self.num_embeddings_per_partition * self.tp_rank
        self.vocab_end_idx = self.vocab_start_idx + self.num_embeddings_per_partition
        self.weight = nn.Parameter(
            torch.empty(self.num_embeddings_per_partition, embedding_dim))
        self.weight.weight_loader = self.weight_loader

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param_data = param.data
        shard_size = param_data.size(0)
        start_idx = self.tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(0, start_idx, shard_size)
        assert param_data.size() == loaded_weight.size()
        param_data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor):
        if self.tp_size > 1:
            mask = (x >= self.vocab_start_idx) & (x < self.vocab_end_idx)
            x = mask * (x - self.vocab_start_idx)
        y = F.embedding(x, self.weight)
        if self.tp_size > 1:
            y = mask.unsqueeze(1) * y
            dist.all_reduce(y, group=self.process_group)
        return y


class ParallelLMHead(VocabParallelEmbedding):
    """Output projection head with TP gather.

    Forward Input:
        x: (num_tokens, hidden_size) — hidden states

    Forward Output:
        (batch_size, vocab_size) on rank 0, None on other ranks

    In PREFILL mode, extracts last-token hidden states per sequence
    before computing logits.
    """

    def __init__(self, num_embeddings: int, embedding_dim: int,
                 process_group: dist.ProcessGroup, bias: bool = False):
        super().__init__(num_embeddings, embedding_dim, process_group)
        if bias:
            self.bias = nn.Parameter(torch.empty(self.num_embeddings_per_partition))
            self.bias.weight_loader = self.weight_loader
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor):
        context = get_context()
        if context.run_type == RunType.PREFILL:
            last_indices = context.cu_seqlens_q[1:] - 1
            x = x[last_indices].contiguous()
        logits = F.linear(x, self.weight, self.bias)
        if self.tp_size > 1:
            all_logits = (
                [torch.empty_like(logits) for _ in range(self.tp_size)]
                if self.tp_rank == 0 else None
            )
            dist.gather(logits, all_logits, 0, group=self.process_group)
            logits = torch.cat(all_logits, -1) if self.tp_rank == 0 else None
        return logits
