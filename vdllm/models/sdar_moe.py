"""SDAR-MoE (Sparse Mixture-of-Experts) model for block diffusion.

Derived from JetEngine by Yihan Bian et al.
Reference: https://github.com/Labman42/JetEngine

Architecture Overview
====================

    SDAR-MoE extends SDAR with sparse MoE layers interleaved with
    dense MLP layers. Uses Switch-Transformer style top-k routing
    with a fused Triton MoE kernel for efficient expert computation.

    ┌──────────────────────────────────────────────────────────────────┐
    │                    SDARMoeForCausalLM                             │
    │                                                                  │
    │  SDARMoeModel                                                    │
    │  ├── VocabParallelEmbedding                                      │
    │  ├── N x SDARMoeDecoderLayer                                     │
    │  │    ├── SDARAttention (shared with dense SDAR)                  │
    │  │    └── MLP (alternating dense / MoE by decoder_sparse_step)   │
    │  │                                                               │
    │  │    Dense layers (non-MoE):                                    │
    │  │    ├── SDARMoeMLP                                             │
    │  │    │    ├── MergedColumnParallelLinear (gate_up_proj)          │
    │  │    │    ├── SiluAndMul                                         │
    │  │    │    └── RowParallelLinear (down_proj)                       │
    │  │    │                                                          │
    │  │    MoE layers (every decoder_sparse_step):                    │
    │  │    └── SDARMoeSparseMoeBlock                                   │
    │  │         ├── ReplicatedLinear (gating network)                   │
    │  │         ├── top-k routing with softmax normalization           │
    │  │         ├── fused_moe kernel (_w1, _w2 pre-allocated)          │
    │  │         └── all_reduce for TP aggregation                      │
    │  │                                                               │
    │  └── RMSNorm (final)                                             │
    │                                                                  │
    │  ParallelLMHead                                                  │
    │                                                                  │
    │  MoE Expert Weight Layout:                                       │
    │    _w1: (num_experts, 2*intermediate/tp, hidden)  gate+up fused  │
    │    _w2: (num_experts, hidden, intermediate/tp)    down proj      │
    │                                                                  │
    │  Routing:                                                        │
    │    logits = gate(x)             (S, num_experts)                 │
    │    probs  = softmax(logits)                                      │
    │    top_p, top_i = topk(probs, k)                                 │
    │    top_p  = top_p / sum(top_p)  (renormalize)                    │
    │    output = fused_moe(x, _w1, _w2, top_p, top_i)                │
    └──────────────────────────────────────────────────────────────────┘


SDARMoeSparseMoeBlock — Tensor Flow
=====================================

    Example: hidden_size=4096, moe_intermediate_size=2048,
             num_experts=8, top_k=2, S=20 tokens, tp_size=2

    hidden_states: (20, 4096) bfloat16
            │
            ▼  gate (ReplicatedLinear, full weight on each rank)
    router_logits: (20, 8) bfloat16        ← one score per expert
            │
            ▼  softmax(dim=-1, dtype=float32)
    probs: (20, 8) float32                 ← expert probabilities
            │
            ▼  topk(probs, k=2, dim=-1)
    top_p: (20, 2) float32                 ← top-2 probabilities
    top_i: (20, 2) int64                   ← top-2 expert indices
            │
            ▼  renormalize: top_p = top_p / top_p.sum(dim=-1, keepdim=True)
    top_p: (20, 2) float32                 ← renormalized weights (sum=1)
            │
            ▼  fused_moe(hidden_states, _w1, _w2, top_p, top_i)
            │
            │  _w1: (8, 2048, 4096) bfloat16    ← gate+up fused per expert
            │  _w2: (8, 4096, 1024) bfloat16    ← down proj per expert
            │
            │  Internal flow per token:
            │    1. Select top-2 experts by top_i
            │    2. x @ expert._w1.T → (1, 2048) → SiLU gating → (1, 1024)
            │    3. result @ expert._w2.T → (1, 4096)
            │    4. Weighted sum by top_p across selected experts
            │
    out: (20, 4096) bfloat16
            │
            ▼  dist.all_reduce(out)            ← aggregate across TP ranks
    output: (20, 4096) bfloat16

    Returns: (output, router_logits)


SDARMoeDecoderLayer — Layer Selection Logic
=============================================

    decoder_sparse_step = 2 means MoE every 2nd layer:

        Layer 0: dense MLP    (0+1=1, 1%2≠0)
        Layer 1: MoE block    (1+1=2, 2%2=0)  ← MoE
        Layer 2: dense MLP    (2+1=3, 3%2≠0)
        Layer 3: MoE block    (3+1=4, 4%2=0)  ← MoE
        ...

    Tensor flow is identical to SDARDecoderLayer (see sdar.py)
    except MoE layers return router_logits alongside hidden_states.


SDARMoeDecoderLayer — Tensor Flow
===================================

    positions: (20,) int64
    hidden_states: (20, 4096) bfloat16
    residual: (20, 4096) bfloat16 | None
            │
            ▼  input_layernorm (RMSNorm, fused add+norm)
    hidden_states: (20, 4096) bfloat16
    residual:      (20, 4096) bfloat16
            │
            ▼  self_attn(positions, hidden_states)
    hidden_states: (20, 4096) bfloat16
            │
            ▼  post_attention_layernorm (fused add+norm)
    hidden_states: (20, 4096) bfloat16
    residual:      (20, 4096) bfloat16
            │
            ├── Dense MLP: mlp(hidden_states) → (20, 4096), router_logits=None
            │
            └── MoE Block: mlp(hidden_states) → ((20, 4096), (20, 8))
                                                  output     router_logits

    Output: (hidden_states, residual, router_logits)


SDARMoeForCausalLM — Full Forward
====================================

    input_ids: (20,) int64
    positions: (20,) int64
            │
            ▼  embed_tokens
    hidden: (20, 4096) bfloat16
            │
            ▼  layer_0 → layer_1 (MoE) → ... → layer_N
    hidden: (20, 4096) bfloat16
    router_logits_accum: tuple of (20, num_experts) per MoE layer
            │
            ▼  norm (final RMSNorm)
    hidden: (20, 4096) bfloat16
            │
            ▼  compute_logits → lm_head
    logits: (B, 151936) float32    ← on rank 0 only
"""

from typing import Optional, Tuple

import torch
from torch import nn
import torch.distributed as dist

from vdllm.layers.activation import SiluAndMul
from vdllm.layers.layernorm import RMSNorm
from vdllm.layers.linear import (
    MergedColumnParallelLinear,
    RowParallelLinear,
    ReplicatedLinear,
)
from vdllm.layers.embed_head import VocabParallelEmbedding, ParallelLMHead
from vdllm.kernels import fused_moe
from vdllm.models.sdar import SDARAttention as SDARMoeAttention


class SDARMoeMLP(nn.Module):
    """Dense MLP used inside both experts and non-MoE layers.

    Input:
        x: (num_tokens, hidden_size)

    Output:
        (num_tokens, hidden_size)
    """

    def __init__(self, hidden_size: int, intermediate_size: int) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size, [intermediate_size] * 2, bias=False)
        self.down_proj = RowParallelLinear(
            intermediate_size, hidden_size, bias=False)
        self.act_fn = SiluAndMul()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        return self.down_proj(x)


class SDARMoeSparseMoeBlock(nn.Module):
    """Top-k sparse MoE block with fused Triton kernel.

    Input:
        hidden_states: (S, hidden_size)

    Output:
        (output, router_logits) tuple
            output:        (S, hidden_size) — MoE output
            router_logits: (S, num_experts) — gating scores (for aux loss)
    """

    def __init__(self, hidden_size: int, intermediate_size: int,
                 num_experts: int, top_k: int, rms_norm_eps: float) -> None:
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.gate = ReplicatedLinear(hidden_size, num_experts, bias=False)
        self.experts = nn.ModuleList(
            [SDARMoeMLP(hidden_size, intermediate_size)
             for _ in range(num_experts)])
        self._prepare_fused_weights = False

    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        S, H = hidden_states.shape
        if not hasattr(self, '_w1') or self._w1 is None:
            raise RuntimeError(
                "Fused MoE weights _w1 not initialized. Ensure load_model was called.")
        if not hasattr(self, '_w2') or self._w2 is None:
            raise RuntimeError(
                "Fused MoE weights _w2 not initialized. Ensure load_model was called.")
        flat = hidden_states.view(-1, H)
        router_logits = self.gate(flat)
        probs = torch.softmax(router_logits, dim=-1, dtype=torch.float)
        top_p, top_i = torch.topk(probs, self.top_k, dim=-1)
        top_p = top_p / top_p.sum(dim=-1, keepdim=True)
        out = fused_moe(
            hidden_states=hidden_states, w1=self._w1, w2=self._w2,
            topk_weights=top_p, topk_ids=top_i, inplace=False)
        if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
            dist.all_reduce(out)
        return out.view(S, H), router_logits


class SDARMoeDecoderLayer(nn.Module):
    """Decoder layer with either dense MLP or MoE block.

    Layers where (layer_idx + 1) % decoder_sparse_step == 0 use MoE.

    Input:
        positions:     (num_tokens,) — position indices
        hidden_states: (num_tokens, hidden_size) — layer input
        residual:      (num_tokens, hidden_size) | None

    Output:
        (hidden_states, residual, router_logits) tuple
    """

    def __init__(self, config, layer_idx: int) -> None:
        super().__init__()
        self.self_attn = SDARMoeAttention(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            max_position=config.max_position_embeddings,
            head_dim=getattr(config, "head_dim", None),
            rms_norm_eps=config.rms_norm_eps,
            qkv_bias=getattr(config, "attention_bias", False),
            rope_theta=getattr(config, "rope_theta", 10000),
            rope_scaling=getattr(config, "rope_scaling", None))

        is_moe_layer = (
            config.num_experts > 0
            and (layer_idx + 1) % config.decoder_sparse_step == 0)
        if is_moe_layer:
            self.mlp = SDARMoeSparseMoeBlock(
                hidden_size=config.hidden_size,
                intermediate_size=config.moe_intermediate_size,
                num_experts=config.num_experts,
                top_k=config.num_experts_per_tok,
                rms_norm_eps=config.rms_norm_eps)
        else:
            self.mlp = SDARMoeMLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size)

        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, positions: torch.Tensor, hidden_states: torch.Tensor,
                residual: Optional[torch.Tensor]
                ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)

        hidden_states = self.self_attn(positions, hidden_states)
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        mlp_out = self.mlp(hidden_states)

        if isinstance(mlp_out, tuple):
            mlp_out, router_logits = mlp_out
        else:
            router_logits = None

        hidden_states = mlp_out
        return hidden_states, residual, router_logits


class SDARMoeModel(nn.Module):
    """Full SDAR-MoE transformer model.

    Input:
        input_ids: (num_tokens,) int64 — token IDs
        positions: (num_tokens,) int64 — position indices

    Output:
        (hidden_states, router_logits) tuple
            hidden_states:  (num_tokens, hidden_size)
            router_logits:  tuple of (S, num_experts) per MoE layer, or None
    """

    def __init__(self, config) -> None:
        super().__init__()
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [SDARMoeDecoderLayer(config, i)
             for i in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, input_ids: torch.Tensor,
                positions: torch.Tensor
                ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, ...] | None]:
        hidden_states = self.embed_tokens(input_ids)
        residual = None
        router_logits_accum: list[torch.Tensor] = []

        for layer in self.layers:
            hidden_states, residual, router_logits = layer(
                positions, hidden_states, residual)
            if router_logits is not None:
                router_logits_accum.append(router_logits)

        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states, tuple(router_logits_accum) if router_logits_accum else None


class SDARMoeForCausalLM(nn.Module):
    """SDAR-MoE model with causal LM head.

    Input:
        input_ids: (num_tokens,) int64
        positions: (num_tokens,) int64

    Output:
        forward():        (num_tokens, hidden_size)
        compute_logits():  (batch_size, vocab_size)
    """

    packed_modules_mapping = {
        "q_proj": ("qkv_proj", "q"),
        "k_proj": ("qkv_proj", "k"),
        "v_proj": ("qkv_proj", "v"),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }

    def __init__(self, config) -> None:
        super().__init__()
        self.model = SDARMoeModel(config)
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)
        if config.tie_word_embeddings:
            self.lm_head.weight.data = self.model.embed_tokens.weight.data

    def forward(self, input_ids: torch.Tensor,
                positions: torch.Tensor) -> torch.Tensor:
        hidden_states, _ = self.model(input_ids, positions)
        return hidden_states

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        logits = self.lm_head(hidden_states)
        return logits
