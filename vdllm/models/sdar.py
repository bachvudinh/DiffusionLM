"""SDAR (Semi-autoregressive Diffusion with Autoregressive Reasoning) model.

Derived from JetEngine by Yihan Bian et al.
Reference: https://github.com/Labman42/JetEngine

Architecture Overview
====================

    SDAR is a dense transformer for block diffusion language modeling.
    It generates text in blocks of tokens, denoising each block iteratively
    while attending to all previously committed blocks.

    ┌──────────────────────────────────────────────────────────────────┐
    │                    SDARForCausalLM                                │
    │                                                                  │
    │  SDARModel                                                       │
    │  ├── VocabParallelEmbedding (TP-sharded)                         │
    │  ├── N x SDARDecoderLayer                                        │
    │  │    ├── input_layernorm (RMSNorm, fused add+norm)              │
    │  │    ├── SDARAttention                                          │
    │  │    │    ├── QKVParallelLinear → fused Q,K,V projection        │
    │  │    │    ├── q_norm, k_norm (per-head RMSNorm)                 │
    │  │    │    ├── RotaryEmbedding (RoPE)                            │
    │  │    │    ├── BlockAttention (staircase prefill / paged denoise) │
    │  │    │    └── RowParallelLinear (o_proj, all-reduce)             │
    │  │    ├── post_attention_layernorm (RMSNorm)                     │
    │  │    └── SDARMLP                                                │
    │  │         ├── MergedColumnParallelLinear (gate_up_proj)          │
    │  │         ├── SiluAndMul (fused gating)                          │
    │  │         └── RowParallelLinear (down_proj, all-reduce)          │
    │  └── RMSNorm (final norm)                                        │
    │                                                                  │
    │  ParallelLMHead (TP-sharded output projection)                   │
    │                                                                  │
    │  Forward Data Flow:                                              │
    │    input_ids ──► embed ──► [layer_0 ... layer_N] ──► norm        │
    │                     │         │                        │          │
    │                     │    hidden + residual stream      │          │
    │                     │                                  ▼          │
    │                     └──────────────────────► compute_logits       │
    │                                              ──► (B, vocab)      │
    │                                                                  │
    │  packed_modules_mapping (for weight loading):                    │
    │    "q_proj" → ("qkv_proj", "q")                                  │
    │    "k_proj" → ("qkv_proj", "k")                                  │
    │    "v_proj" → ("qkv_proj", "v")                                  │
    │    "gate_proj" → ("gate_up_proj", 0)                             │
    │    "up_proj"   → ("gate_up_proj", 1)                             │
    └──────────────────────────────────────────────────────────────────┘


SDARAttention — Tensor Flow
=============================

    Example: hidden_size=4096, num_heads=32, num_kv_heads=8, head_dim=128,
             tp_size=2, N=20 tokens, dtype=bfloat16

    hidden_states: (20, 4096) bfloat16
            │
            ▼  qkv_proj (QKVParallelLinear)
    qkv: (20, 3072) bfloat16      ← per-rank: 16*128 + 4*128 + 4*128 = 3072
            │
            ▼  split([2048, 512, 512], dim=-1)
    q: (20, 2048) bfloat16
    k: (20, 512)  bfloat16
    v: (20, 512)  bfloat16
            │
            ▼  view → (20, 16, 128), (20, 4, 128)
            ▼  q_norm, k_norm  (per-head RMSNorm, head_dim=128)
            ▼  view back → (20, 2048), (20, 512)
            │
            ▼  rotary_emb(positions, q, k)
    q: (20, 2048) bfloat16         ← rotated by position
    k: (20, 512)  bfloat16         ← rotated by position
            │
            ▼  BlockAttention.forward(q, k, v)
            │    (see layers/attention.py for PREFILL vs DENOISE paths)
    o: (20, 2048) bfloat16         ← 16 heads × 128 dim
            │
            ▼  o_proj (RowParallelLinear + all_reduce)
    output: (20, 4096) bfloat16


SDARMLP — Tensor Flow
=======================

    Example: hidden_size=4096, intermediate_size=11008, tp_size=2

    x: (20, 4096) bfloat16
            │
            ▼  gate_up_proj (MergedColumnParallelLinear)
    gate_up: (20, 11008) bfloat16   ← per-rank: 2 × 11008/2 = 11008
            │
            ▼  SiluAndMul
            │    chunk → gate(20, 5504), up(20, 5504)
            │    SiLU(gate) * up
    x: (20, 5504) bfloat16
            │
            ▼  down_proj (RowParallelLinear + all_reduce)
    output: (20, 4096) bfloat16


SDARDecoderLayer — Tensor Flow
================================

    Example: layer_idx=0, first layer (residual=None)

    positions: (20,) int64
    hidden_states: (20, 4096) bfloat16
    residual: None
            │
            ▼  input_layernorm (RMSNorm, no residual for first layer)
            │    residual = hidden_states            ← save for stream
            │    hidden_states = rms_norm(hidden_states)
    hidden_states: (20, 4096) bfloat16    ← normalized
    residual:      (20, 4096) bfloat16    ← original input
            │
            ▼  self_attn(positions, hidden_states)
    hidden_states: (20, 4096) bfloat16    ← attention output
            │
            ▼  post_attention_layernorm (RMSNorm, fused add+norm)
            │    hidden_states, residual = add_rms_forward(hidden_states, residual)
    hidden_states: (20, 4096) bfloat16    ← normalized (attn_out + residual)
    residual:      (20, 4096) bfloat16    ← attn_out + original input
            │
            ▼  mlp(hidden_states)
    hidden_states: (20, 4096) bfloat16    ← MLP output

    Output: (hidden_states, residual)
            Next layer's input_layernorm will add hidden_states to residual.


SDARModel — Full Forward
==========================

    input_ids: (20,) int64
    positions: (20,) int64
            │
            ▼  embed_tokens (VocabParallelEmbedding)
    hidden: (20, 4096) bfloat16
            │
            ▼  layer_0.forward(positions, hidden, residual=None)
    hidden: (20, 4096), residual: (20, 4096)
            │
            ▼  layer_1.forward(positions, hidden, residual)
            ...
            ▼  layer_N.forward(positions, hidden, residual)
    hidden: (20, 4096), residual: (20, 4096)
            │
            ▼  norm (final RMSNorm, fused add+norm)
    hidden: (20, 4096) bfloat16    ← final hidden states


SDARForCausalLM — Full Forward + Logits
==========================================

    input_ids: (20,) int64
    positions: (20,) int64
            │
            ▼  self.model(input_ids, positions)
    hidden_states: (20, 4096) bfloat16
            │
            ▼  compute_logits(hidden_states) → self.lm_head(hidden_states)
            │    PREFILL: extract last tokens → (B, 4096) → logits
            │    DENOISE: all tokens → (B*block_len, 4096) → logits
    logits: (B, 151936) float32       ← on rank 0 only (None on others)
"""

import torch
from torch import nn
import torch.distributed as dist

from vdllm.layers.activation import SiluAndMul
from vdllm.layers.attention import BlockAttention
from vdllm.layers.layernorm import RMSNorm
from vdllm.layers.linear import QKVParallelLinear, MergedColumnParallelLinear, RowParallelLinear
from vdllm.layers.rotary_embedding import get_rope
from vdllm.layers.embed_head import VocabParallelEmbedding, ParallelLMHead


class SDARAttention(nn.Module):
    """Multi-head attention with QKV fusion, per-head norms, RoPE, and block attention.

    Input:
        positions:     (num_tokens,) int64 — position indices
        hidden_states: (num_tokens, hidden_size) — input hidden states

    Output:
        (num_tokens, hidden_size) — attention output after output projection
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        max_position: int = 4096 * 32,
        head_dim: int | None = None,
        rms_norm_eps: float = 1e-06,
        qkv_bias: bool = False,
        rope_theta: float = 10000,
        rope_scaling: tuple | None = None,
        process_group: dist.ProcessGroup = None,
    ) -> None:
        super().__init__()
        tp_size = dist.get_world_size(group=process_group)
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        assert self.total_num_kv_heads % tp_size == 0
        self.num_kv_heads = self.total_num_kv_heads // tp_size
        self.head_dim = head_dim or hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5

        self.qkv_proj = QKVParallelLinear(
            hidden_size, self.head_dim, process_group,
            self.total_num_heads, self.total_num_kv_heads, bias=qkv_bias)
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim, hidden_size,
            process_group, bias=False)
        self.rotary_emb = get_rope(
            self.head_dim, rotary_dim=self.head_dim,
            max_position=max_position, base=rope_theta,
            rope_scaling=rope_scaling)
        self.attn = BlockAttention(
            self.num_heads, self.head_dim, self.scaling, self.num_kv_heads)
        self.q_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)

    def forward(self, positions: torch.Tensor,
                hidden_states: torch.Tensor) -> torch.Tensor:
        qkv = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q_by_head = q.view(-1, self.num_heads, self.head_dim)
        q_by_head = self.q_norm(q_by_head)
        q = q_by_head.view(q.shape)
        k_by_head = k.view(-1, self.num_kv_heads, self.head_dim)
        k_by_head = self.k_norm(k_by_head)
        k = k_by_head.view(k.shape)
        q, k = self.rotary_emb(positions, q, k)
        o = self.attn(q, k, v)
        output = self.o_proj(o)
        return output


class SDARMLP(nn.Module):
    """Dense MLP with fused gate-up projection and SiLU gating.

    Input:
        x: (num_tokens, hidden_size)

    Output:
        (num_tokens, hidden_size)

    Data Flow:
        x ──► gate_up_proj ──► SiLU(gate) * up ──► down_proj ──► output
          (H → 2*I)          (2*I → I)           (I → H)
    """

    def __init__(self, hidden_size: int, intermediate_size: int,
                 hidden_act: str, process_group: dist.ProcessGroup) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size, [intermediate_size] * 2, process_group, bias=False)
        self.down_proj = RowParallelLinear(
            intermediate_size, hidden_size, process_group, bias=False)
        assert hidden_act == "silu"
        self.act_fn = SiluAndMul()

    def forward(self, x):
        gate_up = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x = self.down_proj(x)
        return x


class SDARDecoderLayer(nn.Module):
    """Single transformer decoder layer with pre-norm residual connections.

    Input:
        positions:     (num_tokens,) — position indices
        hidden_states: (num_tokens, hidden_size) — layer input
        residual:      (num_tokens, hidden_size) | None — residual stream

    Output:
        (hidden_states, residual) tuple
    """

    def __init__(self, config, process_group) -> None:
        super().__init__()
        self.self_attn = SDARAttention(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            max_position=config.max_position_embeddings,
            rms_norm_eps=config.rms_norm_eps,
            qkv_bias=getattr(config, 'attention_bias', False),
            head_dim=getattr(config, 'head_dim', None),
            rope_theta=getattr(config, "rope_theta", 1000000),
            rope_scaling=getattr(config, "rope_scaling", None),
            process_group=process_group)
        self.mlp = SDARMLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            process_group=process_group)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, positions: torch.Tensor, hidden_states: torch.Tensor,
                residual: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(positions, hidden_states)
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


class SDARModel(nn.Module):
    """Full SDAR transformer model (embedding + layers + norm).

    Input:
        input_ids: (num_tokens,) int64 — token IDs
        positions: (num_tokens,) int64 — position indices

    Output:
        (num_tokens, hidden_size) — final hidden states
    """

    def __init__(self, config, process_group) -> None:
        super().__init__()
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size, config.hidden_size, process_group)
        self.layers = nn.ModuleList([
            SDARDecoderLayer(config, process_group)
            for _ in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, input_ids: torch.Tensor,
                positions: torch.Tensor) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(positions, hidden_states, residual)
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


class SDARForCausalLM(nn.Module):
    """SDAR model with causal LM head for block diffusion generation.

    Input:
        input_ids: (num_tokens,) int64 — token IDs
        positions: (num_tokens,) int64 — position indices

    Output:
        forward():        (num_tokens, hidden_size) — hidden states
        compute_logits():  (batch_size, vocab_size) — output logits
    """

    packed_modules_mapping = {
        "q_proj": ("qkv_proj", "q"),
        "k_proj": ("qkv_proj", "k"),
        "v_proj": ("qkv_proj", "v"),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }

    def __init__(self, config, process_group) -> None:
        super().__init__()
        self.model = SDARModel(config, process_group)
        self.lm_head = ParallelLMHead(
            config.vocab_size, config.hidden_size, process_group)
        if config.tie_word_embeddings:
            self.lm_head.weight.data = self.model.embed_tokens.weight.data

    def forward(self, input_ids: torch.Tensor,
                positions: torch.Tensor) -> torch.Tensor:
        hidden_states = self.model(input_ids, positions)
        return hidden_states

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        logits = self.lm_head(hidden_states)
        return logits
