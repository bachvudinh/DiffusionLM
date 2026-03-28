"""Rotary Positional Embedding (RoPE) with LRU-cached factory.

Derived from JetEngine by Yihan Bian et al.
Reference: https://github.com/Labman42/JetEngine

Architecture Overview
====================

    RoPE encodes absolute position information by rotating pairs
    of dimensions in the query/key vectors:

    ┌───────────────────────────────────────────────────────────┐
    │                   RotaryEmbedding                          │
    │                                                           │
    │  Initialization:                                          │
    │    inv_freq[i] = 1 / (base^(2i/d))     i = 0..d/2-1     │
    │    freqs[pos, i] = pos * inv_freq[i]                      │
    │    cache = [cos(freqs) | sin(freqs)]    (max_pos, d)     │
    │                                                           │
    │  Forward (compiled):                                      │
    │    cos, sin = cache[positions].chunk(2)                    │
    │    For each head:                                          │
    │      x1, x2 = query_head[:d/2], query_head[d/2:]         │
    │      y1 = x1 * cos - x2 * sin                            │
    │      y2 = x2 * cos + x1 * sin                            │
    │      rotated = [y1, y2]                                   │
    │                                                           │
    │  get_rope() — LRU-cached factory (singleton per config)   │
    └───────────────────────────────────────────────────────────┘


Forward — Tensor Flow
======================

    Example: head_dim=128, num_heads=16 (per-rank), num_kv_heads=4,
             N=20 tokens, base=500000, dtype=bfloat16

    positions: (20,) int64          ← e.g., [0,1,2,...,11, 0,1,...,7]

    cos_sin_cache: (max_pos, 128) float32    ← precomputed buffer
                                               [cos(64) | sin(64)]
            │
            ▼  index by positions
    cos_sin: (20, 128) float32
            │
            ▼  chunk(2, dim=-1)
    cos: (20, 64) float32
    sin: (20, 64) float32


    query: (20, 2048) bfloat16     ← 16 heads × 128 dim
            │
            ▼  view → (20, 16, 128)
            │
            ▼  chunk(2, dim=-1)
    x1: (20, 16, 64) float32
    x2: (20, 16, 64) float32
            │           cos: (20, 1, 64) ← unsqueeze(-2)
            │           sin: (20, 1, 64) ← unsqueeze(-2)
            ▼
    y1 = x1 * cos - x2 * sin     (20, 16, 64) float32
    y2 = x2 * cos + x1 * sin     (20, 16, 64) float32
            │
            ▼  cat → (20, 16, 128) → view → (20, 2048) bfloat16
    rotated_query: (20, 2048) bfloat16

    key: (20, 512) bfloat16       ← 4 kv_heads × 128 dim
            │
            ▼  same rotation as query with (20, 4, 128) view
    rotated_key: (20, 512) bfloat16

    Output: (rotated_query, rotated_key)


Rotation Geometry
==================

    Each pair of dimensions (x1, x2) is rotated by angle theta = pos * inv_freq:

        ┌        ┐   ┌              ┐ ┌    ┐
        │ y1     │ = │ cos  -sin    │ │ x1 │
        │ y2     │   │ sin   cos    │ │ x2 │
        └        ┘   └              ┘ └    ┘

    Low-frequency dims rotate slowly (position-insensitive).
    High-frequency dims rotate fast (capture fine-grained position).
"""

from functools import lru_cache
import torch
from torch import nn


def apply_rotary_emb(x: torch.Tensor, cos: torch.Tensor,
                     sin: torch.Tensor) -> torch.Tensor:
    """Apply rotary embedding to input tensor.

    Input:
        x:   (num_tokens, num_heads, head_dim) — vectors to rotate
        cos: (num_tokens, head_dim//2) — cosine components
        sin: (num_tokens, head_dim//2) — sine components

    Output:
        (num_tokens, num_heads, head_dim) — rotated vectors
    """
    cos = cos.unsqueeze(-2)
    sin = sin.unsqueeze(-2)
    x1, x2 = torch.chunk(x.to(torch.float32), 2, dim=-1)
    y1 = x1 * cos - x2 * sin
    y2 = x2 * cos + x1 * sin
    return torch.cat((y1, y2), dim=-1).to(x.dtype)


class RotaryEmbedding(nn.Module):
    """Rotary positional embedding with pre-computed cache.

    Input:
        head_size:               int — dimension per head (must equal rotary_dim)
        rotary_dim:              int — number of dimensions to rotate
        max_position_embeddings: int — maximum sequence length
        base:                    float — RoPE base frequency (e.g., 10000)
    """

    def __init__(self, head_size: int, rotary_dim: int,
                 max_position_embeddings: int, base: float) -> None:
        super().__init__()
        self.head_size = head_size
        assert rotary_dim == head_size
        inv_freq = 1.0 / (base ** (torch.arange(0, rotary_dim, 2, dtype=torch.float) / rotary_dim))
        t = torch.arange(max_position_embeddings, dtype=torch.float)
        freqs = torch.einsum("i,j -> ij", t, inv_freq)
        cos = freqs.cos()
        sin = freqs.sin()
        cache = torch.cat((cos, sin), dim=-1)
        self.register_buffer("cos_sin_cache", cache, persistent=False)

    @torch.compile
    def forward(self, positions: torch.Tensor, query: torch.Tensor,
                key: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply RoPE to query and key tensors.

        Input:
            positions: (num_tokens,) int64 — position indices
            query:     (num_tokens, num_heads * head_dim) — query vectors
            key:       (num_tokens, num_kv_heads * head_dim) — key vectors

        Output:
            (rotated_query, rotated_key) — same shapes as input
        """
        num_tokens = positions.size(0)
        cos_sin = self.cos_sin_cache[positions]
        cos, sin = cos_sin.chunk(2, dim=-1)
        query_shape = query.shape
        query = query.view(num_tokens, -1, self.head_size)
        query = apply_rotary_emb(query, cos, sin).view(query_shape)
        key_shape = key.shape
        key = key.view(num_tokens, -1, self.head_size)
        key = apply_rotary_emb(key, cos, sin).view(key_shape)
        return query, key


@lru_cache(1)
def get_rope(head_size: int, rotary_dim: int, max_position: int,
             base: float, rope_scaling: dict | None = None):
    """Factory function for RotaryEmbedding (LRU-cached singleton).

    Input:
        head_size:    int — dimension per attention head
        rotary_dim:   int — must equal head_size
        max_position: int — maximum position index
        base:         float — RoPE base frequency
        rope_scaling: dict | None — must be None (scaling not supported)

    Output:
        RotaryEmbedding instance (cached)
    """
    assert rope_scaling is None
    rotary_emb = RotaryEmbedding(head_size, rotary_dim, max_position, base)
    return rotary_emb
