"""Block-local attention for SDAR block diffusion models.

Derived from JetEngine by Yihan Bian et al.
Reference: https://github.com/Labman42/JetEngine

Architecture Overview
====================

    Two attention modes depending on run phase:

    ┌──────────────────────────────────────────────────────────────┐
    │                  BlockAttention Forward                       │
    │                                                              │
    │  PREFILL (full context encoding)                             │
    │  ┌──────────────────────────────────────────────────────┐    │
    │  │  1. Store K,V into paged KV cache via Triton kernel  │    │
    │  │  2. Staircase block-local attention (sparse_attn)    │    │
    │  │                                                      │    │
    │  │  Token layout with block_length=4:                   │    │
    │  │    Tok:  [0  1  2  3 | 4  5  6  7 | 8  9 10 11]    │    │
    │  │    Blk:  [  block 0  |  block 1   |  block 2  ]    │    │
    │  │                                                      │    │
    │  │  Attention mask (staircase, 1=attend, .=masked):     │    │
    │  │                Q                                     │    │
    │  │         0 1 2 3 4 5 6 7 8 9 A B                     │    │
    │  │      0 [1 1 1 1 . . . . . . . .]                    │    │
    │  │    K 4 [1 1 1 1 1 1 1 1 . . . .]                    │    │
    │  │      8 [1 1 1 1 1 1 1 1 1 1 1 1]                    │    │
    │  │                                                      │    │
    │  └──────────────────────────────────────────────────────┘    │
    │                                                              │
    │  DENOISE (iterative block refinement)                        │
    │  ┌──────────────────────────────────────────────────────┐    │
    │  │  Q = noisy block, K/V = paged KV cache + new block   │    │
    │  │  Non-causal attention within the block                │    │
    │  │                                                       │    │
    │  │  Q:  [M  tok  M  M]     (block_length=4)             │    │
    │  │  KV: [cached context ─────────── | new block K,V]     │    │
    │  │       ^-- paged KV cache           ^-- appended       │    │
    │  └──────────────────────────────────────────────────────┘    │
    │                                                              │
    │  store_kvcache Triton Kernel                                 │
    │  ┌──────────────────────────────────────────────────────┐    │
    │  │  For each token i (N programs in parallel):           │    │
    │  │    slot = slot_mapping[i]                              │    │
    │  │    k_cache[slot, :] = key[i, :]    (D values)         │    │
    │  │    v_cache[slot, :] = value[i, :]  (D values)         │    │
    │  └──────────────────────────────────────────────────────┘    │
    └──────────────────────────────────────────────────────────────┘


Prefill Forward — Tensor Flow
==============================

    Example: batch=2 seqs, seq_lens=[12, 8], num_heads=32, num_kv_heads=8,
             head_dim=128, block_length=4, dtype=bfloat16

    q: (20, 4096) float16         ← total_tokens=12+8=20, 32*128=4096
    k: (20, 1024) float16         ← 8*128=1024
    v: (20, 1024) float16
                │
                ▼  view
    q: (20, 32, 128) float16
    k: (20, 8, 128)  float16
    v: (20, 8, 128)  float16
                │
                ├──► store_kvcache(k, v, k_cache, v_cache, slot_mapping)
                │       k_cache: (num_blocks, block_size, 8, 128) float16
                │       slot_mapping: (20,) int32
                │
                └──► sparse_attn_varlen(q, k, v,
                        cu_seqlens_q=[0, 12, 20],      (3,) int32
                        cu_seqlens_k=[0, 12, 20],
                        staircase_size=4)
                            │
                            ▼
                    o: (20, 32, 128) float16
                            │
                            ▼  view
                    output: (20, 4096) float16


Denoise Forward — Tensor Flow
==============================

    Example: batch=3 seqs, block_length=4, cached_lens=[24, 16, 32]

    q: (12, 4096) float16         ← 3*4=12 tokens
    k: (12, 1024) float16
    v: (12, 1024) float16
                │
                ▼  view → (3, 4, 32, 128), (3, 4, 8, 128), (3, 4, 8, 128)
                │
                └──► flash_attn_with_kvcache(
                        q,                         (3, 4, 32, 128) float16
                        k_cache=k_cache,           (num_blocks, block_size, 8, 128)
                        v_cache=v_cache,           same shape
                        k=k, v=v,                  (3, 4, 8, 128) — appended
                        cache_seqlens=[24,16,32],  (3,) int32
                        block_table=block_tables,  (3, max_blocks) int32
                        causal=False)
                            │
                            ▼
                    o: (3, 4, 32, 128) float16
                            │
                            ▼  view
                    output: (12, 4096) float16
"""

import torch
from torch import nn
import triton
import triton.language as tl

from vdllm.utils.context import get_context
from vdllm.engine.sequence import RunType
from vdllm.kernels.triton.attention import sparse_attn_varlen
from flash_attn import flash_attn_with_kvcache
from vdllm.backends.base import AttentionBackend


@triton.jit
def store_kvcache_kernel(
    key_ptr,
    key_stride,
    value_ptr,
    value_stride,
    k_cache_ptr,
    v_cache_ptr,
    slot_mapping_ptr,
    D: tl.constexpr,
):
    """Triton kernel: scatter K,V vectors into paged cache slots.

    Input:
        key_ptr:          pointer to key tensor (N, num_heads * head_dim)
        value_ptr:        pointer to value tensor (N, num_heads * head_dim)
        k_cache_ptr:      pointer to key cache (num_blocks * block_size, D)
        v_cache_ptr:      pointer to value cache (same shape)
        slot_mapping_ptr: pointer to slot indices (N,)
        D:                num_heads * head_dim (compile-time constant)

    Output:
        k_cache and v_cache updated in-place at mapped slots
    """
    idx = tl.program_id(0)
    key_offsets = idx * key_stride + tl.arange(0, D)
    value_offsets = idx * value_stride + tl.arange(0, D)
    key = tl.load(key_ptr + key_offsets)
    value = tl.load(value_ptr + value_offsets)
    slot = tl.load(slot_mapping_ptr + idx)
    cache_offsets = slot * D + tl.arange(0, D)
    tl.store(k_cache_ptr + cache_offsets, key)
    tl.store(v_cache_ptr + cache_offsets, value)


def store_kvcache(key: torch.Tensor, value: torch.Tensor,
                  k_cache: torch.Tensor, v_cache: torch.Tensor,
                  slot_mapping: torch.Tensor):
    """Scatter key/value vectors into paged KV cache.

    Input:
        key:          (N, num_heads, head_dim) — key vectors
        value:        (N, num_heads, head_dim) — value vectors
        k_cache:      (num_blocks * block_size, num_heads * head_dim)
        v_cache:      same shape as k_cache
        slot_mapping: (N,) int32 — physical slot index for each token

    Output:
        k_cache, v_cache updated in-place
    """
    N, num_heads, head_dim = key.shape
    D = num_heads * head_dim
    assert key.stride(-1) == 1 and value.stride(-1) == 1
    assert key.stride(1) == head_dim and value.stride(1) == head_dim
    assert k_cache.stride(1) == D and v_cache.stride(1) == D
    assert slot_mapping.numel() == N
    store_kvcache_kernel[(N,)](
        key, key.stride(0), value, value.stride(0),
        k_cache, v_cache, slot_mapping, D)


class Attention(nn.Module):
    """Base attention module with KV cache placeholders.

    Input:
        num_heads:    int — number of query heads (after TP split)
        head_dim:     int — dimension per head
        scale:        float — attention scale factor (1/sqrt(head_dim))
        num_kv_heads: int — number of KV heads (GQA support)
    """

    def __init__(self, num_heads, head_dim, scale, num_kv_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.k_cache = self.v_cache = torch.tensor([])

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        pass


class BlockAttention(Attention):
    """SDAR block-local attention with staircase prefill and paged denoise.

    Input:
        q: (total_tokens, num_heads * head_dim) — query vectors
        k: (total_tokens, num_kv_heads * head_dim) — key vectors
        v: (total_tokens, num_kv_heads * head_dim) — value vectors

    Output:
        (total_tokens, num_heads * head_dim) — attention output

    Behavior:
        PREFILL: stores KV to cache, uses staircase sparse attention
        DENOISE: reshapes to (batch, block_length, ...), uses FlashAttention
                 with paged KV cache (non-causal within block)

    Backend Dispatch:
        When backend is set, delegates to AttentionBackend protocol.
        Otherwise uses direct Triton/FlashInfer kernels (legacy mode).
    """

    def __init__(
        self,
        num_heads: int,
        head_dim: int,
        scale: float,
        num_kv_heads: int,
        backend: AttentionBackend | None = None,
    ):
        super().__init__(num_heads, head_dim, scale, num_kv_heads)
        self.backend = backend

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        o: torch.Tensor
        q = q.view(-1, self.num_heads, self.head_dim)
        k = k.view(-1, self.num_kv_heads, self.head_dim)
        v = v.view(-1, self.num_kv_heads, self.head_dim)
        context = get_context()
        k_cache, v_cache = self.k_cache, self.v_cache

        should_store_whole = (context.run_type == RunType.PREFILL)
        if should_store_whole and k_cache.numel() and v_cache.numel():
            if self.backend is not None:
                self.backend.reshape_and_cache(
                    k.view(-1, self.num_kv_heads * self.head_dim),
                    v.view(-1, self.num_kv_heads * self.head_dim),
                    k_cache, v_cache, context.slot_mapping)
            else:
                store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)

        if context.run_type == RunType.PREFILL:
            if self.backend is not None:
                o = self.backend.prefill_attention(
                    q.view(-1, self.num_heads * self.head_dim),
                    k.view(-1, self.num_kv_heads * self.head_dim),
                    v.view(-1, self.num_kv_heads * self.head_dim),
                    block_length=context.block_length,
                    staircase=True)
            else:
                o = sparse_attn_varlen(
                    q, k, v,
                    cu_seqlens_q=context.cu_seqlens_q,
                    cu_seqlens_k=context.cu_seqlens_k,
                    staircase_size=context.block_length)
        else:
            if self.backend is not None:
                o = self.backend.denoise_attention(
                    q.view(-1, self.num_heads * self.head_dim),
                    k_cache, v_cache,
                    k.view(-1, self.num_kv_heads * self.head_dim),
                    v.view(-1, self.num_kv_heads * self.head_dim),
                    block_tables=context.block_tables,
                    seq_lens=context.context_lens)
            else:
                q = q.view(-1, context.block_length, self.num_heads, self.head_dim)
                k = k.view(-1, context.block_length, self.num_kv_heads, self.head_dim)
                v = v.view(-1, context.block_length, self.num_kv_heads, self.head_dim)
                o = flash_attn_with_kvcache(
                    q, k_cache=k_cache, v_cache=v_cache, k=k, v=v,
                    cache_seqlens=context.context_lens,
                    block_table=context.block_tables,
                    causal=False)
        o = o.view(-1, self.num_heads * self.head_dim)
        return o
