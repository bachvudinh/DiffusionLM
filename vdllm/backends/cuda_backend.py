"""CUDA attention backend for vdllm block diffusion.

This backend wraps FlashInfer and Triton kernels for high-performance
attention on NVIDIA GPUs.

Architecture Overview
======================

    ┌──────────────────────────────────────────────────────────────┐
    │                CUDAAttentionBackend                            │
    │                                                              │
    │  PREFILL Mode (sparse_attn_varlen - Triton staircase):      │
    │    1. reshape_and_cache: store k,v to paged cache            │
    │    2. sparse_attn_varlen: staircase block-local attention    │
    │                                                              │
    │  DENOISE Mode (flash_attn_with_kvcache):                     │
    │    1. flash_attn_with_kvcache: paged KV cache retrieval      │
    │       + block attention in one kernel                        │
    │                                                              │
    │  Kernel Sources:                                             │
    │    - sparse_attn_varlen: vdllm/kernels/triton/attention/     │
    │      block_prefill_attention_v2.py                           │
    │    - fused_page_attention: vdllm/kernels/triton/attention/   │
    │      fused_page_attention_v6.py                              │
    └──────────────────────────────────────────────────────────────┘
"""

import torch
from typing import Optional

from vdllm.backends.base import AttentionBackend
from vdllm.kernels.triton.attention import sparse_attn_varlen
from vdllm.kernels.triton.attention import fused_kv_cache_attention
from flash_attn import flash_attn_with_kvcache


class CUDAAttentionBackend:
    """CUDA implementation of attention for block diffusion.

    Uses Triton kernels for prefill (staircase attention) and
    FlashAttention with paged KV cache for denoise.

    Attributes:
        num_heads: Number of query heads
        num_kv_heads: Number of key/value heads (GQA support)
        head_dim: Dimension per head
        scale: Attention scale factor (1/sqrt(head_dim))
    """

    def __init__(
        self,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
    ):
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.scale = head_dim ** -0.5

    @property
    def name(self) -> str:
        return "cuda"

    def prefill_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        block_length: int,
        staircase: bool,
    ) -> torch.Tensor:
        """Compute staircase block-local attention for prefill phase.

        Uses Triton sparse_attn_varlen kernel which implements
        block-local attention with a staircase masking pattern.

        Args:
            q: (total_tokens, num_heads * head_dim) Query vectors
            k: (total_tokens, num_kv_heads * head_dim) Key vectors
            v: (total_tokens, num_kv_heads * head_dim) Value vectors
            block_length: Size of each block for staircase masking
            staircase: Whether to apply staircase masking (True for prefill)

        Returns:
            (total_tokens, num_heads * head_dim) Attention output
        """
        # Reshape from (total_tokens, heads * head_dim) to (total_tokens, heads, head_dim)
        q = q.view(-1, self.num_heads, self.head_dim)
        k = k.view(-1, self.num_kv_heads, self.head_dim)
        v = v.view(-1, self.num_kv_heads, self.head_dim)

        # sparse_attn_varlen expects cu_seqlens from context
        # but we need to compute it here from tensor shapes
        # For single sequence prefill, cu_seqlens = [0, total_tokens]
        from vdllm.utils.context import get_context
        context = get_context()

        if staircase:
            o = sparse_attn_varlen(
                q, k, v,
                cu_seqlens_q=context.cu_seqlens_q,
                cu_seqlens_k=context.cu_seqlens_k,
                staircase_size=block_length)
        else:
            # Fallback to standard attention without staircase
            o = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, scale=self.scale)

        return o.view(-1, self.num_heads * self.head_dim)

    def denoise_attention(
        self,
        q: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        k_new: torch.Tensor,
        v_new: torch.Tensor,
        block_tables: torch.Tensor,
        seq_lens: torch.Tensor,
    ) -> torch.Tensor:
        """Compute attention with paged KV cache for denoise phase.

        Uses flash_attn_with_kvcache which efficiently retrieves
        cached KV pairs and computes attention in one kernel.

        Args:
            q: (batch_size * block_length, num_heads * head_dim)
            k_cache: (num_blocks, block_size, num_kv_heads, head_dim)
            v_cache: (num_blocks, block_size, num_kv_heads, head_dim)
            k_new: (batch_size * block_length, num_kv_heads * head_dim)
            v_new: (batch_size * block_length, num_kv_heads * head_dim)
            block_tables: (batch_size, max_blocks_per_seq)
            seq_lens: (batch_size,) number of cached tokens per sequence

        Returns:
            (batch_size * block_length, num_heads * head_dim)
        """
        from vdllm.utils.context import get_context
        context = get_context()

        batch_size = len(seq_lens)
        block_length = context.block_length

        # Reshape for FlashAttention with kvcache
        # q: (batch, block_len, num_heads, head_dim)
        q = q.view(batch_size, block_length, self.num_heads, self.head_dim)
        # k_new, v_new: (batch, block_len, num_kv_heads, head_dim)
        k_new = k_new.view(batch_size, block_length, self.num_kv_heads, self.head_dim)
        v_new = v_new.view(batch_size, block_length, self.num_kv_heads, self.head_dim)

        o = flash_attn_with_kvcache(
            q,
            k_cache=k_cache,
            v_cache=v_cache,
            k=k_new,
            v=v_new,
            cache_seqlens=seq_lens,
            block_table=block_tables,
            causal=False,
        )

        return o.view(-1, self.num_heads * self.head_dim)

    def reshape_and_cache(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> None:
        """Reshape and store key/value tensors into paged KV cache.

        Uses the store_kvcache Triton kernel from attention.py.

        Args:
            k: (total_tokens, num_kv_heads * head_dim)
            v: (total_tokens, num_kv_heads * head_dim)
            k_cache: (num_blocks * block_size, num_kv_heads * head_dim)
            v_cache: (num_blocks * block_size, num_kv_heads * head_dim)
            slot_mapping: (total_tokens,) int32 physical slot indices
        """
        from vdllm.layers.attention import store_kvcache

        k = k.view(-1, self.num_kv_heads, self.head_dim)
        v = v.view(-1, self.num_kv_heads, self.head_dim)

        store_kvcache(k, v, k_cache, v_cache, slot_mapping)


# Export the backend class
__all__ = ["CUDAAttentionBackend"]
