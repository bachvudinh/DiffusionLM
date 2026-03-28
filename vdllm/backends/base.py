"""AttentionBackend Protocol for vdllm attention implementations.

This module defines the contract that all attention backends (CUDA/FlashInfer,
MLX/Metal, MPS, CPU) must implement to work with the BlockAttention module.

Architecture Overview
======================

    ┌─────────────────────────────────────────────────────────────┐
    │                    AttentionBackend                          │
    │                                                             │
    │  Protocol Methods:                                          │
    │    name                    → backend identifier string      │
    │    prefill_attention       → staircase attention for prefill│
    │    denoise_attention       → paged KV cache for denoise      │
    │    reshape_and_cache       → store K,V to paged cache       │
    │                                                             │
    │  Implementations:                                          │
    │    CUDAAttentionBackend    (FlashInfer + Triton)            │
    │    MLXAttentionBackend     (Metal via MX)                   │
    │    MPSAttentionBackend     (PyTorch MPS backend)            │
    │    CPUAttentionBackend     (Naive PyTorch fallback)         │
    └─────────────────────────────────────────────────────────────┘

The Protocol ensures backends can be swapped without changing the
BlockAttention layer code. Each backend handles:
  - PREFILL: staircase block-local attention + KV cache storage
  - DENOISE: paged KV cache retrieval + block-local attention
"""

from typing import Protocol, runtime_checkable
from dataclasses import dataclass
import torch


@runtime_checkable
class AttentionBackend(Protocol):
    """Protocol for attention backends.

    All backends must implement these methods to work with BlockAttention.

    Example:
        class CUDAAttentionBackend:
            @property
            def name(self) -> str:
                return "cuda"

            def prefill_attention(self, q, k, v, block_length, staircase):
                ...

            def denoise_attention(self, q, k_cache, v_cache, k_new, v_new,
                                  block_tables, seq_lens):
                ...

            def reshape_and_cache(self, k, v, k_cache, v_cache, slot_mapping):
                ...
    """

    @property
    def name(self) -> str:
        """Return the backend name identifier (e.g., 'cuda', 'mlx', 'mps', 'cpu')."""
        ...

    def prefill_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        block_length: int,
        staircase: bool,
    ) -> torch.Tensor:
        """Compute staircase block-local attention for prefill phase.

        Args:
            q: Query tensor (total_tokens, num_heads * head_dim)
            k: Key tensor (total_tokens, num_kv_heads * head_dim)
            v: Value tensor (total_tokens, num_kv_heads * head_dim)
            block_length: Size of each block for staircase masking
            staircase: Whether to apply staircase masking (True for prefill)

        Returns:
            Attention output (total_tokens, num_heads * head_dim)
        """
        ...

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

        Args:
            q: Query tensor (batch_size * block_length, num_heads * head_dim)
            k_cache: Paged key cache (num_blocks, block_size, num_kv_heads, head_dim)
            v_cache: Paged value cache (num_blocks, block_size, num_kv_heads, head_dim)
            k_new: New block keys (batch_size * block_length, num_kv_heads * head_dim)
            v_new: New block values (batch_size * block_length, num_kv_heads * head_dim)
            block_tables: Block assignments (batch_size, max_blocks_per_seq)
            seq_lens: Cached sequence lengths (batch_size,)

        Returns:
            Attention output (batch_size * block_length, num_heads * head_dim)
        """
        ...

    def reshape_and_cache(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> None:
        """Reshape and store key/value tensors into paged KV cache.

        Args:
            k: Key tensor (total_tokens, num_kv_heads * head_dim)
            v: Value tensor (total_tokens, num_kv_heads * head_dim)
            k_cache: Paged key cache to write to
            v_cache: Paged value cache to write to
            slot_mapping: (total_tokens,) int32 - physical slot indices
        """
        ...


@dataclass
class CacheConfig:
    """Configuration for KV cache allocation.

    Attributes:
        num_blocks: Total number of paged cache blocks
        block_size: Number of tokens per block
        num_kv_heads: Number of key/value heads (for GQA)
        head_dim: Dimension per attention head
        dtype: Data type for cache (bfloat16, float16, etc.)
    """

    num_blocks: int
    block_size: int
    num_kv_heads: int
    head_dim: int
    dtype: torch.dtype = torch.bfloat16
