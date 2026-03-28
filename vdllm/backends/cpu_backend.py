"""CPU attention backend for vdllm (fallback implementation).

This backend uses standard PyTorch operations for attention computation,
serving as a fallback when CUDA, MLX, and MPS are unavailable.

Architecture Overview
======================

    ┌──────────────────────────────────────────────────────────────┐
    │                CPUAttentionBackend                             │
    │                                                              │
    │  PREFILL Mode:                                              │
    │    1. reshape_and_cache: simple index assignment             │
    │    2. sparse_attention: SDPA with staircase mask            │
    │                                                              │
    │  DENOISE Mode:                                              │
    │    1. gather_kvcache: simple index selection                │
    │    2. block_attention: SDPA with full k,v                   │
    │                                                              │
    │  Note: No kernel fusion or optimization - use for testing   │
    │        and development only.                                  │
    └──────────────────────────────────────────────────────────────┘
"""

import torch
from typing import Optional

from vdllm.backends.base import AttentionBackend


class CPUAttentionBackend:
    """CPU implementation of attention for block diffusion.

    Uses standard PyTorch operations without any hardware acceleration.
    This is a fallback backend for testing and development.

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
        return "cpu"

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
            q: (total_tokens, num_heads * head_dim) Query vectors
            k: (total_tokens, num_kv_heads * head_dim) Key vectors
            v: (total_tokens, num_kv_heads * head_dim) Value vectors
            block_length: Size of each block for staircase masking
            staircase: Whether to apply staircase masking (True for prefill)

        Returns:
            (total_tokens, num_heads * head_dim) Attention output
        """
        # Reshape from flat to (total_tokens, heads, head_dim)
        q = q.view(-1, self.num_heads, self.head_dim)
        k = k.view(-1, self.num_kv_heads, self.head_dim)
        v = v.view(-1, self.num_kv_heads, self.head_dim)

        from vdllm.utils.context import get_context
        context = get_context()

        if staircase:
            # Build staircase mask for prefill
            mask = self._build_staircase_mask(
                q.shape[0],
                context.cu_seqlens_q,
                block_length,
            )

            o = torch.nn.functional.scaled_dot_product_attention(
                q, k, v,
                attn_mask=mask,
                scale=self.scale,
            )
        else:
            o = torch.nn.functional.scaled_dot_product_attention(
                q, k, v,
                scale=self.scale,
            )

        return o.view(-1, self.num_heads * self.head_dim)

    def _build_staircase_mask(
        self,
        total_tokens: int,
        cu_seqlens: torch.Tensor,
        block_length: int,
    ) -> torch.Tensor:
        """Build staircase attention mask for prefill.

        Args:
            total_tokens: Total number of tokens in the batch
            cu_seqlens: Cumulative sequence lengths
            block_length: Size of each block

        Returns:
            Boolean mask of shape (total_tokens, total_tokens)
        """
        mask = torch.ones(total_tokens, total_tokens, dtype=torch.bool)

        num_seqs = len(cu_seqlens) - 1
        for seq_idx in range(num_seqs):
            start = cu_seqlens[seq_idx].item()
            end = cu_seqlens[seq_idx + 1].item()
            seq_len = end - start

            num_blocks = (seq_len + block_length - 1) // block_length

            for block_i in range(num_blocks):
                block_start = start + block_i * block_length
                block_end = min(start + (block_i + 1) * block_length, end)

                for block_j in range(block_i + 1, num_blocks):
                    j_start = start + block_j * block_length
                    j_end = min(start + (block_j + 1) * block_length, end)

                    mask[block_start:block_end, j_start:j_end] = False

        return mask

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
            q: (batch_size * block_length, num_heads * head_dim)
            k_cache: (num_blocks, num_kv_heads, block_size, head_dim)
            v_cache: (num_blocks, num_kv_heads, block_size, head_dim)
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

        # Reshape for attention
        q = q.view(batch_size, block_length, self.num_heads, self.head_dim)
        k_new = k_new.view(batch_size, block_length, self.num_kv_heads, self.head_dim)
        v_new = v_new.view(batch_size, block_length, self.num_kv_heads, self.head_dim)

        # Gather cached KV pairs
        k_cached, v_cached = self._gather_kvcache(
            k_cache, v_cache, block_tables, seq_lens, batch_size, block_length
        )

        # Concatenate cached + new
        k_full = torch.cat([k_cached, k_new], dim=1)
        v_full = torch.cat([v_cached, v_new], dim=1)

        # SDPA - non-causal for block diffusion
        o = torch.nn.functional.scaled_dot_product_attention(
            q, k_full, v_full,
            scale=self.scale,
        )

        return o.view(-1, self.num_heads * self.head_dim)

    def _gather_kvcache(
        self,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        block_tables: torch.Tensor,
        seq_lens: torch.Tensor,
        batch_size: int,
        block_length: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Gather key/value tensors from paged KV cache.

        Args:
            k_cache: (num_blocks, num_kv_heads, block_size, head_dim)
            v_cache: (num_blocks, num_kv_heads, block_size, head_dim)
            block_tables: (batch_size, max_blocks_per_seq)
            seq_lens: (batch_size,) tokens per sequence
            batch_size: Number of sequences
            block_length: Block size

        Returns:
            Tuple of (k, v) tensors with shape (batch_size, max_cached, num_kv_heads, head_dim)
        """
        max_cached = seq_lens.max().item() if seq_lens.numel() > 0 else 0

        device = k_cache.device
        k_out = torch.zeros(
            batch_size, max_cached, self.num_kv_heads, self.head_dim,
            dtype=k_cache.dtype, device=device
        )
        v_out = torch.zeros(
            batch_size, max_cached, self.num_kv_heads, self.head_dim,
            dtype=v_cache.dtype, device=device
        )

        for seq_idx in range(batch_size):
            ctx_len = seq_lens[seq_idx].item()
            if ctx_len == 0:
                continue

            seq_block_table = block_tables[seq_idx]
            num_blocks = (ctx_len + block_length - 1) // block_length

            tokens_copied = 0
            for block_i in range(num_blocks):
                block_global_idx = seq_block_table[block_i].item()

                src_start = 0
                src_end = min(block_length, ctx_len - tokens_copied)

                if src_end <= 0:
                    break

                dst_start = tokens_copied
                dst_end = tokens_copied + src_end

                k_out[seq_idx, dst_start:dst_end] = k_cache[block_global_idx, :, src_start:src_end]
                v_out[seq_idx, dst_start:dst_end] = v_cache[block_global_idx, :, src_start:src_end]

                tokens_copied += src_end

        return k_out, v_out

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
            k: (total_tokens, num_kv_heads * head_dim)
            v: (total_tokens, num_kv_heads * head_dim)
            k_cache: (num_blocks * block_size, num_kv_heads * head_dim)
            v_cache: (num_blocks * block_size, num_kv_heads * head_dim)
            slot_mapping: (total_tokens,) int32 physical slot indices
        """
        # Reshape k, v to (total_tokens, num_kv_heads, head_dim)
        k = k.view(-1, self.num_kv_heads, self.head_dim)
        v = v.view(-1, self.num_kv_heads, self.head_dim)

        total_tokens = k.shape[0]

        # Scatter into cache using slot_mapping
        for i in range(total_tokens):
            slot = slot_mapping[i].item()
            k_cache[slot] = k[i]
            v_cache[slot] = v[i]


# Export the backend class
__all__ = ["CPUAttentionBackend"]
