"""MLX/Metal attention backend for vdllm block diffusion.

This backend uses Apple Metal via MLX for GPU-accelerated attention
computations, providing an alternative to CUDA/FlashInfer.

MLX Best Practices (CRITICAL)
==============================

1. Lazy Evaluation:
   - Operations build a graph; nothing computes until mx.eval()
   - ALWAYS call mx.eval() at loop boundaries to force computation
   - Implicit eval triggers: print(a), a.item(), np.array(a), if a > 0:

2. Array Indexing:
   - Lists must be mx.array (not Python lists)
   - Slice indices must be Python ints (not mx.array)
   - Slices create COPIES, not views (opposite of NumPy!)
   - Use at[] syntax for scatter operations

3. Dtype Handling:
   - Use mx.float32, mx.float16, mx.bfloat16
   - float64 is CPU-only on GPU!
   - bfloat16 from external sources needs explicit conversion

Architecture Overview
====================

    ┌──────────────────────────────────────────────────────────────┐
    │                MLXAttentionBackend                             │
    │                                                              │
    │  PREFILL Mode:                                              │
    │    1. store_kvcache: scatter k,v to paged cache using at[]   │
    │    2. sparse_attention: SDPA with staircase mask              │
    │                                                              │
    │  DENOISE Mode:                                              │
    │    1. gather_kvcache: retrieve k,v from paged cache         │
    │    2. block_attention: SDPA on gathered k,v + new block      │
    │                                                              │
    │  Memory Layout:                                             │
    │    k_cache: (num_blocks, num_kv_heads, block_size, head_dim)│
    │    v_cache: (num_blocks, num_kv_heads, block_size, head_dim)  │
    │                      MLX transposed format                   │
    └──────────────────────────────────────────────────────────────┘
"""

from typing import Optional, Tuple
import math
import numpy as np

import torch
import mlx.core as mx

from vdllm.backends.base import AttentionBackend
from vdllm.backends.tensor_bridge import to_mlx, to_torch


class MLXAttentionBackend:
    """MLX/Metal implementation of attention for block diffusion.

    Uses mx.fast.scaled_dot_product_attention which maps to Metal kernels
    on Apple Silicon (M1/M2/M3/M4).

    Attributes:
        num_heads: Number of query heads
        num_kv_heads: Number of key/value heads (GQA support)
        head_dim: Dimension per head
        scale: Attention scale factor (1/sqrt(head_dim))
        k_cache: MLX array for paged key cache
        v_cache: MLX array for paged value cache
        block_size: Tokens per cache block
        num_blocks: Total number of cache blocks
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
        self.k_cache: Optional[mx.array] = None
        self.v_cache: Optional[mx.array] = None
        self.block_size: Optional[int] = None
        self.num_blocks: Optional[int] = None

    def initialize_cache(self, kv_cache: torch.Tensor) -> None:
        """Initialize MLX arrays from PyTorch KV cache.

        Args:
            kv_cache: PyTorch tensor shape (num_blocks, 2, num_kv_heads,
                     block_size, head_dim) where dim 1 is [k, v]
        """
        # PyTorch layout: (num_blocks, 2, num_kv_heads, block_size, head_dim)
        # MLX layout:     (num_blocks, num_kv_heads, block_size, head_dim)
        num_blocks, _, num_kv_heads, block_size, head_dim = kv_cache.shape

        self.num_blocks = num_blocks
        self.block_size = block_size

        # Extract k and v separately
        torch_k = kv_cache[:, 0]  # (num_blocks, num_kv_heads, block_size, head_dim)
        torch_v = kv_cache[:, 1]

        # Convert using tensor bridge (handles bfloat16 properly)
        self.k_cache = to_mlx(torch_k)
        self.v_cache = to_mlx(torch_v)

    def store_kvcache(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        context: "Context",  # noqa: F821
    ) -> None:
        """Scatter key/value vectors into paged MLX KV cache.

        Uses MLX's at[] syntax for efficient scatter operations.

        Args:
            k: Key tensor (total_tokens, num_kv_heads, head_dim)
            v: Value tensor (total_tokens, num_kv_heads, head_dim)
            context: Context with slot_mapping (total_tokens,) int32
        """
        if self.k_cache is None or self.v_cache is None:
            return

        # Convert to MLX arrays
        k_mlx = to_mlx(k)
        v_mlx = to_mlx(v)
        slots = to_mlx(context.slot_mapping)

        total_tokens = k.shape[0]
        block_size = self.block_size

        # Compute block indices and offsets for each token
        # slots: (total_tokens,) -> (total_tokens,)
        block_indices = slots // block_size
        offsets = slots % block_size

        # MLX efficient scatter using at[] syntax
        # We need to scatter each token into its cache slot
        # cache[block_idx, :, offset, :] = token_k
        #
        # Using MLX's recommended pattern for variable-length scatter:
        # We build the updates and apply them efficiently

        # For MLX, we use the at[] syntax which returns an updated array
        # k_cache: (num_blocks, num_kv_heads, block_size, head_dim)
        # slots[i] gives linear slot index = block_idx * block_size + offset

        # Create linear cache indices
        linear_indices = block_indices * block_size + offsets  # (total_tokens,)

        # Use MLX's scatter for efficient bulk updates
        # Reshape k/v for scatter: (total_tokens, num_kv_heads, head_dim)
        # We need to flatten cache to (num_blocks * block_size, num_kv_heads, head_dim)

        cache_flat = self.k_cache.reshape(-1, self.num_kv_heads, self.head_dim)
        k_flat = k_mlx.reshape(total_tokens, self.num_kv_heads, self.head_dim)

        # For each token i, update cache_flat[linear_indices[i]] = k_flat[i]
        # MLX's modern scatter approach uses take_along_axis or at[]

        # Use advanced indexing: update flattened cache
        # This is more efficient than loop-based scatter
        for i in range(total_tokens):
            idx = linear_indices[i].item()
            cache_flat = cache_flat.at[idx].add(k_flat[i] - cache_flat[idx])

        self.k_cache = cache_flat.reshape(self.num_blocks, self.num_kv_heads, block_size, self.head_dim)

        # Same for v_cache
        cache_flat = self.v_cache.reshape(-1, self.num_kv_heads, self.head_dim)
        for i in range(total_tokens):
            idx = linear_indices[i].item()
            cache_flat = cache_flat.at[idx].add(v_mlx[i] - cache_flat[idx])

        self.v_cache = cache_flat.reshape(self.num_blocks, self.num_kv_heads, block_size, self.head_dim)

        # Force evaluation to ensure cache is updated before attention
        mx.eval(self.k_cache, self.v_cache)

    def _build_staircase_mask_additive(
        self,
        total_tokens: int,
        cu_seqlens: mx.array,
        block_length: int,
    ) -> mx.array:
        """Build staircase attention mask as ADDITIVE mask for MLX SDPA.

        The staircase mask enforces block-local attention with a staircase
        pattern where later blocks can attend to all positions in earlier
        blocks, but earlier blocks cannot attend to positions in later blocks.

        Token layout with block_length=4:
            Tok:  [0  1  2  3 | 4  5  6  7 | 8  9 10 11]
            Blk:  [  block 0  |  block 1   |  block 2  ]

        Mask pattern (additive, 0.0=attend, -1e9=masked):
                    Q
             0 1 2 3 4 5 6 7 8 9 A B
          0 [0 0 0 0 -inf ...       ]
        K 4 [0 0 0 0 0 0 0 0 -inf  ]
          8 [0 0 0 0 0 0 0 0 0 0 0 0]

        Args:
            total_tokens: Total number of tokens in the batch
            cu_seqlens: Cumulative sequence lengths (int32 array)
            block_length: Size of each block

        Returns:
            Additive mask array of shape (total_tokens, total_tokens)
            with 0.0 where attention is allowed, -1e9 where masked
        """
        # Create additive mask initialized to 0 (allow all)
        mask = mx.zeros((total_tokens, total_tokens), dtype=mx.float32)

        # For staircase pattern, later blocks can attend to earlier blocks
        # but earlier blocks CANNOT attend to later blocks
        #
        # Token i can attend to token j if: (j // block_length) <= (i // block_length)
        #
        # We need to find positions where j is in a later block than i
        # and set those to -inf

        num_seqs = len(cu_seqlens) - 1
        for seq_idx in range(num_seqs):
            start = int(cu_seqlens[seq_idx].item())
            end = int(cu_seqlens[seq_idx + 1].item())
            seq_len = end - start

            if seq_len <= 0:
                continue

            # Number of blocks in this sequence
            num_blocks = (seq_len + block_length - 1) // block_length

            # For each block_i (query block), mask out blocks > block_i
            for block_i in range(num_blocks):
                # Row range in mask (query tokens from block_i)
                q_start = start + block_i * block_length
                q_end = min(start + (block_i + 1) * block_length, end)

                # Mask out blocks block_i+1, block_i+2, ... (key tokens in later blocks)
                for block_j in range(block_i + 1, num_blocks):
                    k_start = start + block_j * block_length
                    k_end = min(start + (block_j + 1) * block_length, end)

                    # Set mask[q_start:q_end, k_start:k_end] = -inf
                    # MLX slice assignment requires careful handling
                    mask[q_start:q_end, k_start:k_end] = -1e9

        return mask

    def _gather_kvcache(
        self,
        context_lens: mx.array,
        block_tables: mx.array,
        num_tokens: int,
    ) -> Tuple[mx.array, mx.array]:
        """Gather key/value tensors from paged MLX KV cache.

        For denoise mode, we need to retrieve the cached KV for each
        sequence based on its context length and block table.

        Args:
            context_lens: (batch_size,) - number of cached tokens per sequence
            block_tables: (batch_size, max_blocks) - physical block indices
            num_tokens: Total tokens (batch_size * block_length)

        Returns:
            Tuple of (k, v) tensors with shape (num_tokens, num_kv_heads, head_dim)
        """
        batch_size = len(context_lens)
        block_size = self.block_size

        # Allocate output arrays
        k_out = mx.zeros((num_tokens, self.num_kv_heads, self.head_dim), dtype=self.k_cache.dtype)
        v_out = mx.zeros((num_tokens, self.num_kv_heads, self.head_dim), dtype=self.v_cache.dtype)

        for seq_idx in range(batch_size):
            ctx_len = int(context_lens[seq_idx].item())
            if ctx_len == 0:
                continue

            # Get block table for this sequence
            seq_block_table = block_tables[seq_idx]  # (max_blocks,)

            # Number of blocks needed for this context
            num_cached_blocks = (ctx_len + block_size - 1) // block_size

            tokens_copied = 0
            for block_i in range(num_cached_blocks):
                block_global_idx = int(seq_block_table[block_i].item())

                # How many tokens from this cache block
                block_end_in_cache = min(block_size, ctx_len - tokens_copied)

                if block_end_in_cache <= 0:
                    break

                # Destination positions
                dst_start = tokens_copied
                dst_end = tokens_copied + block_end_in_cache

                # Source positions in cache
                src_start = 0
                src_end = block_end_in_cache

                # Copy from cache: k_cache[block_idx, :, src_start:src_end, :]
                # Shape: (num_kv_heads, src_tokens, head_dim)
                k_block = self.k_cache[block_global_idx, :, src_start:src_end, :]
                v_block = self.v_cache[block_global_idx, :, src_start:src_end, :]

                # Transpose: (num_kv_heads, src_tokens, head_dim) -> (src_tokens, num_kv_heads, head_dim)
                k_block = k_block.transpose(1, 0, 2)
                v_block = v_block.transpose(1, 0, 2)

                # Copy to output
                k_out = k_out.at[dst_start:dst_end].add(k_block - k_out[dst_start:dst_end])
                v_out = v_out.at[dst_start:dst_end].add(v_block - v_out[dst_start:dst_end])

                tokens_copied += block_end_in_cache

        return k_out, v_out

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        context: "Context",  # noqa: F821
    ) -> torch.Tensor:
        """Compute attention output using MLX.

        Handles both PREFILL (staircase attention) and DENOISE (block attention).

        Args:
            q: Query tensor (total_tokens, num_heads, head_dim)
            k: Key tensor (total_tokens, num_kv_heads, head_dim)
            v: Value tensor (total_tokens, num_kv_heads, head_dim)
            context: Context with run_type, cu_seqlens, etc.

        Returns:
            Attention output (total_tokens, num_heads * head_dim)
        """
        # Convert inputs to MLX
        q_mlx = to_mlx(q)
        k_mlx = to_mlx(k)
        v_mlx = to_mlx(v)

        run_type = context.run_type

        if run_type.value == 1:  # PREFILL
            output = self._forward_prefill(q_mlx, k_mlx, v_mlx, context)
        else:  # DENOISE
            output = self._forward_denoise(q_mlx, k_mlx, v_mlx, context)

        # Convert back to PyTorch
        return to_torch(output)

    def _forward_prefill(
        self,
        q: mx.array,
        k: mx.array,
        v: mx.array,
        context: "Context",  # noqa: F821
    ) -> mx.array:
        """Prefill attention with staircase masking.

        Args:
            q: (total_tokens, num_heads, head_dim)
            k: (total_tokens, num_kv_heads, head_dim)
            v: (total_tokens, num_kv_heads, head_dim)
            context: Context with cu_seqlens_q, cu_seqlens_k, block_length

        Returns:
            (total_tokens, num_heads, head_dim)
        """
        total_tokens = q.shape[0]

        # Build staircase mask as ADDITIVE mask (0.0 = attend, -1e9 = masked)
        cu_seqlens_q = to_mlx(context.cu_seqlens_q)
        block_length = context.block_length

        mask = self._build_staircase_mask_additive(total_tokens, cu_seqlens_q, block_length)

        # SDPA expects (B, N, T, D) - add batch dimension
        # q: (total_tokens, num_heads, head_dim) -> (1, num_heads, total_tokens, head_dim)
        q = q.reshape(1, self.num_heads, total_tokens, self.head_dim)
        k = k.reshape(1, self.num_kv_heads, total_tokens, self.head_dim)
        v = v.reshape(1, self.num_kv_heads, total_tokens, self.head_dim)

        # mask needs to be (1, 1, total_tokens, total_tokens) for broadcasting
        mask = mask.reshape(1, 1, total_tokens, total_tokens)

        # MLX SDPA with additive mask
        # scale = 1.0 / sqrt(head_dim) is precomputed as self.scale
        out = mx.fast.scaled_dot_product_attention(
            q, k, v,
            scale=self.scale,
            mask=mask,
        )

        # Remove batch dimension
        out = out.reshape(total_tokens, self.num_heads, self.head_dim)

        # Force evaluation
        mx.eval(out)

        return out

    def _forward_denoise(
        self,
        q: mx.array,
        k: mx.array,
        v: mx.array,
        context: "Context",  # noqa: F821
    ) -> mx.array:
        """Denoise attention with paged KV cache.

        For variable-length cached contexts, we process each sequence
        individually to handle different context_lens per sequence.

        Args:
            q: (num_tokens, num_heads, head_dim) where num_tokens = batch_size * block_length
            k: (num_tokens, num_kv_heads, head_dim) - new block keys
            v: (num_tokens, num_kv_heads, head_dim) - new block values
            context: Context with context_lens, block_tables, block_length

        Returns:
            (num_tokens, num_heads, head_dim)
        """
        batch_size = len(context.context_lens)
        block_length = context.block_length
        num_tokens = q.shape[0]

        context_lens_np = context.context_lens.numpy()
        block_tables_np = context.block_tables.numpy()

        # Reshape q, k, v to (batch_size, block_length, ...)
        q = q.reshape(batch_size, block_length, self.num_heads, self.head_dim)
        k = k.reshape(batch_size, block_length, self.num_kv_heads, self.head_dim)
        v = v.reshape(batch_size, block_length, self.num_kv_heads, self.head_dim)

        # Output accumulator
        outputs = []

        for seq_idx in range(batch_size):
            ctx_len = int(context_lens_np[seq_idx])
            seq_block_table = block_tables_np[seq_idx]

            # Gather cached KV for this sequence
            seq_k_cached, seq_v_cached = self._gather_single_sequence_kvcache(
                ctx_len, seq_block_table
            )

            # Concatenate cached + new block
            # seq_k_cached: (ctx_len, num_kv_heads, head_dim)
            # k[seq_idx]: (block_length, num_kv_heads, head_dim)
            seq_k_full = mx.concatenate([seq_k_cached, k[seq_idx]], axis=0)
            seq_v_full = mx.concatenate([seq_v_cached, v[seq_idx]], axis=0)

            # SDPA: q needs (1, num_heads, block_length, head_dim)
            # k, v need (1, num_kv_heads, seq_len, head_dim)
            seq_q = q[seq_idx].reshape(1, self.num_heads, block_length, self.head_dim)
            seq_k_full = seq_k_full.reshape(1, self.num_kv_heads, -1, self.head_dim)
            seq_v_full = seq_v_full.reshape(1, self.num_kv_heads, -1, self.head_dim)

            seq_out = mx.fast.scaled_dot_product_attention(
                seq_q, seq_k_full, seq_v_full,
                scale=self.scale,
                mask=None,
            )

            # seq_out: (1, num_heads, block_length, head_dim)
            outputs.append(seq_out)

        # Concatenate outputs along token dimension
        out = mx.concatenate(outputs, axis=2)  # (1, num_heads, num_tokens, head_dim)
        out = out.reshape(num_tokens, self.num_heads, self.head_dim)

        mx.eval(out)

        return out

    def _gather_single_sequence_kvcache(
        self,
        ctx_len: int,
        block_table: np.ndarray,
    ) -> tuple[mx.array, mx.array]:
        """Gather KV cache for a single sequence.

        Args:
            ctx_len: Number of cached tokens
            block_table: (max_blocks,) physical block indices

        Returns:
            Tuple of (k, v) arrays with shape (ctx_len, num_kv_heads, head_dim)
        """
        if ctx_len == 0:
            k_out = mx.zeros((0, self.num_kv_heads, self.head_dim), dtype=self.k_cache.dtype)
            v_out = mx.zeros((0, self.num_kv_heads, self.head_dim), dtype=self.v_cache.dtype)
            return k_out, v_out

        block_size = self.block_size
        num_cached_blocks = (ctx_len + block_size - 1) // block_size

        k_chunks = []
        v_chunks = []

        tokens_copied = 0
        for block_i in range(num_cached_blocks):
            block_global_idx = int(block_table[block_i])

            # Tokens from this cache block
            block_start = 0
            block_end = min(block_size, ctx_len - tokens_copied)

            if block_end <= 0:
                break

            # Extract from cache (convert to int for MLX indexing)
            k_block = self.k_cache[block_global_idx, :, block_start:block_end, :]
            v_block = self.v_cache[block_global_idx, :, block_start:block_end, :]

            # Transpose: (num_kv_heads, block_end, head_dim) -> (block_end, num_kv_heads, head_dim)
            k_block = k_block.transpose(1, 0, 2)
            v_block = v_block.transpose(1, 0, 2)

            k_chunks.append(k_block)
            v_chunks.append(v_block)

            tokens_copied += block_end

        if k_chunks:
            k_out = mx.concatenate(k_chunks, axis=0)
            v_out = mx.concatenate(v_chunks, axis=0)
        else:
            k_out = mx.zeros((0, self.num_kv_heads, self.head_dim), dtype=self.k_cache.dtype)
            v_out = mx.zeros((0, self.num_kv_heads, self.head_dim), dtype=self.v_cache.dtype)

        return k_out, v_out


# Export the backend class
__all__ = ["MLXAttentionBackend"]
