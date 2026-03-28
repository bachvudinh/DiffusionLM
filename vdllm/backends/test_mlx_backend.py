"""Standalone test for MLXAttentionBackend.

Run from DiffusionLM root:
    uv run python -m vdllm.backends.test_mlx_backend

This test validates the MLX backend without triggering the full vdllm
import chain (which requires flashinfer, etc.).
"""

import sys
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Optional, List
import torch


# Minimal mock of RunType
class RunType(Enum):
    PREFILL = auto()
    DENOISE = auto()


# Minimal mock of Context
@dataclass
class Context:
    run_type: Optional[RunType] = None
    cu_seqlens_q: Optional[torch.Tensor] = None
    cu_seqlens_k: Optional[torch.Tensor] = None
    max_seqlen_q: int = 0
    max_seqlen_k: int = 0
    slot_mapping: Optional[torch.Tensor] = None
    context_lens: Optional[torch.Tensor] = None
    block_tables: Optional[torch.Tensor] = None
    is_last_denoise_step: List[bool] = field(default_factory=lambda: [False])
    block_length: int = 4


_CONTEXT = Context()


def set_context(run_type, cu_seqlens_q=None, cu_seqlens_k=None,
                max_seqlen_q=0, max_seqlen_k=0, slot_mapping=None,
                context_lens=None, block_tables=None,
                is_last_denoise_step=[False], block_length=4):
    global _CONTEXT
    _CONTEXT = Context(run_type, cu_seqlens_q, cu_seqlens_k,
                       max_seqlen_q, max_seqlen_k, slot_mapping,
                       context_lens, block_tables,
                       is_last_denoise_step, block_length)


def get_context():
    return _CONTEXT


# Now import MLX backend
import mlx.core as mx
import mlx.nn as nn
import numpy as np


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
        num_blocks, _, num_kv_heads, block_size, head_dim = kv_cache.shape

        self.num_blocks = num_blocks
        self.block_size = block_size

        # Move to MLX - handle bfloat16 carefully
        torch_k = kv_cache[:, 0]  # (num_blocks, num_kv_heads, block_size, head_dim)
        torch_v = kv_cache[:, 1]

        # Convert to float32 first if bfloat16 (MLX misinterprets bfloat16 from numpy)
        if torch_k.dtype == torch.bfloat16:
            k_np = torch_k.float().numpy()
            v_np = torch_v.float().numpy()
            self.k_cache = mx.array(k_np, dtype=mx.bfloat16)
            self.v_cache = mx.array(v_np, dtype=mx.bfloat16)
        else:
            self.k_cache = mx.array(torch_k.numpy())
            self.v_cache = mx.array(torch_v.numpy())

    def store_kvcache(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        context: Context,
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

        # Convert to MLX
        k_mlx = mx.array(k.numpy())
        v_mlx = mx.array(v.numpy())

        slots = mx.array(context.slot_mapping.numpy())

        total_tokens = k.shape[0]
        block_size = self.block_size

        block_indices = slots // block_size
        offsets = slots % block_size
        linear_indices = block_indices * block_size + offsets

        # Loop scatter
        for i in range(total_tokens):
            slot = linear_indices[i].item()
            block_idx = slot // block_size
            offset = slot % block_size

            self.k_cache[block_idx, :, offset, :] = k_mlx[i]
            self.v_cache[block_idx, :, offset, :] = v_mlx[i]

        mx.eval(self.k_cache, self.v_cache)

    def _build_staircase_mask(
        self,
        total_tokens: int,
        cu_seqlens: mx.array,
        block_length: int,
    ) -> mx.array:
        """Build staircase attention mask for prefill.

        Token layout with block_length=4:
            Tok:  [0  1  2  3 | 4  5  6  7 | 8  9 10 11]
            Blk:  [  block 0  |  block 1   |  block 2  ]

        Mask pattern (1=attend, 0=masked):
                    Q
             0 1 2 3 4 5 6 7 8 9 A B
          0 [1 1 1 1 . . . . . . . .]
        K 4 [1 1 1 1 1 1 1 1 . . . .]
          8 [1 1 1 1 1 1 1 1 1 1 1 1]

        Args:
            total_tokens: Total number of tokens in the batch
            cu_seqlens: Cumulative sequence lengths (int32 array)
            block_length: Size of each block

        Returns:
            Boolean mask array of shape (total_tokens, total_tokens)
        """
        mask = mx.ones((total_tokens, total_tokens), dtype=mx.bool_)

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

    def _gather_kvcache(
        self,
        context_lens: mx.array,
        block_tables: mx.array,
        num_tokens: int,
    ) -> tuple[mx.array, mx.array]:
        """Gather key/value tensors from paged MLX KV cache.

        Args:
            context_lens: (batch_size,) - number of cached tokens per sequence
            block_tables: (batch_size, max_blocks) - physical block indices
            num_tokens: Total tokens (batch_size * block_length)

        Returns:
            Tuple of (k, v) tensors with shape (num_tokens, num_kv_heads, head_dim)
        """
        batch_size = len(context_lens)
        block_size = self.block_size

        k_out = mx.zeros((num_tokens, self.num_kv_heads, self.head_dim), dtype=self.k_cache.dtype)
        v_out = mx.zeros((num_tokens, self.num_kv_heads, self.head_dim), dtype=self.v_cache.dtype)

        for seq_idx in range(batch_size):
            ctx_len = context_lens[seq_idx].item()
            if ctx_len == 0:
                continue

            seq_block_table = block_tables[seq_idx]
            num_cached_blocks = (ctx_len + block_size - 1) // block_size

            tokens_copied = 0
            for block_i in range(num_cached_blocks):
                block_global_idx = seq_block_table[block_i].item()

                block_end_in_cache = min(block_size, ctx_len - tokens_copied)

                if block_end_in_cache <= 0:
                    break

                dst_start = tokens_copied
                dst_end = tokens_copied + block_end_in_cache

                src_start = 0
                src_end = block_end_in_cache

                k_block = self.k_cache[block_global_idx, :, src_start:src_end, :]
                v_block = self.v_cache[block_global_idx, :, src_start:src_end, :]

                k_block = k_block.transpose(1, 0, 2)
                v_block = v_block.transpose(1, 0, 2)

                k_out[dst_start:dst_end] = k_block[:dst_end - dst_start]
                v_out[dst_start:dst_end] = v_block[:dst_end - dst_start]

                tokens_copied += block_end_in_cache

        return k_out, v_out

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        context: Context,
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
        q_mlx = mx.array(q.numpy())
        k_mlx = mx.array(k.numpy())
        v_mlx = mx.array(v.numpy())

        run_type = context.run_type

        if run_type == RunType.PREFILL:
            output = self._forward_prefill(q_mlx, k_mlx, v_mlx, context)
        else:
            output = self._forward_denoise(q_mlx, k_mlx, v_mlx, context)

        return torch.from_numpy(np.array(output))

    def _forward_prefill(
        self,
        q: mx.array,
        k: mx.array,
        v: mx.array,
        context: Context,
    ) -> mx.array:
        """Prefill attention with staircase masking."""
        total_tokens = q.shape[0]

        cu_seqlens_q = mx.array(context.cu_seqlens_q.numpy())
        block_length = context.block_length

        mask = self._build_staircase_mask(total_tokens, cu_seqlens_q, block_length)

        # SDPA expects (B, N, T, D)
        q = q.reshape(1, self.num_heads, total_tokens, self.head_dim)
        k = k.reshape(1, self.num_kv_heads, total_tokens, self.head_dim)
        v = v.reshape(1, self.num_kv_heads, total_tokens, self.head_dim)

        mask = mask.reshape(1, 1, total_tokens, total_tokens)

        out = mx.fast.scaled_dot_product_attention(
            q, k, v,
            scale=self.scale,
            mask=mask,
        )

        out = out.reshape(total_tokens, self.num_heads, self.head_dim)

        mx.eval(out)

        return out

    def _forward_denoise(
        self,
        q: mx.array,
        k: mx.array,
        v: mx.array,
        context: Context,
    ) -> mx.array:
        """Denoise attention with paged KV cache.

        For variable-length cached contexts, we process each sequence
        individually to handle different context_lens per sequence.
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
            ctx_len = context_lens_np[seq_idx]
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
            block_global_idx = block_table[block_i]

            # Tokens from this cache block
            block_start = 0
            block_end = min(block_size, ctx_len - tokens_copied)

            if block_end <= 0:
                break

            # Extract from cache (convert to int for MLX indexing)
            k_block = self.k_cache[int(block_global_idx), :, block_start:block_end, :]
            v_block = self.v_cache[int(block_global_idx), :, block_start:block_end, :]

            # Transpose: (num_kv_heads, block_end, head_dim) -> (block_end, num_kv_heads, head_dim)
            k_block = k_block.transpose(1, 0, 2)
            v_block = v_block.transpose(1, 0, 2)

            k_chunks.append(k_block)
            v_chunks.append(v_block)

            tokens_copied += block_end

        k_out = mx.concatenate(k_chunks, axis=0) if k_chunks else mx.zeros((0, self.num_kv_heads, self.head_dim), dtype=self.k_cache.dtype)
        v_out = mx.concatenate(v_chunks, axis=0) if v_chunks else mx.zeros((0, self.num_kv_heads, self.head_dim), dtype=self.v_cache.dtype)

        return k_out, v_out


def test_mlx_backend():
    """Test MLXAttentionBackend functionality."""
    print("Testing MLXAttentionBackend...")
    print(f"MLX device: {mx.default_device()}")
    print(f"MLX array dtype: {mx.float32}\n")

    # Create backend
    backend = MLXAttentionBackend(num_heads=32, num_kv_heads=8, head_dim=128)
    print(f"Backend created: {backend.num_heads} heads, {backend.num_kv_heads} KV heads")

    # Initialize cache
    num_blocks = 16
    block_size = 4
    num_kv_heads = 8
    head_dim = 128

    kv_cache = torch.zeros((num_blocks, 2, num_kv_heads, block_size, head_dim), dtype=torch.bfloat16)
    backend.initialize_cache(kv_cache)
    print(f"Cache initialized: {backend.num_blocks} blocks x block_size={backend.block_size}")

    # Test PREFILL
    print("\n--- Testing PREFILL ---")
    seq_lens = [8, 4]  # 2 sequences
    total_tokens = sum(seq_lens)

    set_context(
        run_type=RunType.PREFILL,
        cu_seqlens_q=torch.tensor([0, 8, 12], dtype=torch.int32),
        cu_seqlens_k=torch.tensor([0, 8, 12], dtype=torch.int32),
        slot_mapping=torch.arange(total_tokens, dtype=torch.int32),
        block_length=4,
    )

    ctx = get_context()
    print(f"Context: run_type={ctx.run_type}, block_length={ctx.block_length}")

    # Mock data - use float32 for MLX compatibility
    q = torch.randn(total_tokens, 32, 128, dtype=torch.float32)
    k = torch.randn(total_tokens, 8, 128, dtype=torch.float32)
    v = torch.randn(total_tokens, 8, 128, dtype=torch.float32)

    print(f"Input shapes: q={q.shape}, k={k.shape}, v={v.shape}")

    output = backend.forward(q, k, v, ctx)
    print(f"PREFILL output shape: {output.shape}")
    assert output.shape == (total_tokens, 32, 128), f"Expected {(total_tokens, 32, 128)}, got {output.shape}"

    # Test DENOISE
    print("\n--- Testing DENOISE ---")
    batch_size = 2
    block_length = 4
    num_tokens = batch_size * block_length

    # Populate some cache entries first - use float32 for MLX compatibility
    cache_k = torch.randn(num_blocks, num_kv_heads, block_size, head_dim, dtype=torch.float32)
    cache_v = torch.randn(num_blocks, num_kv_heads, block_size, head_dim, dtype=torch.float32)
    kv_cache = torch.stack([cache_k, cache_v], dim=1)
    backend.initialize_cache(kv_cache)

    # Mock context lens and block tables
    context_lens = torch.tensor([8, 4], dtype=torch.int32)  # seq 0 has 8 cached, seq 1 has 4 cached
    block_tables = torch.tensor([[0, 1, -1, -1], [2, 3, -1, -1]], dtype=torch.int32)  # 4 max_blocks

    set_context(
        run_type=RunType.DENOISE,
        context_lens=context_lens,
        block_tables=block_tables,
        block_length=block_length,
    )

    ctx = get_context()
    print(f"Context: run_type={ctx.run_type}, block_length={ctx.block_length}")

    # New block data (the noisy block to be denoised) - use float32 for MLX
    q = torch.randn(num_tokens, 32, 128, dtype=torch.float32)
    k = torch.randn(num_tokens, 8, 128, dtype=torch.float32)
    v = torch.randn(num_tokens, 8, 128, dtype=torch.float32)

    print(f"Input shapes: q={q.shape}, k={k.shape}, v={v.shape}")

    output = backend.forward(q, k, v, ctx)
    print(f"DENOISE output shape: {output.shape}")
    assert output.shape == (num_tokens, 32, 128), f"Expected {(num_tokens, 32, 128)}, got {output.shape}"

    print("\n" + "="*50)
    print("All tests PASSED!")
    print("="*50)


if __name__ == "__main__":
    test_mlx_backend()
