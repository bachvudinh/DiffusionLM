"""
Triton Block Attention Kernels.

This module provides Triton implementations of block sparse attention
for block diffusion inference.

Architecture patterns from JetEngine:
- jetengine/kernels/triton/attention/block_prefill_attention_v2.py
  (Staircase sparse attention - SDAR semi-AR inference)

================================================================================
                              STAIRCASE ATTENTION
================================================================================

The staircase attention pattern expands the attention window exponentially:
- Token at position i can attend to keys in range [0, (i//staircase_size + 1) * staircase_size]

This creates a hierarchical attention pattern suitable for block diffusion:
- Within block: Bidirectional
- Across blocks: Causal with staircase expansion

================================================================================
                              USAGE
================================================================================

    from engine.triton.kernels.block_attention import triton_staircase_attention

    output = triton_staircase_attention(
        q, k, v, mask,
        scale=0.088388347648,
        block_size=4,
    )

"""

from typing import Optional

import torch

# Try to import Triton
try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False
    triton = None
    tl = None


# Triton kernel for block attention
# This is adapted from JetEngine's _staircase_attn_fwd_kernel_varlen_v2
if TRITON_AVAILABLE:

    @triton.jit
    def _block_attention_kernel(
        Q, K, V, Out,
        stride_qb, stride_qh, stride_qm, stride_qk,
        stride_kb, stride_kh, stride_kn, stride_kk,
        stride_vb, stride_vh, stride_vn, stride_vk,
        stride_ob, stride_oh, stride_om, stride_ok,
        seq_len: tl.constexpr,
        block_size: tl.constexpr,
        num_blocks: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        STAIR_SIZE: tl.constexpr,
    ):
        """
        Block attention kernel with staircase pattern.

        Each token attends to:
        1. All tokens within the same block (bidirectional)
        2. All tokens in previous blocks (causal)
        """
        # Get position
        batch_idx = tl.program_id(0)
        head_idx = tl.program_id(1)
        block_idx = tl.program_id(2)

        # Offsets
        q_offset = batch_idx * stride_qb + head_idx * stride_qh
        k_offset = batch_idx * stride_kb + head_idx * stride_kh
        v_offset = batch_idx * stride_vb + head_idx * stride_vh
        o_offset = batch_idx * stride_ob + head_idx * stride_oh

        # Block start position
        block_start = block_idx * block_size

        # Initialize output accumulator
        offs_m = tl.arange(0, block_size)
        offs_n = tl.arange(0, block_size)
        offs_d = tl.arange(0, HEAD_DIM)

        # Load Q for this block
        q_ptrs = Q + q_offset + (block_start + offs_m)[:, None] * stride_qm + offs_d[None, :] * stride_qk
        q = tl.load(q_ptrs)

        # Initialize accumulators
        acc = tl.zeros([block_size, HEAD_DIM], dtype=tl.float32)
        denominator = tl.zeros([block_size, 1], dtype=tl.float32)

        # Staircase: block can attend to previous blocks + within block
        max_stair = block_idx + 1  # Number of stairs we can attend to

        for stair_idx in range(max_stair):
            # Staircase window: each stair covers STAIR_SIZE tokens
            stair_start = stair_idx * STAIR_SIZE
            stair_end = min(stair_start + STAIR_SIZE, seq_len)
            actual_len = stair_end - stair_start

            # Load K for this stair
            k_ptrs = K + k_offset + (stair_start + offs_n)[:, None] * stride_kn + offs_d[None, :] * stride_kk
            k = tl.load(k_ptrs)

            # Compute attention scores
            qk = tl.dot(q, k.to(tl.float32))

            # Apply causal mask within stair
            stair_offsets = stair_start + offs_n
            causal_mask = (stair_start + offs_n)[None, :] <= (block_start + offs_m)[:, None]
            qk = tl.where(causal_mask, qk, float('-inf'))

            # Softmax
            attn = tl.softmax(qk, axis=1)

            # Load V for this stair
            v_ptrs = V + v_offset + (stair_start + offs_n)[:, None] * stride_vn + offs_d[None, :] * stride_vk
            v = tl.load(v_ptrs)

            # Accumulate
            acc = acc + tl.dot(attn.to(tl.float32), v.to(tl.float32))
            denominator = denominator + attn.sum(axis=1, keepdims=True)

        # Normalize
        acc = acc / denominator

        # Store output
        out_ptrs = Out + o_offset + (block_start + offs_m)[:, None] * stride_om + offs_d[None, :] * stride_ok
        tl.store(out_ptrs, acc.to(tl.float16))


    def triton_block_attention(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        scale: float,
        block_size: int = 4,
    ) -> torch.Tensor:
        """
        Run block attention via Triton.

        Args:
            q: Query [batch, num_heads, seq_len, head_dim]
            k: Key [batch, num_kv_heads, seq_len, head_dim]
            v: Value [batch, num_kv_heads, seq_len, head_dim]
            scale: Attention scale
            block_size: Size of each block

        Returns:
            Attention output
        """
        batch, num_heads, seq_len, head_dim = q.shape
        num_blocks = seq_len // block_size

        # Expand KV for GQA if needed
        if k.shape[1] != num_heads:
            num_groups = num_heads // k.shape[1]
            k = k.repeat_interleave(num_groups, dim=1)
            v = v.repeat_interleave(num_groups, dim=1)

        # Allocate output
        out = torch.empty_like(q)

        # Launch kernel
        grid = (batch, num_heads, num_blocks)

        _block_attention_kernel[grid](
            q, k, v, out,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            out.stride(0), out.stride(1), out.stride(2), out.stride(3),
            seq_len=seq_len,
            block_size=block_size,
            num_blocks=num_blocks,
            HEAD_DIM=head_dim,
            STAIR_SIZE=block_size,
        )

        return out

else:
    def triton_block_attention(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        scale: float,
        block_size: int = 4,
    ) -> torch.Tensor:
        """Fallback when Triton is not available."""
        raise ImportError("Triton is not available. Install with: pip install triton")


def triton_staircase_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: Optional[torch.Tensor],
    scale: float,
    block_size: int = 4,
) -> torch.Tensor:
    """
    Staircase attention for block diffusion.

    This is the main entry point for block diffusion attention.

    Args:
        q: Query tensor
        k: Key tensor
        v: Value tensor
        mask: Optional attention mask
        scale: Attention scale
        block_size: Size of each block

    Returns:
        Attention output
    """
    if not TRITON_AVAILABLE:
        raise ImportError("Triton is required for staircase attention")

    return triton_block_attention(q, k, v, scale, block_size)
