"""
Attention Mask Builders for Block Diffusion.

This module provides attention masks for block diffusion inference.

================================================================================
                              CONCEPTS
================================================================================

1. Block Diffusion Attention:
   - Within block: Bidirectional (tokens see all in block)
   - Across blocks: Causal (token only sees previous blocks)

2. Staircase Attention:
   - Coarser granularity than block
   - Hierarchical attention pattern

3. Causal Mask:
   - Standard autoregressive attention (triangular)

================================================================================
                              USAGE
================================================================================

    from engine.attention import build_block_diffusion_mask, attention_forward

    mask = build_block_diffusion_mask(
        seq_len=1024,
        block_size=8,
        device='mps'
    )
    # mask shape: [1, 1024, 1024]

    # Or use unified attention
    output = attention_forward(q, k, v, mask=mask, device='mps')

================================================================================
"""

import math
from typing import Optional

import torch
import torch.nn.functional as F


def attention_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    scale: Optional[float] = None,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Unified attention forward pass.

    This function provides a consistent attention interface across
    different devices (MPS, CUDA, CPU).

    Args:
        q: Query tensor [..., heads, seq, head_dim]
        k: Key tensor [..., kv_heads, seq, head_dim]
        v: Value tensor [..., kv_heads, seq, head_dim]
        mask: Optional attention mask
        scale: Attention scale (defaults to 1/sqrt(head_dim))
        device: Device for computation

    Returns:
        Attention output with same shape as q
    """
    # Determine scale
    if scale is None:
        head_dim = q.shape[-1]
        scale = 1.0 / math.sqrt(head_dim)

    # Move to device if specified
    if device is not None:
        q = q.to(device)
        k = k.to(device)
        v = v.to(device)
        if mask is not None:
            mask = mask.to(device)

    # Use SDPA for attention
    if mask is not None:
        # Handle mask dimensions for SDPA
        if mask.dim() == 2:
            mask = mask.unsqueeze(0).unsqueeze(0)
        elif mask.dim() == 3:
            mask = mask.unsqueeze(1)
        output = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=mask,
            is_causal=False,
            scale=scale,
        )
    else:
        output = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            is_causal=False,
            scale=scale,
        )

    return output


def build_block_diffusion_mask(
    seq_len: int,
    block_size: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Create block causal attention mask for inference.

    Creates a mask where:
    - Within a block: tokens can attend to each other (bidirectional)
    - Between blocks: later blocks attend to earlier blocks (causal)

    Args:
        seq_len: Total sequence length
        block_size: Size of each block
        device: Device to create tensor on

    Returns:
        Attention mask of shape [1, seq_len, seq_len]
        True = attention allowed, False = blocked

    Example:
        seq_len=12, block_size=4

        Block 0 (pos 0-3):   Can attend to block 0 only
        Block 1 (pos 4-7):   Can attend to blocks 0,1
        Block 2 (pos 8-11):   Can attend to blocks 0,1,2

        Attention pattern:
        [1 1 1 1 | 0 0 0 0 | 0 0 0 0 ]  ← Block 0
        [1 1 1 1 | 1 1 1 1 | 0 0 0 0 ]  ← Block 1
        [1 1 1 1 | 1 1 1 1 | 1 1 1 1 ]  ← Block 2
    """
    num_blocks = seq_len // block_size

    # Create lower triangular block matrix
    block_mask = torch.tril(
        torch.ones(num_blocks, num_blocks, device=device, dtype=torch.bool)
    )

    # Expand: repeat block_mask along both dimensions
    mask = block_mask.repeat_interleave(block_size, dim=0).repeat_interleave(
        block_size, dim=1
    )

    return mask.unsqueeze(0)  # Add batch dimension


def build_staircase_block_mask(
    seq_len: int,
    block_size: int,
    stairs_size: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Create staircase block attention mask.

    Combines block diffusion mask with staircase attention:
    - Token attends to all in same block (bidirectional)
    - Token attends to all in previous stairs (coarse causal)
    - Token attends to all in current stair up to current position (fine causal)

    Args:
        seq_len: Total sequence length
        block_size: Size of each block
        stairs_size: Size of each stair (stair > block)
        device: Device to create tensor on

    Returns:
        Attention mask of shape [1, seq_len, seq_len]
    """
    num_blocks = seq_len // block_size
    num_stairs = seq_len // stairs_size

    mask = torch.zeros(seq_len, seq_len, dtype=torch.bool, device=device)

    for q in range(seq_len):
        q_block = q // block_size
        q_stair = q // stairs_size
        q_within_block = q % block_size

        for k in range(seq_len):
            k_block = k // block_size
            k_stair = k // stairs_size

            # Within same block = bidirectional
            if k_block == q_block:
                mask[q, k] = True
            # In previous stairs = full causal
            elif k_stair < q_stair:
                mask[q, k] = True
            # In same stair, previous blocks = full causal
            elif k_stair == q_stair and k_block < q_block:
                mask[q, k] = True
            # In same stair, same block, previous tokens = fine causal
            elif k_stair == q_stair and k_block == q_block and k <= q:
                mask[q, k] = True

    return mask.unsqueeze(0)


def build_causal_mask(
    seq_len: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Create standard causal attention mask (lower triangular).

    Args:
        seq_len: Sequence length
        device: Device to create tensor on

    Returns:
        Attention mask of shape [1, seq_len, seq_len]
        True = allowed, False = blocked
    """
    # Create lower triangular mask
    mask = torch.tril(
        torch.ones(seq_len, seq_len, dtype=torch.bool, device=device)
    )
    return mask.unsqueeze(0)
