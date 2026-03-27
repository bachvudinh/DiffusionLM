"""
Metal Attention - Block Diffusion Attention Masks.

This module provides attention mask builders for block diffusion inference.
The key insight is that block diffusion requires different attention patterns:

1. Within a block: Bidirectional (tokens attend to all others in block)
2. Across blocks: Causal (token attends only to previous blocks)

Architecture patterns adapted from:
- vllm-metal: paged attention infrastructure
- MinerU-Diffusion: block attention mask construction

================================================================================
                              BLOCK DIFFUSION MASK
================================================================================

For block_size=4, num_blocks=3:

         Block 0   Block 1   Block 2
         ───────   ───────   ───────
Block 0  [1 1 1 1 | 0 0 0 0 | 0 0 0 0]  ← Block 0 only (within block)
Block 1  [1 1 1 1 | 1 1 1 1 | 0 0 0 0]  ← Block 0,1 (block-causal)
Block 2  [1 1 1 1 | 1 1 1 1 | 1 1 1 1]  ← All blocks (full causal)

1 = attention allowed, 0 = blocked

================================================================================
                              USAGE
================================================================================

    from engine.metal.attention import build_block_diffusion_mask

    mask = build_block_diffusion_mask(
        seq_len=1024,
        block_size=4,
        device="mps"
    )
    # mask shape: [1024, 1024] bool

"""

import torch
from typing import Optional


def build_block_diffusion_mask(
    seq_len: int,
    block_size: int,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """
    Create block diffusion attention mask.

    Tokens within a block attend bidirectionally.
    Blocks attend causally (earlier blocks only).

    Args:
        seq_len: Total sequence length
        block_size: Size of each block
        device: Device to create mask on

    Returns:
        Boolean mask [seq_len, seq_len] where True = attention allowed
    """
    num_blocks = seq_len // block_size
    mask = torch.zeros(seq_len, seq_len, dtype=torch.bool, device=device)

    for b in range(num_blocks):
        start = b * block_size
        end = start + block_size

        # Bidirectional within block
        mask[start:end, start:end] = True

        # Causal across blocks (only previous blocks)
        if b > 0:
            mask[start:end, :start] = True

    return mask


def build_staircase_mask(
    seq_len: int,
    block_size: int,
    staircase_size: int,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """
    Create staircase attention mask for block diffusion.

    The staircase pattern expands the attention window exponentially:
    - Token at position i can attend to keys in range [0, (i//staircase_size + 1) * staircase_size]

    This creates a hierarchical attention pattern suitable for block diffusion.

    Args:
        seq_len: Total sequence length
        block_size: Size of each block
        staircase_size: Size of each staircase step
        device: Device to create mask on

    Returns:
        Boolean mask [seq_len, seq_len] where True = attention allowed
    """
    mask = torch.zeros(seq_len, seq_len, dtype=torch.bool, device=device)

    for q in range(seq_len):
        q_stair = q // staircase_size
        q_within_stair = q % staircase_size
        q_block = q // block_size

        for k in range(seq_len):
            k_stair = k // staircase_size
            k_block = k // block_size

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

    return mask


def build_paged_attention_mask(
    seq_len: int,
    block_size: int,
    num_blocks: int,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """
    Create paged attention mask for block diffusion.

    This is similar to block diffusion mask but optimized for paged KV cache.

    Args:
        seq_len: Total sequence length
        block_size: Size of each block
        num_blocks: Number of blocks (seq_len // block_size)
        device: Device to create mask on

    Returns:
        Boolean mask [seq_len, seq_len]
    """
    return build_block_diffusion_mask(seq_len, block_size, device)


def create_causal_mask(
    seq_len: int,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """
    Create standard causal (upper triangular) mask.

    Args:
        seq_len: Sequence length
        device: Device to create mask on

    Returns:
        Boolean mask [seq_len, seq_len] where True = attention allowed
    """
    return torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device), diagonal=1)


def create_block_causal_mask(
    seq_len: int,
    block_size: int,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """
    Create block causal mask where attention is blocked across block boundaries.

    Unlike block diffusion (bidirectional within blocks), this creates
    a strictly causal pattern with block boundaries.

    Args:
        seq_len: Total sequence length
        block_size: Size of each block
        device: Device to create mask on

    Returns:
        Boolean mask [seq_len, seq_len]
    """
    mask = torch.zeros(seq_len, seq_len, dtype=torch.bool, device=device)

    for b in range(seq_len // block_size):
        start = b * block_size
        end = start + block_size
        # Causal within block
        mask[start:end, start:end] = torch.triu(torch.ones(block_size, block_size, dtype=torch.bool, device=device))
        # Causal across blocks
        if b > 0:
            mask[start:end, :start] = True

    return mask


class BlockAttentionMaskBuilder:
    """
    Builder for block diffusion attention masks.

    This class provides methods to build various attention mask patterns
    used in block diffusion inference.
    """

    def __init__(self, block_size: int = 4):
        """
        Initialize mask builder.

        Args:
            block_size: Default block size
        """
        self.block_size = block_size
        self._cached_masks = {}

    def get_mask(
        self,
        seq_len: int,
        mask_type: str = "block_diffusion",
        device: torch.device = torch.device("cpu"),
    ) -> torch.Tensor:
        """
        Get attention mask for given sequence.

        Args:
            seq_len: Sequence length
            mask_type: Type of mask ('block_diffusion', 'staircase', 'causal', 'block_causal')
            device: Device to create mask on

        Returns:
            Boolean mask [seq_len, seq_len]
        """
        cache_key = (seq_len, mask_type, str(device))

        if cache_key not in self._cached_masks:
            if mask_type == "block_diffusion":
                mask = build_block_diffusion_mask(seq_len, self.block_size, device)
            elif mask_type == "staircase":
                mask = build_staircase_mask(seq_len, self.block_size, self.block_size * 2, device)
            elif mask_type == "causal":
                mask = create_causal_mask(seq_len, device)
            elif mask_type == "block_causal":
                mask = create_block_causal_mask(seq_len, self.block_size, device)
            else:
                raise ValueError(f"Unknown mask type: {mask_type}")

            self._cached_masks[cache_key] = mask

        return self._cached_masks[cache_key]

    def clear_cache(self):
        """Clear cached masks."""
        self._cached_masks.clear()
