"""Staircase attention mask for Block Diffusion.

Architecture Overview
====================

    seq_len = 64, block_size = 8  →  n_blocks = 8
           |
           v
    Input is concatenated: [x_t || x_0]     (2L = 128 tokens)
           |                              x_t = noisy half (left)
           |                              x_0 = clean half (right)
           v
    +----------------------------------------------------+
    |                 Staircase Mask                       |
    |                 Shape: (2L, 2L) = (128, 128)         |
    +----------------------------------------------------+
           |
           v
    +----------------------------------------------------+
    |  Noisy Half (rows 0:L)                              |
    |  ─────────────────────────────────────────────────  |
    |  Block i can attend to:                              |
    |    • Noisy tokens in block i (bidirectional)         |
    |    • Clean tokens in blocks < i (block-causal)      |
    +----------------------------------------------------+
           |
           v
    +----------------------------------------------------+
    |  Clean Half (rows L:2L)                             |
    |  ─────────────────────────────────────────────────  |
    |  Block i can attend to:                              |
    |    • Clean tokens in blocks ≤ i (block-causal)      |
    |    • NO clean tokens in block i (no label leakage!) |
    +----------------------------------------------------+


Mask Matrix Visualization (L=16, block_size=4, 4 blocks)
============================================================

    0   4   8   12  16  20  24  28  32  36  40  44  48  52  56  60  64  68  72  76  80  84  88  92  96  100 104 108 112 116 120 124 128
    |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |
    ▼   ▼   ▼   ▼   ▼   ▼   ▼   ▼   ▼   ▼   ▼   ▼   ▼   ▼   ▼   ▼   ▼   ▼   ▼   ▼   ▼   ▼   ▼   ▼   ▼   ▼   ▼   ▼   ▼   ▼
 0 ┌───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬─────────┬───────┬───────┬───────┬───────┐
   │ B │ B │ B │ B │   │   │   │   │   │   │   │   │   │   │   │   │   OBC   │ OBC   │ OBC   │ OBC   │ OBC   │
 4 ├───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼─────────┼───────┼───────┼───────┼───────┤
   │   │ B │ B │ B │ B │   │   │   │   │   │   │   │   │   │   │   │   OBC   │ OBC   │ OBC   │ OBC   │ OBC   │
 8 ├───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼─────────┼───────┼───────┼───────┼───────┤
   │   │   │ B │ B │ B │ B │   │   │   │   │   │   │   │   │   │   │   OBC   │ OBC   │ OBC   │ OBC   │ OBC   │
12 ├───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼─────────┼───────┼───────┼───────┼───────┤
   │   │   │   │ B │ B │ B │ B │   │   │   │   │   │   │   │   │   OBC   │ OBC   │ OBC   │ OBC   │ OBC   │
16 ├───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼─────────┼───────┼───────┼───────┼───────┤
   │   │   │   │   │ BC│ BC│ BC│ BC│   │   │   │   │   │   │   │         │       │       │       │       │
   ... (clean half follows same pattern, but NO self-attention to own block's clean half)


Legend:
    B  = Block-diagonal (bidirectional within block)
    OBC = Offset block-causal (noisy → clean from earlier blocks)
    BC  = Block-causal (clean → clean from current/earlier blocks)
"""

import torch


class StaircaseMask:
    """Build staircase attention masks for Block Diffusion.

    The staircase mask enforces three key constraints:
    1. Within-block bidirectional: tokens in same block can attend to each other
    2. Block-causal: blocks can only attend to earlier blocks (AR ordering)
    3. No label leakage: noisy block i cannot see clean tokens from block i

    Args:
        block_size: Number of tokens per block (L' in BD3-LMs paper)
        seq_len: Length of the original sequence (L). Must be divisible by block_size.
    """

    def __init__(self, block_size: int, seq_len: int):
        self.block_size = block_size
        self.seq_len = seq_len
        self.n_blocks = seq_len // block_size
        assert seq_len % block_size == 0, \
            f"seq_len={seq_len} must be divisible by block_size={block_size}"

    def build(self) -> torch.Tensor:
        """Build the full staircase attention mask (2L x 2L).

        Input:  seq_len=L, block_size=B  →  n_blocks=L//B
        Output: mask: (2L, 2L)  —  True = attention allowed, False = blocked

        The mask is organized as:
            [0:L,   0:L]   → Noisy queries → Noisy keys (block-diagonal)
            [0:L,   L:2L]  → Noisy queries → Clean keys (offset block-causal)
            [L:2L,  0:L]   → Clean queries → Noisy keys (unused, not needed for generation)
            [L:2L,  L:2L]  → Clean queries → Clean keys (block-causal, no self)
        """
        L = self.seq_len
        mask = torch.zeros(2 * L, 2 * L, dtype=torch.bool)

        for b in range(self.n_blocks):
            b_start = b * self.block_size
            b_end = b_start + self.block_size

            # === Noisy half (rows 0:L) ===

            # M_BD: Noisy queries attend bidirectionally within same block
            # Block b noisy → Block b noisy (within-block bidirectional)
            mask[b_start:b_end, b_start:b_end] = True

            # M_OBC: Noisy block b attends to CLEAN keys from STRICTLY earlier blocks
            # Noisy block b → Clean blocks 0..b-1
            for prev_b in range(b):
                noisy_start = b_start
                noisy_end = b_end
                clean_start = prev_b * self.block_size + L
                clean_end = clean_start + self.block_size
                mask[noisy_start:noisy_end, clean_start:clean_end] = True

            # === Clean half (rows L:2L) ===

            # M_BC: Clean block b attends to CLEAN keys from current and earlier blocks
            # Clean block b → Clean blocks 0..b (inclusive)
            for att_b in range(b + 1):
                clean_query_start = b_start + L
                clean_query_end = b_end + L
                clean_key_start = att_b * self.block_size + L
                clean_key_end = clean_key_start + self.block_size
                mask[clean_query_start:clean_query_end, clean_key_start:clean_key_end] = True

            # NOTE: We do NOT add self-attention from clean block b to clean block b
            # This is the "no label leakage" constraint — critical for training!

        return mask

    def to_block_mask(self) -> torch.Tensor:
        """Convert to a FlexAttention-compatible BlockMask or dense float tensor.

        Input:  self.build() → bool mask (2L, 2L)
        Output: float mask (2L, 2L) — 1.0 = attend, 0.0 = don't attend
        """
        bool_mask = self.build()
        return bool_mask.float()
