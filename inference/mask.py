"""Staircase attention mask for Block Diffusion.

The staircase mask is the key innovation of Block Diffusion (BD3-LMs) that enables
KV-caching while preventing label leakage.

Input is concatenated: [x_t || x_0] — noisy half followed by clean half (2L total).

Mask components:
- M_BD (Block-Diagonal): Same half within same block attends bidirectionally
- M_OBC (Offset Block-Causal): Noisy queries attend to clean keys from STRICTLY earlier blocks
- M_BC (Block-Causal): Clean queries attend to clean keys from current/earlier blocks

Critical constraint: NO label leakage — noisy block i must NOT see clean tokens from block i.
This uses STRICT inequality (>) for block comparison, not >=.
"""

import torch


class StaircaseMask:
    """Build staircase attention masks for Block Diffusion.

    Args:
        block_size: Number of tokens per block (L' in BD3-LMs paper)
        seq_len: Length of the original sequence (L)
    """

    def __init__(self, block_size: int, seq_len: int):
        self.block_size = block_size
        self.seq_len = seq_len
        self.n_blocks = seq_len // block_size
        assert seq_len % block_size == 0, "seq_len must be divisible by block_size"

    def build(self) -> torch.Tensor:
        """Build the full staircase attention mask (2L x 2L).

        Returns:
            mask: (2*seq_len, 2*seq_len) — True means attention is ALLOWED
        """
        L = self.seq_len
        mask = torch.zeros(2 * L, 2 * L, dtype=torch.bool)

        for b in range(self.n_blocks):
            b_start = b * self.block_size
            b_end = b_start + self.block_size

            # === Noisy half (rows 0:L) ===

            # M_BD: Noisy queries attend bidirectionally within same block
            # Noisy block b → Noisy block b (within-block bidirectional)
            mask[b_start:b_end, b_start:b_end] = True

            # M_OBC: Noisy block b attends to CLEAN keys from STRICTLY earlier blocks
            # Noisy block b → Clean block < b
            for prev_b in range(b):
                noisy_start = b_start
                noisy_end = b_end
                clean_start = prev_b * self.block_size + L
                clean_end = clean_start + self.block_size
                mask[noisy_start:noisy_end, clean_start:clean_end] = True

            # M_OBC: Noisy block b attends to CLEAN keys from its OWN block
            # BUT only to positions AFTER current position (within-block AR)
            # This is handled by the per-position causal mask within the block

            # === Clean half (rows L:2L) ===

            # M_BC: Clean block b attends to CLEAN keys from current and earlier blocks
            # Clean block b → Clean blocks 0..b (inclusive)
            for att_b in range(b + 1):
                clean_query_start = b_start + L
                clean_query_end = b_end + L
                clean_key_start = att_b * self.block_size + L
                clean_key_end = clean_key_start + self.block_size
                mask[clean_query_start:clean_query_end, clean_key_start:clean_key_end] = True

        return mask

    def build_causal_within_block(self) -> torch.Tensor:
        """Build a causal (lower-triangular) mask for within-block AR generation.

        Returns:
            causal_mask: (seq_len, seq_len) — True means attention is ALLOWED
        """
        L = self.seq_len
        causal = torch.tril(torch.ones(L, L, dtype=torch.bool))
        return causal

    def to_block_mask(self):
        """Convert to a FlexAttention-compatible BlockMask or dense tensor.

        For now returns a dense float mask (1.0 = attend, 0.0 = don't attend).
        Can be upgraded to FlexAttention BlockMask for efficiency.
        """
        bool_mask = self.build()
        return bool_mask.float()  # Dense mask for now
