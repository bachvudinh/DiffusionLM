"""
Semi-AR Sampler for Block Diffusion.

This module provides the semi-AR sampling algorithm for block diffusion.

================================================================================
                              CONCEPT
================================================================================

Semi-AR (SDAR) Sampling:
1. Initialize block with prompt tokens + mask tokens
2. For each denoising step:
   - Compute model predictions
   - Unmask highest confidence positions
   - Resample masked positions
3. Cache block KV after completion
4. Move to next block

================================================================================
                              USAGE
================================================================================

    from engine.diffusion import SemiARUpdater

    updater = SemiARUpdater(
        block_size=4,
        num_steps=4,
        strategy='low_confidence_dynamic',
    )

    # Update block state
    new_block = updater.update(
        block=current_block,
        logits=model_output,
        mask=current_mask,
        step=0,
    )

================================================================================
"""

import torch
import torch.nn.functional as F


class SemiARUpdater:
    """
    Semi-AR sampler for block diffusion.

    Handles the iterative denoising process within each block.
    """

    def __init__(
        self,
        block_size: int,
        num_steps: int,
        strategy: str = "low_confidence_dynamic",
        confidence_threshold: float = 0.85,
    ):
        """
        Initialize semi-AR updater.

        Args:
            block_size: Size of each block
            num_steps: Number of denoising steps per block
            strategy: Unmasking strategy
            confidence_threshold: For dynamic strategy
        """
        self.block_size = block_size
        self.num_steps = num_steps
        self.strategy = strategy
        self.confidence_threshold = confidence_threshold

    def update(
        self,
        block: torch.Tensor,
        logits: torch.Tensor,
        mask: torch.Tensor,
        step: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Update block state at a given denoising step.

        Args:
            block: Current block state [batch, block_size]
            logits: Model logits [batch, block_size, vocab_size]
            mask: Current mask state [batch, block_size] (True = masked)
            step: Current denoising step

        Returns:
            Tuple of (updated_block, updated_mask)
        """
        # Get probabilities
        probs = F.softmax(logits, dim=-1)
        top_probs, top_tokens = probs.max(dim=-1)

        # Compute unmask count for this step
        num_transfer = self._get_num_transfer(step)

        # Find positions to unmask based on strategy
        if self.strategy == "low_confidence_dynamic":
            unmask_idx = self._dynamic_unmask(top_probs, mask, num_transfer)
        elif self.strategy == "low_confidence_static":
            unmask_idx = self._static_unmask(top_probs, mask, num_transfer)
        elif self.strategy == "sequential":
            unmask_idx = self._sequential_unmask(mask, num_transfer)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

        # Apply updates
        new_block = block.clone()
        new_mask = mask.clone()

        for idx in unmask_idx:
            new_block[0, idx] = top_tokens[0, idx]
            new_mask[0, idx] = False

        return new_block, new_mask

    def _get_num_transfer(self, step: int) -> int:
        """Get number of tokens to unmask at this step."""
        base = self.block_size // self.num_steps
        remainder = self.block_size % self.num_steps
        if step < remainder:
            return base + 1
        return base

    def _dynamic_unmask(
        self,
        probs: torch.Tensor,
        mask: torch.Tensor,
        num_transfer: int,
    ) -> list:
        """Unmask highest confidence positions first."""
        # Masked positions with their confidence
        masked_probs = torch.where(mask, probs, torch.zeros_like(probs))

        # If high confidence exceeds threshold, use those
        high_conf_mask = masked_probs > self.confidence_threshold
        n_high_conf = high_conf_mask.sum().item()

        if n_high_conf >= num_transfer:
            # Use top-k of high confidence positions
            _, top_idx = torch.topk(masked_probs, num_transfer)
            return top_idx.tolist()
        else:
            # Use all high confidence + fill with top-k of remaining
            indices = []
            if n_high_conf > 0:
                indices.extend(high_conf_mask.nonzero(as_tuple=True)[0].tolist())
            # Fill remaining
            remaining = num_transfer - len(indices)
            if remaining > 0:
                _, top_idx = torch.topk(masked_probs, num_transfer)
                for idx in top_idx.tolist():
                    if idx not in indices:
                        indices.append(idx)
            return indices[:num_transfer]

    def _static_unmask(
        self,
        probs: torch.Tensor,
        mask: torch.Tensor,
        num_transfer: int,
    ) -> list:
        """Unmask lowest confidence positions first."""
        masked_probs = torch.where(mask, probs, torch.zeros_like(probs))
        _, top_idx = torch.topk(masked_probs, num_transfer)
        return top_idx.tolist()

    def _sequential_unmask(
        self,
        mask: torch.Tensor,
        num_transfer: int,
    ) -> list:
        """Unmask left-to-right."""
        indices = []
        for i in range(mask.shape[1]):
            if mask[0, i] and len(indices) < num_transfer:
                indices.append(i)
        return indices
