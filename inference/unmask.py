"""Confidence-based unmasking strategies for Block Diffusion inference.

During denoising, we iteratively unmask tokens from most confident to least confident.
This gives a natural prioritization: high-confidence predictions are committed early,
while low-confidence positions benefit from more denoising steps.
"""

import torch


def unmask_top_k(
    masked: torch.Tensor,
    confidences: torch.Tensor,
    k: int,
) -> torch.Tensor:
    """Unmask the k positions with highest confidence.

    Positions that are already unmasked (False in masked) stay unmasked.
    Only masked (True) positions can become unmasked.

    Args:
        masked: (batch, seq_len) or (seq_len,) — True means position is currently masked
        confidences: (batch, seq_len) or (seq_len,) — confidence scores per position
        k: number of positions to unmask

    Returns:
        new_mask: same shape as masked — True means position is still masked
    """
    # Handle 1D tensors (single sequence)
    if masked.dim() == 1:
        masked = masked.unsqueeze(0)
        confidences = confidences.unsqueeze(0)
        squeeze_output = True
    else:
        squeeze_output = False

    batch_size, seq_len = masked.shape
    device = masked.device

    # Set unmasked positions to -inf so they won't be selected by topk
    masked_confidences = torch.where(
        masked,
        confidences,
        torch.tensor(-float('inf'), device=device),
    )

    # Determine actual k (can't unmask more than currently masked)
    n_masked = masked.sum().item()
    actual_k = min(k, n_masked)

    if actual_k == 0:
        result = masked.clone()
        return result.squeeze(0) if squeeze_output else result

    # Find top-k indices among masked positions
    _, top_indices = torch.topk(
        masked_confidences.view(batch_size, -1),
        k=actual_k,
        dim=-1,
    )

    # Create decode mask: positions to unmask
    decode_mask = torch.zeros_like(masked, dtype=torch.bool)
    for b in range(batch_size):
        for idx in top_indices[b]:
            decode_mask[b, idx.item()] = True

    # Unmask: set masked positions that were selected to False
    new_mask = masked & ~decode_mask

    return new_mask.squeeze(0) if squeeze_output else new_mask


def unmask_by_threshold(
    masked: torch.Tensor,
    confidences: torch.Tensor,
    threshold: float,
) -> torch.Tensor:
    """Unmask positions where confidence exceeds threshold.

    Args:
        masked: (batch, seq_len) — True means position is currently masked
        confidences: (batch, seq_len) — confidence scores per position
        threshold: minimum confidence to unmask

    Returns:
        new_mask: (batch, seq_len) — True means position is still masked
    """
    # Only consider masked positions
    eligible = masked & (confidences >= threshold)
    return masked & ~eligible


def uniform_schedule(
    n_masked: int,
    step: int,
    total_steps: int,
) -> int:
    """Uniform unmasking schedule: unmask equal number of tokens per step.

    Args:
        n_masked: total number of masked positions
        step: current denoising step (0-indexed)
        total_steps: total number of denoising steps

    Returns:
        number of positions to unmask at this step
    """
    if step >= total_steps - 1:
        return n_masked  # Last step: unmask everything remaining

    tokens_per_step = max(1, n_masked // total_steps)
    return tokens_per_step
