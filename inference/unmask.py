"""Confidence-based unmasking strategies for Block Diffusion inference.

Architecture Overview
====================

    During denoising, we iteratively unmask tokens from most confident to least.
    This gives a natural prioritization: high-confidence predictions committed early,
    while low-confidence positions benefit from more denoising steps.

    Step 0:
        block = [M, M, M, M, M, M, M, M]     M = masked
        conf  = [0.1, 0.9, 0.3, 0.7, 0.5, 0.2, 0.4, 0.6]
               │
               ▼
        unmask_top_k(masked, conf, k=3)
               │
               ▼
        Step 1:
        block = [M, T, M, T, T, M, M, M]     T = token revealed
        conf  = [0.1,  ---, 0.3, ---, ---, 0.2, 0.4, 0.6]
               │
               ▼
        unmask_top_k(masked, conf, k=3)
               │
               ▼
        Step 2:
        block = [M, T, M, T, T, M, T, T]
               │
               ▼
        unmask_top_k(masked, conf, k=2)
               │
               ▼
        Final:
        block = [T, T, T, T, T, T, T, T]


Key Functions:
- unmask_top_k: unmask positions with highest confidence
- unmask_by_threshold: unmask positions exceeding confidence threshold
- uniform_schedule: evenly distribute unmasking across steps
"""

import torch


def unmask_top_k(
    masked: torch.Tensor,
    confidences: torch.Tensor,
    k: int,
) -> torch.Tensor:
    """Unmask the k positions with highest confidence.

    Positions that are already unmasked (masked=False) stay unmasked.
    Only masked (masked=True) positions can become unmasked.

    Data Flow:
    ──────────
    Input:
        masked:      torch.Tensor — shape (B, L) or (L,) — True = masked
        confidences: torch.Tensor — shape (B, L) or (L,) — per-position confidence
        k:           int — number of positions to unmask

    Output:
        torch.Tensor — same shape as masked — True = still masked, False = unmasked

    Example:
        masked =      [True,  True,  True,  True,  True, False, False, False, False, False]
        confidences = [0.1,   0.9,   0.3,   0.7,   0.5,  -1.0,  -1.0,  -1.0,  -1.0,  -1.0]
        k = 3
        ─────────────────────────────────────────────
        Result:       [True,  False, True,  False, False, False, False, False, False, False]
        (positions 1, 3, 4 have highest confidences → unmasked)
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

    Data Flow:
    ──────────
    Input:
        masked:      torch.Tensor — shape (B, L) or (L,) — True = masked
        confidences: torch.Tensor — shape (B, L) or (L,) — per-position confidence
        threshold:   float — minimum confidence to unmask

    Output:
        torch.Tensor — same shape — True = still masked, False = unmasked

    Example:
        masked =      [True,  True,  True,  True,  True]
        confidences = [0.1,   0.9,   0.3,   0.7,   0.5]
        threshold =  0.6
        ─────────────────────────────────────────────
        Result:       [True,  False, True,  False, True]
        (positions 1 and 3 exceed threshold → unmasked)
    """
    # Set unmasked positions to -inf so they're never considered
    eligible = masked & (confidences >= threshold)
    return masked & ~eligible


def uniform_schedule(
    n_masked: int,
    step: int,
    total_steps: int,
) -> int:
    """Uniform unmasking schedule: unmask equal number of tokens per step.

    Data Flow:
    ──────────
    Input:
        n_masked:    int — total number of masked positions
        step:        int — current denoising step (0-indexed)
        total_steps: int — total number of denoising steps

    Output:
        int — number of positions to unmask at this step

    Example:
        n_masked = 10, total_steps = 4
        step=0 → 2    (min 2, 10//4=2)
        step=1 → 2
        step=2 → 2
        step=3 → 4    (last step: take remainder = 10 - 2*3)
    """
    if step >= total_steps - 1:
        return n_masked  # Last step: unmask everything remaining

    tokens_per_step = max(1, n_masked // total_steps)
    return tokens_per_step
