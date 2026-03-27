"""
Rotary Position Embeddings (RoPE).

This module provides Rotary Position Embeddings (RoPE) implementation
for encoding positional information in the attention mechanism.

Based on:
- RoPE paper: https://arxiv.org/abs/2104.09864
- JetEngine's rotary embedding: https://github.com/Jet-Astra/SDAR/blob/main/jetengine/layers/rotary_embedding.py

================================================================================
                              ROTARY POSITION EMBEDDING
================================================================================

RoPE encodes position information by rotating query and key vectors:

    R(d, theta, m) = [[cos(m*theta_d), -sin(m*theta_d)],
                       [sin(m*theta_d),  cos(m*theta_d)]]

For a vector split into pairs of dimensions:
    q' = R(q) = q * cos(m*theta) + (-q_parallel) * sin(m*theta)

The attention score between q at position m and k at position n is:
    <q_m, k_n> = <R(q, m*theta), R(k, n*theta)>

This decays based on the distance |m-n|, providing relative position encoding.

================================================================================
                              USAGE
================================================================================

    from sdar_model.layers.rotary import apply_rotary_pos_emb, precompute_freqs_cis

    # Precompute frequency tensor
    freqs = precompute_freqs_cis(
        seq_len=1024,
        head_dim=128,
        theta=10000.0,
    )

    # Apply RoPE to Q and K
    q_rot, k_rot = apply_rotary_pos_emb(q, k, freqs, position_ids)

"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn


def precompute_freqs_cis(
    seq_len: int,
    head_dim: int,
    theta: float = 10000.0,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Precompute complex frequencies for RoPE.

    Args:
        seq_len: Maximum sequence length
        head_dim: Dimension of each attention head
        theta: Base period of RoPE (default: 10000.0)
        device: Device to create tensor on

    Returns:
        Complex tensor [seq_len, head_dim//2] with frequencies
    """
    # Compute angles for each dimension pair
    freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))

    # Create position indices
    positions = torch.arange(seq_len, device=device)

    # Compute angle = position * frequency
    angles = positions[:, None] * freqs[None, :]

    # Compute complex exponentials: e^(i*angle)
    # This is equivalent to [cos(angle), sin(angle)]
    freqs_cis = torch.polar(torch.ones_like(angles), angles)

    return freqs_cis


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """
    Rotate half the hidden dims of the input.

    This splits the vector into two halves and rotates one half.

    Args:
        x: Input tensor [..., head_dim]

    Returns:
        Rotated tensor
    """
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    freqs_cis: torch.Tensor,
    position_ids: Optional[torch.Tensor] = None,
    unsqueeze_dim: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary position embeddings to query and key tensors.

    Args:
        q: Query tensor [..., seq_len, head_dim]
        k: Key tensor [..., seq_len, head_dim]
        freqs_cis: Precomputed complex frequencies [seq_len, head_dim//2]
        position_ids: Optional position IDs [batch_size, seq_len]
        unsqueeze_dim: Dimension to unsqueeze freqs_cis for broadcasting

    Returns:
        Tuple of (rotated_q, rotated_k)
    """
    # Handle attention implementation from transformers library
    if hasattr(q, "cos") and hasattr(q, "sin"):
        # It's a RotaryEmbedding tensor from transformers
        q_embed = (q * freqs_cis.cos()) + (rotate_half(q) * freqs_cis.sin())
        k_embed = (k * freqs_cis.cos()) + (rotate_half(k) * freqs_cis.sin())
        return q_embed, k_embed

    # Standard PyTorch implementation
    # Reshape q and k for complex multiplication
    # q/k shape: [..., seq_len, head_dim] -> [..., seq_len, head_dim//2, 2]
    dtype = q.dtype
    q_float = q.float()
    k_float = k.float()

    # Reshape to pairs of dimensions
    q_reshaped = q_float.view(*q.shape[:-1], -1, 2)
    k_reshaped = k_float.view(*k.shape[:-1], -1, 2)

    # Handle position_ids
    if position_ids is not None:
        # position_ids shape: [batch_size, seq_len]
        # freqs_cis shape: [seq_len, head_dim//2]
        freqs = freqs_cis[position_ids]  # [batch_size, seq_len, head_dim//2]
    else:
        freqs = freqs_cis

    # Unsqueeze for broadcasting: [..., seq_len, head_dim//2] -> [..., seq_len, head_dim//2, 1]
    freqs = freqs.unsqueeze(unsqueeze_dim)

    # Complex multiplication: (a + bi) * (c + di) = (ac - bd) + (ad + bc)i
    # For rotation: q * cos(angle) + rotate_half(q) * sin(angle)
    q_real = q_reshaped[..., 0]
    q_imag = q_reshaped[..., 1]

    cos = freqs[..., 0]  # [..., seq_len, head_dim//2]
    sin = freqs[..., 1]  # [..., seq_len, head_dim//2]

    q_out_real = q_real * cos - (-q_imag) * sin
    q_out_imag = q_real * sin + q_imag * cos

    q_out = torch.stack([q_out_real, q_out_imag], dim=-1).flatten(-2)

    k_real = k_reshaped[..., 0]
    k_imag = k_reshaped[..., 1]

    k_out_real = k_real * cos - (-k_imag) * sin
    k_out_imag = k_real * sin + k_imag * cos

    k_out = torch.stack([k_out_real, k_out_imag], dim=-1).flatten(-2)

    return q_out.to(dtype), k_out.to(dtype)


class RotaryEmbedding(nn.Module):
    """
    Rotary Position Embedding module.

    This module computes and caches RoPE embeddings for efficient reuse.
    """

    def __init__(
        self,
        head_dim: int,
        max_position: int = 32768,
        base: float = 10000.0,
        device: Optional[torch.device] = None,
    ):
        """
        Initialize RoPE embeddings.

        Args:
            head_dim: Dimension of each attention head
            max_position: Maximum position to precompute
            base: Base period of RoPE
            device: Device to create tensors on
        """
        super().__init__()
        self.head_dim = head_dim
        self.max_position = max_position
        self.base = base

        # Precompute frequencies
        self.register_buffer(
            "freqs_cis",
            precompute_freqs_cis(max_position, head_dim, base, device),
            persistent=False,
        )

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply RoPE to query and key tensors.

        Args:
            q: Query tensor [batch, heads, seq_len, head_dim]
            k: Key tensor [batch, heads, seq_len, head_dim]
            position_ids: Optional position IDs

        Returns:
            Tuple of (rotated_q, rotated_k)
        """
        return apply_rotary_pos_emb(q, k, self.freqs_cis, position_ids)
