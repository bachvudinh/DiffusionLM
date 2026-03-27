"""
RMSNorm - Root Mean Square Layer Normalization.

This module provides RMSNorm implementation, which is equivalent to T5LayerNorm.
RMSNorm is used in SDAR for stable training and inference.

Based on JetEngine's RMSNorm:
https://github.com/Jet-Astra/SDAR/blob/main/jetengine/layers/layernorm.py

================================================================================
                              RMSNORM
================================================================================

RMSNorm formula:
    y = x / RMS(x) * w

where RMS(x) = sqrt(1/N * sum(x_i^2))

Compared to LayerNorm, RMSNorm removes the mean centering operation,
making it more computationally efficient while maintaining similar performance.

================================================================================
                              USAGE
================================================================================

    from sdar_model.layers import RMSNorm

    norm = RMSNorm(hidden_size=4096, eps=1e-6)
    normalized = norm(hidden_states)

"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    """
    RMSNorm - Root Mean Square Layer Normalization.

    This is equivalent to T5LayerNorm and is used in SDAR models.

    Attributes:
        weight: Learnable scale parameter
        variance_epsilon: Small constant for numerical stability
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        """
        Initialize RMSNorm.

        Args:
            hidden_size: Size of the hidden dimension
            eps: Small constant for numerical stability (default: 1e-6)
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Apply RMSNorm transformation.

        Args:
            hidden_states: Input tensor [..., hidden_size]

        Returns:
            Normalized tensor with same shape as input
        """
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)

        # Compute RMS: sqrt(mean(x^2))
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)

        return (self.weight * hidden_states).to(input_dtype)

    def extra_repr(self) -> str:
        """String representation."""
        return f"hidden_size={self.weight.shape[0]}, eps={self.variance_epsilon}"
