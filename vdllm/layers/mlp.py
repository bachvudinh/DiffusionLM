"""
SDAR MLP - Feed-forward Network.

This module provides the MLP (Multi-Layer Perceptron) used in SDAR models,
implementing the SwiGLU activation function.

Based on JetEngine's MLP implementation:
https://github.com/Jet-Astra/SDAR/blob/main/jetengine/layers/mlp.py

================================================================================
                              SWIGLU ACTIVATION
================================================================================

SwiGLU (Swish-Gated Linear Unit) formula:
    SwiGLU(x) = x * sigmoid(beta * x) * gate(x)

For the standard MLP with SwiGLU:
    FFN(x) = DownProj(SiLU(GateProj(x))) * UpProj(x)

Where SiLU(x) = x * sigmoid(x)

================================================================================
                              USAGE
================================================================================

    from sdar_model.layers import SDARMLP

    mlp = SDARMLP(
        hidden_size=4096,
        intermediate_size=22016,
        hidden_act="silu",
    )
    output = mlp(hidden_states)

"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class SDARMLP(nn.Module):
    """
    MLP with SwiGLU activation for SDAR models.

    This implements the feed-forward network used in transformer layers,
    using the SwiGLU (Swish-Gated Linear Unit) activation.

    Attributes:
        gate_proj: Linear layer for gate computation
        up_proj: Linear layer for intermediate values
        down_proj: Linear layer for output projection
        act_fn: SiLU activation function
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str = "silu",
        bias: bool = False,
        device: Optional[torch.device] = None,
    ):
        """
        Initialize MLP.

        Args:
            hidden_size: Hidden dimension size
            intermediate_size: FFN intermediate dimension
            hidden_act: Activation function name (should be "silu")
            bias: Whether to use bias in linear layers
            device: Device for layer creation
        """
        super().__init__()

        assert hidden_act == "silu", f"SDAR only supports silu activation, got {hidden_act}"

        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size

        # Fused gate+up projection (more efficient)
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=bias, device=device)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=bias, device=device)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=bias, device=device)

        # SiLU activation
        self.act_fn = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor [batch_size, seq_len, hidden_size]

        Returns:
            Output tensor [batch_size, seq_len, hidden_size]
        """
        # SwiGLU: x * silu(gate(x)) * up(x)
        gate = self.act_fn(self.gate_proj(x))
        up = self.up_proj(x)
        down = self.down_proj(gate * up)
        return down
