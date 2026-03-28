"""RMS normalization layer with compiled fused variants.

Derived from JetEngine by Yihan Bian et al.
Reference: https://github.com/Labman42/JetEngine

Architecture Overview
====================

    RMSNorm provides two compiled forward paths:

    ┌─────────────────────────────────────────────────────────┐
    │                      RMSNorm                             │
    │                                                          │
    │  Path 1: rms_forward(x)                                  │
    │    x ──► float32 ──► var=mean(x^2) ──► x*rsqrt(var+eps) │
    │       ──► orig_dtype ──► x * weight ──► output           │
    │                                                          │
    │  Path 2: add_rms_forward(x, residual)   [fused add+norm] │
    │    x + residual ──► new_residual                         │
    │    norm(new_residual) * weight ──► output                │
    │                                                          │
    │  Dispatch: forward(x, residual=None)                     │
    │    residual is None → Path 1                             │
    │    residual given   → Path 2                             │
    │                                                          │
    │  Both paths are @torch.compile'd for kernel fusion.      │
    └─────────────────────────────────────────────────────────┘


rms_forward — Tensor Flow
==========================

    Example: hidden_size=4096, eps=1e-6, x dtype=bfloat16

    x: (N, 4096) bfloat16
            │
            ▼  to float32
    x: (N, 4096) float32
            │
            ▼  var = x.pow(2).mean(dim=-1, keepdim=True)
    var: (N, 1) float32
            │
            ▼  x *= rsqrt(var + 1e-6)
    x: (N, 4096) float32           ← normalized
            │
            ▼  to bfloat16, multiply by weight
    out: (N, 4096) bfloat16
                    weight: (4096,) bfloat16  ← learnable scale


add_rms_forward — Tensor Flow (fused residual + norm)
======================================================

    Example: used in decoder layer between attention and MLP

    x: (N, 4096) bfloat16          ← attention output
    residual: (N, 4096) bfloat16   ← running residual stream
            │
            ▼  float32: x = x + residual
    x: (N, 4096) float32
            │
            ├──► new_residual = x.to(bfloat16)    ← saved for next layer
            │       new_residual: (N, 4096) bfloat16
            │
            ▼  var = x.pow(2).mean(dim=-1, keepdim=True)
    var: (N, 1) float32
            │
            ▼  x *= rsqrt(var + eps), to bfloat16, * weight
    normalized: (N, 4096) bfloat16

    Output: (normalized, new_residual)
"""

import torch
from torch import nn


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization.

    Input:
        hidden_size: int — feature dimension
        eps: float — epsilon for numerical stability (default: 1e-6)
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    @torch.compile
    def rms_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Standalone RMS normalization.

        Input:
            x: (..., hidden_size)

        Output:
            (..., hidden_size) — normalized and scaled
        """
        orig_dtype = x.dtype
        x = x.to(torch.float32)
        var = x.pow(2).mean(dim=-1, keepdim=True)
        x.mul_(torch.rsqrt(var + self.eps))
        x = x.to(orig_dtype).mul_(self.weight)
        return x

    @torch.compile
    def add_rms_forward(self, x: torch.Tensor,
                        residual: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Fused residual addition + RMS normalization.

        Input:
            x:        (..., hidden_size) — layer output
            residual: (..., hidden_size) — running residual stream

        Output:
            (normalized, new_residual) tuple
        """
        orig_dtype = x.dtype
        x = x.to(torch.float32).add_(residual.to(torch.float32))
        residual = x.to(orig_dtype)
        var = x.pow(2).mean(dim=-1, keepdim=True)
        x.mul_(torch.rsqrt(var + self.eps))
        x = x.to(orig_dtype).mul_(self.weight)
        return x, residual

    def forward(self, x: torch.Tensor,
                residual: torch.Tensor | None = None
                ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            return self.rms_forward(x)
        else:
            return self.add_rms_forward(x, residual)
