"""Fused SiLU-gated activation using Liger kernel.

Derived from JetEngine by Yihan Bian et al.
Reference: https://github.com/Labman42/JetEngine

Architecture Overview
====================

    SiLU-gated linear unit used in LLaMA-style MLP blocks:

    ┌──────────────────────────────────────────────────────┐
    │                    SiluAndMul                         │
    │                                                      │
    │  Input x: (..., 2 * intermediate_size)               │
    │                                                      │
    │  x_gate, x_up = chunk(x, 2, dim=-1)                 │
    │  output = SiLU(x_gate) * x_up                        │
    │                                                      │
    │  Uses LigerSiLUMulFunction for fused CUDA kernel     │
    │  Output: (..., intermediate_size)                     │
    └──────────────────────────────────────────────────────┘


Tensor Flow
============

    Example: intermediate_size=5504 (per-rank, tp_size=2 from 11008)

    x: (N, 11008) bfloat16       ← output of gate_up_proj
            │
            ▼  chunk(2, dim=-1)
    x_gate: (N, 5504) bfloat16
    x_up:   (N, 5504) bfloat16
            │
            ▼  SiLU(x_gate) = x_gate * sigmoid(x_gate)
    activated: (N, 5504) bfloat16
            │
            ▼  activated * x_up  (element-wise)
    output: (N, 5504) bfloat16

    SiLU curve:
        x < -5  → ~0
        x = 0   → 0
        x > 5   → ~x
        Smooth approximation to ReLU with negative values near zero.
"""

import torch
from torch import nn
from liger_kernel.ops.swiglu import LigerSiLUMulFunction


class SiluAndMul(nn.Module):
    """Fused SiLU gating: SiLU(x_gate) * x_up.

    Input:
        x: (..., 2 * intermediate_size) — concatenated gate and up projections

    Output:
        (..., intermediate_size)
    """

    def __init__(self):
        super().__init__()

    @torch.compile
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, y = x.chunk(2, -1)
        return LigerSiLUMulFunction.apply(x, y)
