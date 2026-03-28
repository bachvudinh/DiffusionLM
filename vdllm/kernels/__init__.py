"""Triton and CUDA kernels for vdllm.

Derived from JetEngine by Yihan Bian et al.
Reference: https://github.com/Labman42/JetEngine
"""
from .triton.fused_moe import fused_moe  # re-export

__all__ = ["fused_moe"]
