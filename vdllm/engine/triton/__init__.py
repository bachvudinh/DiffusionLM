"""
Triton Backend for CUDA GPUs.

This module provides Triton-specific implementations for attention
and KV cache management on NVIDIA GPUs.

Architecture patterns from JetEngine:
- jetengine/kernels/triton/attention/block_prefill_attention_v2.py - Staircase attention
- jetengine/kernels/triton/fused_page_attention_v3.py - Paged KV cache attention

================================================================================
                              KEY COMPONENTS
================================================================================

1. TritonBackend - Main backend class for CUDA + Triton
2. BlockAttention - Block sparse attention with Triton kernels
3. PagedAttention - Paged KV cache attention with Triton kernels

================================================================================
                              USAGE
================================================================================

    from engine.triton import TritonBackend

    backend = TritonBackend(num_heads=32, head_dim=128)
    output = backend.forward(q, k, v, mask=None)

"""

from .backend import TritonBackend
from .attention import triton_attention

__all__ = ["TritonBackend", "triton_attention"]
