"""
Metal Backend for MPS (Metal Performance Shaders).

This module provides Metal-specific implementations for attention,
KV cache management, and memory handling on Apple Silicon.

Architecture patterns adapted from vllm-metal:
https://github.com/vllm-project/vllm-metal

================================================================================
                              KEY COMPONENTS
================================================================================

1. MetalBackend - Main backend class wrapping PyTorch MPS
2. MetalKVCache - Block-wise KV cache with Metal memory management
3. MetalAttention - Attention operations via SDPA on MPS

================================================================================
                              USAGE
================================================================================

    from engine.metal import MetalBackend

    backend = MetalBackend()
    output = backend.attention(q, k, v, mask=None)

"""

from .backend import MetalBackend
from .kvcache import BlockKVCache, PagedKVCache
from .attention import (
    build_block_diffusion_mask,
    build_staircase_mask,
    build_paged_attention_mask,
    create_causal_mask,
    create_block_causal_mask,
    BlockAttentionMaskBuilder,
)

__all__ = [
    "MetalBackend",
    "BlockKVCache",
    "PagedKVCache",
    "build_block_diffusion_mask",
    "build_staircase_mask",
    "build_paged_attention_mask",
    "create_causal_mask",
    "create_block_causal_mask",
    "BlockAttentionMaskBuilder",
]
