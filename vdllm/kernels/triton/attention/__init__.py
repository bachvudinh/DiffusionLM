"""Triton attention kernels for block diffusion inference.

Derived from JetEngine by Yihan Bian et al.
Reference: https://github.com/Labman42/JetEngine

Provides:
    sparse_attn_varlen:       Staircase block-local attention for prefill
    fused_kv_cache_attention: Paged KV cache attention for denoise
"""
from .block_prefill_attention_v2 import sparse_attn_varlen_v2 as sparse_attn_varlen
from .fused_page_attention_v3 import fused_kv_cache_attention

__all__ = [
    "sparse_attn_varlen",
    "fused_kv_cache_attention",
]
