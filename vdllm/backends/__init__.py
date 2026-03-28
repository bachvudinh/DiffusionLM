"""Attention backends for vdllm.

Provides a factory function get_backend() that automatically selects
the best available backend based on hardware and software availability.

Backend Priority
================

    1. CUDA:   NVIDIA GPU via torch.cuda.is_available()
    2. MLX:    Apple Silicon via mx.metal.is_available()
    3. MPS:    Apple GPU via torch.backends.mps.is_available()
    4. CPU:    Naive PyTorch fallback

Usage
=====

    from vdllm.backends import get_backend

    backend = get_backend(num_heads=32, num_kv_heads=8, head_dim=128)
    print(backend.name)  # e.g., "cuda"

    # Use backend methods
    output = backend.prefill_attention(q, k, v, block_length=4, staircase=True)
    backend.reshape_and_cache(k, v, k_cache, v_cache, slot_mapping)
    output = backend.denoise_attention(q, k_cache, v_cache, k_new, v_new,
                                        block_tables, seq_lens)
"""

import torch
from typing import Optional

from vdllm.backends.base import AttentionBackend, CacheConfig

# Lazy import for MLX - only loaded when needed and available
_mlx_backend = None


def _get_mlx_backend():
    """Lazily import and return MLX backend if available."""
    global _mlx_backend
    if _mlx_backend is None:
        try:
            import mlx.core as mx
            if mx.metal.is_available():
                from vdllm.backends.mlx_backend import MLXAttentionBackend
                _mlx_backend = MLXAttentionBackend
        except ImportError:
            _mlx_backend = None
    return _mlx_backend


def get_backend(
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    backend_type: Optional[str] = None,
) -> AttentionBackend:
    """Factory function to get the best available attention backend.

    Args:
        num_heads: Number of query heads (after TP split)
        num_kv_heads: Number of key/value heads (GQA support)
        head_dim: Dimension per attention head
        backend_type: Optional explicit backend selection ('cuda', 'mlx', 'mps', 'cpu')
                     If None, auto-detects based on hardware availability.

    Returns:
        An AttentionBackend instance compatible with the available hardware.

    Raises:
        ValueError: If an unknown backend_type is specified or no backends available.

    Backend Detection Logic
    =======================

        Priority order (first available wins):

        1. CUDA:   torch.cuda.is_available()
        2. MLX:    mx.metal.is_available() (Apple Silicon MLX)
        3. MPS:    torch.backends.mps.is_available() (Apple GPU via PyTorch)
        4. CPU:    Always available as fallback
    """
    if backend_type is not None:
        # Explicit backend selection
        return _create_backend(backend_type, num_heads, num_kv_heads, head_dim)

    # Auto-detection based on hardware priority
    # Priority: CUDA > MLX > MPS > CPU

    # 1. Try CUDA first
    if torch.cuda.is_available():
        from vdllm.backends.cuda_backend import CUDAAttentionBackend
        return CUDAAttentionBackend(num_heads, num_kv_heads, head_dim)

    # 2. Try MLX (Apple Silicon with MLX)
    mlx_backend_cls = _get_mlx_backend()
    if mlx_backend_cls is not None:
        return mlx_backend_cls(num_heads, num_kv_heads, head_dim)

    # 3. Try MPS (Apple GPU via PyTorch)
    if torch.backends.mps.is_available():
        from vdllm.backends.mps_backend import MPSAttentionBackend
        return MPSAttentionBackend(num_heads, num_kv_heads, head_dim)

    # 4. CPU fallback
    from vdllm.backends.cpu_backend import CPUAttentionBackend
    return CPUAttentionBackend(num_heads, num_kv_heads, head_dim)


def _create_backend(
    backend_type: str,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
) -> AttentionBackend:
    """Create a specific backend by type name.

    Args:
        backend_type: One of 'cuda', 'mlx', 'mps', 'cpu'
        num_heads: Number of query heads
        num_kv_heads: Number of key/value heads
        head_dim: Dimension per head

    Returns:
        The requested AttentionBackend instance

    Raises:
        ValueError: If backend_type is unknown or unavailable
    """
    if backend_type == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("CUDA backend requested but torch.cuda.is_available() is False")
        from vdllm.backends.cuda_backend import CUDAAttentionBackend
        return CUDAAttentionBackend(num_heads, num_kv_heads, head_dim)

    elif backend_type == "mlx":
        mlx_cls = _get_mlx_backend()
        if mlx_cls is None:
            raise ValueError("MLX backend requested but MLX is not available")
        return mlx_cls(num_heads, num_kv_heads, head_dim)

    elif backend_type == "mps":
        if not torch.backends.mps.is_available():
            raise ValueError("MPS backend requested but torch.backends.mps.is_available() is False")
        from vdllm.backends.mps_backend import MPSAttentionBackend
        return MPSAttentionBackend(num_heads, num_kv_heads, head_dim)

    elif backend_type == "cpu":
        from vdllm.backends.cpu_backend import CPUAttentionBackend
        return CPUAttentionBackend(num_heads, num_kv_heads, head_dim)

    else:
        available = ["cuda", "mlx", "mps", "cpu"]
        raise ValueError(
            f"Unknown backend_type '{backend_type}'. Available: {available}"
        )


def list_available_backends() -> list[str]:
    """Return list of available backend names in priority order.

    Returns:
        List of backend names that can be selected, in detection priority order.
    """
    available = []

    if torch.cuda.is_available():
        available.append("cuda")

    if _get_mlx_backend() is not None:
        available.append("mlx")

    if torch.backends.mps.is_available():
        available.append("mps")

    available.append("cpu")

    return available


__all__ = [
    "AttentionBackend",
    "CacheConfig",
    "get_backend",
    "list_available_backends",
]
