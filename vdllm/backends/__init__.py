"""Attention backends for vdllm.

Provides a factory function get_backend() that selects the best available
attention backend based on hardware.

Currently supported:
    - CUDA: NVIDIA GPU (FlashInfer + Triton kernels)

The MLX path uses a separate engine (vdllm.engine.mlx_engine) and does not
go through the backend abstraction.
"""

from typing import Optional

from vdllm.backends.base import AttentionBackend, CacheConfig


def get_backend(
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    backend_type: Optional[str] = None,
) -> AttentionBackend:
    """Factory function to get the best available attention backend.

    Args:
        num_heads: Number of query heads
        num_kv_heads: Number of key/value heads (GQA support)
        head_dim: Dimension per attention head
        backend_type: Optional explicit backend selection ('cuda')

    Returns:
        An AttentionBackend instance.

    Raises:
        ValueError: If backend is unavailable or unknown.
    """
    import torch

    if backend_type is not None:
        if backend_type == "cuda":
            if not torch.cuda.is_available():
                raise ValueError("CUDA backend requested but CUDA is not available")
            from vdllm.backends.cuda_backend import CUDAAttentionBackend
            return CUDAAttentionBackend(num_heads, num_kv_heads, head_dim)
        raise ValueError(f"Unknown backend_type '{backend_type}'. Available: ['cuda']")

    if torch.cuda.is_available():
        from vdllm.backends.cuda_backend import CUDAAttentionBackend
        return CUDAAttentionBackend(num_heads, num_kv_heads, head_dim)

    raise ValueError(
        "No attention backend available. CUDA required for the backend API. "
        "For Apple Silicon, use LLM(model, backend='mlx') which uses MLXEngine directly."
    )


def list_available_backends() -> list[str]:
    """Return list of available backend names."""
    available = []
    try:
        import torch
        if torch.cuda.is_available():
            available.append("cuda")
    except ImportError:
        pass
    return available


__all__ = [
    "AttentionBackend",
    "CacheConfig",
    "get_backend",
    "list_available_backends",
]
