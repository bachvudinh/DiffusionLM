"""Shared pytest fixtures for vdllm backend tests.

Run from DiffusionLM root:
    uv run python -m pytest tests/ -v
"""

import sys
import os
from enum import Enum, auto

import pytest
import torch
import mlx.core as mx


# ============================================================================
# Test Configuration
# ============================================================================

DIFFUSIONLM_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BACKENDS_PATH = os.path.join(DIFFUSIONLM_ROOT, "vdllm", "backends")
sys.path.insert(0, BACKENDS_PATH)


# ============================================================================
# Minimal Context class (matches what BlockAttention uses)
# ============================================================================

class RunType(Enum):
    """Context run type enumeration."""
    PREFILL = auto()
    DENOISE = auto()


class Context:
    """Minimal Context class for testing attention backends.

    This mirrors the Context used in the actual BlockAttention module.
    """
    def __init__(
        self,
        run_type=None,
        cu_seqlens_q=None,
        cu_seqlens_k=None,
        max_seqlen_q=0,
        max_seqlen_k=0,
        slot_mapping=None,
        context_lens=None,
        block_tables=None,
        is_last_denoise_step=None,
        block_length=4,
    ):
        self.run_type = run_type
        self.cu_seqlens_q = cu_seqlens_q
        self.cu_seqlens_k = cu_seqlens_k
        self.max_seqlen_q = max_seqlen_q
        self.max_seqlen_k = max_seqlen_k
        self.slot_mapping = slot_mapping
        self.context_lens = context_lens
        self.block_tables = block_tables
        self.is_last_denoise_step = is_last_denoise_step or [False]
        self.block_length = block_length


# ============================================================================
# Pytest Fixtures
# ============================================================================

@pytest.fixture
def mlx_backend():
    """Create MLXAttentionBackend instance for testing."""
    from mlx_backend import MLXAttentionBackend
    return MLXAttentionBackend(num_heads=32, num_kv_heads=8, head_dim=128)


@pytest.fixture
def mlx_backend_small():
    """Create MLXAttentionBackend with smaller config for quick tests."""
    from mlx_backend import MLXAttentionBackend
    return MLXAttentionBackend(num_heads=8, num_kv_heads=2, head_dim=64)


@pytest.fixture
def kv_cache():
    """Create a mock KV cache tensor.

    Shape: (num_blocks, 2, num_kv_heads, block_size, head_dim)
    """
    num_blocks = 16
    block_size = 4
    num_kv_heads = 8
    head_dim = 128
    return torch.zeros(
        (num_blocks, 2, num_kv_heads, block_size, head_dim),
        dtype=torch.bfloat16
    )


@pytest.fixture
def kv_cache_small():
    """Create a small mock KV cache tensor for quick tests."""
    num_blocks = 4
    block_size = 4
    num_kv_heads = 2
    head_dim = 64
    return torch.zeros(
        (num_blocks, 2, num_kv_heads, block_size, head_dim),
        dtype=torch.bfloat16
    )


@pytest.fixture
def prefill_context():
    """Create a minimal prefill context."""
    def _make(total_tokens, block_length=4):
        return Context(
            run_type=RunType.PREFILL,
            cu_seqlens_q=torch.tensor([0, total_tokens], dtype=torch.int32),
            cu_seqlens_k=torch.tensor([0, total_tokens], dtype=torch.int32),
            slot_mapping=torch.arange(total_tokens, dtype=torch.int32),
            block_length=block_length,
        )
    return _make


@pytest.fixture
def denoise_context():
    """Create a minimal denoise context."""
    def _make(batch_size, context_lens, block_tables, block_length=4):
        return Context(
            run_type=RunType.DENOISE,
            context_lens=torch.tensor(context_lens, dtype=torch.int32),
            block_tables=torch.tensor(block_tables, dtype=torch.int32),
            block_length=block_length,
        )
    return _make


# ============================================================================
# Helper Functions
# ============================================================================

def tensor_to_mlx(tensor):
    """Convert PyTorch tensor to MLX array using the tensor bridge."""
    from tensor_bridge import to_mlx
    return to_mlx(tensor)


def mlx_to_tensor(mlx_arr):
    """Convert MLX array to PyTorch tensor using the tensor bridge."""
    from tensor_bridge import to_torch
    return to_torch(mlx_arr)


def create_random_qkv(total_tokens, num_heads, num_kv_heads, head_dim, dtype=torch.float32):
    """Create random Q, K, V tensors for testing."""
    q = torch.randn(total_tokens, num_heads, head_dim, dtype=dtype)
    k = torch.randn(total_tokens, num_kv_heads, head_dim, dtype=dtype)
    v = torch.randn(total_tokens, num_kv_heads, head_dim, dtype=dtype)
    return q, k, v


def assert_no_nan_or_inf(tensor):
    """Assert that a PyTorch tensor contains no NaN or Inf values."""
    if isinstance(tensor, mx.array):
        tensor = mlx_to_tensor(tensor)
    assert not torch.isnan(tensor).any(), "Tensor contains NaN values"
    assert not torch.isinf(tensor).any(), "Tensor contains Inf values"


def assert_close(a, b, rtol=1e-5, atol=1e-5):
    """Assert two tensors are close within tolerance."""
    if isinstance(a, mx.array):
        a = mlx_to_tensor(a)
    if isinstance(b, mx.array):
        b = mlx_to_tensor(b)
    assert torch.allclose(a, b, rtol=rtol, atol=atol), \
        f"Tensors not close:\n{a}\n vs \n{b}"
