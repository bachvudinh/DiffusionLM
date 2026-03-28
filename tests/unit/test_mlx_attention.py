"""Unit tests for MLX attention backend.

Run from DiffusionLM root:
    uv run python tests/unit/test_mlx_attention.py

These tests validate the MLX backend implementation including:
- Prefill attention with staircase masking
- Denoise attention with paged KV cache
- Proper MLX ↔ PyTorch conversion
- GQA (grouped query attention) support
"""

import sys
import os
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Optional, List
import importlib.util

import torch
import mlx.core as mx
import numpy as np


# ============================================================================
# Direct module loading to avoid triggering vdllm.__init__ (requires flashinfer)
# ============================================================================

def import_module_from_path(module_name: str, file_path: str):
    """Import a module directly from a file path, avoiding package __init__ files."""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


# Get base path for vdllm
vdllm_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
backends_path = os.path.join(vdllm_path, "vdllm", "backends")

# Import tensor_bridge first (no dependencies)
tensor_bridge = import_module_from_path(
    "vdllm.backends.tensor_bridge",
    os.path.join(backends_path, "tensor_bridge.py")
)
to_mlx = tensor_bridge.to_mlx
to_torch = tensor_bridge.to_torch
TensorBridge = tensor_bridge.TensorBridge
ensure_mlx_array = tensor_bridge.ensure_mlx_array
ensure_torch_tensor = tensor_bridge.ensure_torch_tensor
mlx_to_torch_dtype = tensor_bridge.mlx_to_torch_dtype
torch_to_mlx_dtype = tensor_bridge.torch_to_mlx_dtype

# Import base module
base_module = import_module_from_path(
    "vdllm.backends.base",
    os.path.join(backends_path, "base.py")
)
AttentionBackend = base_module.AttentionBackend
CacheConfig = base_module.CacheConfig

# Import mlx_backend (depends on base and tensor_bridge)
mlx_backend = import_module_from_path(
    "vdllm.backends.mlx_backend",
    os.path.join(backends_path, "mlx_backend.py")
)
MLXAttentionBackend = mlx_backend.MLXAttentionBackend


# ============================================================================
# Test Fixtures and Helpers
# ============================================================================

class RunType(Enum):
    """Minimal RunType enum for testing."""
    PREFILL = auto()
    DENOISE = auto()


@dataclass
class Context:
    """Minimal Context class for testing."""
    run_type: Optional[RunType] = None
    cu_seqlens_q: Optional[torch.Tensor] = None
    cu_seqlens_k: Optional[torch.Tensor] = None
    max_seqlen_q: int = 0
    max_seqlen_k: int = 0
    slot_mapping: Optional[torch.Tensor] = None
    context_lens: Optional[torch.Tensor] = None
    block_tables: Optional[torch.Tensor] = None
    is_last_denoise_step: List[bool] = field(default_factory=lambda: [False])
    block_length: int = 4


def make_backend():
    """Create an MLXAttentionBackend instance for testing."""
    return MLXAttentionBackend(num_heads=32, num_kv_heads=8, head_dim=128)


def make_kv_cache():
    """Create a mock KV cache tensor.

    Shape: (num_blocks, 2, num_kv_heads, block_size, head_dim)
    """
    num_blocks = 16
    block_size = 4
    num_kv_heads = 8
    head_dim = 128
    return torch.zeros((num_blocks, 2, num_kv_heads, block_size, head_dim),
                       dtype=torch.bfloat16)


def run_tests():
    """Run all tests and report results."""
    print("=" * 70)
    print("MLX Attention Backend - Unit Tests")
    print("=" * 70)

    passed = 0
    failed = 0

    # Tensor Bridge Tests
    print("\n--- Tensor Bridge Tests ---")
    for test_name, test_fn in [
        ("test_to_mlx_float32", test_to_mlx_float32),
        ("test_to_mlx_float16", test_to_mlx_float16),
        ("test_to_mlx_bfloat16", test_to_mlx_bfloat16),
        ("test_to_torch_float32", test_to_torch_float32),
        ("test_to_torch_bfloat16", test_to_torch_bfloat16),
        ("test_roundtrip_float32", test_roundtrip_float32),
        ("test_roundtrip_bfloat16", test_roundtrip_bfloat16),
        ("test_ensure_mlx_array_from_mlx", test_ensure_mlx_array_from_mlx),
        ("test_ensure_mlx_array_from_torch", test_ensure_mlx_array_from_torch),
        ("test_dtype_conversion_functions", test_dtype_conversion_functions),
    ]:
        try:
            test_fn()
            print(f"  PASS: {test_name}")
            passed += 1
        except Exception as e:
            print(f"  FAIL: {test_name}: {e}")
            failed += 1

    # Staircase Mask Tests
    print("\n--- Staircase Mask Tests ---")
    for test_name, test_fn in [
        ("test_staircase_mask_single_sequence", test_staircase_mask_single_sequence),
        ("test_staircase_mask_multiple_sequences", test_staircase_mask_multiple_sequences),
    ]:
        try:
            test_fn()
            print(f"  PASS: {test_name}")
            passed += 1
        except Exception as e:
            print(f"  FAIL: {test_name}: {e}")
            failed += 1

    # Backend Initialization Tests
    print("\n--- Backend Initialization Tests ---")
    for test_name, test_fn in [
        ("test_backend_creation", test_backend_creation),
        ("test_initialize_cache", test_initialize_cache),
    ]:
        try:
            test_fn()
            print(f"  PASS: {test_name}")
            passed += 1
        except Exception as e:
            print(f"  FAIL: {test_name}: {e}")
            failed += 1

    # Prefill Attention Tests
    print("\n--- Prefill Attention Tests ---")
    for test_name, test_fn in [
        ("test_prefill_single_sequence", test_prefill_single_sequence),
        ("test_prefill_multiple_sequences", test_prefill_multiple_sequences),
    ]:
        try:
            test_fn()
            print(f"  PASS: {test_name}")
            passed += 1
        except Exception as e:
            print(f"  FAIL: {test_name}: {e}")
            failed += 1

    # Denoise Attention Tests
    print("\n--- Denoise Attention Tests ---")
    for test_name, test_fn in [
        ("test_denoise_basic", test_denoise_basic),
        ("test_denoise_variable_context_lengths", test_denoise_variable_context_lengths),
    ]:
        try:
            test_fn()
            print(f"  PASS: {test_name}")
            passed += 1
        except Exception as e:
            print(f"  FAIL: {test_name}: {e}")
            failed += 1

    # GQA Tests
    print("\n--- GQA Support Tests ---")
    for test_name, test_fn in [
        ("test_gqa_different_num_heads", test_gqa_different_num_heads),
    ]:
        try:
            test_fn()
            print(f"  PASS: {test_name}")
            passed += 1
        except Exception as e:
            print(f"  FAIL: {test_name}: {e}")
            failed += 1

    # Integration Tests
    print("\n--- Integration Tests ---")
    for test_name, test_fn in [
        ("test_prefill_then_denoise", test_prefill_then_denoise),
    ]:
        try:
            test_fn()
            print(f"  PASS: {test_name}")
            passed += 1
        except Exception as e:
            print(f"  FAIL: {test_name}: {e}")
            failed += 1

    # Summary
    print("\n" + "=" * 70)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 70)

    return failed == 0


# ============================================================================
# Tensor Bridge Tests
# ============================================================================

def test_to_mlx_float32():
    """Test conversion of float32 tensors."""
    pt_tensor = torch.randn(4, 8, 16, dtype=torch.float32)
    mlx_arr = to_mlx(pt_tensor)

    assert mlx_arr.shape == (4, 8, 16)
    assert mlx_arr.dtype == mx.float32
    mx.eval(mlx_arr)  # Force evaluation


def test_to_mlx_float16():
    """Test conversion of float16 tensors."""
    pt_tensor = torch.randn(4, 8, 16, dtype=torch.float16)
    mlx_arr = to_mlx(pt_tensor)

    assert mlx_arr.shape == (4, 8, 16)
    assert mlx_arr.dtype == mx.float16
    mx.eval(mlx_arr)


def test_to_mlx_bfloat16():
    """Test conversion of bfloat16 tensors (MLX bug workaround)."""
    pt_tensor = torch.randn(4, 8, 16, dtype=torch.bfloat16)
    mlx_arr = to_mlx(pt_tensor)

    assert mlx_arr.shape == (4, 8, 16)
    assert mlx_arr.dtype == mx.bfloat16
    mx.eval(mlx_arr)


def test_to_torch_float32():
    """Test conversion of float32 arrays."""
    mlx_arr = mx.random.normal((4, 8, 16), dtype=mx.float32)
    pt_tensor = to_torch(mlx_arr)

    assert pt_tensor.shape == (4, 8, 16)
    assert pt_tensor.dtype == torch.float32


def test_to_torch_bfloat16():
    """Test conversion of bfloat16 arrays."""
    mlx_arr = mx.random.normal((4, 8, 16), dtype=mx.bfloat16)
    pt_tensor = to_torch(mlx_arr)

    assert pt_tensor.shape == (4, 8, 16)
    assert pt_tensor.dtype == torch.bfloat16


def test_roundtrip_float32():
    """Test roundtrip conversion float32 -> MLX -> float32."""
    original = torch.randn(10, 20, dtype=torch.float32)
    mlx_arr = to_mlx(original)
    recovered = to_torch(mlx_arr)

    assert recovered.dtype == torch.float32
    mx.eval(mlx_arr)
    # Values should be close (may not be exact due to float32 conversions)
    assert torch.allclose(original, recovered, rtol=1e-5, atol=1e-5)


def test_roundtrip_bfloat16():
    """Test roundtrip conversion bfloat16 -> MLX -> bfloat16."""
    original = torch.randn(10, 20, dtype=torch.bfloat16)
    mlx_arr = to_mlx(original)
    recovered = to_torch(mlx_arr)

    assert recovered.dtype == torch.bfloat16
    mx.eval(mlx_arr)
    # Values should be close
    assert torch.allclose(original, recovered, rtol=1e-3, atol=1e-3)


def test_ensure_mlx_array_from_mlx():
    """Test ensure_mlx_array with MLX input."""
    arr = mx.random.normal((10, 20))
    result = ensure_mlx_array(arr)
    assert result is arr


def test_ensure_mlx_array_from_torch():
    """Test ensure_mlx_array with torch input."""
    tensor = torch.randn(10, 20)
    result = ensure_mlx_array(tensor)
    assert isinstance(result, mx.array)


def test_dtype_conversion_functions():
    """Test dtype conversion helper functions."""
    assert mlx_to_torch_dtype(mx.float32) == torch.float32
    assert mlx_to_torch_dtype(mx.float16) == torch.float16
    assert mlx_to_torch_dtype(mx.bfloat16) == torch.bfloat16
    assert torch_to_mlx_dtype(torch.float32) == mx.float32
    assert torch_to_mlx_dtype(torch.float16) == mx.float16
    assert torch_to_mlx_dtype(torch.bfloat16) == mx.bfloat16


# ============================================================================
# Staircase Mask Tests
# ============================================================================

def test_staircase_mask_single_sequence():
    """Test staircase mask with a single sequence.

    block_length=4, seq_len=12 (3 blocks)

    Token layout:
        Tok:  [0  1  2  3 | 4  5  6  7 | 8  9 10 11]
        Blk:  [  block 0  |  block 1   |  block 2  ]

    Expected mask (0=attend, -inf=masked):
                Q
         0 1 2 3 4 5 6 7 8 9 A B
      0 [0 0 0 0 . . . . . . . .]
    K 4 [0 0 0 0 0 0 0 0 . . . .]
      8 [0 0 0 0 0 0 0 0 0 0 0 0]
    """
    backend = MLXAttentionBackend(num_heads=32, num_kv_heads=8, head_dim=128)
    backend.block_size = 4

    total_tokens = 12
    block_length = 4
    cu_seqlens = mx.array([0, 12], dtype=mx.int32)

    mask = backend._build_staircase_mask_additive(total_tokens, cu_seqlens, block_length)
    mx.eval(mask)

    # Block 0 (rows 0-3) should only attend to block 0 (cols 0-3)
    for i in range(4):
        for j in range(4):
            assert mask[i, j].item() == 0.0, f"Block 0: mask[{i},{j}] should be 0.0"
        for j in range(4, 12):
            assert mask[i, j].item() == -1e9, f"Block 0: mask[{i},{j}] should be -inf"

    # Block 1 (rows 4-7) should attend to blocks 0 and 1
    for i in range(4, 8):
        for j in range(8):
            assert mask[i, j].item() == 0.0, f"Block 1: mask[{i},{j}] should be 0.0"
        for j in range(8, 12):
            assert mask[i, j].item() == -1e9, f"Block 1: mask[{i},{j}] should be -inf"

    # Block 2 (rows 8-11) should attend to all blocks
    for i in range(8, 12):
        for j in range(12):
            assert mask[i, j].item() == 0.0, f"Block 2: mask[{i},{j}] should be 0.0"


def test_staircase_mask_multiple_sequences():
    """Test staircase mask with multiple sequences.

    seq_lens = [8, 4] (total=12)
    block_length = 4

    Sequence 0: tokens 0-7 (2 blocks)
    Sequence 1: tokens 8-11 (1 block)

    Each sequence should have independent staircase masking.
    """
    backend = MLXAttentionBackend(num_heads=32, num_kv_heads=8, head_dim=128)
    backend.block_size = 4

    total_tokens = 12
    block_length = 4
    # Two sequences: [0, 8) and [8, 12)
    cu_seqlens = mx.array([0, 8, 12], dtype=mx.int32)

    mask = backend._build_staircase_mask_additive(total_tokens, cu_seqlens, block_length)
    mx.eval(mask)

    # Sequence 0 (tokens 0-7):
    # Block 0 (rows 0-3) should only attend to block 0 (cols 0-3)
    for i in range(4):
        for j in range(4):
            assert mask[i, j].item() == 0.0
        for j in range(4, 8):
            assert mask[i, j].item() == -1e9

    # Block 1 (rows 4-7) should attend to blocks 0 and 1
    for i in range(4, 8):
        for j in range(8):
            assert mask[i, j].item() == 0.0

    # Sequence 1 (tokens 8-11):
    # Single block should attend to all within the block
    for i in range(8, 12):
        for j in range(8, 12):
            assert mask[i, j].item() == 0.0


# ============================================================================
# Backend Initialization Tests
# ============================================================================

def test_backend_creation():
    """Test basic backend creation."""
    backend = MLXAttentionBackend(num_heads=32, num_kv_heads=8, head_dim=128)

    assert backend.num_heads == 32
    assert backend.num_kv_heads == 8
    assert backend.head_dim == 128
    assert backend.scale == (128 ** -0.5)
    assert backend.k_cache is None
    assert backend.v_cache is None


def test_initialize_cache():
    """Test KV cache initialization."""
    backend = make_backend()
    kv_cache = make_kv_cache()
    backend.initialize_cache(kv_cache)

    assert backend.k_cache is not None
    assert backend.v_cache is not None
    assert backend.num_blocks == 16
    assert backend.block_size == 4

    # Force evaluation
    mx.eval(backend.k_cache, backend.v_cache)


# ============================================================================
# Prefill Attention Tests
# ============================================================================

def test_prefill_single_sequence():
    """Test prefill with a single sequence.

    Sequence: 8 tokens, block_length=4
    """
    backend = make_backend()
    kv_cache = make_kv_cache()
    backend.initialize_cache(kv_cache)

    total_tokens = 8
    seq_lens = [8]

    ctx = Context(
        run_type=RunType.PREFILL,
        cu_seqlens_q=torch.tensor([0, 8], dtype=torch.int32),
        cu_seqlens_k=torch.tensor([0, 8], dtype=torch.int32),
        slot_mapping=torch.arange(total_tokens, dtype=torch.int32),
        block_length=4,
    )

    # Create mock Q, K, V
    q = torch.randn(total_tokens, 32, 128, dtype=torch.float32)
    k = torch.randn(total_tokens, 8, 128, dtype=torch.float32)
    v = torch.randn(total_tokens, 8, 128, dtype=torch.float32)

    output = backend.forward(q, k, v, ctx)

    assert output.shape == (total_tokens, 32, 128)
    mx.eval(backend.k_cache, backend.v_cache)  # Ensure no lazy eval issues


def test_prefill_multiple_sequences():
    """Test prefill with multiple sequences.

    Sequences: [8, 4] tokens, block_length=4
    """
    backend = make_backend()
    kv_cache = make_kv_cache()
    backend.initialize_cache(kv_cache)

    total_tokens = 12
    seq_lens = [8, 4]

    ctx = Context(
        run_type=RunType.PREFILL,
        cu_seqlens_q=torch.tensor([0, 8, 12], dtype=torch.int32),
        cu_seqlens_k=torch.tensor([0, 8, 12], dtype=torch.int32),
        slot_mapping=torch.arange(total_tokens, dtype=torch.int32),
        block_length=4,
    )

    q = torch.randn(total_tokens, 32, 128, dtype=torch.float32)
    k = torch.randn(total_tokens, 8, 128, dtype=torch.float32)
    v = torch.randn(total_tokens, 8, 128, dtype=torch.float32)

    output = backend.forward(q, k, v, ctx)

    assert output.shape == (total_tokens, 32, 128)
    mx.eval(backend.k_cache, backend.v_cache)


# ============================================================================
# Denoise Attention Tests
# ============================================================================

def test_denoise_basic():
    """Test basic denoise attention.

    Batch: 2 sequences, each with 8 cached tokens
    New block: 4 tokens per sequence
    """
    backend = MLXAttentionBackend(num_heads=32, num_kv_heads=8, head_dim=128)

    num_blocks = 16
    block_size = 4
    num_kv_heads = 8
    head_dim = 128

    # Initialize cache with some data
    cache_k = torch.randn(num_blocks, num_kv_heads, block_size, head_dim, dtype=torch.float32)
    cache_v = torch.randn(num_blocks, num_kv_heads, block_size, head_dim, dtype=torch.float32)
    kv_cache = torch.stack([cache_k, cache_v], dim=1)
    backend.initialize_cache(kv_cache)

    batch_size = 2
    num_tokens = batch_size * block_size  # 8 tokens total

    # Context: each seq has 8 cached tokens, 4 new tokens per seq
    context_lens = torch.tensor([8, 8], dtype=torch.int32)
    block_tables = torch.tensor([
        [0, 1, -1, -1],  # seq 0 uses blocks 0, 1
        [2, 3, -1, -1],  # seq 1 uses blocks 2, 3
    ], dtype=torch.int32)

    ctx = Context(
        run_type=RunType.DENOISE,
        context_lens=context_lens,
        block_tables=block_tables,
        block_length=block_size,
    )

    # New block data
    q = torch.randn(num_tokens, 32, 128, dtype=torch.float32)
    k = torch.randn(num_tokens, 8, 128, dtype=torch.float32)
    v = torch.randn(num_tokens, 8, 128, dtype=torch.float32)

    output = backend.forward(q, k, v, ctx)

    assert output.shape == (num_tokens, 32, 128)
    mx.eval(output)


def test_denoise_variable_context_lengths():
    """Test denoise with different context lengths per sequence.

    Batch: 3 sequences with [4, 8, 12] cached tokens
    """
    backend = MLXAttentionBackend(num_heads=32, num_kv_heads=8, head_dim=128)

    num_blocks = 16
    block_size = 4
    num_kv_heads = 8
    head_dim = 128

    cache_k = torch.randn(num_blocks, num_kv_heads, block_size, head_dim, dtype=torch.float32)
    cache_v = torch.randn(num_blocks, num_kv_heads, block_size, head_dim, dtype=torch.float32)
    kv_cache = torch.stack([cache_k, cache_v], dim=1)
    backend.initialize_cache(kv_cache)

    batch_size = 3
    num_tokens = batch_size * block_size

    context_lens = torch.tensor([4, 8, 12], dtype=torch.int32)
    block_tables = torch.tensor([
        [0, -1, -1, -1],
        [1, 2, -1, -1],
        [3, 4, 5, -1],
    ], dtype=torch.int32)

    ctx = Context(
        run_type=RunType.DENOISE,
        context_lens=context_lens,
        block_tables=block_tables,
        block_length=block_size,
    )

    q = torch.randn(num_tokens, 32, 128, dtype=torch.float32)
    k = torch.randn(num_tokens, 8, 128, dtype=torch.float32)
    v = torch.randn(num_tokens, 8, 128, dtype=torch.float32)

    output = backend.forward(q, k, v, ctx)

    assert output.shape == (num_tokens, 32, 128)
    mx.eval(output)


# ============================================================================
# GQA (Grouped Query Attention) Tests
# ============================================================================

def test_gqa_different_num_heads():
    """Test attention with different number of Q vs KV heads.

    num_heads = 32 (query)
    num_kv_heads = 8 (key/value)
    GQA ratio = 4
    """
    backend = MLXAttentionBackend(num_heads=32, num_kv_heads=8, head_dim=128)

    num_blocks = 8
    block_size = 4
    head_dim = 128

    cache_k = torch.randn(num_blocks, 8, block_size, head_dim, dtype=torch.float32)
    cache_v = torch.randn(num_blocks, 8, block_size, head_dim, dtype=torch.float32)
    kv_cache = torch.stack([cache_k, cache_v], dim=1)
    backend.initialize_cache(kv_cache)

    total_tokens = 16
    ctx = Context(
        run_type=RunType.PREFILL,
        cu_seqlens_q=torch.tensor([0, 16], dtype=torch.int32),
        cu_seqlens_k=torch.tensor([0, 16], dtype=torch.int32),
        slot_mapping=torch.arange(total_tokens, dtype=torch.int32),
        block_length=4,
    )

    q = torch.randn(total_tokens, 32, 128, dtype=torch.float32)
    k = torch.randn(total_tokens, 8, 128, dtype=torch.float32)
    v = torch.randn(total_tokens, 8, 128, dtype=torch.float32)

    output = backend.forward(q, k, v, ctx)

    # Output should have same number of heads as query
    assert output.shape == (total_tokens, 32, 128)


# ============================================================================
# Store KV Cache Tests
# ============================================================================

def test_store_kvcache_basic():
    """Test basic KV cache storage."""
    backend = make_backend()
    kv_cache = make_kv_cache()
    backend.initialize_cache(kv_cache)

    total_tokens = 8
    k = torch.randn(total_tokens, 8, 128, dtype=torch.float32)
    v = torch.randn(total_tokens, 8, 128, dtype=torch.float32)

    ctx = Context(
        run_type=RunType.PREFILL,
        cu_seqlens_q=torch.tensor([0, 8], dtype=torch.int32),
        cu_seqlens_k=torch.tensor([0, 8], dtype=torch.int32),
        slot_mapping=torch.arange(total_tokens, dtype=torch.int32),
        block_length=4,
    )

    backend.store_kvcache(k, v, ctx)

    # Force evaluation to ensure storage completed
    mx.eval(backend.k_cache, backend.v_cache)


# ============================================================================
# Integration Tests
# ============================================================================

def test_prefill_then_denoise():
    """Test a complete prefill followed by denoise workflow.

    This simulates:
    1. Prefill: encode 8 tokens (2 blocks)
    2. Denoise: process 2 new blocks with cached context
    """
    backend = MLXAttentionBackend(num_heads=32, num_kv_heads=8, head_dim=128)

    num_blocks = 16
    block_size = 4
    num_kv_heads = 8
    head_dim = 128

    cache_k = torch.randn(num_blocks, num_kv_heads, block_size, head_dim, dtype=torch.float32)
    cache_v = torch.randn(num_blocks, num_kv_heads, block_size, head_dim, dtype=torch.float32)
    kv_cache = torch.stack([cache_k, cache_v], dim=1)
    backend.initialize_cache(kv_cache)

    # Step 1: Prefill 8 tokens
    prefill_tokens = 8
    prefill_ctx = Context(
        run_type=RunType.PREFILL,
        cu_seqlens_q=torch.tensor([0, prefill_tokens], dtype=torch.int32),
        cu_seqlens_k=torch.tensor([0, prefill_tokens], dtype=torch.int32),
        slot_mapping=torch.arange(prefill_tokens, dtype=torch.int32),
        block_length=block_size,
    )

    q_prefill = torch.randn(prefill_tokens, 32, 128, dtype=torch.float32)
    k_prefill = torch.randn(prefill_tokens, 8, 128, dtype=torch.float32)
    v_prefill = torch.randn(prefill_tokens, 8, 128, dtype=torch.float32)

    prefill_output = backend.forward(q_prefill, k_prefill, v_prefill, prefill_ctx)
    assert prefill_output.shape == (prefill_tokens, 32, 128)

    # Step 2: Denoise 4 tokens (one new block)
    denoise_tokens = 4
    context_lens = torch.tensor([prefill_tokens], dtype=torch.int32)
    block_tables = torch.tensor([[0, 1, -1, -1]], dtype=torch.int32)

    denoise_ctx = Context(
        run_type=RunType.DENOISE,
        context_lens=context_lens,
        block_tables=block_tables,
        block_length=block_size,
    )

    q_denoise = torch.randn(denoise_tokens, 32, 128, dtype=torch.float32)
    k_denoise = torch.randn(denoise_tokens, 8, 128, dtype=torch.float32)
    v_denoise = torch.randn(denoise_tokens, 8, 128, dtype=torch.float32)

    denoise_output = backend.forward(q_denoise, k_denoise, v_denoise, denoise_ctx)
    assert denoise_output.shape == (denoise_tokens, 32, 128)


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
