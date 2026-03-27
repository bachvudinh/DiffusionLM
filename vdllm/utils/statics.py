"""
KV Cache Statistics and Estimation.

This module provides utilities for estimating KV cache memory usage.

Based on JetEngine's statics.py:
https://github.com/Jet-Astra/SDAR/blob/main/jetengine/utils/statics.py

================================================================================
                              USAGE
================================================================================

    from vdllm.utils import estimate_kv_cache_usage

    num_blocks, total_bytes = estimate_kv_cache_usage(
        num_layers=32,
        num_kv_heads=8,
        head_dim=128,
        max_model_len=4096,
        max_num_seqs=1,
        block_size=16,
        dtype=torch.float16,
    )
    print(f"Need {num_blocks} blocks, {total_bytes / 1e9:.2f} GB")

"""

import math
from typing import Optional

import torch


def estimate_kv_cache_usage(
    num_layers: int,
    num_kv_heads: int,
    head_dim: int,
    max_model_len: int,
    max_num_seqs: int = 1,
    block_size: int = 16,
    dtype: torch.dtype = torch.float16,
    tensor_parallel_size: int = 1,
) -> tuple[int, int]:
    """
    Estimate KV cache usage for a given configuration.

    Args:
        num_layers: Number of transformer layers
        num_kv_heads: Number of key/value attention heads
        head_dim: Dimension of each attention head
        max_model_len: Maximum model sequence length
        max_num_seqs: Maximum number of sequences in batch
        block_size: Size of each KV cache block
        dtype: Data type for KV cache
        tensor_parallel_size: Tensor parallel size

    Returns:
        Tuple of (total_blocks, total_bytes)
    """
    tokens_per_sequence = max_model_len
    blocks_per_sequence = math.ceil(tokens_per_sequence / block_size)
    total_blocks = blocks_per_sequence * max_num_seqs

    num_kv_heads = num_kv_heads // tensor_parallel_size

    # Calculate bytes per block: 2 (K+V) * layers * block_size * kv_heads * head_dim * dtype_size
    block_bytes = (
        2
        * num_layers
        * block_size
        * num_kv_heads
        * head_dim
        * dtype.itemsize
    )

    total_bytes = total_blocks * block_bytes

    return total_blocks, total_bytes


def actual_estimate_kv_cache_usage(
    max_lengths: int,
    batch_size: int,
    num_layers: int,
    num_kv_heads: int,
    head_dim: int,
    block_size: int = 16,
    dtype: torch.dtype = torch.float16,
    tensor_parallel_size: int = 1,
) -> tuple[int, int]:
    """
    Estimate KV cache usage for actual generation scenario.

    Args:
        max_lengths: Maximum sequence length for generation
        batch_size: Batch size
        num_layers: Number of transformer layers
        num_kv_heads: Number of key/value attention heads
        head_dim: Dimension of each attention head
        block_size: Size of each KV cache block
        dtype: Data type for KV cache
        tensor_parallel_size: Tensor parallel size

    Returns:
        Tuple of (total_blocks, total_bytes)
    """
    tokens_per_sequence = max_lengths
    blocks_per_sequence = math.ceil(tokens_per_sequence / block_size)
    total_blocks = blocks_per_sequence * batch_size

    num_kv_heads = num_kv_heads // tensor_parallel_size

    block_bytes = (
        2
        * num_layers
        * block_size
        * num_kv_heads
        * head_dim
        * dtype.itemsize
    )

    total_bytes = total_blocks * block_bytes

    return total_blocks, total_bytes


def format_bytes(num_bytes: int) -> str:
    """Format bytes as human-readable string."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if abs(num_bytes) < 1024.0:
            return f"{num_bytes:.2f} {unit}"
        num_bytes /= 1024.0
    return f"{num_bytes:.2f} PB"
