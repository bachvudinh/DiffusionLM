"""KV cache memory estimation utilities.

Derived from JetEngine by Yihan Bian et al.
Reference: https://github.com/Labman42/JetEngine

Computes worst-case and actual KV cache memory requirements
based on model config, batch size, and sequence length.
"""

import math
from vdllm.config import Config


def _estimate_kv_cache_usage(config: Config) -> tuple[int, int]:
    """Estimate worst-case KV cache usage (all seqs at max length).

    Input:
        config: Config — engine configuration

    Output:
        (total_blocks, total_bytes) tuple
    """
    tokens_per_sequence = config.max_model_len
    blocks_per_sequence = math.ceil(tokens_per_sequence / config.kvcache_block_size)
    total_blocks = blocks_per_sequence * config.max_num_seqs

    num_kv_heads = config.num_key_value_heads // config.tensor_parallel_size
    block_bytes = (
        2 * config.num_hidden_layers * config.kvcache_block_size
        * num_kv_heads * config.head_dim * config.torch_dtype.itemsize
    )
    total_bytes = total_blocks * block_bytes
    return total_blocks, total_bytes


def _actual_estimate_kv_cache_usage(max_lengths: int, batch_size: int,
                                     config: Config) -> tuple[int, int]:
    """Estimate KV cache usage for a specific batch size and max length.

    Input:
        max_lengths: int — maximum sequence length
        batch_size:  int — number of concurrent sequences
        config:      Config — engine configuration

    Output:
        (total_blocks, total_bytes) tuple
    """
    tokens_per_sequence = max_lengths
    blocks_per_sequence = math.ceil(tokens_per_sequence / config.kvcache_block_size)
    total_blocks = blocks_per_sequence * batch_size

    num_kv_heads = config.num_key_value_heads // config.tensor_parallel_size
    block_bytes = (
        2 * config.num_hidden_layers * config.kvcache_block_size
        * num_kv_heads * config.head_dim * config.torch_dtype.itemsize
    )
    total_bytes = total_blocks * block_bytes
    return total_blocks, total_bytes
