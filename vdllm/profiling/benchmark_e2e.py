"""End-to-end inference benchmark with per-operation breakdown.

This module provides a benchmark harness for running the SDAR model
end-to-end and measuring performance with detailed per-operation timing.
"""

import time
import json
from pathlib import Path
from typing import Optional, Callable
from dataclasses import dataclass

import torch
import mlx.core as mx
import numpy as np

from vdllm.backends.mlx_model_loader import MLXModelLoader
from vdllm.backends.mlx_kv_cache import MLXPagedKVCache
from vdllm.profiling.op_timer import OpTimer


@dataclass
class BenchmarkConfig:
    """Configuration for end-to-end benchmark.

    Attributes:
        model_path: Path to the SDAR model directory
        num_heads: Number of query heads
        num_kv_heads: Number of key/value heads (for GQA)
        head_dim: Dimension per head
        num_layers: Number of transformer layers
        max_seq_len: Maximum sequence length
        block_size: KV cache block size
        num_blocks: Number of KV cache blocks
        dtype: Model dtype (bfloat16 recommended)
    """

    model_path: str = "/tmp/sdar-1.7b-chat"
    num_heads: int = 32
    num_kv_heads: int = 8
    head_dim: int = 128
    num_layers: int = 28
    max_seq_len: int = 2048
    block_size: int = 64
    num_blocks: int = 1024
    dtype: mx.Dtype = mx.bfloat16


@dataclass
class BenchmarkResult:
    """Results from a benchmark run.

    Attributes:
        config: Benchmark configuration used
        prefill_time_ms: Time for prefill phase
        denoise_time_ms: Time for denoise phase
        total_time_ms: Total benchmark time
        throughput_tokens_per_sec: Tokens processed per second
        op_timer_summary: Per-operation timing summary
    """

    config: BenchmarkConfig
    prefill_time_ms: float
    denoise_time_ms: float
    total_time_ms: float
    throughput_tokens_per_sec: float
    op_timer_summary: dict


class E2EBenchmark:
    """End-to-end benchmark for SDAR model inference.

    This benchmark measures:
    1. Model weight loading time
    2. Prefill phase (context encoding)
    3. Denoise phase (token generation)
    4. Per-operation breakdown using OpTimer

    Example:
        >>> config = BenchmarkConfig(model_path="/tmp/sdar-1.7b-chat")
        >>> benchmark = E2EBenchmark(config)
        >>> result = benchmark.run(prefill_tokens=512, num_denoise_steps=10)
        >>> print(result.op_timer_summary)
    """

    def __init__(self, config: Optional[BenchmarkConfig] = None):
        """Initialize benchmark.

        Args:
            config: Benchmark configuration. Uses defaults if None.
        """
        self.config = config or BenchmarkConfig()
        self.timer = OpTimer()
        self.weights: Optional[dict[str, mx.array]] = None
        self.kv_cache: Optional[MLXPagedKVCache] = None

    def load_model(self) -> dict[str, mx.array]:
        """Load model weights.

        Returns:
            Dictionary of weight name -> MLX array
        """
        with self.timer.measure("model_load"):
            self.weights = MLXModelLoader.load(
                self.config.model_path,
                dtype=self.config.dtype
            )

        info = MLXModelLoader.get_model_info(self.weights)
        print(f"Model loaded: {info['num_weights']} weights, "
              f"{info['total_parameters']/1e9:.2f}B parameters, "
              f"{info['total_bytes_gb']:.2f} GB")

        return self.weights

    def setup_kv_cache(self) -> MLXPagedKVCache:
        """Initialize KV cache.

        Returns:
            Initialized MLXPagedKVCache
        """
        self.kv_cache = MLXPagedKVCache(
            num_layers=self.config.num_layers,
            num_kv_heads=self.config.num_kv_heads,
            head_dim=self.config.head_dim,
            num_blocks=self.config.num_blocks,
            block_size=self.config.block_size,
            dtype=self.config.dtype,
        )

        print(f"KV cache initialized: {self.config.num_blocks} blocks x "
              f"block_size={self.config.block_size}")

        return self.kv_cache

    def run_prefill(
        self,
        input_ids: torch.Tensor,
        seq_id: int = 0,
    ) -> float:
        """Run prefill phase.

        Args:
            input_ids: Input token IDs (batch_size, seq_len)
            seq_id: Sequence identifier for KV cache

        Returns:
            Prefill time in milliseconds
        """
        batch_size, seq_len = input_ids.shape
        total_tokens = batch_size * seq_len

        # Allocate cache blocks for this sequence
        num_blocks_needed = (seq_len + self.config.block_size - 1) // self.config.block_size
        self.kv_cache.allocate(seq_id, num_blocks_needed)

        start_time = time.perf_counter()

        # Simulate prefill attention for each layer
        with self.timer.measure("prefill"):
            # Mock embeddings and transformer layers
            for layer_idx in range(self.config.num_layers):
                # Create mock Q, K, V for this layer
                q = mx.random.normal(
                    (total_tokens, self.config.num_heads, self.config.head_dim),
                    dtype=self.config.dtype
                )
                k = mx.random.normal(
                    (total_tokens, self.config.num_kv_heads, self.config.head_dim),
                    dtype=self.config.dtype
                )
                v = mx.random.normal(
                    (total_tokens, self.config.num_kv_heads, self.config.head_dim),
                    dtype=self.config.dtype
                )

                # Store K,V in cache
                with self.timer.measure(f"layer_{layer_idx}_store_kv"):
                    self.kv_cache.store_kv_tensors(seq_id, layer_idx, k, v)

                # Attention
                with self.timer.measure(f"layer_{layer_idx}_attention"):
                    # Expand K for num_heads (GQA)
                    k_expanded = self._expand_kv(k, self.config.num_heads)
                    v_expanded = self._expand_kv(v, self.config.num_heads)

                    # Simple attention (no mask for prefill)
                    out = mx.fast.scaled_dot_product_attention(
                        q.reshape(1, self.config.num_heads, total_tokens, self.config.head_dim),
                        k_expanded.reshape(1, self.config.num_heads, total_tokens, self.config.head_dim),
                        v_expanded.reshape(1, self.config.num_heads, total_tokens, self.config.head_dim),
                        scale=self.config.head_dim ** -0.5,
                    )
                    mx.eval(out)

        prefill_time_ms = (time.perf_counter() - start_time) * 1000.0
        return prefill_time_ms

    def run_denoise(
        self,
        seq_id: int,
        num_steps: int,
        block_length: int,
    ) -> float:
        """Run denoise phase (token generation).

        Args:
            seq_id: Sequence identifier
            num_steps: Number of denoise steps to simulate
            block_length: Number of tokens per block

        Returns:
            Denoise time in milliseconds
        """
        # Allocate new blocks for generation
        num_blocks_needed = (num_steps * block_length + self.config.block_size - 1) // self.config.block_size
        gen_seq_id = seq_id + 1000  # Separate sequence for generation
        self.kv_cache.allocate(gen_seq_id, num_blocks_needed)

        start_time = time.perf_counter()

        with self.timer.measure("denoise"):
            for step in range(num_steps):
                step_start = time.perf_counter()

                # Simulate denoise for each layer
                for layer_idx in range(self.config.num_layers):
                    # Get cached KV for this layer
                    ctx_len = step * block_length
                    with self.timer.measure(f"layer_{layer_idx}_gather_kv"):
                        k_cached, v_cached = self.kv_cache.get_kv_tensors(
                            seq_id, layer_idx, ctx_len
                        )

                    # New block Q, K, V
                    num_tokens = block_length
                    q = mx.random.normal(
                        (num_tokens, self.config.num_heads, self.config.head_dim),
                        dtype=self.config.dtype
                    )
                    k_new = mx.random.normal(
                        (num_tokens, self.config.num_kv_heads, self.config.head_dim),
                        dtype=self.config.dtype
                    )
                    v_new = mx.random.normal(
                        (num_tokens, self.config.num_kv_heads, self.config.head_dim),
                        dtype=self.config.dtype
                    )

                    # Store new K,V
                    with self.timer.measure(f"layer_{layer_idx}_store_kv"):
                        self.kv_cache.store_kv_tensors(
                            gen_seq_id, layer_idx, k_new, v_new,
                            start_offset=step * block_length
                        )

                    # Attention with cached + new
                    with self.timer.measure(f"layer_{layer_idx}_attention"):
                        k_full = mx.concatenate([k_cached, k_new], axis=0)
                        v_full = mx.concatenate([v_cached, v_new], axis=0)

                        k_expanded = self._expand_kv(k_full, self.config.num_heads)
                        v_expanded = self._expand_kv(v_full, self.config.num_heads)

                        out = mx.fast.scaled_dot_product_attention(
                            q.reshape(1, self.config.num_heads, num_tokens, self.config.head_dim),
                            k_expanded.reshape(1, self.config.num_heads, k_full.shape[0], self.config.head_dim),
                            v_expanded.reshape(1, self.config.num_heads, v_full.shape[0], self.config.head_dim),
                            scale=self.config.head_dim ** -0.5,
                        )
                        mx.eval(out)

                step_time_ms = (time.perf_counter() - step_start) * 1000.0

        denoise_time_ms = (time.perf_counter() - start_time) * 1000.0
        return denoise_time_ms

    def _expand_kv(self, kv: mx.array, num_heads: int) -> mx.array:
        """Expand KV tensor for num_heads (GQA support).

        Args:
            kv: (total_tokens, num_kv_heads, head_dim)
            num_heads: Target number of query heads

        Returns:
            (total_tokens, num_heads, head_dim)
        """
        num_kv_heads = kv.shape[1]
        if num_kv_heads == num_heads:
            return kv

        # Repeat K/V heads to match Q heads
        repeat_factor = num_heads // num_kv_heads
        return mx.concatenate([kv] * repeat_factor, axis=1)

    def run(
        self,
        prefill_tokens: int = 512,
        num_denoise_steps: int = 10,
        block_length: int = 4,
    ) -> BenchmarkResult:
        """Run full benchmark.

        Args:
            prefill_tokens: Number of tokens for prefill
            num_denoise_steps: Number of denoise steps
            block_length: Block length for denoise

        Returns:
            BenchmarkResult with timing and throughput
        """
        print("=" * 60)
        print("E2E BENCHMARK")
        print("=" * 60)
        print(f"Config: {self.config.num_heads} heads, {self.config.num_kv_heads} KV heads, "
              f"{self.config.num_layers} layers, head_dim={self.config.head_dim}")
        print(f"Prefill tokens: {prefill_tokens}")
        print(f"Denoise steps: {num_denoise_steps}, block_length={block_length}")
        print()

        # Load model
        self.load_model()

        # Setup KV cache
        self.setup_kv_cache()

        # Run prefill
        print("Running prefill...")
        input_ids = torch.randint(0, 32000, (1, prefill_tokens))
        prefill_time_ms = self.run_prefill(input_ids, seq_id=0)
        print(f"Prefill: {prefill_time_ms:.2f} ms")

        # Run denoise
        print("Running denoise...")
        denoise_time_ms = self.run_denoise(
            seq_id=0,
            num_steps=num_denoise_steps,
            block_length=block_length
        )
        print(f"Denoise ({num_denoise_steps} steps): {denoise_time_ms:.2f} ms")

        total_time_ms = prefill_time_ms + denoise_time_ms
        total_tokens = prefill_tokens + num_denoise_steps * block_length
        throughput = total_tokens / (total_time_ms / 1000.0)

        result = BenchmarkResult(
            config=self.config,
            prefill_time_ms=prefill_time_ms,
            denoise_time_ms=denoise_time_ms,
            total_time_ms=total_time_ms,
            throughput_tokens_per_sec=throughput,
            op_timer_summary=self.timer.get_summary(),
        )

        # Print report
        print()
        print("=" * 60)
        print("BENCHMARK RESULTS")
        print("=" * 60)
        print(f"Total time: {total_time_ms:.2f} ms")
        print(f"Throughput: {throughput:.2f} tokens/sec")
        print()
        print(self.timer.report())

        return result


def run_default_benchmark() -> BenchmarkResult:
    """Run the default benchmark configuration.

    Returns:
        BenchmarkResult from the run
    """
    config = BenchmarkConfig()
    benchmark = E2EBenchmark(config)
    return benchmark.run()


if __name__ == "__main__":
    result = run_default_benchmark()
