"""
MPS Performance Benchmark Tests.

This module provides benchmarking tests to measure and prove
the performance of Metal (MPS) implementation for block diffusion inference.

Run with: python -m pytest tests/test_mps_benchmark.py -v -s

================================================================================
                              BENCHMARK RESULTS
================================================================================

This module outputs TPS (Tokens Per Second) metrics for:
- Model loading time
- Token generation/inference time
- End-to-end TPS (including model loading)

================================================================================
"""

import time
import torch
import pytest
from dataclasses import dataclass


def format_tps(time_seconds: float, num_tokens: int) -> str:
    """Format TPS (Tokens Per Second) for display."""
    if time_seconds <= 0:
        return "inf TPS"
    tps = num_tokens / time_seconds
    return f"{tps:.2f} tok/s"


@dataclass
class TinySDARConfig:
    """Minimal SDAR config for fast testing."""
    vocab_size: int = 32000
    hidden_size: int = 512
    intermediate_size: int = 1408
    num_hidden_layers: int = 4
    num_attention_heads: int = 8
    num_key_value_heads: int = 4  # GQA
    head_dim: int = 64
    hidden_act: str = "silu"
    rms_norm_eps: float = 1e-6
    max_position_embeddings: int = 2048
    rope_theta: float = 10000.0
    rope_scaling: int = None
    attention_bias: bool = False
    attention_dropout: float = 0.0
    use_sliding_window: bool = False
    sliding_window: int = 4096
    max_window_layers: int = 28
    block_size: int = 4
    mask_token_id: int = 151669
    torch_dtype: str = "float32"
    initializer_range: float = 0.02
    use_cache: bool = True
    tie_word_embeddings: bool = False

    @property
    def num_key_value_groups(self) -> int:
        return self.num_attention_heads // self.num_key_value_heads


class TestDeviceDetection:
    """Test device detection and backend selection."""

    def test_device_detection(self):
        """Test that MPS device is properly detected."""
        if torch.backends.mps.is_available():
            expected = "mps"
        else:
            expected = "cpu"

        from vdllm.engine.device import get_device_name
        device_name = get_device_name()

        assert device_name in ("metal", "mps", "cpu", "triton"), f"Unknown device: {device_name}"
        print(f"\n[Device Detection] Detected: {device_name}")

    def test_backend_selection(self):
        """Test backend selection logic."""
        from vdllm.engine import get_backend

        backend = get_backend(num_heads=32, head_dim=128)
        print(f"\n[Backend Selection] Selected: {backend.name} on {backend.device}")


class TestAttentionPerformance:
    """Benchmark attention forward pass performance."""

    @pytest.fixture
    def device(self):
        """Get MPS device if available, else CPU."""
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def test_attention_forward_speed(self, device):
        """Benchmark attention forward pass."""
        batch_size = 1
        num_heads = 16
        seq_len = 256
        head_dim = 128

        # Create tensors
        q = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device)
        k = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device)
        v = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device)

        # Warmup
        for _ in range(3):
            _ = torch.nn.functional.scaled_dot_product_attention(q, k, v)

        if device.type == "mps":
            torch.mps.synchronize()

        # Benchmark
        iterations = 10
        start = time.perf_counter()
        for _ in range(iterations):
            _ = torch.nn.functional.scaled_dot_product_attention(q, k, v)

        if device.type == "mps":
            torch.mps.synchronize()

        elapsed = time.perf_counter() - start
        avg_time = elapsed / iterations

        # Calculate tokens per second for attention
        tokens_processed = batch_size * seq_len * iterations
        tps = tokens_processed / elapsed if elapsed > 0 else 0

        print(f"\n[Attention Forward] Device: {device.type}")
        print(f"  Sequence length: {seq_len}")
        print(f"  Average time: {avg_time*1000:.2f}ms")
        print(f"  Throughput: {tps:.0f} tokens/sec")

        assert avg_time < 1.0, f"Attention too slow: {avg_time*1000:.2f}ms"

    def test_large_seq_attention(self, device):
        """Benchmark attention with larger sequences (block diffusion typical)."""
        batch_size = 1
        num_heads = 32
        seq_len = 512  # Typical block diffusion sequence
        head_dim = 128

        q = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device)
        k = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device)
        v = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device)

        # Warmup
        for _ in range(5):
            _ = torch.nn.functional.scaled_dot_product_attention(q, k, v)

        if device.type == "mps":
            torch.mps.synchronize()

        # Benchmark
        iterations = 20
        start = time.perf_counter()
        for _ in range(iterations):
            _ = torch.nn.functional.scaled_dot_product_attention(q, k, v)

        if device.type == "mps":
            torch.mps.synchronize()

        elapsed = time.perf_counter() - start
        avg_time = elapsed / iterations

        tokens_processed = batch_size * seq_len * iterations
        tps = tokens_processed / elapsed if elapsed > 0 else 0

        print(f"\n[Large Attention] Device: {device.type}")
        print(f"  Sequence length: {seq_len}")
        print(f"  Average time: {avg_time*1000:.2f}ms")
        print(f"  Throughput: {tps:.0f} tokens/sec")


class TestMaskCreation:
    """Benchmark block diffusion mask creation."""

    @pytest.fixture
    def device(self):
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def test_block_diffusion_mask_creation(self, device):
        """Benchmark block diffusion mask creation."""
        seq_len = 1024
        block_size = 4

        # Warmup
        for _ in range(3):
            num_blocks = seq_len // block_size
            block_mask = torch.tril(torch.ones(num_blocks, num_blocks, device=device, dtype=torch.bool))
            mask = block_mask.repeat_interleave(block_size, dim=0).repeat_interleave(block_size, dim=1)

        # Benchmark
        iterations = 10
        start = time.perf_counter()
        for _ in range(iterations):
            num_blocks = seq_len // block_size
            block_mask = torch.tril(torch.ones(num_blocks, num_blocks, device=device, dtype=torch.bool))
            mask = block_mask.repeat_interleave(block_size, dim=0).repeat_interleave(block_size, dim=1)

        if device.type == "mps":
            torch.mps.synchronize()

        elapsed = time.perf_counter() - start
        avg_time = elapsed / iterations

        # Calculate TPS: how many masks can we create per second
        masks_per_sec = iterations / elapsed if elapsed > 0 else 0

        print(f"\n[Mask Creation] Device: {device.type}")
        print(f"  Sequence length: {seq_len}, Block size: {block_size}")
        print(f"  Average time: {avg_time*1000:.2f}ms")
        print(f"  Throughput: {masks_per_sec:.0f} masks/sec")

        assert avg_time < 0.1, f"Mask creation too slow: {avg_time*1000:.2f}ms"


class TestMemoryAllocation:
    """Benchmark memory allocation."""

    @pytest.fixture
    def device(self):
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def test_memory_allocation(self, device):
        """Benchmark memory allocation speed."""
        sizes = [1024, 4096]

        for size in sizes:
            # Warmup
            for _ in range(3):
                x = torch.randn(size, size, device=device)

            # Benchmark
            iterations = 5
            start = time.perf_counter()
            for _ in range(iterations):
                x = torch.randn(size, size, device=device)

            if device.type == "mps":
                torch.mps.synchronize()

            elapsed = time.perf_counter() - start
            avg_time = elapsed / iterations

            print(f"\n[Memory Allocation {size}x{size}] Device: {device.type}")
            print(f"  Average time: {avg_time*1000:.2f}ms")


class TestMPSCorrectness:
    """Test MPS produces correct results."""

    @pytest.fixture
    def mps_device(self):
        if torch.backends.mps.is_available():
            return torch.device("mps")
        pytest.skip("MPS not available")

    def test_attention_output_matches_cpu(self, mps_device):
        """Verify MPS attention produces same results as CPU."""
        batch_size = 1
        num_heads = 8
        seq_len = 64
        head_dim = 64

        q_cpu = torch.randn(batch_size, num_heads, seq_len, head_dim)
        k_cpu = torch.randn(batch_size, num_heads, seq_len, head_dim)
        v_cpu = torch.randn(batch_size, num_heads, seq_len, head_dim)

        q_mps = q_cpu.to(mps_device)
        k_mps = k_cpu.to(mps_device)
        v_mps = v_cpu.to(mps_device)

        out_cpu = torch.nn.functional.scaled_dot_product_attention(q_cpu, k_cpu, v_cpu)
        out_mps = torch.nn.functional.scaled_dot_product_attention(q_mps, k_mps, v_mps)

        torch.mps.synchronize()

        # Results should be close (within numerical tolerance)
        assert torch.allclose(out_cpu, out_mps.to("cpu"), atol=1e-5)


class TestSpeedComparison:
    """Compare MPS vs CPU speed for key operations."""

    @pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS not available")
    def test_mps_vs_cpu_attention(self):
        """Compare attention speed: MPS vs CPU."""
        batch_size = 1
        num_heads = 32
        seq_len = 512
        head_dim = 128

        results = {}

        for device_type in ["cpu", "mps"]:
            device = torch.device(device_type)

            q = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device)
            k = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device)
            v = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device)

            # Warmup
            for _ in range(5):
                _ = torch.nn.functional.scaled_dot_product_attention(q, k, v)

            if device_type == "mps":
                torch.mps.synchronize()

            # Benchmark
            iterations = 20
            start = time.perf_counter()
            for _ in range(iterations):
                _ = torch.nn.functional.scaled_dot_product_attention(q, k, v)

            if device_type == "mps":
                torch.mps.synchronize()

            elapsed = time.perf_counter() - start
            avg_time = elapsed / iterations
            results[device_type] = avg_time

        speedup = results["cpu"] / results["mps"]
        tokens_processed = batch_size * seq_len * iterations

        print(f"\n{'='*60}")
        print(f"[Attention Speed Comparison (seq_len={seq_len})]")
        print(f"  CPU time: {results['cpu']*1000:.2f}ms")
        print(f"  CPU TPS:  {format_tps(results['cpu'], batch_size * seq_len)}")
        print(f"  MPS time: {results['mps']*1000:.2f}ms")
        print(f"  MPS TPS:  {format_tps(results['mps'], batch_size * seq_len)}")
        print(f"  Speedup: {speedup:.2f}x")

        if speedup > 1.0:
            print(f"  Result: MPS is {speedup:.2f}x faster than CPU")
        else:
            print(f"  Result: CPU is {1/speedup:.2f}x faster than MPS")
        print(f"{'='*60}")


class TestEndToEndTPS:
    """End-to-end TPS benchmark: model loading + inference."""

    @pytest.fixture
    def device(self):
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def test_model_load_time(self, device):
        """Benchmark model instantiation/loading time."""
        from vdllm.models import SDARForCausalLM, SDARConfig

        config = SDARConfig(
            vocab_size=32000,
            hidden_size=512,
            intermediate_size=1408,
            num_hidden_layers=4,
            num_attention_heads=8,
            num_key_value_heads=4,
            head_dim=64,
        )

        # Warmup
        _ = SDARForCausalLM(config)

        # Benchmark model creation
        iterations = 5
        start = time.perf_counter()
        for _ in range(iterations):
            model = SDARForCausalLM(config)

        model_creation_time = (time.perf_counter() - start) / iterations

        print(f"\n[Model Creation]")
        print(f"  Device: {device.type}")
        print(f"  Config: {config.num_hidden_layers}L, hidden={config.hidden_size}")
        print(f"  Avg creation time: {model_creation_time*1000:.2f}ms")

    def test_forward_pass_tps(self, device):
        """Benchmark forward pass throughput."""
        from vdllm.models import SDARForCausalLM, SDARConfig

        config = SDARConfig(
            vocab_size=32000,
            hidden_size=512,
            intermediate_size=1408,
            num_hidden_layers=4,
            num_attention_heads=8,
            num_key_value_heads=4,
            head_dim=64,
        )

        model = SDARForCausalLM(config).to(device)
        model.eval()

        batch_size = 1
        seq_len = 128
        vocab_size = config.vocab_size

        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
        positions = torch.arange(seq_len, device=device).unsqueeze(0)

        # Warmup
        for _ in range(3):
            with torch.no_grad():
                _ = model(input_ids=input_ids, positions=positions)

        if device.type == "mps":
            torch.mps.synchronize()

        # Benchmark forward passes
        iterations = 20
        start = time.perf_counter()
        tokens_processed = 0
        for _ in range(iterations):
            with torch.no_grad():
                _ = model(input_ids=input_ids, positions=positions)
            tokens_processed += batch_size * seq_len

        if device.type == "mps":
            torch.mps.synchronize()

        elapsed = time.perf_counter() - start
        tps = tokens_processed / elapsed if elapsed > 0 else 0

        print(f"\n[Forward Pass TPS]")
        print(f"  Device: {device.type}")
        print(f"  Batch: {batch_size}, Seq Len: {seq_len}")
        print(f"  Tokens processed: {tokens_processed}")
        print(f"  Total time: {elapsed*1000:.2f}ms")
        print(f"  Throughput: {tps:.0f} tokens/sec")

    @pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS not available")
    def test_mps_vs_cpu_end_to_end(self):
        """Compare MPS vs CPU end-to-end: model load + forward pass."""
        from vdllm.models import SDARForCausalLM, SDARConfig

        config = SDARConfig(
            vocab_size=32000,
            hidden_size=512,
            intermediate_size=1408,
            num_hidden_layers=4,
            num_attention_heads=8,
            num_key_value_heads=4,
            head_dim=64,
        )

        batch_size = 1
        seq_len = 128
        iterations = 10

        results = {}

        for device_name in ["cpu", "mps"]:
            device = torch.device(device_name)

            # Measure model loading
            load_start = time.perf_counter()
            model = SDARForCausalLM(config).to(device)
            model.eval()
            load_time = time.perf_counter() - load_start

            # Prepare input
            input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len), device=device)
            positions = torch.arange(seq_len, device=device).unsqueeze(0)

            # Warmup
            for _ in range(3):
                with torch.no_grad():
                    _ = model(input_ids=input_ids, positions=positions)

            if device_name == "mps":
                torch.mps.synchronize()

            # Benchmark
            tokens_processed = 0
            inf_start = time.perf_counter()
            for _ in range(iterations):
                with torch.no_grad():
                    _ = model(input_ids=input_ids, positions=positions)
                tokens_processed += batch_size * seq_len

            if device_name == "mps":
                torch.mps.synchronize()

            inf_time = time.perf_counter() - inf_start
            total_time = load_time + inf_time

            results[device_name] = {
                "load_time": load_time,
                "inf_time": inf_time,
                "total_time": total_time,
                "tokens": tokens_processed,
                "tps": tokens_processed / inf_time if inf_time > 0 else 0,
                "tps_total": tokens_processed / total_time if total_time > 0 else 0,
            }

        print(f"\n{'='*60}")
        print(f"[End-to-End TPS Comparison]")
        print(f"  Model: {config.num_hidden_layers}L, hidden={config.hidden_size}, seq_len={seq_len}")
        print(f"{'='*60}")
        print(f"  CPU:")
        print(f"    Model load: {results['cpu']['load_time']*1000:.2f}ms")
        print(f"    Inference:  {results['cpu']['inf_time']*1000:.2f}ms ({results['cpu']['tps']:.0f} tok/s)")
        print(f"    Total:      {results['cpu']['total_time']*1000:.2f}ms ({results['cpu']['tps_total']:.0f} tok/s incl. load)")
        print(f"  MPS:")
        print(f"    Model load: {results['mps']['load_time']*1000:.2f}ms")
        print(f"    Inference:  {results['mps']['inf_time']*1000:.2f}ms ({results['mps']['tps']:.0f} tok/s)")
        print(f"    Total:      {results['mps']['total_time']*1000:.2f}ms ({results['mps']['tps_total']:.0f} tok/s incl. load)")

        speedup = results['cpu']['inf_time'] / results['mps']['inf_time']
        print(f"{'='*60}")
        print(f"  MPS Speedup (inference only): {speedup:.2f}x")
        print(f"{'='*60}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
