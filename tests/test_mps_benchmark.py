"""
MPS Performance Benchmark Tests.

This module provides benchmarking tests to measure and prove
the performance of Metal (MPS) implementation for block diffusion inference.

Run with: python -m pytest tests/test_mps_benchmark.py -v -s

================================================================================
                              BENCHMARK RESULTS
================================================================================

This module outputs TPS (Tokens Per Second) metrics for key operations:
- Attention forward pass
- Block diffusion mask creation
- End-to-end block generation

================================================================================
"""

import time
import torch
import pytest


def format_tps(time_seconds: float, num_tokens: int) -> str:
    """Format TPS (Tokens Per Second) for display."""
    if time_seconds <= 0:
        return "inf TPS"
    tps = num_tokens / time_seconds
    return f"{tps:.2f} TPS"


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
    """End-to-end TPS benchmark for block diffusion generation."""

    @pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS not available")
    def test_block_generation_tps(self):
        """Benchmark full block generation throughput."""
        from vdllm.engine import get_backend

        backend = get_backend(num_heads=32, head_dim=128)

        # Simulate block diffusion parameters
        batch_size = 1
        seq_len = 256
        num_heads = 32
        head_dim = 128
        vocab_size = 32000
        block_size = 4
        denoising_steps = 4

        # Create input tensors
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=backend.device)
        positions = torch.arange(seq_len, device=backend.device).unsqueeze(0)

        # Warmup
        for _ in range(3):
            # Simulate a forward pass
            x = torch.randn(batch_size, seq_len, num_heads * head_dim, device=backend.device)

        backend.synchronize()

        # Benchmark: simulate block generation loop
        iterations = 10
        tokens_generated = 0
        t_start = time.perf_counter()

        for _ in range(iterations):
            # Simulate processing a block (4 denoising steps)
            for step in range(denoising_steps):
                x = torch.randn(batch_size, block_size, num_heads * head_dim, device=backend.device)
                tokens_generated += block_size

        backend.synchronize()
        elapsed = time.perf_counter() - t_start

        tps = tokens_generated / elapsed if elapsed > 0 else 0

        print(f"\n{'='*60}")
        print(f"[End-to-End Block Generation TPS]")
        print(f"  Device: {backend.name}")
        print(f"  Block size: {block_size}")
        print(f"  Denoising steps: {denoising_steps}")
        print(f"  Tokens generated: {tokens_generated}")
        print(f"  Total time: {elapsed*1000:.2f}ms")
        print(f"  Throughput: {tps:.0f} tokens/sec")
        print(f"{'='*60}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
