"""MLX/Metal profiling utilities for performance analysis.

Professional MLX profiling workflow:
1. mx.synchronize() + time.perf_counter() for wall-clock timing
2. mx.metal.start_capture() for GPU kernel timeline (view in Xcode)
3. mx.get_active_memory() / get_peak_memory() for memory analysis
4. Warmup runs before measurement (critical for compiled functions)

Usage:
    # Time a single op
    from vdllm.profiling.mlx_profiler import time_mlx_op
    with time_mlx_op("forward_pass"):
        mx.eval(model(inputs))

    # Benchmark a function
    from vdllm.profiling.mlx_profiler import benchmark_function
    result = benchmark_function(lambda: mx.eval(model(inputs)))

    # GPU capture (open in Xcode)
    from vdllm.profiling.mlx_profiler import metal_capture
    with metal_capture("/tmp/trace.gputrace"):
        outputs = llm.generate(prompts, params)

    # Full benchmark
    python -m vdllm.profiling.mlx_profiler --model /tmp/sdar-1.7b-chat
"""

import argparse
import time
import gc
from pathlib import Path
from contextlib import contextmanager
from dataclasses import dataclass

import mlx.core as mx


# ---------------------------------------------------------------------------
# Core timing utilities
# ---------------------------------------------------------------------------

@contextmanager
def metal_capture(trace_path: str):
    """Context manager for Metal GPU capture.

    Then open the .gputrace file in Xcode (File > Open).
    Requires MTL_CAPTURE_ENABLED=1 environment variable.
    """
    print(f"[Profiler] Starting Metal capture to {trace_path}")
    mx.metal.start_capture(trace_path)
    try:
        yield
    finally:
        mx.metal.stop_capture()
        print(f"[Profiler] Metal capture saved to {trace_path}")
        print(f"[Profiler] Open in Xcode: File > Open > {trace_path}")


@contextmanager
def time_mlx_op(name: str):
    """Time a single MLX operation with proper synchronization.

    Forces mx.synchronize() before and after to ensure accurate GPU timing.

    Usage:
        with time_mlx_op("attention"):
            out = mx.fast.scaled_dot_product_attention(q, k, v)
            mx.eval(out)
    """
    mx.synchronize()
    start = time.perf_counter()
    yield
    mx.synchronize()
    elapsed_ms = (time.perf_counter() - start) * 1000
    print(f"[{name}] {elapsed_ms:.2f}ms")


def benchmark_function(
    fn,
    warmup: int = 3,
    iterations: int = 10,
    verbose: bool = True,
) -> dict:
    """Benchmark a callable with warmup and statistics.

    Args:
        fn: Callable that performs the operation to benchmark
        warmup: Number of warmup runs (critical for mx.compile first-call)
        iterations: Number of timed runs
        verbose: Print progress

    Returns:
        dict with avg_ms, min_ms, max_ms, std_ms, all_ms
    """
    # Warmup — compiles Metal shaders, warms caches
    for _ in range(warmup):
        fn()
        mx.synchronize()

    # Timed runs
    timings = []
    for i in range(iterations):
        gc.collect()
        mx.synchronize()
        start = time.perf_counter()
        fn()
        mx.synchronize()
        elapsed = (time.perf_counter() - start) * 1000
        timings.append(elapsed)
        if verbose:
            print(f"  iter {i+1}/{iterations}: {elapsed:.2f}ms")

    avg = sum(timings) / len(timings)
    min_t = min(timings)
    max_t = max(timings)
    std = (sum((t - avg) ** 2 for t in timings) / len(timings)) ** 0.5

    if verbose:
        print(f"[Benchmark] avg={avg:.2f}ms, min={min_t:.2f}ms, "
              f"max={max_t:.2f}ms, std={std:.2f}ms")

    return {
        "avg_ms": avg,
        "min_ms": min_t,
        "max_ms": max_t,
        "std_ms": std,
        "iterations": iterations,
        "warmup": warmup,
        "all_ms": timings,
    }


# ---------------------------------------------------------------------------
# Memory utilities
# ---------------------------------------------------------------------------

def profile_memory(label: str = ""):
    """Print current MLX memory usage."""
    mx.synchronize()
    active = mx.get_active_memory() / (1024 ** 3)
    peak = mx.get_peak_memory() / (1024 ** 3)
    cache = mx.get_cache_memory() / (1024 ** 3)
    prefix = f" {label}" if label else ""
    print(f"[Memory{prefix}] active={active:.2f}GB, peak={peak:.2f}GB, "
          f"cache={cache:.2f}GB")


def reset_memory_stats():
    """Reset MLX peak memory counters."""
    mx.reset_peak_memory()


class MemoryTracker:
    """Track memory usage across operations.

    Usage:
        tracker = MemoryTracker()
        tracker.start("prefill")
        # ... run prefill ...
        tracker.end("prefill")
        tracker.report()
    """

    def __init__(self):
        self._starts = {}
        self._ends = {}

    def start(self, label: str):
        mx.synchronize()
        mx.reset_peak_memory()
        self._starts[label] = {
            "active": mx.get_active_memory(),
            "time": time.perf_counter(),
        }

    def end(self, label: str):
        if label not in self._starts:
            return
        mx.synchronize()
        self._ends[label] = {
            "active": mx.get_active_memory(),
            "peak": mx.get_peak_memory(),
            "time": time.perf_counter(),
        }

    def report(self):
        print("\n" + "=" * 60)
        print("MEMORY REPORT")
        print("=" * 60)
        for label in self._starts:
            if label not in self._ends:
                continue
            s = self._starts[label]
            e = self._ends[label]
            delta = (e["active"] - s["active"]) / (1024 ** 3)
            peak = e["peak"] / (1024 ** 3)
            elapsed = (e["time"] - s["time"]) * 1000
            print(f"{label:20s}  delta={delta:+.2f}GB  "
                  f"peak={peak:.2f}GB  time={elapsed:.1f}ms")


# ---------------------------------------------------------------------------
# Benchmark result
# ---------------------------------------------------------------------------

@dataclass
class BenchmarkResult:
    """Results from a profiling run."""
    model_load_s: float
    prefill_ms: dict  # {avg, min, max}
    decode_ms: dict   # {avg, min, max}
    total_ms: dict    # {avg, min, max}
    prefill_tps: float
    decode_tps: float
    peak_memory_gb: float
    num_runs: int
    prompt_tokens: int
    generated_tokens: int

    def __str__(self):
        lines = [
            "=" * 60,
            "BENCHMARK RESULTS",
            "=" * 60,
            f"Model load:     {self.model_load_s:.1f}s",
            f"Peak memory:    {self.peak_memory_gb:.2f} GB",
            "",
            f"Prefill:        {self.prefill_ms['avg']:.1f}ms avg "
            f"({self.prefill_ms['min']:.1f}-{self.prefill_ms['max']:.1f}ms) "
            f"| {self.prefill_tps:.0f} tok/s",
            f"Decode:         {self.decode_ms['avg']:.1f}ms avg "
            f"({self.decode_ms['min']:.1f}-{self.decode_ms['max']:.1f}ms) "
            f"| {self.decode_tps:.0f} tok/s",
            f"Total:          {self.total_ms['avg']:.1f}ms avg "
            f"({self.total_ms['min']:.1f}-{self.total_ms['max']:.1f}ms)",
            "",
            f"Prompt tokens:  {self.prompt_tokens}",
            f"Gen tokens:     {self.generated_tokens}",
            f"Runs:           {self.num_runs}",
            "=" * 60,
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# End-to-end profiling using real model via LLM API
# ---------------------------------------------------------------------------

def profile_model_load(model_path: str) -> float:
    """Profile model loading time.

    Returns:
        Load time in seconds
    """
    from vdllm.models.mlx_sdar import load_sdar_model

    print(f"[Profiler] Loading model from {model_path}...")
    reset_memory_stats()

    t0 = time.perf_counter()
    model, config = load_sdar_model(model_path, dtype=mx.bfloat16)
    mx.eval(model.parameters())
    mx.synchronize()
    load_time = time.perf_counter() - t0

    peak = mx.get_peak_memory() / (1024 ** 3)
    num_params = sum(p.size for _, p in model.parameters().items())
    print(f"[Profiler] Model loaded in {load_time:.1f}s "
          f"({num_params/1e9:.2f}B params, peak {peak:.2f}GB)")
    return load_time


def profile_generation(
    model_path: str,
    prompt: str = "Hello, world!",
    gen_length: int = 128,
    block_length: int = 4,
    denoising_steps: int = 4,
    num_runs: int = 3,
    use_chat: bool = False,
) -> BenchmarkResult:
    """Profile end-to-end generation with real model.

    Runs warmup, then timed iterations with per-phase breakdown.
    """
    from vdllm import LLM, SamplingParams

    # --- Model load ---
    print(f"[Profiler] Initializing LLM from {model_path}...")
    reset_memory_stats()
    t0 = time.perf_counter()
    llm = LLM(model_path, backend="mlx")
    mx.synchronize()
    model_load_s = time.perf_counter() - t0

    params = SamplingParams(
        max_tokens=gen_length,
        block_length=block_length,
        denoising_steps=denoising_steps,
    )

    # Apply chat template if requested
    if use_chat:
        messages = [{"role": "user", "content": prompt}]
        prompt_text = llm.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
    else:
        prompt_text = prompt

    print(f"[Profiler] Prompt: '{prompt_text[:80]}...'")
    print(f"[Profiler] gen_length={gen_length}, block_length={block_length}, "
          f"steps={denoising_steps}")

    # --- Warmup (1 full run) ---
    print("[Profiler] Warming up (1 run)...")
    _ = llm.generate([prompt_text], params, use_tqdm=False)
    mx.synchronize()

    # --- Timed runs ---
    print(f"[Profiler] Benchmarking ({num_runs} runs)...")
    prefill_times = []
    decode_times = []
    total_times = []
    prompt_tokens = 0
    generated_tokens = 0

    for i in range(num_runs):
        gc.collect()
        mx.synchronize()
        reset_memory_stats()

        t0 = time.perf_counter()
        outputs = llm.generate([prompt_text], params, use_tqdm=False)
        mx.synchronize()
        total = (time.perf_counter() - t0) * 1000

        timing = outputs[0].get("timing", {})
        prefill = timing.get("prefill_time", 0) * 1000
        decode = timing.get("decode_time", 0) * 1000
        prompt_tokens = timing.get("prompt_tokens", 0)
        generated_tokens = timing.get("generated_tokens", 0)

        prefill_times.append(prefill)
        decode_times.append(decode)
        total_times.append(total)

        print(f"  run {i+1}: total={total:.1f}ms "
              f"(prefill={prefill:.1f}ms, decode={decode:.1f}ms)")

    peak_memory_gb = mx.get_peak_memory() / (1024 ** 3)

    def _stats(times):
        return {"avg": sum(times)/len(times), "min": min(times), "max": max(times)}

    avg_prefill = _stats(prefill_times)["avg"]
    avg_decode = _stats(decode_times)["avg"]
    prefill_tps = (prompt_tokens / (avg_prefill / 1000)) if avg_prefill > 0 else 0
    decode_tps = (generated_tokens / (avg_decode / 1000)) if avg_decode > 0 else 0

    result = BenchmarkResult(
        model_load_s=model_load_s,
        prefill_ms=_stats(prefill_times),
        decode_ms=_stats(decode_times),
        total_ms=_stats(total_times),
        prefill_tps=prefill_tps,
        decode_tps=decode_tps,
        peak_memory_gb=peak_memory_gb,
        num_runs=num_runs,
        prompt_tokens=prompt_tokens,
        generated_tokens=generated_tokens,
    )

    print()
    print(result)
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="MLX Performance Profiler")
    parser.add_argument("--model", type=str, default="/tmp/sdar-1.7b-chat",
                        help="Path to SDAR model")
    parser.add_argument("--prompt", type=str, default="Hello, world!",
                        help="Input prompt")
    parser.add_argument("--gen-length", type=int, default=128,
                        help="Generation length")
    parser.add_argument("--block-length", type=int, default=4,
                        help="Block length")
    parser.add_argument("--denoising-steps", type=int, default=4,
                        help="Denoising steps per block")
    parser.add_argument("--runs", type=int, default=3,
                        help="Number of benchmark runs")
    parser.add_argument("--profile-load", action="store_true",
                        help="Profile model loading only")
    parser.add_argument("--capture", type=str, default=None,
                        help="Output path for Metal GPU capture (.gputrace)")
    parser.add_argument("--chat", action="store_true",
                        help="Use chat template formatting")

    args = parser.parse_args()

    if not Path(args.model).exists():
        print(f"[Error] Model path not found: {args.model}")
        print("Please download or specify a valid model path.")
        return

    if args.profile_load:
        profile_model_load(args.model)
        return

    if args.capture:
        with metal_capture(args.capture):
            profile_generation(
                args.model, args.prompt, args.gen_length,
                args.block_length, args.denoising_steps, args.runs, args.chat
            )
    else:
        profile_generation(
            args.model, args.prompt, args.gen_length,
            args.block_length, args.denoising_steps, args.runs, args.chat
        )


if __name__ == "__main__":
    main()
