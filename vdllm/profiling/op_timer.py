"""Per-operation profiling timer with MLX synchronization."""

import time
import mlx.core as mx
from dataclasses import dataclass, field
from typing import Dict, List
from contextlib import contextmanager


@dataclass
class OpProfile:
    """Timing data for a single operation.

    Attributes:
        name: Operation name
        calls: Number of times operation was measured
        total_ms: Total wall-clock time in milliseconds
        min_ms: Minimum single-call time in milliseconds
        max_ms: Maximum single-call time in milliseconds
        history: List of individual call times in milliseconds
    """

    name: str
    calls: int = 0
    total_ms: float = 0.0
    min_ms: float = float("inf")
    max_ms: float = 0.0
    history: List[float] = field(default_factory=list)


class OpTimer:
    """Measures wall-clock time per operation with MLX synchronization.

    This timer forces mx.synchronize() before and after each measurement
    to ensure accurate GPU time when using MLX on Apple Silicon.

    Example:
        >>> timer = OpTimer()
        >>> with timer.measure("attention"):
        ...     # MLX attention computation
        ...     out = mx.fast.scaled_dot_product_attention(q, k, v)
        ...
        >>> print(timer.report())

    Notes:
        - Uses wall-clock time, not GPU time (mx.synchronize() ensures
          all pending GPU work is complete before timing)
        - Thread-safe for single-threaded use cases
        - Histories can grow unbounded - call clear_history() periodically
    """

    def __init__(self):
        self.ops: Dict[str, OpProfile] = {}
        self.enabled = True
        self._total_time_ms: float = 0.0

    @contextmanager
    def measure(self, op_name: str):
        """Time an operation with MLX synchronization.

        Forces mx.synchronize() before and after timing to ensure
        accurate measurement of GPU operations.

        Args:
            op_name: Name to identify this operation in reports

        Yields:
            None - the context manager handles timing

        Example:
            >>> timer = OpTimer()
            >>> with timer.measure("my_op"):
            ...     result = some_computation()
        """
        if not self.enabled:
            yield
            return

        # Ensure any pending GPU work is complete
        mx.synchronize()

        start_time = time.perf_counter()

        try:
            yield
        finally:
            # Ensure GPU work for this operation is complete
            mx.synchronize()

            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            self._record(op_name, elapsed_ms)

    def _record(self, op_name: str, elapsed_ms: float) -> None:
        """Record a timing measurement.

        Args:
            op_name: Operation name
            elapsed_ms: Elapsed time in milliseconds
        """
        self._total_time_ms += elapsed_ms

        if op_name not in self.ops:
            self.ops[op_name] = OpProfile(name=op_name)

        profile = self.ops[op_name]
        profile.calls += 1
        profile.total_ms += elapsed_ms
        profile.min_ms = min(profile.min_ms, elapsed_ms)
        profile.max_ms = max(profile.max_ms, elapsed_ms)
        profile.history.append(elapsed_ms)

    def report(self) -> str:
        """Generate a formatted report of operation timings.

        Returns:
            String with ranked breakdown of time per operation
        """
        if not self.ops:
            return "No operations timed yet."

        lines = []
        lines.append("=" * 70)
        lines.append("OPERATION TIMING REPORT")
        lines.append("=" * 70)

        # Sort by total time descending
        sorted_ops = sorted(
            self.ops.values(),
            key=lambda p: p.total_ms,
            reverse=True
        )

        total_time = sum(p.total_ms for p in self.ops.values())

        lines.append(f"\n{'Operation':<30} {'Calls':>8} {'Total ms':>12} {'%':>8} {'Avg ms':>10}")
        lines.append("-" * 70)

        for profile in sorted_ops:
            pct = (profile.total_ms / total_time * 100) if total_time > 0 else 0
            avg_ms = profile.total_ms / profile.calls if profile.calls > 0 else 0
            lines.append(
                f"{profile.name:<30} {profile.calls:>8} "
                f"{profile.total_ms:>12.2f} {pct:>7.1f}% {avg_ms:>10.2f}"
            )

        lines.append("-" * 70)
        lines.append(f"{'TOTAL':<30} {'':<8} {total_time:>12.2f} {'100.0%':>8}")
        lines.append("=" * 70)

        return "\n".join(lines)

    def get_bottlenecks(self, threshold_pct: float = 15.0) -> List[OpProfile]:
        """Return ops consuming more than threshold_pct of total time.

        Args:
            threshold_pct: Percentage threshold (default 15.0)

        Returns:
            List of OpProfile objects exceeding threshold, sorted by time
        """
        total_time = sum(p.total_ms for p in self.ops.values())
        if total_time == 0:
            return []

        threshold_ms = total_time * (threshold_pct / 100.0)

        bottlenecks = [
            p for p in self.ops.values()
            if p.total_ms >= threshold_ms
        ]

        return sorted(bottlenecks, key=lambda p: p.total_ms, reverse=True)

    def clear(self) -> None:
        """Clear all recorded timings."""
        self.ops.clear()
        self._total_time_ms = 0.0

    def clear_history(self) -> None:
        """Clear history arrays to free memory."""
        for profile in self.ops.values():
            profile.history.clear()

    def disable(self) -> None:
        """Disable timing measurements."""
        self.enabled = False

    def enable(self) -> None:
        """Enable timing measurements."""
        self.enabled = True

    def get_summary(self) -> Dict[str, dict]:
        """Get summary statistics as a dictionary.

        Returns:
            Dictionary mapping operation name -> summary stats
        """
        return {
            name: {
                "calls": p.calls,
                "total_ms": p.total_ms,
                "min_ms": p.min_ms,
                "max_ms": p.max_ms,
                "avg_ms": p.total_ms / p.calls if p.calls > 0 else 0,
            }
            for name, p in self.ops.items()
        }
