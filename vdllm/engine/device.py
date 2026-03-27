"""
Device Detection and Unified Backend Selection.

Unified device backend with priority: Triton (CUDA) > Metal (MPS) > CPU.

Architecture patterns from vllm-metal:
- https://github.com/vllm-project/vllm-metal

================================================================================
                              DEVICE PRIORITY
================================================================================

1. Triton (CUDA): NVIDIA GPUs with Triton kernels for attention
2. Metal (MPS): Apple Silicon via PyTorch MPS + Metal kernels
3. CPU: Fallback for all other platforms

================================================================================
                              USAGE
================================================================================

    from engine.device import DeviceBackend, get_device

    backend = DeviceBackend()
    print(f"Using: {backend.name}")  # 'triton', 'metal', or 'cpu'
    device = backend.device          # torch.device

"""

import torch
from typing import Optional


class DeviceBackend:
    """
    Unified device backend with automatic detection and fallback.

    Priority: Triton (CUDA) > Metal (MPS) > CPU

    This class detects available compute backends and selects the optimal one
    based on hardware availability. It provides a consistent interface for
    device operations across different backends.
    """

    _instance: Optional["DeviceBackend"] = None

    def __init__(self, force_device: Optional[str] = None):
        """
        Initialize device backend.

        Args:
            force_device: Force a specific device ('triton', 'metal', 'cpu').
                         Useful for testing or debugging.
        """
        self._force_device = force_device
        self._backend = self._detect()
        self._setup()

    def _detect(self) -> str:
        """Detect best available backend."""
        if self._force_device is not None:
            return self._force_device

        # Priority 1: Triton (CUDA)
        if torch.cuda.is_available():
            return "triton"

        # Priority 2: Metal (MPS)
        if torch.backends.mps.is_available():
            return "metal"

        # Priority 3: CPU
        return "cpu"

    def _setup(self):
        """Initialize backend-specific components."""
        pass

    @property
    def name(self) -> str:
        """Backend name: 'triton', 'metal', or 'cpu'."""
        return self._backend

    @property
    def device(self) -> torch.device:
        """PyTorch device object."""
        if self._backend == "triton":
            return torch.device("cuda")
        elif self._backend == "metal":
            return torch.device("mps")
        return torch.device("cpu")

    @property
    def supports_memory_efficient_attention(self) -> bool:
        """Whether backend supports memory-efficient attention kernels."""
        return self._backend in ("triton", "metal")

    @property
    def supports_flash_attention(self) -> bool:
        """Whether backend supports FlashAttention."""
        return self._backend == "triton"

    def synchronize(self):
        """Synchronize device operations if needed."""
        if self._backend == "triton":
            torch.cuda.synchronize()
        elif self._backend == "metal":
            torch.mps.synchronize()

    def empty_cache(self):
        """Empty device cache."""
        if self._backend == "triton":
            torch.cuda.empty_cache()
        elif self._backend == "metal":
            # MPS doesn't have explicit empty_cache
            pass

    def mem_info(self) -> tuple[int, int]:
        """Get memory info (free, total) in bytes.

        Returns:
            Tuple of (free memory, total memory) in bytes.
            On CPU/MPS, returns approximate values.
        """
        if self._backend == "triton":
            free, total = torch.cuda.mem_get_info()
            return free, total
        elif self._backend == "metal":
            # MPS doesn't provide mem_info directly
            # Try mlx if available
            try:
                import mlx.core as mx
                info = mx.metal.DeviceInfo()
                return info["memory_size"], info["memory_size"]
            except ImportError:
                pass
        return 0, 0

    def __repr__(self) -> str:
        return f"DeviceBackend('{self._backend}')"


def get_device(force: Optional[str] = None) -> DeviceBackend:
    """
    Get the global DeviceBackend instance.

    Args:
        force: Force a specific backend ('triton', 'metal', 'cpu')

    Returns:
        DeviceBackend instance
    """
    if DeviceBackend._instance is None or force is not None:
        DeviceBackend._instance = DeviceBackend(force_device=force)
    return DeviceBackend._instance


def get_device_name() -> str:
    """Get current device name."""
    return get_device().name
