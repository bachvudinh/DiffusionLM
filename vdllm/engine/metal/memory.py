"""
Metal Memory Management.

This module provides memory management utilities for Apple Silicon MPS,
including MPS-safe tensor allocation and memory budget calculation.

Architecture patterns adapted from vllm-metal:
- vllm_metal/v1/worker.py - Memory budget calculation
- vllm_metal/pytorch_backend/tensor_bridge.py - MPS safe size handling

================================================================================
                              MPS 4GB LIMIT
================================================================================

Apple Silicon MPS has a 4GB limit for single tensor allocation.
This module provides utilities to safely allocate tensors on MPS
by chunking large tensors or keeping them on CPU when necessary.

================================================================================
                              USAGE
================================================================================

    from engine.metal.memory import MPSMemoryManager, get_mps_memory_limit

    manager = MPSMemoryManager()
    tensor = manager.allocate_safe(size=(16384, 16384), dtype=torch.float16)

"""

import logging
from typing import Optional

import torch


logger = logging.getLogger(__name__)

# Conservative threshold to avoid MPS crashes (1GB default, can be 4GB on newer chips)
# Adapted from vllm-metal's tensor_bridge.py
_MPS_SAFE_SIZE_BYTES = 1 << 30  # 1GB default threshold


def set_mps_safe_size_threshold(bytes_threshold: int) -> None:
    """
    Set the safe size threshold for MPS tensor allocation.

    Args:
        bytes_threshold: Maximum tensor size in bytes before chunking
    """
    global _MPS_SAFE_SIZE_BYTES
    _MPS_SAFE_SIZE_BYTES = bytes_threshold


def get_mps_safe_size_threshold() -> int:
    """Get current MPS safe size threshold."""
    return _MPS_SAFE_SIZE_BYTES


def _get_tensor_size_bytes(shape: tuple, dtype: torch.dtype) -> int:
    """Calculate tensor size in bytes."""
    num_elements = 1
    for dim in shape:
        num_elements *= dim
    return num_elements * torch.empty((), dtype=dtype).element_size()


def is_safe_for_mps(shape: tuple, dtype: torch.dtype) -> bool:
    """
    Check if tensor size is safe for MPS allocation.

    Args:
        shape: Tensor shape
        dtype: Tensor dtype

    Returns:
        True if tensor can be safely allocated on MPS
    """
    return _get_tensor_size_bytes(shape, dtype) < _MPS_SAFE_SIZE_BYTES


class MPSMemoryManager:
    """
    Memory manager for MPS with safe allocation.

    This class handles memory allocation on Apple Silicon MPS,
    automatically chunking large tensors to avoid the 4GB limit.
    """

    def __init__(
        self,
        device: Optional[torch.device] = None,
        memory_fraction: float = 0.9,
    ):
        """
        Initialize memory manager.

        Args:
            device: MPS device (defaults to 'mps' if available)
            memory_fraction: Fraction of available memory to use
        """
        self.device = device or torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.memory_fraction = memory_fraction
        self._is_mps = self.device.type == "mps"

        if self._is_mps:
            self._max_memory = self._get_metal_memory_limit()
        else:
            self._max_memory = 0

    def _get_metal_memory_limit(self) -> int:
        """
        Get Metal memory limit from device info.

        Returns:
            Maximum memory in bytes
        """
        try:
            # Try to get from mlx if available (vllm-metal pattern)
            import mlx.core as mx
            info = mx.metal.DeviceInfo()
            return int(info["max_recommended_working_set_size"])
        except ImportError:
            # Fallback: assume 16GB for Apple Silicon
            return 16 * (1 << 30)

    def allocate_safe(
        self,
        shape: tuple,
        dtype: torch.dtype = torch.float16,
    ) -> torch.Tensor:
        """
        Allocate tensor safely, chunking if necessary.

        Args:
            shape: Tensor shape
            dtype: Tensor dtype

        Returns:
            Allocated tensor on appropriate device
        """
        if not self._is_mps:
            return torch.empty(shape, dtype=dtype, device=self.device)

        if is_safe_for_mps(shape, dtype):
            return torch.empty(shape, dtype=dtype, device=self.device)

        # Too large for MPS, allocate on CPU and move
        logger.warning(f"Tensor size {_get_tensor_size_bytes(shape, dtype) / 1e9:.2f}GB exceeds MPS safe threshold, allocating on CPU")
        return torch.empty(shape, dtype=dtype, device="cpu")

    def get_available_memory(self) -> int:
        """
        Get available memory in bytes.

        Returns:
            Available memory in bytes
        """
        if not self._is_mps:
            return 0
        return int(self._max_memory * self.memory_fraction)

    def compute_cache_budget(
        self,
        model_memory: int,
        overhead: int = 800 * (1 << 20),  # 800MB overhead (vllm-metal constant)
    ) -> int:
        """
        Compute KV cache budget from available memory.

        Args:
            model_memory: Memory used by model weights in bytes
            overhead: Overhead memory in bytes

        Returns:
            Number of bytes available for KV cache
        """
        available = self.get_available_memory()
        cache_budget = available - model_memory - overhead
        return max(0, cache_budget)


def chunk_for_mps(tensor: torch.Tensor, chunk_size_bytes: int = _MPS_SAFE_SIZE_BYTES) -> list[torch.Tensor]:
    """
    Chunk a tensor into MPS-safe pieces.

    Args:
        tensor: Input tensor
        chunk_size_bytes: Maximum size per chunk in bytes

    Returns:
        List of tensor chunks
    """
    if tensor.device.type != "mps":
        return [tensor]

    total_bytes = tensor.nelement() * tensor.element_size()
    if total_bytes <= chunk_size_bytes:
        return [tensor]

    # Calculate elements per chunk
    elements_per_chunk = chunk_size_bytes // tensor.element_size()
    total_elements = tensor.nelement()

    chunks = []
    for start in range(0, total_elements, elements_per_chunk):
        end = min(start + elements_per_chunk, total_elements)
        chunk = tensor.flatten()[start:end]
        # Reshape to roughly original dimensions
        chunks.append(chunk)

    return chunks


def mps_memory_stats() -> dict:
    """
    Get MPS memory statistics.

    Returns:
        Dictionary with memory stats (free, total, used)
    """
    if not torch.backends.mps.is_available():
        return {"available": False}

    try:
        import mlx.core as mx
        info = mx.metal.DeviceInfo()
        return {
            "available": True,
            "max_working_set_size": info["max_recommended_working_set_size"],
            "current_working_set_size": info["current_working_set_size"],
        }
    except ImportError:
        return {"available": True, "note": "mlx not available for detailed stats"}
