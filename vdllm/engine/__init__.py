"""
Unified Engine for Block Diffusion Inference.

================================================================================
                              OVERVIEW
================================================================================

This package provides unified device backends for block diffusion inference
across different hardware platforms:

    Priority: Triton (CUDA) > Metal (MPS) > CPU

Components:
    - device.py: Device detection and unified backend selection
    - attention.py: Block diffusion attention masks
    - metal/: Metal backend for Apple Silicon
    - triton/: Triton backend for NVIDIA GPUs

================================================================================
                              USAGE
================================================================================

    from vdllm.engine import get_backend, build_block_diffusion_mask

    # Get unified backend
    backend = get_backend(num_heads=32, head_dim=128)

    # Build attention mask
    mask = build_block_diffusion_mask(
        seq_len=1024,
        block_size=4,
        device=backend.device
    )

    # Run attention
    output = backend.forward(q, k, v, mask=mask)

================================================================================
"""

import torch
from typing import Optional

# Device detection and unified backend
from .device import DeviceBackend, get_device, get_device_name

# Attention masks
from .attention import (
    build_block_diffusion_mask,
    build_staircase_block_mask,
    build_causal_mask,
)

# Unified attention function
from .attention import attention_forward


def get_backend(
    num_heads: int = 32,
    num_kv_heads: int = 8,
    head_dim: int = 128,
    force: Optional[str] = None,
) -> "UnifiedAttentionBackend":
    """
    Get unified attention backend for current device.

    Args:
        num_heads: Number of attention heads
        num_kv_heads: Number of key/value heads (GQA)
        head_dim: Dimension of each head
        force: Force backend ('triton', 'metal', 'cpu')

    Returns:
        UnifiedAttentionBackend instance
    """
    return UnifiedAttentionBackend(
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        force=force,
    )


class UnifiedAttentionBackend:
    """
    Unified attention backend that dispatches to appropriate implementation.

    This class provides a consistent interface for attention operations
    across different backends (Triton, Metal, CPU).
    """

    def __init__(
        self,
        num_heads: int = 32,
        num_kv_heads: int = 8,
        head_dim: int = 128,
        force: Optional[str] = None,
    ):
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim

        # Detect device
        self._device_backend = DeviceBackend(force_device=force)
        self._backend_name = self._device_backend.name

        # Initialize appropriate backend
        if self._backend_name == "triton":
            from .triton.backend import TritonBackend
            self._backend = TritonBackend(
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
            )
        elif self._backend_name == "metal":
            from .metal.backend import MetalBackend
            self._backend = MetalBackend(
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
            )
        else:
            # CPU fallback
            self._backend = None

    @property
    def device(self) -> torch.device:
        """Get PyTorch device."""
        return self._device_backend.device

    @property
    def name(self) -> str:
        """Get backend name."""
        return self._backend_name

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Run attention forward pass.

        Args:
            q: Query tensor [batch, heads, seq, head_dim]
            k: Key tensor [batch, kv_heads, seq, head_dim]
            v: Value tensor [batch, kv_heads, seq, head_dim]
            mask: Optional attention mask

        Returns:
            Attention output
        """
        if self._backend is not None:
            return self._backend.forward(q, k, v, mask)
        else:
            # CPU fallback
            return attention_forward(q, k, v, mask)

    def synchronize(self) -> None:
        """Synchronize device."""
        self._device_backend.synchronize()


__all__ = [
    # Device
    "DeviceBackend",
    "get_device",
    "get_device_name",
    "get_backend",
    # Attention masks
    "build_block_diffusion_mask",
    "build_staircase_block_mask",
    "build_causal_mask",
    # Unified backend
    "UnifiedAttentionBackend",
]
