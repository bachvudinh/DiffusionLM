"""
Metal Backend - PyTorch MPS with Metal Kernels.

This module provides the Metal backend implementation for Apple Silicon,
using PyTorch's MPS (Metal Performance Shaders) backend with custom
Metal kernels where beneficial.

Architecture patterns adapted from vllm-metal:
- vllm_metal/pytorch_backend/tensor_bridge.py - MLX/PyTorch tensor conversion
- vllm_metal/metal_kernel_backend/attention_sdpa.py - SDPA forward
- vllm_metal/metal_kernel_backend/cache.py - MetalKVCache

================================================================================
                              MEMORY MANAGEMENT
================================================================================

MPS has a 4GB limit for single tensors. Large tensors are kept on CPU
and moved to MPS as needed. See _MPS_SAFE_SIZE_BYTES.

================================================================================
                              USAGE
================================================================================

    from engine.metal import MetalBackend

    backend = MetalBackend(num_heads=32, head_dim=128)
    output = backend.forward(q, k, v, block_mask=None)

"""

import math
from typing import Optional, Tuple

import torch
import torch.nn.functional as F


# Conservative threshold to avoid MPS crashes (1GB)
# Adapted from vllm-metal's tensor_bridge.py
_MPS_SAFE_SIZE_BYTES = 1 << 30


def _is_safe_for_mps(tensor: torch.Tensor) -> bool:
    """Check if tensor size is safe for MPS allocation."""
    size_bytes = tensor.nelement() * tensor.element_size()
    return size_bytes < _MPS_SAFE_SIZE_BYTES


def _get_tensor_size_bytes(tensor: torch.Tensor) -> int:
    """Get tensor size in bytes."""
    return tensor.nelement() * tensor.element_size()


class MetalBackend:
    """
    Metal backend using PyTorch MPS.

    This backend wraps PyTorch's MPS (Metal Performance Shaders) backend
    for Apple Silicon GPUs. It uses SDPA (Scaled Dot Product Attention)
    for attention computation and provides memory-safe tensor operations.

    Attributes:
        num_heads: Number of attention heads
        num_kv_heads: Number of key/value heads (for GQA)
        head_dim: Dimension of each attention head
        device: PyTorch device (mps or cpu)
    """

    def __init__(
        self,
        num_heads: int = 32,
        num_kv_heads: int = 8,
        head_dim: int = 128,
        device: Optional[torch.device] = None,
    ):
        """
        Initialize Metal backend.

        Args:
            num_heads: Number of query attention heads
            num_kv_heads: Number of key/value attention heads (GQA)
            head_dim: Dimension of each attention head
            device: PyTorch device (defaults to MPS if available)
        """
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.scale = 1.0 / math.sqrt(head_dim)

        # Determine device
        if device is None:
            self.device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
        else:
            self.device = device

        self._is_mps = self.device.type == "mps"

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
            q: Query tensor [batch, num_heads, seq_len, head_dim]
            k: Key tensor [batch, num_kv_heads, seq_len, head_dim]
            v: Value tensor [batch, num_kv_heads, seq_len, head_dim]
            mask: Optional attention mask [batch, seq_len, seq_len] or [seq_len, seq_len]

        Returns:
            Attention output [batch, num_heads, seq_len, head_dim]
        """
        # Handle GQA: repeat KV heads to match Q heads
        if self.num_kv_heads < self.num_heads:
            num_groups = self.num_heads // self.num_kv_heads
            k = self._repeat_kv(k, num_groups)
            v = self._repeat_kv(v, num_groups)

        # Move to device if needed and safe
        q = self._to_device_safe(q)
        k = self._to_device_safe(k)
        v = self._to_device_safe(v)
        if mask is not None:
            mask = self._to_device_safe(mask)

        # Synchronize before MPS operation
        if self._is_mps:
            torch.mps.synchronize()

        # Use SDPA for attention
        if mask is not None:
            # Expand mask for SDPA broadcasting: [B, 1, Seq, Seq] or [1, 1, Seq, Seq]
            if mask.dim() == 2:
                mask = mask.unsqueeze(0).unsqueeze(0)
            elif mask.dim() == 3:
                mask = mask.unsqueeze(1)
            output = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=mask,
                is_causal=False,
                scale=self.scale,
            )
        else:
            output = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,
                is_causal=False,
                scale=self.scale,
            )

        # Synchronize after MPS operation
        if self._is_mps:
            torch.mps.synchronize()

        return output

    def forward_with_cache(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        kv_cache: Tuple[torch.Tensor, torch.Tensor],
        slot_mapping: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Run attention with KV cache storage.

        This is used during generation where we need to:
        1. Store new K/V to cache
        2. Run attention with cached K/V

        Args:
            q: Query tensor [batch, num_heads, 1, head_dim] (single token during decode)
            k: Key tensor [batch, num_kv_heads, 1, head_dim]
            v: Value tensor [batch, num_kv_heads, 1, head_dim]
            kv_cache: Tuple of (key_cache, value_cache)
            slot_mapping: Optional mapping of positions to cache slots

        Returns:
            Attention output for the current query
        """
        key_cache, value_cache = kv_cache

        # Store K/V to cache
        if slot_mapping is not None:
            # Paged attention: store at specific slots
            self._store_kv_paged(k, v, key_cache, value_cache, slot_mapping)
        else:
            # Standard: append to end of cache
            self._store_kv(k, v, key_cache, value_cache)

        # Get cached K/V for attention
        k_cached = self._to_device_safe(key_cache[:, :, :q.shape[2] + 1, :])
        v_cached = self._to_device_safe(value_cache[:, :, :q.shape[2] + 1, :])

        return self.forward(q, k_cached, v_cached, mask=None)

    def _repeat_kv(self, x: torch.Tensor, n_rep: int) -> torch.Tensor:
        """
        Repeat KV heads to match Q heads for GQA.

        Args:
            x: Input tensor [batch, num_kv_heads, seq_len, head_dim]
            n_rep: Number of repetitions

        Returns:
            Repeated tensor [batch, num_heads, seq_len, head_dim]
        """
        if n_rep == 1:
            return x
        batch, num_kv, seq_len, head_dim = x.shape
        x = x[:, :, None, :, :].expand(batch, num_kv, n_rep, seq_len, head_dim)
        return x.reshape(batch, num_kv * n_rep, seq_len, head_dim)

    def _to_device_safe(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Move tensor to device safely, handling MPS size limits.

        Args:
            tensor: Input tensor

        Returns:
            Tensor on the appropriate device
        """
        if tensor.device == self.device:
            return tensor

        size_bytes = _get_tensor_size_bytes(tensor)

        if self._is_mps and not _is_safe_for_mps(tensor):
            # Tensor too large for MPS, keep on CPU (will slow down)
            # This matches vllm-metal's approach
            return tensor.to("cpu")
        return tensor.to(self.device)

    def _store_kv(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
    ) -> None:
        """Store key/value tensors in cache."""
        seq_len = k.shape[2]
        key_cache[:, :, :seq_len, :] = k
        value_cache[:, :, :seq_len, :] = v

    def _store_kv_paged(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> None:
        """
        Store key/value tensors in paged cache at specific slots.

        Args:
            k: Key tensor [batch, num_kv_heads, 1, head_dim]
            v: Value tensor [batch, num_kv_heads, 1, head_dim]
            key_cache: Key cache [num_blocks, num_kv_heads, block_size, head_dim]
            value_cache: Value cache [num_blocks, num_kv_heads, block_size, head_dim]
            slot_mapping: Position to slot mapping
        """
        batch_size = k.shape[0]
        num_kv_heads = k.shape[1]
        head_dim = k.shape[3]

        for i in range(batch_size):
            slot = slot_mapping[i].item()
            block_idx = slot // key_cache.shape[2]  # block_size
            offset = slot % key_cache.shape[2]
            key_cache[block_idx, :, offset, :] = k[i]
            value_cache[block_idx, :, offset, :] = v[i]

    def synchronize(self) -> None:
        """Synchronize MPS device."""
        if self._is_mps:
            torch.mps.synchronize()

    def empty_cache(self) -> None:
        """Empty MPS cache."""
        if self._is_mps:
            torch.mps.empty_cache()


# Convenience function for simple attention
def metal_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    num_heads: int = 32,
    num_kv_heads: int = 8,
    head_dim: int = 128,
) -> torch.Tensor:
    """
    Convenience function for Metal attention.

    Args:
        q: Query tensor
        k: Key tensor
        v: Value tensor
        mask: Optional attention mask
        num_heads: Number of attention heads
        num_kv_heads: Number of key/value heads
        head_dim: Head dimension

    Returns:
        Attention output
    """
    backend = MetalBackend(
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
    )
    return backend.forward(q, k, v, mask)
