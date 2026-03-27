"""
Triton Backend - CUDA with Triton Kernels.

This module provides the Triton backend implementation for NVIDIA GPUs,
using Triton kernels for optimized attention computation.

Architecture patterns from JetEngine:
- jetengine/kernels/triton/attention/block_prefill_attention_v2.py - Staircase sparse attention
- jetengine/kernels/triton/fused_page_attention_v3.py - Paged KV cache attention

================================================================================
                              TRITON KERNELS
================================================================================

This backend provides:
1. Standard attention via FlashAttention or SDPA
2. Block sparse attention (staircase pattern) via Triton
3. Paged KV cache attention via Triton

================================================================================
                              USAGE
================================================================================

    from engine.triton import TritonBackend

    backend = TritonBackend(num_heads=32, head_dim=128)
    output = backend.forward(q, k, v, mask=None)

"""

import math
from typing import Optional

import torch
import torch.nn.functional as F


class TritonBackend:
    """
    Triton backend using CUDA with optional Triton kernels.

    This backend wraps PyTorch CUDA operations with optional Triton
    kernel acceleration for attention patterns specific to block diffusion.

    Attributes:
        num_heads: Number of attention heads
        num_kv_heads: Number of key/value heads (for GQA)
        head_dim: Dimension of each attention head
        device: PyTorch device (cuda)
        use_triton_attention: Whether to use Triton kernels (auto-detected)
    """

    def __init__(
        self,
        num_heads: int = 32,
        num_kv_heads: int = 8,
        head_dim: int = 128,
        device: Optional[torch.device] = None,
    ):
        """
        Initialize Triton backend.

        Args:
            num_heads: Number of query attention heads
            num_kv_heads: Number of key/value attention heads (GQA)
            head_dim: Dimension of each attention head
            device: PyTorch device (defaults to CUDA if available)
        """
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.scale = 1.0 / math.sqrt(head_dim)

        # Determine device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        self._is_cuda = self.device.type == "cuda"

        # Check for Triton availability
        self.use_triton_attention = self._check_triton()

    def _check_triton(self) -> bool:
        """Check if Triton is available."""
        try:
            import triton
            return True
        except ImportError:
            return False

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
            mask: Optional attention mask

        Returns:
            Attention output [batch, num_heads, seq_len, head_dim]
        """
        if not self._is_cuda:
            return self._cpu_attention(q, k, v, mask)

        # Handle GQA: repeat KV heads to match Q heads
        if self.num_kv_heads < self.num_heads:
            num_groups = self.num_heads // self.num_kv_heads
            k = self._repeat_kv(k, num_groups)
            v = self._repeat_kv(v, num_groups)

        # Move to device
        q = q.to(self.device)
        k = k.to(self.device)
        v = v.to(self.device)
        if mask is not None:
            mask = mask.to(self.device)

        # Use SDPA for standard attention
        if mask is not None:
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

        return output

    def block_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        block_mask: torch.Tensor,
        block_size: int = 4,
    ) -> torch.Tensor:
        """
        Run block sparse attention.

        This uses the staircase attention pattern where tokens attend
        to exponentially expanding windows.

        Args:
            q: Query tensor [batch, num_heads, seq_len, head_dim]
            k: Key tensor [batch, num_kv_heads, seq_len, head_dim]
            v: Value tensor [batch, num_kv_heads, seq_len, head_dim]
            block_mask: Block diffusion mask
            block_size: Size of each block

        Returns:
            Attention output
        """
        if not self.use_triton_attention:
            # Fallback to standard attention
            return self.forward(q, k, v, block_mask)

        # Use Triton staircase attention
        from .attention import triton_staircase_attention
        return triton_staircase_attention(
            q, k, v, block_mask,
            scale=self.scale,
            block_size=block_size,
        )

    def _repeat_kv(self, x: torch.Tensor, n_rep: int) -> torch.Tensor:
        """Repeat KV heads to match Q heads for GQA."""
        if n_rep == 1:
            return x
        batch, num_kv, seq_len, head_dim = x.shape
        x = x[:, :, None, :, :].expand(batch, num_kv, n_rep, seq_len, head_dim)
        return x.reshape(batch, num_kv * n_rep, seq_len, head_dim)

    def _cpu_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """CPU fallback."""
        scale = 1.0 / math.sqrt(q.shape[-1])
        scores = torch.einsum("bqhd,bkhd->bhqk", q, k) * scale
        if mask is not None:
            scores = scores.masked_fill(~mask, float("-inf"))
        attn = torch.softmax(scores, dim=-1)
        return torch.einsum("bhqk,bkhd->bqhd", attn, v)

    def synchronize(self) -> None:
        """Synchronize CUDA device."""
        if self._is_cuda:
            torch.cuda.synchronize()

    def empty_cache(self) -> None:
        """Empty CUDA cache."""
        if self._is_cuda:
            torch.cuda.empty_cache()


# Convenience function
def triton_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    num_heads: int = 32,
    num_kv_heads: int = 8,
    head_dim: int = 128,
) -> torch.Tensor:
    """Convenience function for Triton attention."""
    backend = TritonBackend(
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
    )
    return backend.forward(q, k, v, mask)
