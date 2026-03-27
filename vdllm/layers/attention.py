"""
SDAR Attention - Multi-head Attention with GQA.

This module provides the attention mechanism for SDAR models, including
support for Grouped Query Attention (GQA) and rotary position embeddings.

Based on JetEngine's attention implementation:
https://github.com/Jet-Astra/SDAR/blob/main/jetengine/layers/attention.py

================================================================================
                              GROUPED QUERY ATTENTION
================================================================================

GQA uses fewer key/value heads than query heads:
- num_query_heads: 32 (for example)
- num_kv_heads: 8 (for example)
- num_groups: 32 / 8 = 4

Each KV head is shared by multiple Q heads, reducing computation.

================================================================================
                              USAGE
================================================================================

    from sdar_model.layers import SDARAttention

    attn = SDARAttention(
        hidden_size=4096,
        num_heads=32,
        num_kv_heads=8,
        head_dim=128,
    )
    output = attn(positions, hidden_states)

"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .rmsnorm import RMSNorm
from .rotary import apply_rotary_pos_emb


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    Repeat key/value heads to match query heads for GQA.

    Args:
        hidden_states: Input tensor [batch, num_kv_heads, seq_len, head_dim]
        n_rep: Number of repetitions (num_q_heads // num_kv_heads)

    Returns:
        Repeated tensor [batch, num_q_heads, seq_len, head_dim]
    """
    if n_rep == 1:
        return hidden_states

    batch, num_kv_heads, seq_len, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_kv_heads, n_rep, seq_len, head_dim
    )
    return hidden_states.reshape(batch, num_kv_heads * n_rep, seq_len, head_dim)


class SDARAttention(nn.Module):
    """
    Multi-head attention with GQA support and RoPE.

    This attention module implements:
    - Grouped Query Attention (GQA) for memory efficiency
    - Rotary Position Embeddings (RoPE)
    - RMSNorm for Q/K normalization

    Attributes:
        num_heads: Number of query attention heads
        num_kv_heads: Number of key/value attention heads
        head_dim: Dimension of each attention head
        scaling: Attention scale factor
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: Optional[int] = None,
        rope_theta: float = 10000.0,
        rope_scaling: Optional[dict] = None,
        rms_norm_eps: float = 1e-6,
        bias: bool = False,
        device: Optional[torch.device] = None,
    ):
        """
        Initialize SDAR attention.

        Args:
            hidden_size: Hidden dimension size
            num_heads: Number of query attention heads
            num_kv_heads: Number of key/value attention heads
            head_dim: Dimension per head (defaults to hidden_size // num_heads)
            rope_theta: RoPE base period
            rope_scaling: Optional RoPE scaling config
            rms_norm_eps: RMSNorm epsilon
            bias: Whether to use bias in QKV projection
            device: Device for layer creation
        """
        super().__init__()

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim or (hidden_size // num_heads)
        self.scaling = self.head_dim ** -0.5

        assert num_heads % num_kv_heads == 0, "num_heads must be divisible by num_kv_heads"
        self.num_groups = num_heads // num_kv_heads

        # Q/K/V projections
        self.q_proj = nn.Linear(hidden_size, num_heads * self.head_dim, bias=bias, device=device)
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * self.head_dim, bias=bias, device=device)
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * self.head_dim, bias=bias, device=device)
        self.o_proj = nn.Linear(num_heads * self.head_dim, hidden_size, bias=False, device=device)

        # Q/K RMSNorm (per-head normalization before RoPE)
        self.q_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)

        # RoPE is applied in the forward pass using the rotary module

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        store_kv: bool = True,
    ) -> torch.Tensor:
        """
        Forward pass for SDAR attention.

        Args:
            positions: Position indices [batch_size, seq_len]
            hidden_states: Input hidden states [batch_size, seq_len, hidden_size]
            attention_mask: Optional attention mask
            store_kv: Whether to store KV for cache (not used in this implementation)

        Returns:
            Attention output [batch_size, seq_len, hidden_size]
        """
        batch_size, seq_len, _ = hidden_states.shape

        # Q/K/V projection
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        # Reshape for multi-head attention
        # [batch, seq, hidden] -> [batch, seq, num_heads, head_dim]
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = k.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        v = v.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)

        # Apply Q/K RMSNorm
        q = self.q_norm(q)
        k = self.k_norm(k)

        # Apply RoPE
        q, k = apply_rotary_pos_emb(q, k, None, positions)

        # Transpose for attention: [batch, num_heads, seq, head_dim]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Repeat KV heads for GQA
        k = repeat_kv(k, self.num_groups)
        v = repeat_kv(v, self.num_groups)

        # Attention mask handling
        if attention_mask is not None:
            # Expand mask for SDPA: [..., seq, seq] -> [..., 1, seq, seq]
            if attention_mask.dim() == 2:
                attention_mask = attention_mask.unsqueeze(0).unsqueeze(0)
            elif attention_mask.dim() == 3:
                attention_mask = attention_mask.unsqueeze(1)

        # Compute attention with SDPA
        output = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attention_mask,
            is_causal=False,
            scale=self.scaling,
        )

        # Reshape output: [batch, num_heads, seq, head_dim] -> [batch, seq, hidden_size]
        output = output.transpose(1, 2).contiguous()
        output = output.view(batch_size, seq_len, self.num_heads * self.head_dim)

        # Output projection
        output = self.o_proj(output)

        return output


class Attention(nn.Module):
    """
    Simplified attention module for SDAR.

    This is a wrapper around SDARAttention that provides a simpler interface.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: Optional[int] = None,
        head_dim: Optional[int] = None,
        rope_theta: float = 10000.0,
        rms_norm_eps: float = 1e-6,
        device: Optional[torch.device] = None,
    ):
        """
        Initialize Attention.

        Args:
            hidden_size: Hidden dimension size
            num_heads: Number of attention heads
            num_kv_heads: Number of KV heads (defaults to num_heads for MHA)
            head_dim: Dimension per head
            rope_theta: RoPE base period
            rms_norm_eps: RMSNorm epsilon
            device: Device for layer creation
        """
        super().__init__()

        num_kv_heads = num_kv_heads or num_heads

        self.attn = SDARAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            rope_theta=rope_theta,
            rms_norm_eps=rms_norm_eps,
            device=device,
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            positions: Position indices
            hidden_states: Input hidden states
            attention_mask: Optional attention mask

        Returns:
            Attention output
        """
        return self.attn(positions, hidden_states, attention_mask)
