"""
SDAR model in MLX, following mlx-lm/models/qwen2.py patterns.
"""
import math
from dataclasses import dataclass
from typing import Optional, List

import mlx.core as mx
import mlx.nn as nn


@dataclass
class SDARModelArgs:
    hidden_size: int = 2048
    num_hidden_layers: int = 28
    num_attention_heads: int = 16
    num_key_value_heads: int = 8
    head_dim: int = 128
    intermediate_size: int = 6144
    vocab_size: int = 151936
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1000000.0
    max_position_embeddings: int = 32768
    attention_bias: bool = False

    @classmethod
    def from_dict(cls, params: dict):
        import inspect
        valid_keys = inspect.signature(cls).parameters
        filtered = {k: v for k, v in params.items() if k in valid_keys}
        return cls(**filtered)


class Attention(nn.Module):
    """GQA attention with RoPE, q_norm, k_norm — SDAR pattern."""

    def __init__(self, args: SDARModelArgs):
        super().__init__()
        self.num_heads = args.num_attention_heads
        self.num_kv_heads = args.num_key_value_heads
        self.head_dim = args.head_dim
        self.scale = self.head_dim ** -0.5
        self.n_rep = self.num_heads // self.num_kv_heads

        self.q_proj = nn.Linear(args.hidden_size, self.num_heads * self.head_dim, bias=args.attention_bias)
        self.k_proj = nn.Linear(args.hidden_size, self.num_kv_heads * self.head_dim, bias=args.attention_bias)
        self.v_proj = nn.Linear(args.hidden_size, self.num_kv_heads * self.head_dim, bias=args.attention_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, args.hidden_size, bias=args.attention_bias)

        # SDAR-specific: q_norm and k_norm applied per-head before RoPE
        self.q_norm = nn.RMSNorm(self.head_dim, eps=args.rms_norm_eps)
        self.k_norm = nn.RMSNorm(self.head_dim, eps=args.rms_norm_eps)

        self.rope = nn.RoPE(
            dims=self.head_dim,
            traditional=False,
            base=args.rope_theta,
        )

    def __call__(self, x, mask=None, cache=None):
        B, L, _ = x.shape

        # QKV projections
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Reshape: [B, L, hidden] -> [B, L, num_heads, head_dim]
        q = q.reshape(B, L, self.num_heads, self.head_dim)
        k = k.reshape(B, L, self.num_kv_heads, self.head_dim)
        v = v.reshape(B, L, self.num_kv_heads, self.head_dim)

        # Apply q_norm and k_norm (per-head RMSNorm on head_dim)
        q = self.q_norm(q)
        k = self.k_norm(k)

        # Transpose: [B, L, num_heads, head_dim] -> [B, num_heads, L, head_dim]
        q = q.transpose(0, 2, 1, 3)
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)

        # Apply RoPE
        if cache is not None:
            q = self.rope(q, offset=cache.offset)
            k = self.rope(k, offset=cache.offset)
            k, v = cache.update_and_fetch(k, v)
        else:
            q = self.rope(q)
            k = self.rope(k)

        # GQA repeat K, V to match Q heads
        if self.n_rep > 1:
            k = mx.repeat(k, self.n_rep, axis=1)
            v = mx.repeat(v, self.n_rep, axis=1)

        # SDPA
        output = mx.fast.scaled_dot_product_attention(
            q, k, v, scale=self.scale, mask=mask
        )

        # Reshape back: [B, num_heads, L, head_dim] -> [B, L, hidden]
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(output)


class MLP(nn.Module):
    """SiLU-gated MLP with fused gate+up projection.

    Fuses gate_proj and up_proj into a single weight matrix for memory bandwidth savings.
    The fused weight is [hidden_size, 2 * intermediate_size], where:
      - columns 0..intermediate_size-1 = gate_proj
      - columns intermediate_size.. = up_proj
    """

    def __init__(self, args: SDARModelArgs):
        super().__init__()
        self.intermediate_size = args.intermediate_size
        # Fused: single weight matrix [hidden_size, 2 * intermediate_size]
        self.gate_up_proj = nn.Linear(args.hidden_size, 2 * args.intermediate_size, bias=False)
        self.down_proj = nn.Linear(args.intermediate_size, args.hidden_size, bias=False)

    def __call__(self, x):
        gate_up = self.gate_up_proj(x)
        gate = gate_up[..., :self.intermediate_size]
        up = gate_up[..., self.intermediate_size:]
        return self.down_proj(nn.silu(gate) * up)


class KVCache:
    """Simple KV cache for autoregressive generation.

    Compatible with MLX 0.31.1 which doesn't have nn.KVCache.
    Matches the interface expected by Attention.__call__:
    - offset: int - current position offset
    - update_and_fetch(k, v) -> (k_updated, v_updated)
    """

    def __init__(self):
        self._k = None
        self._v = None
        self._offset = 0

    @property
    def offset(self):
        return self._offset

    def update_and_fetch(self, k, v):
        """Append new k, v to cache and return full cached k, v."""
        if self._k is None:
            self._k = k
            self._v = v
        else:
            # Concatenate along sequence dimension
            self._k = mx.concatenate([self._k, k], axis=2)
            self._v = mx.concatenate([self._v, v], axis=2)
        self._offset += k.shape[2]
        return self._k, self._v


class TransformerBlock(nn.Module):
    """Single decoder layer — same as Qwen2 TransformerBlock."""

    def __init__(self, args: SDARModelArgs):
        super().__init__()
        self.self_attn = Attention(args)
        self.mlp = MLP(args)
        self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    def __call__(self, x, mask=None, cache=None):
        r = self.self_attn(self.input_layernorm(x), mask=mask, cache=cache)
        h = x + r
        r = self.mlp(self.post_attention_layernorm(h))
        return h + r


class SDARModel(nn.Module):
    """Full SDAR transformer — same as Qwen2Model."""

    def __init__(self, args: SDARModelArgs):
        super().__init__()
        self.args = args
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [TransformerBlock(args) for _ in range(args.num_hidden_layers)]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    def __call__(self, inputs, cache=None, mask=None):
        h = self.embed_tokens(inputs)
        if cache is None:
            cache = [None] * len(self.layers)
        for i, layer in enumerate(self.layers):
            h = layer(h, mask=mask, cache=cache[i] if cache else None)
        return self.norm(h)


class Model(nn.Module):
    """Full SDAR model with lm_head — same as Qwen2 Model."""

    def __init__(self, args: SDARModelArgs):
        super().__init__()
        self.args = args
        self.model = SDARModel(args)
        self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(self, inputs, cache=None, mask=None):
        h = self.model(inputs, cache=cache, mask=mask)
        return self.lm_head(h)

    def sanitize(self, weights):
        """Remove any weights that don't match the model structure.

        Also fuses gate_proj and up_proj into a single gate_up_proj for the MLP.
        The safetensors have separate weights: gate_proj.weight [intermediate_size, hidden_size]
        and up_proj.weight [intermediate_size, hidden_size]. We concatenate them along dim=0
        to get gate_up_proj.weight [2 * intermediate_size, hidden_size].
        """
        # First, fuse gate_proj and up_proj into gate_up_proj
        fused_weights = {}
        for name, arr in weights.items():
            if name.endswith(".gate_proj.weight") and name.replace(".gate_proj.", ".up_proj.") in weights:
                # This is a gate_proj weight - will be fused
                base_name = name.replace(".gate_proj.weight", "")
                gate_key = f"{base_name}.gate_proj.weight"
                up_key = f"{base_name}.up_proj.weight"
                if gate_key in weights and up_key in weights:
                    gate = weights[gate_key]
                    up = weights[up_key]
                    # Concatenate along dim=0: [intermediate_size, hidden] + [intermediate_size, hidden]
                    # -> [2 * intermediate_size, hidden]
                    fused_weights[f"{base_name}.gate_up_proj.weight"] = mx.concatenate([gate, up], axis=0)

        # Add non-fused weights and the fused weights
        result = {}
        for name, arr in weights.items():
            base_name = name.replace(".weight", "").replace(".bias", "")
            if name.endswith(".gate_proj.weight"):
                # Skip individual gate_proj - will use gate_up_proj instead
                continue
            if name.endswith(".up_proj.weight"):
                # Skip individual up_proj - fused into gate_up_proj
                continue
            result[name] = arr

        # Add fused weights
        result.update(fused_weights)

        return result

    def make_cache(self):
        """Create KV caches for each layer."""
        return [KVCache() for _ in range(self.args.num_hidden_layers)]

    @staticmethod
    def build_causal_mask(seq_len: int) -> mx.array:
        """Build causal mask: tokens can only attend to previous tokens.

        Returns a boolean mask where True means "can attend".
        For MLX SDPA, use "causal" string or pass a boolean mask.
        """
        positions = mx.arange(seq_len)
        mask = positions[:, None] >= positions[None, :]
        return mask


def load_sdar_model(model_path: str, dtype: mx.Dtype = mx.bfloat16):
    """Load SDAR model from a directory with safetensors weights.

    Args:
        model_path: Path to model directory containing config.json and safetensors
        dtype: Target MLX dtype (default bfloat16)

    Returns:
        Tuple of (model, config)
    """
    import json
    from pathlib import Path

    model_dir = Path(model_path)

    # Load config
    with open(model_dir / "config.json") as f:
        config = json.load(f)

    args = SDARModelArgs.from_dict(config)
    model = Model(args)

    # Load weights from safetensors
    weight_files = sorted(model_dir.glob("*.safetensors"))
    weights = {}
    for wf in weight_files:
        tensors = mx.load(str(wf))
        for name, arr in tensors.items():
            if arr.dtype != dtype:
                arr = arr.astype(dtype)
            weights[name] = arr

    # Sanitize and load weights
    weights = model.sanitize(weights)
    model.load_weights(list(weights.items()), strict=False)
    mx.eval(model.parameters())

    return model, config
