"""SDAR model implemented in MLX for Apple Silicon inference."""

from dataclasses import dataclass

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
    tie_word_embeddings: bool = False

    @classmethod
    def from_dict(cls, params: dict):
        import inspect
        valid_keys = inspect.signature(cls).parameters
        filtered = {k: v for k, v in params.items() if k in valid_keys}
        return cls(**filtered)


class SDARAttention(nn.Module):
    """GQA attention with per-head q/k norms and RoPE."""

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

        self.q_norm = nn.RMSNorm(self.head_dim, eps=args.rms_norm_eps)
        self.k_norm = nn.RMSNorm(self.head_dim, eps=args.rms_norm_eps)

        self.rope = nn.RoPE(
            dims=self.head_dim,
            traditional=False,
            base=args.rope_theta,
        )

    def __call__(self, x, mask=None, cache=None, position_ids=None, store_kv=True):
        B, L, _ = x.shape

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.reshape(B, L, self.num_heads, self.head_dim)
        k = k.reshape(B, L, self.num_kv_heads, self.head_dim)
        v = v.reshape(B, L, self.num_kv_heads, self.head_dim)

        q = self.q_norm(q)
        k = self.k_norm(k)

        q = q.transpose(0, 2, 1, 3)
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)

        # Determine RoPE offset
        if position_ids is not None:
            # Use the starting position from position_ids for RoPE offset.
            # This works because within a block, positions are contiguous.
            offset = position_ids[0, 0].item()
        elif cache is not None:
            offset = cache.offset
        else:
            offset = 0

        q = self.rope(q, offset=offset)
        k = self.rope(k, offset=offset)

        if cache is not None:
            if store_kv:
                # Prefill / commit: update cache with new K, V
                k, v = cache.update_and_fetch(k, v)
            else:
                # Denoise: read cached K, V and concatenate, but do NOT update cache
                k, v = cache.fetch_and_concat(k, v)

        if self.n_rep > 1:
            k = mx.repeat(k, self.n_rep, axis=1)
            v = mx.repeat(v, self.n_rep, axis=1)

        output = mx.fast.scaled_dot_product_attention(
            q, k, v, scale=self.scale, mask=mask
        )

        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(output)


class SDARMLP(nn.Module):
    """SiLU-gated MLP with fused gate+up projection."""

    def __init__(self, args: SDARModelArgs):
        super().__init__()
        self.intermediate_size = args.intermediate_size
        self.gate_up_proj = nn.Linear(args.hidden_size, 2 * args.intermediate_size, bias=False)
        self.down_proj = nn.Linear(args.intermediate_size, args.hidden_size, bias=False)

    def __call__(self, x):
        gate_up = self.gate_up_proj(x)
        gate = gate_up[..., :self.intermediate_size]
        up = gate_up[..., self.intermediate_size:]
        return self.down_proj(nn.silu(gate) * up)


class KVCache:
    """KV cache supporting both store (prefill) and read-only (denoise) modes."""

    def __init__(self):
        self._k = None
        self._v = None
        self._offset = 0

    @property
    def offset(self):
        return self._offset

    def update_and_fetch(self, k, v):
        """Store new K, V into cache and return full K, V (prefill mode)."""
        if self._k is None:
            self._k = k
            self._v = v
        else:
            self._k = mx.concatenate([self._k, k], axis=2)
            self._v = mx.concatenate([self._v, v], axis=2)
        self._offset += k.shape[2]
        return self._k, self._v

    def fetch_and_concat(self, k, v):
        """Concatenate cached K, V with current K, V without updating cache (denoise mode)."""
        if self._k is None:
            return k, v
        full_k = mx.concatenate([self._k, k], axis=2)
        full_v = mx.concatenate([self._v, v], axis=2)
        return full_k, full_v


class SDARDecoderLayer(nn.Module):
    """Single SDAR decoder layer with pre-norm residual connections."""

    def __init__(self, args: SDARModelArgs):
        super().__init__()
        self.self_attn = SDARAttention(args)
        self.mlp = SDARMLP(args)
        self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    def __call__(self, x, mask=None, cache=None, position_ids=None, store_kv=True):
        r = self.self_attn(
            self.input_layernorm(x),
            mask=mask, cache=cache,
            position_ids=position_ids, store_kv=store_kv,
        )
        h = x + r
        r = self.mlp(self.post_attention_layernorm(h))
        return h + r


class SDARModel(nn.Module):
    """SDAR transformer: embedding + N decoder layers + final norm."""

    def __init__(self, args: SDARModelArgs):
        super().__init__()
        self.args = args
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [SDARDecoderLayer(args) for _ in range(args.num_hidden_layers)]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    def __call__(self, inputs, cache=None, mask=None, position_ids=None, store_kv=True):
        h = self.embed_tokens(inputs)
        if cache is None:
            cache = [None] * len(self.layers)
        for i, layer in enumerate(self.layers):
            h = layer(h, mask=mask, cache=cache[i],
                      position_ids=position_ids, store_kv=store_kv)
        return self.norm(h)


class SDARForCausalLM(nn.Module):
    """SDAR model with LM head for causal language modeling."""

    def __init__(self, args: SDARModelArgs):
        super().__init__()
        self.args = args
        self.model = SDARModel(args)
        if args.tie_word_embeddings:
            self.lm_head = None
        else:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(self, inputs, cache=None, mask=None, position_ids=None, store_kv=True):
        h = self.model(inputs, cache=cache, mask=mask,
                       position_ids=position_ids, store_kv=store_kv)
        if self.lm_head is not None:
            return self.lm_head(h)
        return self.model.embed_tokens.as_linear(h)

    def sanitize(self, weights):
        """Fuse gate_proj + up_proj into gate_up_proj and drop unneeded keys."""
        fused = {}
        for name in list(weights):
            if name.endswith(".gate_proj.weight"):
                base = name.replace(".gate_proj.weight", "")
                up_key = f"{base}.up_proj.weight"
                if up_key in weights:
                    fused[f"{base}.gate_up_proj.weight"] = mx.concatenate(
                        [weights[name], weights[up_key]], axis=0
                    )

        result = {}
        for name, arr in weights.items():
            if name.endswith(".gate_proj.weight") or name.endswith(".up_proj.weight"):
                continue
            result[name] = arr
        result.update(fused)
        return result

    def make_cache(self):
        return [KVCache() for _ in range(self.args.num_hidden_layers)]

    @staticmethod
    def build_causal_mask(seq_len: int) -> mx.array:
        positions = mx.arange(seq_len)
        return positions[:, None] >= positions[None, :]


def load_sdar_model(model_path: str, dtype: mx.Dtype = mx.bfloat16):
    """Load SDAR model from a directory with safetensors weights."""
    import json
    from pathlib import Path

    model_dir = Path(model_path)

    with open(model_dir / "config.json") as f:
        config = json.load(f)

    args = SDARModelArgs.from_dict(config)
    model = SDARForCausalLM(args)

    weight_files = sorted(model_dir.glob("*.safetensors"))
    weights = {}
    for wf in weight_files:
        tensors = mx.load(str(wf))
        for name, arr in tensors.items():
            if arr.dtype != dtype:
                arr = arr.astype(dtype)
            weights[name] = arr

    weights = model.sanitize(weights)
    model.load_weights(list(weights.items()), strict=False)
    mx.eval(model.parameters())

    return model, config
