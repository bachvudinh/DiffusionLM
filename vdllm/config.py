"""Engine configuration for vdllm.

Derived from JetEngine by Yihan Bian et al.
Reference: https://github.com/Labman42/JetEngine

Backend-agnostic: no torch/mlx imports at module level.
Supports CUDA (NVIDIA GPU), MLX (Apple Silicon), MPS, and CPU backends.
"""

import os
from dataclasses import dataclass, field
from typing import Any

from vdllm.utils.hardware import detect_backend


def _get_cfg_alias(cfg, name, *candidates):
    """Return the first matching attribute from cfg."""
    for key in (name, *candidates):
        if hasattr(cfg, key):
            return getattr(cfg, key)
    raise AttributeError(f"{name} not found on config")


@dataclass
class Config:
    """Unified config for all vdllm backends.

    Fields marked [CUDA-only] are ignored by the MLX engine.
    """
    model: str

    # Backend: 'cuda', 'mlx', 'mps', 'cpu', or 'auto'
    backend: str = "auto"

    # Dtype: 'auto', 'bfloat16', 'float16', 'float32'
    dtype: str = "auto"

    # Block diffusion
    block_length: int = 4
    mask_token_id: int = -1
    eos: int = -1

    # Resource limits [CUDA-only]
    max_num_batched_tokens: int = 1024 * 128
    max_num_seqs: int = 512
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.8

    # Parallelism [CUDA-only]
    tensor_parallel_size: int = 1

    # Execution mode [CUDA-only]
    enforce_eager: bool = False

    # Sampling diversity [CUDA-only]
    diversity_enforce: bool = False
    epsilon_greedy: bool = False
    epsilon: float = 0.1
    diversity_enforce_barrier: int = 100

    # KV cache [CUDA-only]
    kvcache_block_size: int = 256
    num_kvcache_blocks: int = -1

    # Populated by __post_init__
    hf_config: Any = field(default=None, repr=False)

    def __post_init__(self):
        assert os.path.isdir(self.model), f"Model path does not exist: {self.model}"

        # Auto-detect backend
        if self.backend == "auto":
            self.backend = detect_backend()

        # Load HF config (transformers is always available)
        from transformers import AutoConfig
        self.hf_config = AutoConfig.from_pretrained(
            self.model, trust_remote_code=True)

        # Extract model architecture from HF config
        cfg = self.hf_config
        self.hidden_size = _get_cfg_alias(cfg, "hidden_size")
        self.num_attention_heads = _get_cfg_alias(cfg, "num_attention_heads")
        self.num_key_value_heads = _get_cfg_alias(cfg, "num_key_value_heads")
        self.num_hidden_layers = _get_cfg_alias(cfg, "num_hidden_layers")
        self.max_position_embeddings = _get_cfg_alias(
            cfg, "max_position_embeddings")
        self.head_dim = _get_cfg_alias(cfg, "head_dim")

        # Backend-specific validation
        if self.backend == "cuda":
            self._init_cuda()
        elif self.backend == "mlx":
            self._init_mlx()

    def _init_cuda(self):
        """CUDA-specific initialization."""
        import torch

        assert self.kvcache_block_size % 256 == 0
        assert 1 <= self.tensor_parallel_size <= 8

        # Resolve torch_dtype
        if self.dtype == "bfloat16":
            self.torch_dtype = torch.bfloat16
        elif self.dtype == "float16":
            self.torch_dtype = torch.float16
        elif self.dtype == "float32":
            self.torch_dtype = torch.float32
        else:
            if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
                self.torch_dtype = torch.bfloat16
            else:
                self.torch_dtype = torch.float16

        self.max_model_len = min(
            self.max_model_len, self.max_position_embeddings)
        assert self.max_num_batched_tokens >= self.max_model_len

    def _init_mlx(self):
        """MLX-specific initialization."""
        import mlx.core as mx

        if self.dtype == "float16":
            self.mlx_dtype = mx.float16
        elif self.dtype == "float32":
            self.mlx_dtype = mx.float32
        else:
            self.mlx_dtype = mx.bfloat16
