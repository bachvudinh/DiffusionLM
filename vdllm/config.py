"""Engine configuration for vdllm.

Derived from JetEngine by Yihan Bian et al.
Reference: https://github.com/Labman42/JetEngine

Architecture Overview
====================

    Config is the central dataclass that governs every aspect of inference:

    ┌────────────────────────────────────────────────────────────────────────┐
    │                         Config Dataclass                               │
    │                                                                        │
    │  Model Identity                                                        │
    │  ├── model: str              path to HuggingFace checkpoint            │
    │  ├── hf_config: AutoConfig   loaded HF config (auto-populated)         │
    │  └── dtype: str              'auto' | 'bfloat16' | 'float16'           │
    │                                                                        │
    │  Resource Limits                                                       │
    │  ├── max_num_batched_tokens   max tokens per forward pass (128K)       │
    │  ├── max_num_seqs             max concurrent sequences (512)           │
    │  ├── max_model_len            max context length per seq               │
    │  └── gpu_memory_utilization   fraction of VRAM to use (0.8)            │
    │                                                                        │
    │  Parallelism                                                           │
    │  └── tensor_parallel_size     number of GPUs for TP (1-8)              │
    │                                                                        │
    │  Block Diffusion                                                       │
    │  ├── block_length             tokens per denoising block (4)           │
    │  ├── kvcache_block_size       physical KV cache block size (256)       │
    │  └── mask_token_id            special [MASK] token ID                  │
    │                                                                        │
    │  Execution Mode                                                        │
    │  └── enforce_eager            skip CUDA graph capture if True          │
    │                                                                        │
    │  Derived (auto-populated from hf_config)                               │
    │  ├── hidden_size, num_attention_heads, num_key_value_heads             │
    │  ├── num_hidden_layers, head_dim, max_position_embeddings             │
    │  └── torch_dtype: torch.dtype                                          │
    └────────────────────────────────────────────────────────────────────────┘

    Input:
        model: str — path to HF model directory (must exist on disk)
        **kwargs   — any field above can be overridden

    Output:
        Config instance with validated and auto-populated fields
"""

import os
import torch
from dataclasses import dataclass
from transformers import AutoConfig


def get_cfg_alias(cfg, name, *candidates):
    """Return the first matching attribute from cfg."""
    for key in (name, *candidates):
        if hasattr(cfg, key):
            return getattr(cfg, key)
    raise AttributeError(f"{name} not found on config")


@dataclass
class Config:
    model: str
    max_num_batched_tokens: int = 1024 * 128
    max_num_seqs: int = 512
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.8
    tensor_parallel_size: int = 1
    enforce_eager: bool = False
    diversity_enforce: bool = False
    epsilon_greedy: bool = False
    epsilon: float = 0.1
    diversity_enforce_barrier: int = 100
    hf_config: AutoConfig | None = None
    eos: int = -1
    kvcache_block_size: int = 256
    num_kvcache_blocks: int = -1
    mask_token_id: int = -1
    block_length: int = 4
    dtype: str = 'auto'

    def __post_init__(self):
        assert os.path.isdir(self.model)
        assert self.kvcache_block_size % 256 == 0
        assert 1 <= self.tensor_parallel_size <= 8
        cfg = AutoConfig.from_pretrained(self.model, trust_remote_code=True)
        self.hf_config = cfg

        # Determine torch_dtype
        if self.dtype == "bfloat16":
            self.torch_dtype = torch.bfloat16
        elif self.dtype == "float16":
            self.torch_dtype = torch.float16
        else:
            if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
                self.torch_dtype = torch.bfloat16
            else:
                self.torch_dtype = torch.float16

        # Standard HF config fields (SDAR models)
        self.hidden_size = get_cfg_alias(cfg, "hidden_size")
        self.num_attention_heads = get_cfg_alias(cfg, "num_attention_heads")
        self.num_key_value_heads = get_cfg_alias(cfg, "num_key_value_heads")
        self.num_hidden_layers = get_cfg_alias(cfg, "num_hidden_layers")
        self.max_position_embeddings = get_cfg_alias(cfg, "max_position_embeddings")
        self.head_dim = get_cfg_alias(cfg, "head_dim")

        self.max_model_len = min(self.max_model_len, self.max_position_embeddings)
        assert self.max_num_batched_tokens >= self.max_model_len
        assert self.mask_token_id != -1, "Mask token ID must be set"
