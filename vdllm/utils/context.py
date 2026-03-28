"""Global context for passing runtime information to Triton kernels.

Derived from JetEngine by Yihan Bian et al.
Reference: https://github.com/Labman42/JetEngine

Architecture Overview
====================

    The Context system provides a thread-global mechanism for passing
    runtime state (PREFILL vs DENOISE, KV cache metadata) to attention
    kernels without threading it through every function call.

    ┌─────────────────────────────────────────────────────────────┐
    │                   Context Lifecycle                          │
    │                                                              │
    │  ModelRunner.prepare_prefill() / prepare_denoise()           │
    │         │                                                    │
    │         ▼                                                    │
    │  set_context(run_type, cu_seqlens, slot_mapping, ...)        │
    │         │                                                    │
    │         ▼                                                    │
    │  model.forward() → layers use get_context() internally       │
    │    ├── BlockAttention reads run_type, cu_seqlens             │
    │    ├── store_kvcache reads slot_mapping                      │
    │    └── ParallelLMHead reads cu_seqlens for last-token        │
    │         │                                                    │
    │         ▼                                                    │
    │  reset_context()                                             │
    └─────────────────────────────────────────────────────────────┘

    Context Fields:
        run_type:     RunType.PREFILL or RunType.DENOISE
        cu_seqlens_q: cumulative sequence lengths for queries (prefill)
        cu_seqlens_k: cumulative sequence lengths for keys (prefill)
        slot_mapping:  physical KV cache slot indices (prefill)
        context_lens:  cached token counts per sequence (denoise)
        block_tables:  physical block assignments per sequence (denoise)
        block_length:  block size for denoising
"""

from dataclasses import dataclass, field
from typing import List
import torch

from vdllm.engine.sequence import RunType


@dataclass
class Context:
    run_type: RunType | None = None
    cu_seqlens_q: torch.Tensor | None = None
    cu_seqlens_k: torch.Tensor | None = None
    max_seqlen_q: int = 0
    max_seqlen_k: int = 0
    slot_mapping: torch.Tensor | None = None
    context_lens: torch.Tensor | None = None
    block_tables: torch.Tensor | None = None
    is_last_denoise_step: List[bool] = field(default_factory=lambda: [False])
    block_length: int = 4


_CONTEXT = Context()


def get_context():
    return _CONTEXT


def set_context(run_type, cu_seqlens_q=None, cu_seqlens_k=None,
                max_seqlen_q=0, max_seqlen_k=0, slot_mapping=None,
                context_lens=None, block_tables=None,
                is_last_denoise_step=[False], block_length=4):
    global _CONTEXT
    _CONTEXT = Context(run_type, cu_seqlens_q, cu_seqlens_k,
                       max_seqlen_q, max_seqlen_k, slot_mapping,
                       context_lens, block_tables,
                       is_last_denoise_step, block_length)


def reset_context():
    global _CONTEXT
    _CONTEXT = Context()
