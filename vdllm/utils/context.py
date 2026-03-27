"""
Context Management for Block Diffusion.

This module provides thread-local-like global context for tracking
sequence metadata during PREFILL and DENOISE runs.

Based on JetEngine's context.py:
https://github.com/Jet-Astra/SDAR/blob/main/jetengine/utils/context.py

================================================================================
                              USAGE
================================================================================

    from vdllm.utils import set_context, get_context

    # During prefill
    set_context(run_type="PREFILL", cu_seqlens_q=..., max_seqlen_q=1024)

    # During denoise
    set_context(run_type="DENOISE", block_length=4, is_last_denoise_step=False)

    ctx = get_context()
    print(ctx.run_type)

"""

from dataclasses import dataclass, field
from typing import List, Optional
import torch


@dataclass
class Context:
    """Context for tracking sequence metadata during inference."""
    run_type: Optional[str] = None  # "PREFILL" or "DENOISE"
    cu_seqlens_q: Optional[torch.Tensor] = None  # Cumulative sequence lengths for Q
    cu_seqlens_k: Optional[torch.Tensor] = None  # Cumulative sequence lengths for K
    max_seqlen_q: int = 0
    max_seqlen_k: int = 0
    slot_mapping: Optional[torch.Tensor] = None  # Position to cache slot mapping
    context_lens: Optional[torch.Tensor] = None  # Length of each context
    block_tables: Optional[torch.Tensor] = None  # Block allocation table
    is_last_denoise_step: List[bool] = field(default_factory=lambda: [False])
    block_length: int = 4


# Global context instance
_CONTEXT = Context()


def get_context() -> Context:
    """Get the current global context."""
    return _CONTEXT


def set_context(
    run_type: Optional[str] = None,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    cu_seqlens_k: Optional[torch.Tensor] = None,
    max_seqlen_q: int = 0,
    max_seqlen_k: int = 0,
    slot_mapping: Optional[torch.Tensor] = None,
    context_lens: Optional[torch.Tensor] = None,
    block_tables: Optional[torch.Tensor] = None,
    is_last_denoise_step: List[bool] = None,
    block_length: int = 4,
) -> None:
    """Set the global context."""
    global _CONTEXT
    _CONTEXT = Context(
        run_type=run_type,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        slot_mapping=slot_mapping,
        context_lens=context_lens,
        block_tables=block_tables,
        is_last_denoise_step=is_last_denoise_step or [False],
        block_length=block_length,
    )


def reset_context() -> None:
    """Reset the global context to defaults."""
    global _CONTEXT
    _CONTEXT = Context()
