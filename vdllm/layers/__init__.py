"""
SDAR Model Layers.

This package provides the building blocks for the SDAR model, implemented
in pure PyTorch without transformers dependency.

Based on JetEngine's layer implementations:
https://github.com/Jet-Astra/SDAR

Layers:
    - rmsnorm.py: RMSNorm normalization
    - attention.py: Multi-head attention with GQA
    - mlp.py: Feed-forward network (SiLU/SWiGLU)
    - rotary.py: Rotary position embeddings (RoPE)
    - embedding.py: Token embeddings

================================================================================
"""

from .rmsnorm import RMSNorm
from .attention import SDARAttention, Attention
from .mlp import SDARMLP
from .rotary import apply_rotary_pos_emb, precompute_freqs_cis
from .embedding import TokenEmbedding, LMHead

__all__ = [
    "RMSNorm",
    "SDARAttention",
    "Attention",
    "SDARMLP",
    "apply_rotary_pos_emb",
    "precompute_freqs_cis",
    "TokenEmbedding",
    "LMHead",
]
