"""
VDLLM Models.

This package provides the SDAR model implementations.

Based on JetEngine's SDAR implementation:
https://github.com/Jet-Astra/SDAR
"""

from .config import SDARConfig
from .sdar import SDARModel, SDARForCausalLM

__all__ = [
    "SDARConfig",
    "SDARModel",
    "SDARForCausalLM",
]
