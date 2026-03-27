"""
SDAR Model - Synergy of Diffusion and AutoRegression

This module provides SDAR model implementation and inference.
Based on: https://github.com/JetAstra/SDAR
"""
from .modeling_sdar import SDARForCausalLM, SDARModel, SDARConfig
from .configuration_sdar import SDARConfig

__all__ = ["SDARForCausalLM", "SDARModel", "SDARConfig"]
