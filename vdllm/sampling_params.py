"""Sampling parameters for block diffusion decoding.

Derived from JetEngine by Yihan Bian et al.
Reference: https://github.com/Labman42/JetEngine

Architecture Overview
====================

    SamplingParams controls how tokens are sampled and how masked positions
    are selected for unmasking during the block diffusion process.

    ┌────────────────────────────────────────────────────────────────────────┐
    │                     Block Diffusion Decoding                           │
    │                                                                        │
    │  For each block of `block_length` tokens:                              │
    │                                                                        │
    │    Step 0: [MASK, MASK, MASK, MASK]  ← all masked                     │
    │                    │                                                    │
    │                    ▼                                                    │
    │            model(block) → logits (block_length, V)                     │
    │                    │                                                    │
    │                    ▼                                                    │
    │            temperature / topk / topp → sample → confidences            │
    │                    │                                                    │
    │                    ▼                                                    │
    │            remasking_strategy selects which positions to unmask:        │
    │                                                                        │
    │    ┌──────────────────────────────────────────────────────────────┐    │
    │    │  sequential:            left-to-right unmasking              │    │
    │    │  low_confidence_static: unmask highest-confidence first      │    │
    │    │  low_confidence_dynamic: threshold + fallback                │    │
    │    │  entropy_bounded:       unmask lowest-entropy first          │    │
    │    │  random:                random selection                     │    │
    │    └──────────────────────────────────────────────────────────────┘    │
    │                    │                                                    │
    │    Step 1: [tok_0, MASK, tok_2, MASK]  ← partially unmasked           │
    │                    │                                                    │
    │                    ▼  ... repeat for `denoising_steps` ...              │
    │                    │                                                    │
    │    Final:  [tok_0, tok_1, tok_2, tok_3]  ← fully denoised             │
    └────────────────────────────────────────────────────────────────────────┘

    Input:
        All fields are optional with sensible defaults.

    Output:
        SamplingParams instance controlling generation behavior.
"""

from dataclasses import dataclass
from typing import Literal


@dataclass
class SamplingParams:
    temperature: float = 1.0
    max_tokens: int = 64
    ignore_eos: bool = False

    # Block Diffusion Parameters
    block_length: int = 4
    denoising_steps: int = 4
    dynamic_threshold: float = 0.9
    eb_threshold: float = 0.35
    topk: int = 0
    topp: float = 1
    repetition_penalty: float = 1.0
    remasking_strategy: Literal[
        'sequential',
        'low_confidence_static',
        'low_confidence_dynamic',
        'entropy_bounded',
        'random',
    ] = 'low_confidence_static'
    stop_words: list[int] | None = None
