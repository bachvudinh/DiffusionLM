"""Sampling utilities for block diffusion token generation.

Derived from JetEngine by Yihan Bian et al.
Reference: https://github.com/Labman42/JetEngine

Architecture Overview
====================

    Provides temperature-scaled sampling with top-k and top-p filtering:

    ┌──────────────────────────────────────────────────────────────┐
    │          sample_with_temperature_topk_topp                   │
    │                                                              │
    │  logits: (batch, block_len, vocab_size)                      │
    │          │                                                   │
    │          ▼                                                   │
    │  1. Scale: logits / temperature                              │
    │          │                                                   │
    │          ▼                                                   │
    │  2. Top-k filter: keep k highest logits, mask rest to -inf   │
    │          │                                                   │
    │          ▼                                                   │
    │  3. Top-p filter: keep cumulative prob <= p, mask rest       │
    │          │                                                   │
    │          ▼                                                   │
    │  4. Softmax → probs                                          │
    │          │                                                   │
    │          ▼                                                   │
    │  5. Multinomial sample → token_ids, token_probs              │
    │                                                              │
    │  Output: (token_ids, token_probs) both (batch, block_len)    │
    └──────────────────────────────────────────────────────────────┘


Tensor Flow — Detailed Example
================================

    Example: batch=4, block_len=8, vocab_size=151936,
             temperature=0.7, top_k=50, top_p=0.9

    logits: (4, 8, 151936) float32
                │
                ▼  reshape → (32, 151936)
    logits: (32, 151936) float32
                │
                ▼  logits / 0.7
    logits: (32, 151936) float32         ← scaled (sharper distribution)
                │
                ▼  top_k_logits(logits, k=50)
                │    values, _ = topk(logits, 50)       → (32, 50)
                │    min_values = values[:, -1:]         → (32, 1)
                │    logits[logits < min_values] = -inf
    logits: (32, 151936) float32         ← only top-50 remain finite
                │
                ▼  top_p_logits(logits, p=0.9)
                │    sorted_logits = sort(logits, descending=True)
                │    cumprobs = cumsum(softmax(sorted))
                │    mask positions where cumprobs > 0.9
    logits: (32, 151936) float32         ← nucleus filtered
                │
                ▼  softmax(logits, dim=-1)
    probs: (32, 151936) float32          ← proper probability distribution
                │
                ▼  multinomial(probs, num_samples=1)
    token: (32, 1) int64
                │
                ▼  gather(probs, -1, token)
    token_prob: (32, 1) float32          ← prob of sampled token
                │
                ▼  view → (4, 8), (4, 8)
    token_ids:  (4, 8) int64
    token_probs: (4, 8) float32
"""

import torch
from torch.nn import functional as F


def top_k_logits(logits, k):
    """Zero out logits below the top-k threshold.

    Input:
        logits: (..., vocab_size)
        k:      int — number of top logits to keep (0 = no filtering)

    Output:
        (..., vocab_size) — filtered logits with -inf for removed entries
    """
    if k <= 0:
        return logits
    else:
        values, _ = torch.topk(logits, k)
        min_values = values[..., -1, None]
        return torch.where(logits < min_values,
                           torch.full_like(logits, float('-inf')), logits)


def top_p_logits(logits, p):
    """Zero out logits outside the nucleus (top-p) set.

    Input:
        logits: (..., vocab_size)
        p:      float — cumulative probability threshold (1.0 = no filtering)

    Output:
        (..., vocab_size) — filtered logits
    """
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    sorted_mask = cumulative_probs > p
    sorted_mask[..., 1:] = sorted_mask[..., :-1].clone()
    sorted_mask[..., 0] = False
    mask_indices = torch.scatter(
        torch.full_like(logits, False, dtype=torch.bool),
        -1, sorted_indices, sorted_mask)
    logits = logits.masked_fill(mask_indices, float('-inf'))
    return logits


def sample_with_temperature_topk_topp(logits, temperature=1.0, top_k=0, top_p=1.0):
    """Sample tokens from logits with temperature, top-k, and top-p.

    Input:
        logits:      (batch, block_len, vocab_size) or (batch*block_len, vocab_size)
        temperature: float — sampling temperature (1.0 = no scaling)
        top_k:       int — top-k filtering threshold (0 = disabled)
        top_p:       float — nucleus sampling threshold (1.0 = disabled)

    Output:
        (token_ids, token_probs) — both shaped (batch, block_len)
            token_ids:   int64 — sampled token indices
            token_probs: float — probability of each sampled token
    """
    orig_shape = logits.shape[:-1]
    vocab_size = logits.shape[-1]

    logits = logits.reshape(-1, vocab_size)

    if temperature != 1.0:
        logits = logits / temperature
    if top_k > 0:
        logits = top_k_logits(logits, top_k)
    if top_p < 1.0:
        logits = top_p_logits(logits, top_p)
    probs = F.softmax(logits, dim=-1)
    assert probs.dim() == 2
    token = torch.multinomial(probs, num_samples=1)
    token_prob = torch.gather(probs, -1, token)

    return token.view(*orig_shape), token_prob.view(*orig_shape)
