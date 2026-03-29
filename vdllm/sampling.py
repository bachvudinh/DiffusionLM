"""Sampling utilities for block diffusion generation in MLX."""

from functools import partial

import mlx.core as mx


# Compiled categorical sampling — fuses random draw + softmax + gather into
# one Metal kernel.  Random state is declared so mx.compile tracks the PRNG
# properly (same pattern as mlx-lm's sample_utils.py).
@partial(mx.compile, inputs=mx.random.state, outputs=mx.random.state)
def _compiled_sample(logits: mx.array) -> tuple:
    """Sample tokens and extract their probabilities (compiled)."""
    tokens = mx.random.categorical(logits)
    probs = mx.softmax(logits, axis=-1)
    token_probs = mx.take_along_axis(probs, tokens[:, None], axis=-1).squeeze(-1)
    return tokens, token_probs


def top_k_logits(logits: mx.array, k: int) -> mx.array:
    """Mask logits to keep only top-k values, setting the rest to -inf."""
    if k <= 0:
        return logits
    # Get the k-th largest value along the last axis
    # mx.topk returns the top-k values in ascending order
    top_values = mx.topk(logits, k=k, axis=-1)
    # The minimum of top-k is the first element (ascending order)
    min_values = top_values[..., 0:1]
    return mx.where(logits < min_values, float("-inf"), logits)


def top_p_logits(logits: mx.array, p: float) -> mx.array:
    """Nucleus (top-p) filtering: mask logits outside cumulative probability p."""
    # Sort descending
    sorted_indices = mx.argsort(-logits, axis=-1)
    sorted_logits = mx.take_along_axis(logits, sorted_indices, axis=-1)

    # Compute cumulative probabilities
    cumulative_probs = mx.cumsum(mx.softmax(sorted_logits, axis=-1), axis=-1)

    # Create mask: True for tokens to remove (cumulative prob > p)
    sorted_mask = cumulative_probs > p
    # Shift mask right by 1 so the first token above threshold is kept
    sorted_mask = mx.concatenate(
        [mx.zeros(sorted_mask[..., :1].shape, dtype=mx.bool_), sorted_mask[..., :-1]],
        axis=-1,
    )

    # Scatter the mask back to original positions
    inverse_indices = mx.argsort(sorted_indices, axis=-1)
    mask_original = mx.take_along_axis(sorted_mask, inverse_indices, axis=-1)

    return mx.where(mask_original, float("-inf"), logits)


def sample_with_temperature_topk_topp(
    logits: mx.array,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
) -> tuple:
    """Sample tokens from logits with temperature, top-k, and top-p filtering.

    Args:
        logits: shape (..., vocab_size)
        temperature: sampling temperature
        top_k: top-k filtering (0 = disabled)
        top_p: nucleus sampling threshold (1.0 = disabled)

    Returns:
        (tokens, token_probs) each with shape (...)
    """
    orig_shape = logits.shape[:-1]
    vocab_size = logits.shape[-1]

    # Flatten to 2D for processing
    logits = logits.reshape(-1, vocab_size)

    if temperature != 1.0:
        logits = logits / temperature

    if top_k > 0:
        logits = top_k_logits(logits, top_k)

    if top_p < 1.0:
        logits = top_p_logits(logits, top_p)

    # Use compiled sampling (fused random + softmax + gather)
    tokens, token_probs = _compiled_sample(logits)

    return tokens.reshape(orig_shape), token_probs.reshape(orig_shape)
