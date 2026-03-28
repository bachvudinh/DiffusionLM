"""Block diffusion generation pipeline for SDAR models in MLX."""

import time

import mlx.core as mx
import numpy as np

from .models.mlx_sdar import KVCache
from .sampling import sample_with_temperature_topk_topp


def get_num_transfer_tokens(block_length: int, steps: int) -> list:
    """Compute how many tokens to unmask at each denoising step.

    Distributes block_length tokens across steps as evenly as possible,
    with earlier steps getting one extra token if there's a remainder.

    Returns a Python list of ints.
    """
    base = block_length // steps
    remainder = block_length % steps
    result = [base + (1 if i < remainder else 0) for i in range(steps)]
    return result


def _build_block_attention_mask(num_blocks: int, block_length: int) -> mx.array:
    """Build the block-causal attention mask.

    Returns a boolean mask of shape (1, total_length, total_length) where
    True means "attend" and False means "don't attend".
    Each block can attend to itself and all previous blocks.
    """
    # Block-level lower-triangular mask
    block_mask = mx.array(np.tril(np.ones((num_blocks, num_blocks), dtype=np.bool_)))
    # Expand to token level by repeating along both axes
    # block_mask[i, j] -> token_mask[i*B:(i+1)*B, j*B:(j+1)*B]
    token_mask = mx.repeat(block_mask, block_length, axis=0)
    token_mask = mx.repeat(token_mask, block_length, axis=1)
    return token_mask[None, :, :]  # (1, total_len, total_len)


def block_diffusion_generate(
    model,
    input_ids: mx.array,
    mask_id: int,
    gen_length: int = 128,
    block_length: int = 8,
    denoising_steps: int = 8,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
    remasking_strategy: str = "low_confidence_dynamic",
    confidence_threshold: float = 0.85,
    eb_threshold: float = None,
    stopping_criteria_idx: list = None,
):
    """Generate text using block diffusion with the SDAR model.

    Args:
        model: SDARForCausalLM instance
        input_ids: prompt token ids, shape (1, prompt_length)
        mask_id: token id used for [MASK]
        gen_length: number of tokens to generate
        block_length: size of each diffusion block
        denoising_steps: number of denoising iterations per block
        temperature: sampling temperature
        top_k: top-k filtering
        top_p: nucleus sampling threshold
        remasking_strategy: one of 'sequential', 'low_confidence_static',
            'low_confidence_dynamic', 'entropy_bounded'
        confidence_threshold: threshold for low_confidence_dynamic strategy
        eb_threshold: entropy budget for entropy_bounded strategy
        stopping_criteria_idx: list of token ids that signal early stopping

    Returns:
        (output_ids, timing) where output_ids is mx.array of shape (1, total_length)
        and timing is a dict with prefill_time, decode_time, prompt_tokens, generated_tokens.
    """
    prompt_length = input_ids.shape[1]

    # Compute block-aligned total length
    num_blocks = (prompt_length + gen_length + block_length - 1) // block_length
    total_length = num_blocks * block_length

    # Build full block-causal attention mask (boolean: True=attend)
    block_attn_mask = _build_block_attention_mask(num_blocks, block_length)

    # Position ids
    position_ids = mx.arange(total_length).reshape(1, -1)

    # Initialize sequence: prompt + mask tokens
    x_np = np.full((1, total_length), mask_id, dtype=np.int32)
    x_np[0, :prompt_length] = np.array(input_ids[0])
    x = mx.array(x_np)

    # Create KV cache
    cache = model.make_cache()

    # Prefill: process complete prompt blocks
    prefill_blocks = prompt_length // block_length
    prefill_length = prefill_blocks * block_length

    prefill_start = time.perf_counter()
    if prefill_length > 0:
        cur_x = x[:, :prefill_length]
        # Slice attention mask for prefill region
        cur_attn_mask = block_attn_mask[:, :prefill_length, :prefill_length]
        cur_position_ids = position_ids[:, :prefill_length]

        # Run prefill (store_kv=True to populate cache)
        model(cur_x, cache=cache,
              mask=_bool_to_additive(cur_attn_mask),
              position_ids=cur_position_ids, store_kv=True)
        mx.eval(*[c._k for c in cache if c._k is not None],
                *[c._v for c in cache if c._v is not None])
    prefill_time = time.perf_counter() - prefill_start

    # Compute transfer schedule
    num_transfer_tokens = get_num_transfer_tokens(block_length, denoising_steps)

    decode_start = time.perf_counter()
    generated_tokens = 0

    # Decode: process each block with iterative denoising
    for num_block in range(prefill_blocks, num_blocks):
        block_start = num_block * block_length
        block_end = (num_block + 1) * block_length

        # Extract current block tokens
        cur_x = mx.array(np.array(x[0, block_start:block_end])).reshape(1, -1)

        # Attention mask: current block attending to all tokens up to and including this block
        cur_attn_mask = block_attn_mask[:, block_start:block_end, :block_end]
        cur_position_ids = position_ids[:, block_start:block_end]

        # When using cache, the mask for SDPA needs shape [1, block_length, cache_len + block_length]
        # The cache stores tokens 0..block_start-1 (if prefilled), so we only need
        # the portion of the mask covering [cache_offset..block_end)
        # But our cache contains exactly the prefilled + previously committed blocks.
        cache_len = cache[0].offset if cache[0]._k is not None else 0
        # The attention mask for the current query tokens over [cached_keys, current_keys]
        # = cur_attn_mask[:, :, :cache_len + block_length] but we need to slice from the
        # full mask appropriately. cached keys cover positions 0..cache_len-1, current cover
        # cache_len..cache_len+block_length-1 which equals block_start..block_end-1.
        # So the mask cols are [0..cache_len-1, block_start..block_end-1]
        if cache_len > 0:
            mask_cached = cur_attn_mask[:, :, :cache_len]
            mask_current = cur_attn_mask[:, :, block_start:block_end]
            denoise_attn_mask = mx.concatenate([mask_cached, mask_current], axis=-1)
        else:
            denoise_attn_mask = cur_attn_mask[:, :, block_start:block_end]

        for step in range(denoising_steps + 1):
            # Check which positions are still masked
            cur_x_np = np.array(cur_x[0])
            mask_index = cur_x_np == mask_id  # (block_length,)

            if mask_index.sum() == 0:
                # All tokens unmasked - commit block to cache and move on
                model(cur_x, cache=cache,
                      mask=_bool_to_additive(denoise_attn_mask),
                      position_ids=cur_position_ids, store_kv=True)
                mx.eval(*[c._k for c in cache if c._k is not None],
                        *[c._v for c in cache if c._v is not None])
                break

            # Run model in denoise mode (don't update cache)
            logits = model(cur_x, cache=cache,
                          mask=_bool_to_additive(denoise_attn_mask),
                          position_ids=cur_position_ids, store_kv=False)

            # Sample
            x0, x0_p = sample_with_temperature_topk_topp(
                logits, temperature=temperature, top_k=top_k, top_p=top_p
            )
            mx.eval(x0, x0_p)

            x0_np = np.array(x0[0].astype(mx.int32))       # (block_length,)
            x0_p_np = np.array(x0_p[0].astype(mx.float32))   # (block_length,)

            if step == denoising_steps:
                # Last step: unmask everything remaining
                transfer_index = mask_index.copy()
            elif remasking_strategy == "sequential":
                transfer_index = np.zeros_like(mask_index)
                if mask_index.any():
                    first_mask = np.argmax(mask_index)
                    end = min(first_mask + num_transfer_tokens[step], len(mask_index))
                    transfer_index[first_mask:end] = True

            elif remasking_strategy == "low_confidence_static":
                confidence = np.where(mask_index, x0_p_np, -np.inf)
                n_transfer = num_transfer_tokens[step]
                top_indices = np.argsort(-confidence)[:n_transfer]
                transfer_index = np.zeros_like(mask_index)
                transfer_index[top_indices] = True

            elif remasking_strategy == "low_confidence_dynamic":
                confidence = np.where(mask_index, x0_p_np, -np.inf)
                high_conf = confidence > confidence_threshold
                num_high = high_conf.sum()
                n_transfer = num_transfer_tokens[step]

                if num_high >= n_transfer:
                    transfer_index = high_conf
                else:
                    top_indices = np.argsort(-confidence)[:n_transfer]
                    transfer_index = np.zeros_like(mask_index)
                    transfer_index[top_indices] = True

            elif remasking_strategy == "entropy_bounded":
                # Compute per-token entropy from probabilities
                eps = 1e-12
                p_clamped = np.clip(x0_p_np, eps, None)
                entropies = -(p_clamped * np.log(p_clamped))
                entropies = np.where(mask_index, entropies, np.inf)

                order = np.argsort(entropies)
                ent_sorted = entropies[order]
                cumsum = np.cumsum(ent_sorted)

                k = np.searchsorted(cumsum, eb_threshold, side="left")
                k = max(1, min(k, int(mask_index.sum())))
                transfer_index = np.zeros_like(mask_index)
                transfer_index[order[:k]] = True
            else:
                raise ValueError(f"Unknown remasking strategy: {remasking_strategy}")

            # Apply transfer: update cur_x where transfer_index is True
            cur_x_np = np.array(cur_x[0])
            cur_x_np[transfer_index] = x0_np[transfer_index]
            cur_x = mx.array(cur_x_np).reshape(1, -1)

        # Commit the denoised block into x
        x_np = np.array(x)
        x_np[0, block_start:block_end] = np.array(cur_x[0])
        x = mx.array(x_np)

        # If we didn't break early (all unmasked), we need to commit to cache
        # Check if the last step was denoising_steps+1 (loop completed without break)
        if cache[0].offset < block_end:
            # Need to commit this block to cache for subsequent blocks
            model(cur_x, cache=cache,
                  mask=_bool_to_additive(denoise_attn_mask),
                  position_ids=cur_position_ids, store_kv=True)
            mx.eval(*[c._k for c in cache if c._k is not None],
                    *[c._v for c in cache if c._v is not None])

        generated_tokens += block_length

        # Early stopping check
        if stopping_criteria_idx is not None:
            generated = np.array(x[0, prompt_length:])
            if any(int(stop_id) in generated for stop_id in stopping_criteria_idx):
                break

    decode_time = time.perf_counter() - decode_start

    timing = {
        "prefill_time": prefill_time,
        "decode_time": decode_time,
        "prompt_tokens": prefill_length,
        "generated_tokens": generated_tokens,
    }
    return x, timing


def _bool_to_additive(mask: mx.array, dtype=mx.bfloat16) -> mx.array:
    """Convert boolean mask (True=attend) to additive mask (0=attend, -inf=don't attend).

    The input is expected to be shape (1, query_len, key_len).
    The output is (1, 1, query_len, key_len) for broadcasting over heads.
    dtype should match the model's compute dtype.
    """
    # Convert: True -> 0.0, False -> -inf
    zeros = mx.zeros(mask.shape, dtype=dtype)
    neginf = mx.full(mask.shape, float("-inf"), dtype=dtype)
    additive = mx.where(mask, zeros, neginf)
    # Add head dimension: (1, Q, K) -> (1, 1, Q, K)
    return additive[:, None, :, :]
