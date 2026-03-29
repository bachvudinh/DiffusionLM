"""Block diffusion generation pipeline for SDAR models in MLX."""

import time

import mlx.core as mx

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
    # Block-level lower-triangular mask (pure MLX, no numpy)
    indices = mx.arange(num_blocks)
    block_mask = indices[:, None] >= indices[None, :]
    # Expand to token level by repeating along both axes
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

    # Initialize sequence: prompt + mask tokens (pure MLX)
    pad_length = total_length - prompt_length
    if pad_length > 0:
        x = mx.concatenate([
            input_ids,
            mx.full((1, pad_length), mask_id, dtype=mx.int32)
        ], axis=1)
    else:
        x = input_ids[:, :total_length]

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

    # Pre-compute position indices for remasking (reused every block)
    block_positions = mx.arange(block_length)

    # Decode: process each block with iterative denoising
    for num_block in range(prefill_blocks, num_blocks):
        block_start = num_block * block_length
        block_end = (num_block + 1) * block_length

        # Extract current block tokens
        cur_x = x[:, block_start:block_end]

        # Attention mask: current block attending to all tokens up to and including this block
        cur_attn_mask = block_attn_mask[:, block_start:block_end, :block_end]
        cur_position_ids = position_ids[:, block_start:block_end]

        # Build denoise attention mask accounting for KV cache
        cache_len = cache[0].offset if cache[0]._k is not None else 0
        if cache_len > 0:
            mask_cached = cur_attn_mask[:, :, :cache_len]
            mask_current = cur_attn_mask[:, :, block_start:block_end]
            denoise_attn_mask = mx.concatenate([mask_cached, mask_current], axis=-1)
        else:
            denoise_attn_mask = cur_attn_mask[:, :, block_start:block_end]

        for step in range(denoising_steps + 1):
            # Check which positions are still masked (stay in MLX)
            mask_index = (cur_x[0] == mask_id)  # (block_length,) bool
            num_masked = mx.sum(mask_index)
            mx.eval(num_masked)

            if num_masked.item() == 0:
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

            x0 = x0[0].astype(mx.int32)        # (block_length,)
            x0_p = x0_p[0].astype(mx.float32)   # (block_length,)

            if step == denoising_steps:
                # Last step: unmask everything remaining
                transfer_index = mask_index

            elif remasking_strategy == "sequential":
                # Unmask n_transfer positions starting from first masked
                first_mask = mx.argmax(mask_index.astype(mx.int32))
                mx.eval(first_mask)
                n_transfer = num_transfer_tokens[step]
                first_pos = first_mask.item()
                transfer_index = (
                    mask_index
                    & (block_positions >= first_pos)
                    & (block_positions < first_pos + n_transfer)
                )

            elif remasking_strategy == "low_confidence_static":
                confidence = mx.where(mask_index, x0_p, float('-inf'))
                n_transfer = num_transfer_tokens[step]
                # rank[i] = position of element i in descending-confidence order
                rank = mx.argsort(mx.argsort(-confidence))
                transfer_index = (rank < n_transfer) & mask_index

            elif remasking_strategy == "low_confidence_dynamic":
                confidence = mx.where(mask_index, x0_p, float('-inf'))
                high_conf = confidence > confidence_threshold
                num_high = mx.sum(high_conf)
                mx.eval(num_high)
                n_transfer = num_transfer_tokens[step]

                if num_high.item() >= n_transfer:
                    transfer_index = high_conf
                else:
                    rank = mx.argsort(mx.argsort(-confidence))
                    transfer_index = (rank < n_transfer) & mask_index

            elif remasking_strategy == "entropy_bounded":
                # Per-token entropy from sampled probability
                p_clamped = mx.maximum(x0_p, 1e-12)
                entropies = -(p_clamped * mx.log(p_clamped))
                entropies = mx.where(mask_index, entropies, float('inf'))

                # Cumulative entropy in ascending order
                order = mx.argsort(entropies)
                ent_sorted = entropies[order]
                cumsum = mx.cumsum(ent_sorted)

                # Count tokens within entropy budget
                within_budget = cumsum < eb_threshold
                k = mx.sum(within_budget)
                mx.eval(k)
                k = max(1, min(int(k.item()), int(num_masked.item())))

                rank = mx.argsort(mx.argsort(entropies))
                transfer_index = (rank < k) & mask_index
            else:
                raise ValueError(f"Unknown remasking strategy: {remasking_strategy}")

            # Apply transfer: update cur_x where transfer_index is True
            cur_x = mx.where(
                transfer_index.reshape(1, -1),
                x0.reshape(1, -1),
                cur_x
            )

        # Commit the denoised block into x (pure MLX concatenation)
        if block_end < total_length:
            x = mx.concatenate([x[:, :block_start], cur_x, x[:, block_end:]], axis=1)
        else:
            x = mx.concatenate([x[:, :block_start], cur_x], axis=1)

        # If we didn't break early, commit to cache for subsequent blocks
        if cache[0].offset < block_end:
            model(cur_x, cache=cache,
                  mask=_bool_to_additive(denoise_attn_mask),
                  position_ids=cur_position_ids, store_kv=True)
            mx.eval(*[c._k for c in cache if c._k is not None],
                    *[c._v for c in cache if c._v is not None])

        generated_tokens += block_length

        # Early stopping check (pure MLX)
        if stopping_criteria_idx is not None:
            generated = x[0, prompt_length:]
            stop_ids = mx.array(
                [int(s) for s in stopping_criteria_idx], dtype=mx.int32
            )
            # Check if any generated token matches any stop id
            matches = generated[:, None] == stop_ids[None, :]
            has_stop = mx.any(matches)
            mx.eval(has_stop)
            if has_stop.item():
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
