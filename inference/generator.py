"""Full generation pipeline for Block Diffusion Language Models.

High-level generation flow:
1. Encode prompt → token IDs
2. Warmup KV cache with full prompt blocks
3. For each generation block:
   a. Initialize: prompt remainder + [MASK] tokens
   b. Denoise loop: forward → Gumbel sample → confidence unmask
   c. Cache finalized block's KV
   d. Extract generated tokens
4. Decode tokens → text

The model must implement:
- __call__(x, pos_offset) → (logits, None)
- set_cache_mode(enabled: bool)
- reset_kv_cache()
"""

import time
from typing import Callable, Optional

import torch

from .denoiser import BlockDenoiser


@torch.no_grad()
def generate(
    model,
    encode_fn: Callable[[str], list[int]],
    decode_fn: Callable[[list[int]], str],
    prompt: str = '',
    max_new_tokens: int = 512,
    block_size: int = 32,
    mask_token_id: int = 0,
    eos_token_id: int = 1,
    pad_token_id: int = 2,
    denoise_steps: int = 10,
    temperature: float = 0.7,
    top_k: int = 50,
    verbose: bool = True,
) -> str:
    """Generate text using Block Diffusion LM.

    Args:
        model: Language model with forward(x, pos_offset) → (logits, _)
        encode_fn: str → list[int] tokenization function
        decode_fn: list[int] → str detokenization function
        prompt: Input prompt string
        max_new_tokens: Maximum tokens to generate
        block_size: Number of tokens per diffusion block
        mask_token_id: Token ID for [MASK]
        eos_token_id: Token ID for end-of-sequence
        pad_token_id: Token ID for padding
        denoise_steps: Denoising steps per block
        temperature: Sampling temperature (0 = greedy)
        top_k: Top-k filtering (0 = disabled)
        verbose: Print generation stats

    Returns:
        Generated text string (detokenized)
    """
    was_training = getattr(model, 'training', False)
    if was_training:
        model.eval()

    try:
        # Initialize model state
        if hasattr(model, 'reset_kv_cache'):
            model.reset_kv_cache()
        if hasattr(model, 'set_cache_mode'):
            model.set_cache_mode(False)

        device = next(iter(model.parameters())).device

        # Encode prompt
        prompt_ids = encode_fn(prompt) if prompt else []
        prompt_len = len(prompt_ids)
        all_tokens = list(prompt_ids)

        # Split prompt into blocks
        n_full_prompt_blocks = prompt_len // block_size
        prompt_remainder = prompt_len % block_size
        pos_offset = 0

        # Warmup KV cache with full prompt blocks
        if hasattr(model, 'set_cache_mode'):
            model.set_cache_mode(True)
        for i in range(n_full_prompt_blocks):
            start = i * block_size
            end = start + block_size
            block_ids = torch.tensor(
                [prompt_ids[start:end]],
                dtype=torch.long,
                device=device,
            )
            model(block_ids, pos_offset=pos_offset)
            pos_offset += block_size
        if hasattr(model, 'set_cache_mode'):
            model.set_cache_mode(False)

        # Create denoiser
        denoiser = BlockDenoiser(
            block_size=block_size,
            mask_token_id=mask_token_id,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            denoise_steps=denoise_steps,
            temperature=temperature,
            top_k=top_k,
        )

        # Block-by-block generation
        tokens_generated = 0
        total_steps = 0
        done = False
        t_start = time.time()

        def logits_callback(m, block, offset):
            return m(block, pos_offset=offset)

        while tokens_generated < max_new_tokens and not done:
            # How many prompt tokens to include in this block
            fill_from_prompt = min(prompt_remainder, block_size)

            # Initialize block: prompt remainder + [MASK]
            block, masked = denoiser.init_block(prompt_remainder=fill_from_prompt)

            # Track which positions need decoding (masked ones)
            n_masked = masked.sum().item()

            # Denoise the block
            for step in range(denoise_steps):
                if not masked.any():
                    break
                total_steps += 1

                # Forward pass
                logits, _ = logits_callback(model, block, pos_offset)

                # Suppress special tokens
                logits[:, :, mask_token_id] = -float('inf')
                logits[:, :, pad_token_id] = -float('inf')

                # Gumbel-max sampling
                from .sampler import GumbelSampler
                sampler = GumbelSampler(temperature=temperature)
                samples = sampler.sample(logits)

                # Confidence
                probs = torch.softmax(logits.float(), dim=-1)
                confidences = torch.gather(probs, -1, samples.unsqueeze(-1)).squeeze(-1)

                # How many to unmask
                if step == denoise_steps - 1:
                    n_to_unmask = int(masked.sum().item())
                else:
                    n_to_unmask = denoiser.step_unmask_count(
                        t=1.0 - step / denoise_steps,
                        s=1.0 - (step + 1) / denoise_steps,
                        n_masked=n_masked,
                        step=step,
                        total_steps=denoise_steps,
                    )
                n_to_unmask = max(1, n_to_unmask)

                # Unmask
                from .unmask import unmask_top_k
                masked = unmask_top_k(masked, confidences, n_to_unmask)

                # Update block
                block = torch.where(masked == False, samples, block)

            # Cache this block's KV for next block
            if hasattr(model, 'set_cache_mode'):
                model.set_cache_mode(True)
            model(block, pos_offset=pos_offset)
            if hasattr(model, 'set_cache_mode'):
                model.set_cache_mode(False)
            pos_offset += block_size

            # Consume prompt remainder (only affects first generated block)
            prompt_remainder = 0

            # Extract new tokens
            new_tokens = block[0, fill_from_prompt:].tolist()

            # Truncate at EOS
            for i, tok in enumerate(new_tokens):
                if tok == eos_token_id:
                    new_tokens = new_tokens[:i]
                    done = True
                    break

            all_tokens.extend(new_tokens)
            tokens_generated += len(new_tokens)

        elapsed = time.time() - t_start
        tok_per_sec = tokens_generated / elapsed if elapsed > 0 else float('inf')
        if verbose:
            print(f'Generated {tokens_generated} tokens in {total_steps} denoise steps '
                  f'({tok_per_sec:.1f} tok/s)')

        # Filter special tokens and decode
        decoded_tokens = [t for t in all_tokens if t >= 14]  # skip special IDs 0-13
        return decode_fn(decoded_tokens)

    finally:
        # Reset model state
        if hasattr(model, 'set_cache_mode'):
            model.set_cache_mode(False)
        if hasattr(model, 'reset_kv_cache'):
            model.reset_kv_cache()
        if was_training:
            try:
                model.train()
            except Exception:
                pass
