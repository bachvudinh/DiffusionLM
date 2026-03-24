"""Full generation pipeline for Block Diffusion Language Models.

Architecture Overview
====================

    ┌─────────────────────────────────────────────────────────────────────────────┐
    │                           generate()                                         │
    │                                                                              │
    │  Input:                                                                       │
    │    • prompt: str  — "Hello, world!"                                         │
    │    • max_new_tokens: int — e.g., 512                                        │
    │    • block_size: int — e.g., 32                                             │
    │                                                                              │
    │  ┌────────────────────────────────────────────────────────────────────┐    │
    │  │ 1. ENCODE PROMPT                                                       │    │
    │  │    prompt → [1, 234, 567, 890, ...]  (list of token IDs)            │    │
    │  └────────────────────────────────────────────────────────────────────┘    │
    │                                  │                                            │
    │                                  ▼                                            │
    │  ┌────────────────────────────────────────────────────────────────────┐    │
    │  │ 2. KV CACHE WARMUP (prompt blocks only)                              │    │
    │  │                                                                       │    │
    │  │    For each FULL prompt block:                                        │    │
    │  │      model.set_cache_mode(True)                                       │    │
    │  │      model(block_ids, pos_offset) → caches K,V                         │    │
    │  │                                                                       │    │
    │  │    After warmup: all prompt K,V are cached                           │    │
    │  └────────────────────────────────────────────────────────────────────┘    │
    │                                  │                                            │
    │                                  ▼                                            │
    │  ┌────────────────────────────────────────────────────────────────────┐    │
    │  │ 3. BLOCK-BY-BLOCK GENERATION                                         │    │
    │  │                                                                       │    │
    │  │    while tokens_generated < max_new_tokens:                          │    │
    │  │                                                                       │    │
    │  │      ┌──────────────────────────────────────────────────────────┐    │    │
    │  │      │ 3a. INIT BLOCK                                           │    │    │
    │  │      │     • prompt_remainder (if any) at start                  │    │    │
    │  │      │     • rest = [MASK] tokens                               │    │    │
    │  │      └──────────────────────────────────────────────────────────┘    │    │
    │  │      │                                                             │    │
    │  │      ▼                                                             │    │
    │  │      ┌──────────────────────────────────────────────────────────┐    │    │
    │  │      │ 3b. DENOISE LOOP (denoise_steps iterations)              │    │    │
    │  │      │                                                         │    │    │
    │  │      │     For each step:                                       │    │    │
    │  │      │       1. model(block) → logits                          │    │    │
    │  │      │       2. Suppress MASK/PAD tokens                       │    │    │
    │  │      │       3. Gumbel sample → tokens                         │    │    │
    │  │      │       4. Compute confidence per position                 │    │    │
    │  │      │       5. Unmask top-k most confident                    │    │    │
    │  │      │       6. Update block with sampled tokens               │    │    │
    │  │      └──────────────────────────────────────────────────────────┘    │    │
    │  │      │                                                             │    │
    │  │      ▼                                                             │    │
    │  │      ┌──────────────────────────────────────────────────────────┐    │    │
    │  │      │ 3c. CACHE BLOCK KV                                       │    │    │
    │  │      │     model.set_cache_mode(True)                            │    │    │
    │  │      │     model(final_block, pos_offset) → cache K,V             │    │    │
    │  │      └──────────────────────────────────────────────────────────┘    │    │
    │  │      │                                                             │    │
    │  │      ▼                                                             │    │
    │  │      ┌──────────────────────────────────────────────────────────┐    │    │
    │  │      │ 3d. EXTRACT NEW TOKENS                                    │    │    │
    │  │      │     new_tokens = block[prompt_remainder:]                  │    │    │
    │  │      │     Truncate at EOS                                       │    │    │
    │  │      └──────────────────────────────────────────────────────────┘    │    │
    │  │                                                                       │    │
    │  └────────────────────────────────────────────────────────────────────┘    │
    │                                  │                                            │
    │                                  ▼                                            │
    │  ┌────────────────────────────────────────────────────────────────────┐    │
    │  │ 4. DECODE & RETURN                                                  │    │
    │  │    tokens → "generated text..."                                     │    │
    │  └────────────────────────────────────────────────────────────────────┘    │
    │                                                                              │
    └─────────────────────────────────────────────────────────────────────────────┘


Timeline Example
===============

    prompt = "Hello"  (3 tokens),  block_size = 4,  denoise_steps = 3

    Step 1: KV Cache Warmup
    ───────────────────────
    prompt_ids = [10, 20, 30]
    n_full_prompt_blocks = 3 // 4 = 0
    prompt_remainder = 3

    Step 2: Generate Block 0
    ─────────────────────────
    fill_from_prompt = min(3, 4) = 3
    Initial block = [token_10, token_20, token_30, MASK]

    Denoise Step 0:
        logits → sample → unmask top-k(k=1)
        block = [token_10, token_20, token_30, token_42]  (one token revealed)
    Denoise Step 1:
        logits → sample → unmask top-k(k=1)
        block = [token_10, token_20, token_30, token_88]  (another revealed)
    Denoise Step 2 (last):
        logits → sample → unmask ALL
        block = [token_10, token_20, token_30, token_99]  (fully denoised)

    Cache KV for block 0
    new_tokens = [token_99]

    Step 3: Generate Block 1
    ─────────────────────────
    fill_from_prompt = 0  (prompt fully consumed)
    Initial block = [MASK, MASK, MASK, MASK]

    Denoise Loop...
    Cache KV for block 1
    new_tokens = [...]

    Continue until max_new_tokens reached


Model Interface Required
========================

    The model passed to generate() must implement:

        def __call__(self, x: Tensor, pos_offset: int) -> tuple[Tensor, None]:
            '''
            x:        (B, L) — token IDs
            pos_offset: int — starting position for RoPE
            Returns: (logits, _) — logits: (B, L, vocab_size)
            '''

        def set_cache_mode(self, enabled: bool) -> None:
            '''
            enabled=True  → accumulate K,V in cache
            enabled=False → don't accumulate
            '''

        def reset_kv_cache(self) -> None:
            '''Clear all cached K,V'''

        def parameters(self) -> Iterator[Parameter]:
            '''Standard torch.nn.Module interface'''
"""

import time
from typing import Callable

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

    Input:
    ──────
        model:           object — language model with forward(x, pos_offset) → (logits, _)
        encode_fn:       callable — str → list[int] tokenization function
        decode_fn:       callable — list[int] → str detokenization function
        prompt:          str — input prompt string
        max_new_tokens:  int — maximum tokens to generate
        block_size:      int — number of tokens per diffusion block
        mask_token_id:   int — token ID for [MASK]
        eos_token_id:    int — token ID for end-of-sequence
        pad_token_id:    int — token ID for padding
        denoise_steps:   int — denoising steps per block
        temperature:     float — sampling temperature (0 = greedy)
        top_k:           int — top-k filtering (0 = disabled)
        verbose:         bool — print generation stats

    Output:
    ──────
        str — generated text string (detokenized), special tokens filtered out
    """
    was_training = getattr(model, 'training', False)
    if was_training:
        model.eval()

    try:
        # ─── Initialize model state ───────────────────────────────────────
        if hasattr(model, 'reset_kv_cache'):
            model.reset_kv_cache()
        if hasattr(model, 'set_cache_mode'):
            model.set_cache_mode(False)

        device = next(iter(model.parameters())).device

        # ─── 1. ENCODE PROMPT ─────────────────────────────────────────────
        prompt_ids = encode_fn(prompt) if prompt else []
        prompt_len = len(prompt_ids)
        all_tokens = list(prompt_ids)

        # ─── 2. KV CACHE WARMUP ───────────────────────────────────────────
        n_full_prompt_blocks = prompt_len // block_size
        prompt_remainder = prompt_len % block_size
        pos_offset = 0

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

        # ─── 3. CREATE DENOISER ───────────────────────────────────────────
        denoiser = BlockDenoiser(
            block_size=block_size,
            mask_token_id=mask_token_id,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            denoise_steps=denoise_steps,
            temperature=temperature,
            top_k=top_k,
        )

        # ─── 4. BLOCK-BY-BLOCK GENERATION ─────────────────────────────────
        tokens_generated = 0
        total_steps = 0
        done = False
        t_start = time.time()

        def logits_callback(m, block, offset):
            return m(block, pos_offset=offset)

        while tokens_generated < max_new_tokens and not done:
            # ── 3a. INIT BLOCK ──────────────────────────────────────────
            fill_from_prompt = min(prompt_remainder, block_size)
            block, masked = denoiser.init_block(prompt_remainder=fill_from_prompt)
            n_masked = masked.sum().item()

            # ── 3b. DENOISE LOOP ─────────────────────────────────────────
            for step in range(denoise_steps):
                if not masked.any():
                    break
                total_steps += 1

                # Forward pass
                logits, _ = logits_callback(model, block, pos_offset)

                # Suppress special tokens
                logits = logits.clone()
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

            # ── 3c. CACHE BLOCK KV ────────────────────────────────────────
            if hasattr(model, 'set_cache_mode'):
                model.set_cache_mode(True)
            model(block, pos_offset=pos_offset)
            if hasattr(model, 'set_cache_mode'):
                model.set_cache_mode(False)
            pos_offset += block_size

            # Consume prompt remainder (only affects first generated block)
            prompt_remainder = 0

            # ── 3d. EXTRACT NEW TOKENS ───────────────────────────────────
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

        # ─── 5. DECODE & RETURN ────────────────────────────────────────────
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
