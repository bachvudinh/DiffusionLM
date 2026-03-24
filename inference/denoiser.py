"""Block denoiser: orchestrates the per-block denoising loop.

Architecture Overview
====================

    ┌─────────────────────────────────────────────────────────────────────┐
    │                    Block Denoising (per block)                       │
    │                                                                      │
    │  Input:                                                              │
    │    • prompt_remainder: N tokens from prompt at block start           │
    │    • rest of block: filled with [MASK] tokens                       │
    │                                                                      │
    │  ┌──────────────────────────────────────────────────────────────┐   │
    │  │              Denoise Loop (denoise_steps iterations)           │   │
    │  │                                                               │   │
    │  │   block, masked ──► model ──► logits ──► Gumbel sample         │   │
    │  │       ▲               │             │                         │   │
    │  │       │               ▼             ▼                         │   │
    │  │       │         (B, block_size, V)  confidences                │   │
    │  │       │               │             │                         │   │
    │  │       │               ▼             ▼                         │   │
    │  │       │         unmask_top_k ──► new_masked                    │   │
    │  │       │               │             │                         │   │
    │  │       └───────────────┴─────────────┘                         │   │
    │  │                      │                                         │   │
    │  │                      ▼                                         │   │
    │  │               block = where(new_masked=False, samples, block)  │   │
    │  └──────────────────────────────────────────────────────────────┘   │
    │                              │                                     │
    │                              ▼                                     │
    │  Output:                                                             │
    │    • final_block: (1, block_size) — fully denoised                  │
    │    • new_tokens: list[int] — generated token IDs                     │
    └─────────────────────────────────────────────────────────────────────┘


Block Initialization (init_block)
================================

    prompt_remainder = 3, block_size = 8

    Before:
        [token_0, token_1, token_2, M, M, M, M, M]
                         └─ 3 ─┘└──── 5 ────┘
                       prompt   masks

    After init_block:
        block  = [token_0, token_1, token_2, 0,    0,    0,    0,    0   ]
                             (mask_token_id=0)
        masked = [False,   False,   False,   True, True, True, True, True]


Denoise Step Data Flow
=======================

    Step s, block=(1, 8), masked=(1, 8)

        masked = [False, False, False,  True,  True,  True,  True,  True]
        block  = [token_0, token_1, token_2, M,    M,    M,    M,    M   ]

                          │
                          ▼
                   model(block)
                          │
                          ▼
                   logits: (1, 8, V)   — V = vocab_size
                          │
                          ▼
                   ┌─────────────┐
                   │ Suppress     │  mask_token_id → -inf
                   │ MASK/PAD    │  pad_token_id → -inf
                   └─────────────┘
                          │
                          ▼
                   logits: (1, 8, V)   — -inf at special token positions
                          │
                          ▼
                   GumbelSampler(temperature)
                          │
                          ▼
                   samples: (1, 8)   — sampled token IDs
                          │
                          ▼
                   probs = softmax(logits, dim=-1)
                   confidences = gather(probs, samples) → (1, 8)
                          │
                          ▼
                   confidences = [?, ?, ?, 0.3, 0.8, 0.5, 0.6, 0.4]
                          │
                          ▼
                   unmask_top_k(masked, confidences, k=3)
                          │
                          ▼
                   new_masked = [False, False, False, True, False, True, False, True]
                          │
                          ▼
                   block = where(new_masked == False, samples, block)
                          │
                          ▼
                   block = [token_0, token_1, token_2, token_4, M, token_6, M, M]
"""

import torch
import torch.nn.functional as F

from .sampler import GumbelSampler
from .unmask import unmask_top_k, uniform_schedule


class BlockDenoiser:
    """Handles per-block denoising for Block Diffusion generation.

    Attributes:
        block_size: Number of tokens per block
        mask_token_id: Token ID for [MASK]
        pad_token_id: Token ID for padding
        eos_token_id: Token ID for end-of-sequence
        denoise_steps: Number of denoising steps per block
        temperature: Sampling temperature (0 = greedy)
        top_k: Top-k filtering (0 = no filtering)
    """

    def __init__(
        self,
        block_size: int,
        mask_token_id: int = 0,
        pad_token_id: int = 2,
        eos_token_id: int = 1,
        denoise_steps: int = 10,
        temperature: float = 0.7,
        top_k: int = 50,
    ):
        self.block_size = block_size
        self.mask_token_id = mask_token_id
        self.pad_token_id = pad_token_id
        self.eos_token_id = eos_token_id
        self.denoise_steps = denoise_steps
        self.temperature = temperature
        self.top_k = top_k
        self.sampler = GumbelSampler(temperature=temperature)

    def init_block(self, prompt_remainder: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
        """Initialize a block: prompt remainder + [MASK] tokens.

        Input:
            prompt_remainder: int — number of prompt tokens at start of block (0 to block_size)

        Output:
            block:  torch.Tensor — shape (1, block_size)
                    First prompt_remainder positions = prompt tokens (not masked)
                    Rest = mask_token_id
            masked: torch.Tensor — shape (1, block_size)
                    True = masked (needs denoising), False = already known
        """
        block = torch.full(
            (1, self.block_size),
            self.mask_token_id,
            dtype=torch.long,
        )
        masked = torch.ones_like(block, dtype=torch.bool)
        masked[:, :prompt_remainder] = False  # Prompt tokens are not masked
        return block, masked

    def step_unmask_count(
        self,
        t: float,
        s: float,
        n_masked: int,
        step: int,
        total_steps: int,
    ) -> int:
        """Compute how many tokens to unmask at this denoising step.

        Uses uniform schedule: equal tokens per step, last step takes remainder.

        Input:
            t:          float — current timestep (1.0 = no noise, 0.0 = fully masked)
            s:          float — next timestep
            n_masked:   int   — number of currently masked positions
            step:       int   — current step index (0-indexed)
            total_steps: int  — total denoising steps

        Output:
            int — number of positions to unmask at this step
        """
        if step >= total_steps - 1:
            return n_masked
        return uniform_schedule(n_masked, step, total_steps)

    def denoise_block(
        self,
        model,
        block: torch.Tensor,
        masked: torch.Tensor,
        pos_offset: int,
        logits_callback,
    ) -> tuple[torch.Tensor, list[int]]:
        """Run the full denoising loop for one block.

        Input:
            model:          object — language model with forward(x, pos_offset) → (logits, _)
            block:          torch.Tensor — shape (1, block_size) — current block state
            masked:         torch.Tensor — shape (1, block_size) — True for masked positions
            pos_offset:     int — position offset for RoPE
            logits_callback: callable — function(model, block, pos_offset) → (logits, extra)

        Output:
            final_block: torch.Tensor — shape (1, block_size) — denoised block
            new_tokens:  list[int] — newly generated token IDs (excluding prompt remainder)
        """
        for step in range(self.denoise_steps):
            if not masked.any():
                break

            # Run model forward pass
            logits, _ = logits_callback(model, block, pos_offset)

            # Suppress mask/padding tokens
            logits = logits.clone()
            logits[:, :, self.mask_token_id] = -float('inf')
            logits[:, :, self.pad_token_id] = -float('inf')

            # Top-k filtering
            if self.top_k > 0:
                top_k_vals = torch.topk(logits, min(self.top_k, logits.size(-1)))[0]
                logits[logits < top_k_vals[:, :, [-1]]] = -float('inf')

            # Gumbel-max sampling
            samples = self.sampler.sample(logits)  # (1, block_size)

            # Compute confidence = probability of sampled token
            probs = F.softmax(logits.float(), dim=-1)
            confidences = torch.gather(probs, -1, samples.unsqueeze(-1)).squeeze(-1)

            # Determine how many to unmask this step
            n_masked = masked.sum().item()
            n_to_unmask = self.step_unmask_count(
                t=1.0 - step / self.denoise_steps,
                s=1.0 - (step + 1) / self.denoise_steps,
                n_masked=n_masked,
                step=step,
                total_steps=self.denoise_steps,
            )

            # Unmask top-k most confident masked positions
            masked = unmask_top_k(masked, confidences, n_to_unmask)

            # Update block with sampled tokens at unmasked positions
            block = torch.where(masked == False, samples, block)

        return block, block[0].tolist()

    def extract_generated_tokens(
        self,
        block: torch.Tensor,
        prompt_remainder: int,
    ) -> list[int]:
        """Extract newly generated tokens from a denoised block.

        Input:
            block:           torch.Tensor — shape (1, block_size) — denoised block
            prompt_remainder: int — number of prompt tokens at start of block

        Output:
            list[int] — newly generated token IDs (excluding prompt remainder)
        """
        tokens = block[0].tolist()
        return tokens[prompt_remainder:]
