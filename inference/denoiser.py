"""Block denoiser: orchestrates the per-block denoising loop.

Block Diffusion generates one block at a time (left-to-right, AR).
Within each block, it runs T denoising steps to progressively unmask tokens.

Key components:
1. Initialize block: prompt remainder (if any) + [MASK] tokens
2. Denoise loop: for each step, run model forward, compute confidences, unmask top-k
3. Commit: after all steps, cache KV for this block
"""

import torch
import torch.nn.functional as F

from .sampler import GumbelSampler
from .unmask import unmask_top_k, uniform_schedule


class BlockDenoiser:
    """Handles per-block denoising for Block Diffusion generation.

    Args:
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

        Args:
            prompt_remainder: Number of prompt tokens at the start of this block

        Returns:
            block: (1, block_size) — initialized block with mask tokens
            masked: (1, block_size) — True for masked positions
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

        Args:
            t: current timestep (1.0 = no noise, 0.0 = fully masked)
            s: next timestep
            n_masked: number of currently masked positions
            step: current step index
            total_steps: total denoising steps

        Returns:
            number of positions to unmask this step
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

        Args:
            model: The language model (with forward(x, pos_offset) -> logits)
            block: (1, block_size) — current block state
            masked: (1, block_size) — True for masked positions
            pos_offset: Position offset for RoPE
            logits_callback: function(model, block, pos_offset) -> (logits, extra)

        Returns:
            final_block: (1, block_size) — denoised block
            new_tokens: list of newly generated token IDs (excluding prompt)
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

        Args:
            block: (1, block_size) — denoised block
            prompt_remainder: Number of prompt tokens at the start of this block

        Returns:
            new_tokens: list of token IDs (excluding prompt remainder)
        """
        tokens = block[0].tolist()
        return tokens[prompt_remainder:]
