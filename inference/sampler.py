"""Gumbel-max sampling for discrete token generation.

Gumbel-max sampling provides a differentiable way to sample from a categorical
distribution by adding Gumbel noise to logits:
    gumbel = -log(-log(uniform(0,1)))
    sample = argmax(logits + gumbel)

This is equivalent to sampling from softmax(logits) but gives us a continuous
relaxation for gradient flow during training.
"""

import torch


class GumbelSampler:
    """Gumbel-max sampler with configurable temperature.

    Temperature=0 gives greedy (argmax) sampling.
    Higher temperature gives more diverse samples.
    """

    def __init__(self, temperature: float = 1.0):
        self.temperature = temperature

    def sample(self, logits: torch.Tensor) -> torch.Tensor:
        """Sample tokens from logits using Gumbel-max.

        Args:
            logits: (batch, seq_len, vocab_size) — unnormalized log probabilities

        Returns:
            samples: (batch, seq_len) — sampled token indices
        """
        if self.temperature == 0:
            return torch.argmax(logits, dim=-1)

        # Work in float64 for numerical stability
        logits_64 = logits.to(torch.float64)
        uniform = torch.rand_like(logits_64, dtype=torch.float64).clamp(min=1e-20)
        gumbel_noise = (-torch.log(uniform)) ** self.temperature
        noisy_logits = logits_64.exp() / gumbel_noise

        return noisy_logits.to(logits.dtype).argmax(dim=-1)

    def probabilities(self, logits: torch.Tensor) -> torch.Tensor:
        """Compute softmax probabilities from logits.

        Args:
            logits: (batch, seq_len, vocab_size)

        Returns:
            probs: (batch, seq_len, vocab_size) — softmax probabilities
        """
        return torch.softmax(logits.float(), dim=-1)

    def sample_with_confidence(
        self, logits: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample tokens and return confidence (probability of sampled token).

        Args:
            logits: (batch, seq_len, vocab_size)

        Returns:
            samples: (batch, seq_len) — sampled token indices
            confidence: (batch, seq_len) — probability of sampled token
        """
        probs = self.probabilities(logits)
        samples = torch.argmax(probs, dim=-1)
        confidence = torch.gather(probs, -1, samples.unsqueeze(-1)).squeeze(-1)
        return samples, confidence
