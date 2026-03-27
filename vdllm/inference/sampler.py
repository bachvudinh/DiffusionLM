"""Gumbel-max sampling for discrete token generation.

Architecture Overview
====================

    logits                                    (B, L, V)
      |                                        B=batch, L=seq_len, V=vocab_size
      v
    +------------------------+
    | Softmax (optional)      |  probs = softmax(logits, dim=-1)
    +------------------------+
      |
      v
    +------------------------+
    | Gumbel Noise            |  gumbel = (-log(-log(uniform))) ^ temperature
    +------------------------+
      |
      v
    logits_exp / gumbel         (B, L, V)
      |
      v
    +------------------------+
    | Argmax                  |  samples = argmax(..., dim=-1)
    +------------------------+
      |
      v
    samples                              (B, L)  — token indices


Key Properties:
- temperature=0 → pure argmax (greedy)
- temperature=1 → standard Gumbel-max
- temperature→∞ → approaches uniform sampling

Gumbel-max provides differentiable sampling:
    sample ~ softmax(logits + gumbel)
    This gives gradients wrt logits even at discrete choices.
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

        Input:
            logits: torch.Tensor — shape (B, L, V)
                B = batch size
                L = sequence length
                V = vocabulary size

        Output:
            torch.Tensor — shape (B, L)
                sampled token indices in [0, V-1]
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

        Input:
            logits: torch.Tensor — shape (B, L, V)

        Output:
            torch.Tensor — shape (B, L, V)
                softmax probabilities per token
        """
        return torch.softmax(logits.float(), dim=-1)

    def sample_with_confidence(
        self, logits: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample tokens and return confidence (probability of sampled token).

        Input:
            logits: torch.Tensor — shape (B, L, V)

        Output:
            samples: torch.Tensor — shape (B, L)
            confidence: torch.Tensor — shape (B, L)
                confidence[i] = prob[ samples[i] ]  (probability of chosen token)
        """
        probs = self.probabilities(logits)
        samples = torch.argmax(probs, dim=-1)
        confidence = torch.gather(probs, -1, samples.unsqueeze(-1)).squeeze(-1)
        return samples, confidence
