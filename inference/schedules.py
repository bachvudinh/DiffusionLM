"""Noise schedules for Block Diffusion Language Models.

Two schedules supported:
- Linear: mask_prob = t (used in MDLM, LLaDA)
- Cosine: alpha(t) = cos(pi*t/2)^2, mask_prob = 1 - alpha(t) (used in SEDD)
"""

import torch


class LinearSchedule:
    """Linear noise schedule: mask_prob = t.

    For linear schedule, the ELBO weight trivially equals 1/t.
    """

    def mask_prob(self, t: torch.Tensor) -> torch.Tensor:
        return t.clamp(0.0, 1.0)

    def elbo_weight(self, t: torch.Tensor) -> torch.Tensor:
        return 1.0 / t.clamp(min=1e-5)


class CosineSchedule:
    """Cosine schedule from SEDD paper.

    alpha(t) = cos(pi * t / 2)^2
    mask_prob(t) = 1 - alpha(t)

    This is smoother than linear, giving more weight to low-noise regimes.
    """

    def mask_prob(self, t: torch.Tensor) -> torch.Tensor:
        alpha = torch.cos(torch.pi * t / 2) ** 2
        return (1.0 - alpha).clamp(0.0, 1.0)

    def alpha(self, t: torch.Tensor) -> torch.Tensor:
        return torch.cos(torch.pi * t / 2) ** 2

    def elbo_weight(self, t: torch.Tensor) -> torch.Tensor:
        alpha = self.alpha(t)
        d_alpha = -torch.pi / 2 * torch.sin(torch.pi * t) * torch.cos(torch.pi * t / 2)
        return (d_alpha / (alpha + 1e-5)).clamp(0.0, 100.0)
