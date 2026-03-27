"""Noise schedules for Block Diffusion Language Models.

Architecture Overview
====================

    Timestep t ~ U[0, 1]
           |
           v
    +----------------------+
    |  LinearSchedule       |  mask_prob(t) = t
    |  OR CosineSchedule    |  mask_prob(t) = 1 - cos²(πt/2)
    +----------------------+
           |
           v
    mask_prob: float         [0.0, 1.0]
           |
           v
    +----------------------+
    |  ELBO Weight          |  weight(t) = 1 / mask_prob(t)
    +----------------------+
           |
           v
    weight: float            [1.0, 10.0] typically


Two schedules supported:
- Linear: mask_prob = t (used in MDLM, LLaDA)
- Cosine: alpha(t) = cos(πt/2)², mask_prob = 1 - alpha(t) (used in SEDD)
"""

import torch


class LinearSchedule:
    """Linear noise schedule: mask_prob = t.

    For linear schedule, the ELBO weight trivially equals 1/t.

    Example:
        t = torch.tensor([0.1, 0.5, 1.0])
        schedule = LinearSchedule()
        mask_prob = schedule.mask_prob(t)       # [0.1, 0.5, 1.0]
        elbo_weight = schedule.elbo_weight(t)   # [10.0, 2.0, 1.0]
    """

    def mask_prob(self, t: torch.Tensor) -> torch.Tensor:
        """Compute mask probability from timestep.

        Input:
            t: torch.Tensor  — timestep in [0, 1], any shape

        Output:
            torch.Tensor — mask probability, same shape as t, clamped to [0, 1]
        """
        return t.clamp(0.0, 1.0)

    def elbo_weight(self, t: torch.Tensor) -> torch.Tensor:
        """Compute ELBO loss weight from timestep.

        For linear schedule: weight = 1 / mask_prob = 1 / t

        Input:
            t: torch.Tensor — timestep in (0, 1]

        Output:
            torch.Tensor — weight, same shape as t
        """
        return 1.0 / t.clamp(min=1e-5)


class CosineSchedule:
    """Cosine schedule from SEDD paper.

    alpha(t) = cos(πt/2)²
    mask_prob(t) = 1 - alpha(t)

    This is smoother than linear, giving more weight to low-noise regimes.

    Example:
        t = torch.tensor([0.0, 0.5, 1.0])
        schedule = CosineSchedule()
        alpha = schedule.alpha(t)         # [1.0, 0.5, 0.0]
        mask_prob = schedule.mask_prob(t) # [0.0, 0.5, 1.0]
    """

    def mask_prob(self, t: torch.Tensor) -> torch.Tensor:
        """Compute mask probability from timestep using cosine schedule.

        Input:
            t: torch.Tensor — timestep in [0, 1], any shape

        Output:
            torch.Tensor — mask probability = 1 - cos²(πt/2), same shape as t
        """
        alpha = torch.cos(torch.pi * t / 2) ** 2
        return (1.0 - alpha).clamp(0.0, 1.0)

    def alpha(self, t: torch.Tensor) -> torch.Tensor:
        """Compute alpha (clean probability) from timestep.

        Input:
            t: torch.Tensor — timestep in [0, 1], any shape

        Output:
            torch.Tensor — alpha = cos²(πt/2), same shape as t
        """
        return torch.cos(torch.pi * t / 2) ** 2

    def elbo_weight(self, t: torch.Tensor) -> torch.Tensor:
        """Compute ELBO loss weight from timestep.

        Uses d(alpha)/dt for proper evidence lower bound weighting.

        Input:
            t: torch.Tensor — timestep in [0, 1]

        Output:
            torch.Tensor — weight = d_alpha / alpha, clamped to [0, 100]
        """
        alpha = self.alpha(t)
        d_alpha = -torch.pi / 2 * torch.sin(torch.pi * t) * torch.cos(torch.pi * t / 2)
        return (d_alpha / (alpha + 1e-5)).clamp(0.0, 100.0)
