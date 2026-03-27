"""
Cosine Noise Schedule for DDPM-style Diffusion.

This module provides noise schedule implementations for block diffusion.

================================================================================
                              COSINE SCHEDULE
================================================================================

The cosine schedule is from "Improved Denoising Diffusion Probabilistic Models":

    ᾱ(t) = cos(t * π/2)^2

This provides smoother interpolation than linear schedules.

================================================================================
                              USAGE
================================================================================

    from engine.diffusion import CosineNoiseSchedule

    schedule = CosineNoiseSchedule(eps=1e-3)
    alphas = schedule.get_schedule(t)  # t in [0, 1]

================================================================================
"""

import torch


class CosineNoiseSchedule:
    """
    Cosine noise schedule for diffusion.

    This schedule provides smooth interpolation and works well
    for block diffusion where we want gradual denoising.
    """

    def __init__(self, eps: float = 1e-3, device: str = "cpu"):
        """
        Initialize cosine noise schedule.

        Args:
            eps: Minimum value for ᾱ (prevents division by zero)
            device: Device for tensor creation
        """
        self.eps = eps
        self.device = device

    def get_schedule(self, t: torch.Tensor) -> torch.Tensor:
        """
        Get ᾱ (cumulative alpha) values for timesteps t.

        Formula: ᾱ(t) = cos(t * π/2)^2

        Args:
            t: Timesteps in [0, 1] — shape can be arbitrary

        Returns:
            ᾱ values in [eps, 1]
        """
        t = t.to(self.device)
        alpha_bar = torch.cos(t * torch.pi / 2) ** 2
        return alpha_bar.clamp(min=self.eps, max=1.0)

    def get_alpha(self, t: torch.Tensor) -> torch.Tensor:
        """
        Get α (alpha) values for timesteps t.

        α(t) = d(ᾱ)/dt ≈ discrete difference of cumulative alpha

        Args:
            t: Timesteps in [0, 1]

        Returns:
            α values
        """
        t = t.to(self.device)
        alpha = -torch.sin(t * torch.pi / 2) * torch.pi / 2
        return alpha.abs()

    def forward_noise(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """
        Add noise to x0 at timestep t.

        x(t) = √(ᾱ(t)) * x0 + √(1 - ᾱ(t)) * ε

        Args:
            x0: Clean data
            t: Timestep in [0, 1]
            noise: Noise to add

        Returns:
            Noisy data at timestep t
        """
        alpha_bar = self.get_schedule(t)
        sqrt_alpha_bar = alpha_bar.sqrt().view(*[-1] + [1] * (x0.dim() - 1))
        sqrt_one_minus = (1 - alpha_bar).sqrt().view(*[-1] + [1] * (x0.dim() - 1))
        return sqrt_alpha_bar * x0 + sqrt_one_minus * noise


class LinearNoiseSchedule:
    """Linear noise schedule for comparison."""

    def __init__(self, eps: float = 1e-3, device: str = "cpu"):
        self.eps = eps
        self.device = device

    def get_schedule(self, t: torch.Tensor) -> torch.Tensor:
        """Linear schedule: ᾱ(t) = 1 - t"""
        t = t.to(self.device)
        return (1 - t).clamp(min=self.eps, max=1.0)
