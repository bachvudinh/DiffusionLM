"""
DDPM Transition Functions for Block Diffusion.

This module provides transition functions for the diffusion process.

================================================================================
                              CONCEPT
================================================================================

DDPM Forward Process:
    q(x_t | x_{t-1}) = N(x_t; √(1 - β_t) * x_{t-1}, β_t * I)

DDPM Reverse Process (denoising):
    p_θ(x_{t-1} | x_t) = N(x_{t-1}; μ_θ(x_t, t), Σ_θ(x_t, t))

================================================================================
                              USAGE
================================================================================

    from engine.diffusion import DDPMTransition

    transition = DDPMTransition(beta_start=0.0001, beta_end=0.02)

    # Compute noised sample
    x_t = transition.forward(x_0, t, noise)

    # Get denoised prediction
    x_pred = transition.predict_x0(x_t, model_output, t)

================================================================================
"""

import torch


class DDPMTransition:
    """
    DDPM-style transition functions.

    Provides forward and reverse process transitions for block diffusion.
    """

    def __init__(
        self,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        num_steps: int = 1000,
        device: str = "cpu",
    ):
        """
        Initialize DDPM transitions.

        Args:
            beta_start: Starting β value
            beta_end: Ending β value
            num_steps: Number of diffusion steps
            device: Device for tensor creation
        """
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.num_steps = num_steps
        self.device = device

        # Precompute β schedule
        self.betas = torch.linspace(beta_start, beta_end, num_steps, device=device)

        # Precompute ᾱ (cumulative product of 1 - β)
        alphas = 1 - self.betas
        self.alphas_bar = torch.cumprod(alphas, dim=0)

    def forward(
        self,
        x0: torch.Tensor,
        t: int,
        noise: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward process: add noise to x0 at timestep t.

        x_t = √(ᾱ_t) * x0 + √(1 - ᾱ_t) * noise

        Args:
            x0: Clean data
            t: Timestep (integer index)
            noise: Noise to add

        Returns:
            Noisy data at timestep t
        """
        alpha_bar = self.alphas_bar[t]
        sqrt_alphas_bar = alpha_bar.sqrt()
        sqrt_one_minus = (1 - alpha_bar).sqrt()

        return (
            sqrt_alphas_bar.view(-1, *([1] * (x0.dim() - 1))) * x0
            + sqrt_one_minus.view(-1, *([1] * (x0.dim() - 1))) * noise
        )

    def predict_x0(
        self,
        x_t: torch.Tensor,
        model_output: torch.Tensor,
        t: int,
    ) -> torch.Tensor:
        """
        Predict clean data x0 from model output.

        Uses DDPM parameterization:
        x0 = (x_t - √(1 - ᾱ_t) * ε_θ) / √(ᾱ_t)

        Args:
            x_t: Noisy data at timestep t
            model_output: Model's noise prediction ε_θ
            t: Timestep

        Returns:
            Predicted clean data x0
        """
        alphas_bar_t = self.alphas_bar[t]
        sqrt_alphas_bar_t = alphas_bar_t.sqrt()
        sqrt_one_minus = (1 - alphas_bar_t).sqrt()

        return (x_t - sqrt_one_minus * model_output) / sqrt_alphas_bar_t.clamp(min=1e-8)

    def q_posterior(
        self,
        x0: torch.Tensor,
        x_t: torch.Tensor,
        t: int,
    ) -> torch.Tensor:
        """
        Compute q(x_{t-1} | x_t, x_0) - posterior mean.

        Args:
            x0: Clean data
            x_t: Noisy data
            t: Timestep

        Returns:
            Posterior mean
        """
        alphas_bar_t = self.alphas_bar[t]
        alphas_bar_prev = self.alphas_bar[t - 1] if t > 0 else torch.tensor(1.0, device=self.device)

        posterior_mean = (
            alphas_bar_prev.sqrt() * self.betas[t] / (1 - alphas_bar_t)
        ) * x0 + (
            alphas_bar_t.sqrt() * (1 - alphas_bar_prev) / (1 - alphas_bar_t)
        ) * x_t

        return posterior_mean
