"""
Diffusion-specific components for semi-AR inference on MPS.

These components are used by the block diffusion generation algorithm.

Components:
    - noise_schedules.py: Cosine noise schedule for DDPM transitions
    - transitions.py: DDPM transition computation
    - sampler.py: Semi-AR sampler with block-wise denoising

Note: These are optional utilities. The main inference module (inference/)
provides the complete generation pipeline.
"""

from .noise_schedules import CosineNoiseSchedule
from .transitions import DDPMTransition
from .sampler import SemiARUpdater

__all__ = [
    "CosineNoiseSchedule",
    "DDPMTransition",
    "SemiARUpdater",
]
