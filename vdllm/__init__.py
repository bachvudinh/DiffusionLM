"""
VDLLM - Block Diffusion Inference Engine.

Pure PyTorch implementation of SDAR (Synergy of Diffusion and AutoRegression)
block diffusion model inference, optimized for Metal (Apple Silicon) and Triton (NVIDIA).

Based on:
- JetEngine SDAR: https://github.com/Jet-Astra/SDAR
- vllm-metal: https://github.com/vllm-project/vllm-metal

================================================================================
                              ARCHITECTURE
================================================================================

vdllm/
├── engine/       # Device backends (Metal, Triton, CPU)
├── layers/       # Model layers (attention, MLP, RMSNorm, RoPE)
├── models/       # Model implementations (SDAR)
├── inference/    # Inference pipeline
└── utils/        # Utilities

================================================================================
                              USAGE
================================================================================

    from vdllm.models import SDARForCausalLM, SDARConfig
    from vdllm.engine import get_backend, build_block_diffusion_mask

    # Create model
    config = SDARConfig()
    model = SDARForCausalLM(config)

    # Get compute backend
    backend = get_backend(num_heads=32, head_dim=128)

    # Build attention mask
    mask = build_block_diffusion_mask(seq_len=1024, block_size=4)

"""

from .models import SDARConfig, SDARModel, SDARForCausalLM
from .engine import get_backend, get_device_name

__all__ = [
    "SDARConfig",
    "SDARModel",
    "SDARForCausalLM",
    "get_backend",
    "get_device_name",
]
