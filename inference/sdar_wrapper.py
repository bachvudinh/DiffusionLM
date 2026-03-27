"""SDAR Model Wrapper for Block Diffusion Inference.

This module wraps the SDAR model to provide a unified interface for our
block diffusion generation pipeline.

IMPORTANT: This implementation uses a local SDAR model that is MPS-compatible.
The official HuggingFace SDAR model requires flash_attn (CUDA-only), so we
use our custom implementation in sdar_model/ which has MPS-compatible attention.

Architecture:
    SDAR Model (local sdar_model/) → SDARWrapper
    ├── Input:  token IDs (B, seq_len) with mask tokens
    ├── Output: logits (B, seq_len, vocab_size)
    └── KV caching supported via DynamicCache

Example:
    from inference.sdar_wrapper import SDARWrapper

    wrapper = SDARWrapper(model_path='/path/to/SDAR-1.7B-Chat')
    wrapper.set_cache_mode(True)
    wrapper.reset_kv_cache()

    # Forward pass
    x = torch.tensor([[1, 2, 3, 151669, 151669]])  # prompt + masks
    logits, _ = wrapper(x, pos_offset=0)
    # logits shape: (1, 5, 151936)
"""

import sys
from pathlib import Path

import torch
import torch.nn as nn
from safetensors.torch import load_file

# Add parent to path for sdar_model import
sys.path.insert(0, str(Path(__file__).parent.parent))

from sdar_model import SDARForCausalLM, SDARConfig


class SDARWrapper(nn.Module):
    """Wraps SDAR model for block diffusion generation.

    Required interface for generate():
        __call__(x, pos_offset) → (logits, _)
        set_cache_mode(enabled)
        reset_kv_cache()
        parameters()

    Args:
        model_path: Path to SDAR model directory
        device: Device to load model on ('cpu', 'mps', 'cuda')
        dtype: Data type for model weights ('bfloat16', 'float16')
    """

    def __init__(
        self,
        model_path: str,
        device: str = 'cpu',
        dtype: str = 'bfloat16',
    ):
        super().__init__()

        self.model_path = model_path
        self.device = torch.device(device)
        self.dtype = torch.bfloat16 if dtype == 'bfloat16' else torch.float16

        # Load config
        self.config = SDARConfig.from_pretrained(model_path)

        # Create model
        self.model = SDARForCausalLM(self.config)

        # Load weights from safetensors
        safetensors_path = f'{model_path}/model.safetensors'
        print(f"Loading weights from {safetensors_path}...")
        state_dict = load_file(safetensors_path)
        self.model.load_state_dict(state_dict, strict=False)

        # Move to device and dtype
        self.model = self.model.to(device=self.device)
        self.model = self.model.to(dtype=self.dtype)
        self.model.eval()

        # Cache attributes
        self._vocab_size = self.config.vocab_size
        self._hidden_size = self.config.hidden_size

    def __call__(self, x: torch.Tensor, pos_offset: int = 0, **kwargs):
        """Forward pass through SDAR model.

        Args:
            x: torch.Tensor — shape (B, seq_len) — token IDs
            pos_offset: int — position offset for RoPE (currently unused in SDAR)

        Returns:
            tuple: (logits, _) — logits shape (B, seq_len, vocab_size)
        """
        # Ensure input is on correct device
        x = x.to(device=self.device)

        # Remove position_ids from kwargs if present (will be handled by SDARModel or passed via kwargs)
        kwargs.pop('position_ids', None)

        with torch.no_grad():
            outputs = self.model(
                input_ids=x,
                **kwargs
            )
            logits = outputs.logits if hasattr(outputs, 'logits') else outputs

        return logits, None

    def set_cache_mode(self, enabled: bool):
        """Enable/disable KV cache usage.

        Args:
            enabled: bool — True to accumulate KV cache
        """
        # SDAR uses DynamicCache internally via use_cache parameter
        pass

    def reset_kv_cache(self):
        """Reset KV cache for generation."""
        # DynamicCache is recreated on each forward with use_cache=True
        pass

    def parameters(self, recurse: bool = True):
        """Return model parameters."""
        return self.model.parameters(recurse=recurse)

    @property
    def vocab_size(self) -> int:
        """Vocabulary size."""
        return self._vocab_size

    @property
    def hidden_size(self) -> int:
        """Hidden size."""
        return self._hidden_size

    @property
    def config(self):
        """Model configuration."""
        return self._config

    @config.setter
    def config(self, value):
        self._config = value


def create_sdar_wrapper(
    model_path: str,
    device: str = 'cpu',
    dtype: str = 'bfloat16',
) -> SDARWrapper:
    """Factory function to create SDAR wrapper.

    Args:
        model_path: Path to SDAR model directory
        device: Device to load model on ('cpu', 'mps', 'cuda')
        dtype: Data type for model weights ('bfloat16', 'float16')

    Returns:
        SDARWrapper instance
    """
    return SDARWrapper(model_path=model_path, device=device, dtype=dtype)
