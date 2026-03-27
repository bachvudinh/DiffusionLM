"""
SDAR Model Configuration.

This module provides the SDAR model configuration, defining the model
architecture parameters for the SDAR (Synergy of Diffusion and AutoRegression) model.

Based on JetEngine's SDAR implementation:
https://github.com/Jet-Astra/SDAR

================================================================================
                              CONFIGURATION
================================================================================

Default configuration for SDAR-1.7B:
- vocab_size: 151936
- hidden_size: 4096
- intermediate_size: 22016
- num_hidden_layers: 32
- num_attention_heads: 32
- num_key_value_heads: 8 (GQA - Grouped Query Attention)
- head_dim: 128
- max_position_embeddings: 32768

================================================================================
                              USAGE
================================================================================

    from sdar_model.config import SDARConfig

    config = SDARConfig(
        vocab_size=151936,
        hidden_size=4096,
        num_hidden_layers=32,
    )

"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class SDARConfig:
    """
    Configuration class for SDAR model.

    Defines the architecture of the SDAR (Synergy of Diffusion and AutoRegression)
    block diffusion model.
    """

    # Model architecture
    vocab_size: int = 151936
    hidden_size: int = 4096
    intermediate_size: int = 22016
    num_hidden_layers: int = 32
    num_attention_heads: int = 32
    num_key_value_heads: int = 8  # GQA: fewer KV heads than Q heads
    head_dim: int = 128

    # Activation and normalization
    hidden_act: str = "silu"
    rms_norm_eps: float = 1e-6

    # Position embeddings
    max_position_embeddings: int = 32768
    rope_theta: float = 10000.0
    rope_scaling: Optional[dict] = None

    # Attention configuration
    attention_bias: bool = False
    attention_dropout: float = 0.0

    # Sliding window attention (for long contexts)
    use_sliding_window: bool = False
    sliding_window: int = 4096
    max_window_layers: int = 28

    # Block diffusion configuration
    block_size: int = 4
    mask_token_id: int = 151669

    # Device and dtype
    torch_dtype: str = "bfloat16"

    # Model initialization
    initializer_range: float = 0.02

    # Use cache during inference
    use_cache: bool = True

    @property
    def num_key_value_groups(self) -> int:
        """Number of Q heads per KV head (for GQA)."""
        return self.num_attention_heads // self.num_key_value_heads

    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError(
                f"hidden_size ({self.hidden_size}) must be divisible by "
                f"num_attention_heads ({self.num_attention_heads})"
            )

        if self.num_key_value_heads > self.num_attention_heads:
            raise ValueError(
                f"num_key_value_heads ({self.num_key_value_heads}) must be <= "
                f"num_attention_heads ({self.num_attention_heads})"
            )

        if self.head_dim is None:
            self.head_dim = self.hidden_size // self.num_attention_heads

        # Validate rope_scaling if provided
        if self.rope_scaling is not None:
            rope_type = self.rope_scaling.get("rope_type", self.rope_scaling.get("type"))
            if rope_type not in ("default", "linear", "dynamic", "yarn", "longrope", "llama3"):
                raise ValueError(f"Unknown rope type: {rope_type}")

    @classmethod
    def from_pretrained(cls, pretrained_config) -> "SDARConfig":
        """
        Create config from a pretrained config object or dictionary.

        Args:
            pretrained_config: PretrainedConfig object or dictionary

        Returns:
            SDARConfig instance
        """
        if hasattr(pretrained_config, "to_dict"):
            # It's a HuggingFace PretrainedConfig
            config_dict = pretrained_config.to_dict()
        elif isinstance(pretrained_config, dict):
            config_dict = pretrained_config
        else:
            raise ValueError(f"Cannot create config from {type(pretrained_config)}")

        # Map relevant fields
        config_params = {
            "vocab_size": config_dict.get("vocab_size", 151936),
            "hidden_size": config_dict.get("hidden_size", 4096),
            "intermediate_size": config_dict.get("intermediate_size", 22016),
            "num_hidden_layers": config_dict.get("num_hidden_layers", 32),
            "num_attention_heads": config_dict.get("num_attention_heads", 32),
            "num_key_value_heads": config_dict.get("num_key_value_heads", 8),
            "head_dim": config_dict.get("head_dim", 128),
            "hidden_act": config_dict.get("hidden_act", "silu"),
            "rms_norm_eps": config_dict.get("rms_norm_eps", 1e-6),
            "max_position_embeddings": config_dict.get("max_position_embeddings", 32768),
            "rope_theta": config_dict.get("rope_theta", 10000.0),
            "rope_scaling": config_dict.get("rope_scaling"),
            "attention_bias": config_dict.get("attention_bias", False),
            "attention_dropout": config_dict.get("attention_dropout", 0.0),
            "use_sliding_window": config_dict.get("use_sliding_window", False),
            "sliding_window": config_dict.get("sliding_window", 4096),
            "max_window_layers": config_dict.get("max_window_layers", 28),
            "initializer_range": config_dict.get("initializer_range", 0.02),
            "use_cache": config_dict.get("use_cache", True),
        }

        return cls(**config_params)
