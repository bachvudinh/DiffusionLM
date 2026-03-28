"""MLX model weight loading for SDAR models.

Loads model weights from safetensors or PyTorch .bin files into MLX arrays.

Supports:
- Safetensors (preferred - direct MLX load via mx.load)
- PyTorch .bin files (via numpy bridge)
"""

import mlx.core as mx
from pathlib import Path
from typing import Optional
import json
import numpy as np
import torch


class MLXModelLoader:
    """Load SDAR model weights into MLX arrays.

    Supports:
    - Safetensors (preferred - direct MLX load)
    - PyTorch .bin files (via numpy bridge)
    """

    @staticmethod
    def load(model_path: str, dtype: mx.Dtype = mx.bfloat16) -> dict[str, mx.array]:
        """Load all model weights from a directory.

        Args:
            model_path: Path to model directory
            dtype: Target MLX dtype (default bfloat16)

        Returns:
            Dictionary mapping weight name -> MLX array
        """
        model_dir = Path(model_path)

        # Check for safetensors files (preferred)
        safetensors_files = list(model_dir.glob("*.safetensors"))
        if safetensors_files:
            return MLXModelLoader._load_safetensors(safetensors_files, dtype)

        # Fall back to PyTorch .bin files
        bin_files = list(model_dir.glob("*.bin"))
        if bin_files:
            return MLXModelLoader._load_pytorch(bin_files, dtype)

        raise FileNotFoundError(f"No model files found in {model_path}")

    @staticmethod
    def _load_safetensors(
        files: list[Path], dtype: mx.Dtype = mx.bfloat16
    ) -> dict[str, mx.array]:
        """Load safetensors directly into MLX (fastest path).

        Args:
            files: List of safetensors file paths
            dtype: Target dtype

        Returns:
            Dictionary mapping weight name -> MLX array
        """
        weights = {}

        for file_path in files:
            # mx.load handles safetensors natively on Apple Silicon
            tensors = mx.load(str(file_path))

            for name, arr in tensors.items():
                # Ensure correct dtype (MLX may load as float32 by default)
                if arr.dtype != dtype:
                    arr = arr.astype(dtype)
                weights[name] = arr

        return weights

    @staticmethod
    def _load_pytorch(
        files: list[Path], dtype: mx.Dtype = mx.bfloat16
    ) -> dict[str, mx.array]:
        """Load PyTorch weights and convert to MLX via numpy.

        Args:
            files: List of PyTorch .bin file paths
            dtype: Target dtype

        Returns:
            Dictionary mapping weight name -> MLX array
        """
        weights = {}

        for file_path in files:
            # Load PyTorch state dict
            state_dict = torch.load(str(file_path), map_location="cpu", weights_only=True)

            for name, tensor in state_dict.items():
                # Convert PyTorch -> numpy -> MLX
                # Handle bfloat16 by going through float32
                if tensor.dtype == torch.bfloat16:
                    tensor = tensor.float()

                np_arr = tensor.numpy()
                mlx_arr = mx.array(np_arr, dtype=dtype)
                weights[name] = mlx_arr

            # Clear reference to avoid memory retention
            del state_dict

        return weights

    @staticmethod
    def load_config(model_path: str) -> dict:
        """Load model configuration from config.json.

        Args:
            model_path: Path to model directory

        Returns:
            Configuration dictionary
        """
        config_path = Path(model_path) / "config.json"
        if config_path.exists():
            with open(config_path) as f:
                return json.load(f)
        return {}

    @staticmethod
    def get_model_info(weights: dict[str, mx.array]) -> dict:
        """Get information about loaded model weights.

        Args:
            weights: Dictionary of weight name -> MLX array

        Returns:
            Dictionary with model statistics
        """
        total_params = 0
        total_bytes = 0
        dtype_counts = {}

        for name, arr in weights.items():
            num_params = arr.size
            bytes_per_elem = arr.dtype == mx.bfloat16 and 2 or 4
            total_params += num_params
            total_bytes += num_params * bytes_per_elem

            dtype_str = str(arr.dtype)
            dtype_counts[dtype_str] = dtype_counts.get(dtype_str, 0) + num_params

        return {
            "total_parameters": total_params,
            "total_bytes": total_bytes,
            "total_bytes_gb": total_bytes / (1024**3),
            "num_weights": len(weights),
            "dtype_counts": dtype_counts,
        }
