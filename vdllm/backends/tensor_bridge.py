"""Tensor bridge between MLX and PyTorch for vdllm.

This module provides zero-copy (where possible) and safe conversion
between MLX arrays and PyTorch tensors, handling the tricky edge cases:

1. bfloat16: MLX misinterprets bfloat16 from numpy - must convert via float32
2. Unified memory: Apple Silicon unified memory allows zero-copy in many cases
3. Dtype mapping: PyTorch and MLX dtype systems differ slightly

MLX ↔ PyTorch Conversion Rules
===============================

| PyTorch dtype | → MLX dtype | Notes                              |
|--------------|-------------|-------------------------------------|
| float32      | float32     | Direct conversion                   |
| float16      | float16     | Direct conversion                   |
| bfloat16     | bfloat16    | Must go through float32 (MLX bug)  |
| int32        | int32       | Direct conversion                   |
| int64        | int64       | Direct conversion                   |

Zero-Copy Strategy (Apple Silicon)
==================================

On Apple Silicon with unified memory, PyTorch tensors and MLX arrays can
share the same underlying memory buffer when:

1. The tensor is contiguous
2. The tensor is on CPU (or MPS with proper setup)
3. The dtype is compatible (not bfloat16)

This module uses numpy as an intermediary for zero-copy when possible,
falling back to explicit copies when necessary.

Example Usage
============

    import torch
    import mlx.core as mx
    from vdllm.backends.tensor_bridge import to_mlx, to_torch

    # PyTorch → MLX
    pt_tensor = torch.randn(10, 20, dtype=torch.float32)
    mlx_array = to_mlx(pt_tensor)

    # MLX → PyTorch
    mlx_array = mx.random.normal((10, 20))
    pt_tensor = to_torch(mlx_array)

    # bfloat16 (handled correctly)
    pt_tensor = torch.randn(10, 20, dtype=torch.bfloat16)
    mlx_array = to_mlx(pt_tensor)  # No error!
"""

from typing import Union
import math

import torch
import mlx.core as mx
import numpy as np


# MLX dtype from PyTorch dtype
_MLX_DTYPE_FROM_TORCH = {
    torch.float32: mx.float32,
    torch.float16: mx.float16,
    torch.bfloat16: mx.bfloat16,
    torch.int32: mx.int32,
    torch.int64: mx.int64,
    torch.uint8: mx.uint8,
    torch.int8: mx.int8,
}

# PyTorch dtype from MLX dtype
_TORCH_DTYPE_FROM_MLX = {v: k for k, v in _MLX_DTYPE_FROM_TORCH.items()}


def to_mlx(
    tensor: torch.Tensor,
    dtype: mx.Dtype | None = None,
    strict: bool = True,
) -> mx.array:
    """Convert PyTorch tensor to MLX array.

    This function handles the tricky edge cases:

    1. bfloat16: MLX has a bug where bfloat16 from numpy is misinterpreted.
       We work around by converting through float32.

    2. Device: MLX on Apple Silicon uses unified memory. If the PyTorch tensor
       is on CPU and contiguous, we can often do zero-copy via numpy view.

    3. Non-contiguous: For non-contiguous tensors, we make a contiguous copy.

    Args:
        tensor: PyTorch tensor to convert
        dtype: Optional MLX dtype. If None, inferred from PyTorch tensor.
              For bfloat16, this is required due to MLX bug.
        strict: If True, raise error on unsupported dtypes. If False,
                silently convert to closest supported dtype.

    Returns:
        MLX array with the same shape and (converted) dtype

    Raises:
        ValueError: If dtype is unsupported and strict=True

    Example:
        >>> import torch
        >>> from vdllm.backends.tensor_bridge import to_mlx
        >>> pt = torch.randn(4, 8, 16, dtype=torch.bfloat16)
        >>> mlx_arr = to_mlx(pt)  # Works despite MLX bfloat16 bug!
    """
    if tensor.numel() == 0:
        # Empty tensor - return empty MLX array
        shape = tuple(tensor.shape)
        mlx_dtype = dtype or _MLX_DTYPE_FROM_TORCH.get(tensor.dtype, mx.float32)
        return mx.array([], dtype=mlx_dtype).reshape(shape)

    # Get target dtype
    if dtype is not None:
        target_dtype = dtype
    else:
        target_dtype = _MLX_DTYPE_FROM_TORCH.get(tensor.dtype, None)
        if target_dtype is None:
            if strict:
                raise ValueError(
                    f"Unsupported PyTorch dtype {tensor.dtype} for MLX conversion. "
                    f"Supported: {list(_MLX_DTYPE_FROM_TORCH.keys())}"
                )
            # Fall back to float32
            target_dtype = mx.float32

    # Handle bfloat16 specially (MLX bug workaround)
    # MLX misinterprets bfloat16 data from numpy - we must convert via float32
    if tensor.dtype == torch.bfloat16:
        # Convert bfloat16 → float32 → numpy → MLX bfloat16
        # This is safe because we explicitly specify dtype
        float32_np = tensor.float().numpy()
        return mx.array(float32_np, dtype=mx.bfloat16)

    # For other dtypes, try zero-copy via numpy
    # On Apple Silicon with unified memory, if tensor is on CPU and contiguous,
    # the numpy view shares memory with the PyTorch tensor
    if tensor.is_contiguous():
        try:
            # Try zero-copy via numpy view
            np_view = tensor.numpy()

            # Check if numpy dtype matches MLX expectations
            # For float32, float16, int32, etc., direct conversion works
            if target_dtype in [mx.float32, mx.float16, mx.int32, mx.int64,
                                 mx.uint8, mx.int8]:
                return mx.array(np_view, dtype=target_dtype)
            else:
                # Fall through to explicit copy for other types
                pass

        except (TypeError, ValueError) as e:
            # Numpy view failed (e.g., unsupported dtype) - fall through
            pass

    # Explicit copy conversion
    np_arr = tensor.detach().numpy()
    return mx.array(np_arr, dtype=target_dtype)


def to_torch(
    array: mx.array,
    dtype: torch.dtype | None = None,
    device: torch.device | None = None,
    strict: bool = True,
) -> torch.Tensor:
    """Convert MLX array to PyTorch tensor.

    This function handles conversion from MLX arrays to PyTorch tensors,
    with proper dtype handling and device placement.

    Args:
        array: MLX array to convert
        dtype: Optional PyTorch dtype. If None, inferred from MLX array.
        device: Target device. If None, uses CPU.
        strict: If True, raise error on unsupported dtypes. If False,
                silently convert to closest supported dtype.

    Returns:
        PyTorch tensor with the same shape and (converted) dtype

    Raises:
        ValueError: If dtype is unsupported and strict=True

    Example:
        >>> import mlx.core as mx
        >>> from vdllm.backends.tensor_bridge import to_torch
        >>> mlx_arr = mx.random.normal((4, 8, 16))
        >>> pt = to_torch(mlx_arr)  # Gets float32 by default
    """
    if array.size == 0:
        # Empty array
        shape = tuple(array.shape)
        torch_dtype = dtype or _TORCH_DTYPE_FROM_MLX.get(array.dtype, torch.float32)
        return torch.empty(shape, dtype=torch_dtype, device=device or 'cpu')

    # Get target dtype
    if dtype is not None:
        target_dtype = dtype
    else:
        target_dtype = _TORCH_DTYPE_FROM_MLX.get(array.dtype, None)
        if target_dtype is None:
            if strict:
                raise ValueError(
                    f"Unsupported MLX dtype {array.dtype} for PyTorch conversion. "
                    f"Supported: {list(_TORCH_DTYPE_FROM_MLX.keys())}"
                )
            # Fall back to float32
            target_dtype = torch.float32

    # Convert MLX → numpy → PyTorch
    # MLX arrays are always on CPU after ops unless explicitly moved
    #
    # IMPORTANT: MLX bfloat16 needs special handling because:
    # 1. MLX stores bfloat16 with 2 bytes per element
    # 2. numpy doesn't have native bfloat16, so np.array() creates a
    #    malformed buffer (reporting itemsize=1 for the PEP 3118 format)
    # 3. We must go through float32 to get correct values
    if array.dtype == mx.bfloat16:
        # Convert bfloat16 -> float32 via MLX, then to torch
        float32_array = array.astype(mx.float32)
        np_arr = np.array(float32_array)
        pt_tensor = torch.from_numpy(np_arr).to(dtype=torch.bfloat16, device=device or 'cpu')
        return pt_tensor

    np_arr = np.array(array)

    # Direct conversion for other types
    pt_tensor = torch.from_numpy(np_arr).to(dtype=target_dtype, device=device or 'cpu')
    return pt_tensor


def mlx_to_torch_dtype(dtype: mx.Dtype) -> torch.dtype:
    """Convert MLX dtype to PyTorch dtype.

    Args:
        dtype: MLX dtype to convert

    Returns:
        Corresponding PyTorch dtype

    Raises:
        ValueError: If dtype is unsupported
    """
    torch_dtype = _TORCH_DTYPE_FROM_MLX.get(dtype, None)
    if torch_dtype is None:
        raise ValueError(f"Unsupported MLX dtype {dtype} for PyTorch conversion")
    return torch_dtype


def torch_to_mlx_dtype(dtype: torch.dtype) -> mx.Dtype:
    """Convert PyTorch dtype to MLX dtype.

    Args:
        dtype: PyTorch dtype to convert

    Returns:
        Corresponding MLX dtype

    Raises:
        ValueError: If dtype is unsupported
    """
    mlx_dtype = _MLX_DTYPE_FROM_TORCH.get(dtype, None)
    if mlx_dtype is None:
        raise ValueError(f"Unsupported PyTorch dtype {dtype} for MLX conversion")
    return mlx_dtype


class TensorBridge:
    """Context manager for mixed MLX/PyTorch operations.

    This class provides a convenient way to convert tensors between
    MLX and PyTorch within a context, ensuring proper cleanup and
    avoiding memory leaks.

    Example:
        >>> import torch
        >>> import mlx.core as mx
        >>> from vdllm.backends.tensor_bridge import TensorBridge
        >>>
        >>> with TensorBridge() as bridge:
        ...     pt_tensor = torch.randn(10, 20)
        ...     mlx_array = bridge.to_mlx(pt_tensor)
        ...     # ... do MLX operations ...
        ...     result = bridge.to_torch(mlx_array)
        ...
        >>> # Tensors are cleaned up when exiting context

    Note:
        This is primarily useful for debugging and testing. In production,
        prefer direct to_mlx() / to_torch() calls.
    """

    def __init__(self, device: str = "cpu"):
        self.device = device
        self._mlx_tensors: list[mx.array] = []
        self._torch_tensors: list[torch.Tensor] = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Clear references (MLX/PyTorch handle their own memory)
        self._mlx_tensors.clear()
        self._torch_tensors.clear()
        return False

    def to_mlx(self, tensor: torch.Tensor, dtype: mx.Dtype | None = None) -> mx.array:
        """Convert PyTorch tensor to MLX array (tracked)."""
        arr = to_mlx(tensor, dtype=dtype)
        self._mlx_tensors.append(arr)
        return arr

    def to_torch(self, array: mx.array, dtype: torch.dtype | None = None,
                 device: torch.device | None = None) -> torch.Tensor:
        """Convert MLX array to PyTorch tensor (tracked)."""
        tensor = to_torch(array, dtype=dtype, device=device or self.device)
        self._torch_tensors.append(tensor)
        return tensor


def ensure_mlx_array(arr_or_tensor: Union[mx.array, torch.Tensor]) -> mx.array:
    """Ensure the input is an MLX array, converting if necessary.

    Args:
        arr_or_tensor: Either an MLX array or PyTorch tensor

    Returns:
        MLX array

    Example:
        >>> import torch
        >>> import mlx.core as mx
        >>> from vdllm.backends.tensor_bridge import ensure_mlx_array
        >>>
        >>> mlx_arr = ensure_mlx_array(mx.randn(10, 20))  # Already MLX
        >>> mlx_arr = ensure_mlx_array(torch.randn(10, 20))  # Converted
    """
    if isinstance(arr_or_tensor, mx.array):
        return arr_or_tensor
    elif isinstance(arr_or_tensor, torch.Tensor):
        return to_mlx(arr_or_tensor)
    else:
        raise TypeError(
            f"Expected mx.array or torch.Tensor, got {type(arr_or_tensor)}"
        )


def ensure_torch_tensor(arr_or_tensor: Union[mx.array, torch.Tensor]) -> torch.Tensor:
    """Ensure the input is a PyTorch tensor, converting if necessary.

    Args:
        arr_or_tensor: Either an MLX array or PyTorch tensor

    Returns:
        PyTorch tensor

    Example:
        >>> import torch
        >>> import mlx.core as mx
        >>> from vdllm.backends.tensor_bridge import ensure_torch_tensor
        >>>
        >>> pt_tensor = ensure_torch_tensor(torch.randn(10, 20))  # Already Torch
        >>> pt_tensor = ensure_torch_tensor(mx.randn(10, 20))  # Converted
    """
    if isinstance(arr_or_tensor, torch.Tensor):
        return arr_or_tensor
    elif isinstance(arr_or_tensor, mx.array):
        return to_torch(arr_or_tensor)
    else:
        raise TypeError(
            f"Expected mx.array or torch.Tensor, got {type(arr_or_tensor)}"
        )


# Export public API
__all__ = [
    "to_mlx",
    "to_torch",
    "mlx_to_torch_dtype",
    "torch_to_mlx_dtype",
    "TensorBridge",
    "ensure_mlx_array",
    "ensure_torch_tensor",
]
