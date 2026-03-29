"""Hardware detection for vdllm backend dispatch.

Uses lazy imports so this module can be imported without torch or mlx installed.
"""


def detect_backend() -> str:
    """Detect the best available backend: 'cuda', 'mlx', 'mps', or 'cpu'."""
    # CUDA (NVIDIA GPU)
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass

    # MLX (Apple Silicon)
    try:
        import mlx.core as mx
        if mx.metal.is_available():
            return "mlx"
    except ImportError:
        pass

    # MPS (Apple GPU via PyTorch)
    try:
        import torch
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass

    return "cpu"
