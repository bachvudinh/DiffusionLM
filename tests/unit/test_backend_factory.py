"""Unit tests for backend factory auto-detection.

Run from DiffusionLM root:
    uv run python -m pytest tests/unit/test_backend_factory.py -v
"""

import sys
import os

import pytest
import torch


class TestBackendFactory:
    """Test the get_backend() factory function."""

    def test_list_available_backends(self):
        """Test that list_available_backends returns expected structure."""
        from vdllm.backends import list_available_backends

        backends = list_available_backends()

        # Should always have 'cpu' as fallback
        assert "cpu" in backends

        # Should be in priority order
        priority_order = ["cuda", "mlx", "mps", "cpu"]
        backend_positions = {b: backends.index(b) for b in backends}
        for i, higher in enumerate(priority_order[:-1]):
            for lower in priority_order[i+1:]:
                if higher in backends and lower in backends:
                    assert backend_positions[higher] < backend_positions[lower], \
                        f"{higher} should come before {lower}"

    def test_explicit_cuda_backend(self):
        """Test explicit cuda backend selection."""
        from vdllm.backends import get_backend

        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        backend = get_backend(32, 8, 128, backend_type="cuda")
        assert backend.name == "cuda"

    def test_explicit_mlx_backend(self):
        """Test explicit mlx backend selection."""
        from vdllm.backends import get_backend

        try:
            import mlx.core as mx
            if not mx.metal.is_available():
                pytest.skip("MLX Metal not available")
        except ImportError:
            pytest.skip("MLX not installed")

        backend = get_backend(32, 8, 128, backend_type="mlx")
        assert backend.name == "mlx"

    def test_explicit_mps_backend(self):
        """Test explicit mps backend selection."""
        from vdllm.backends import get_backend

        if not torch.backends.mps.is_available():
            pytest.skip("MPS not available")

        backend = get_backend(32, 8, 128, backend_type="mps")
        assert backend.name == "mps"

    def test_explicit_cpu_backend(self):
        """Test explicit cpu backend selection."""
        from vdllm.backends import get_backend

        backend = get_backend(32, 8, 128, backend_type="cpu")
        assert backend.name == "cpu"

    def test_unknown_backend_raises(self):
        """Test that unknown backend type raises ValueError."""
        from vdllm.backends import get_backend

        with pytest.raises(ValueError) as exc_info:
            get_backend(32, 8, 128, backend_type="unknown_backend")
        assert "Unknown backend_type" in str(exc_info.value)

    def test_cuda_unavailable_raises(self):
        """Test that requesting unavailable CUDA raises error."""
        from vdllm.backends import get_backend

        if torch.cuda.is_available():
            pytest.skip("CUDA is available, can't test unavailability")

        with pytest.raises(ValueError) as exc_info:
            get_backend(32, 8, 128, backend_type="cuda")
        assert "CUDA backend requested" in str(exc_info.value)

    def test_mlx_unavailable_raises(self):
        """Test that requesting unavailable MLX raises error."""
        from vdllm.backends import get_backend

        try:
            import mlx.core as mx
            if mx.metal.is_available():
                pytest.skip("MLX Metal is available")
        except ImportError:
            pass  # MLX not installed, test will pass

        with pytest.raises(ValueError) as exc_info:
            get_backend(32, 8, 128, backend_type="mlx")
        assert "MLX backend requested" in str(exc_info.value)

    def test_mps_unavailable_raises(self):
        """Test that requesting unavailable MPS raises error."""
        from vdllm.backends import get_backend

        if torch.backends.mps.is_available():
            pytest.skip("MPS is available")

        with pytest.raises(ValueError) as exc_info:
            get_backend(32, 8, 128, backend_type="mps")
        assert "MPS backend requested" in str(exc_info.value)

    def test_auto_detection_returns_valid_backend(self):
        """Test that auto-detection returns a backend with correct interface."""
        from vdllm.backends import get_backend
        from vdllm.backends.base import AttentionBackend

        backend = get_backend(32, 8, 128)

        # Should be an AttentionBackend
        assert isinstance(backend, AttentionBackend)

        # Should have required methods
        assert hasattr(backend, "name")
        assert hasattr(backend, "forward")

        # Name should be valid
        assert backend.name in ["cuda", "mlx", "mps", "cpu"]


class TestBackendInterface:
    """Test that backends implement the correct interface."""

    def test_mlx_backend_interface(self):
        """Test MLX backend implements AttentionBackend protocol."""
        try:
            import mlx.core as mx
            if not mx.metal.is_available():
                pytest.skip("MLX Metal not available")
        except ImportError:
            pytest.skip("MLX not installed")

        from vdllm.backends import get_backend
        from vdllm.backends.base import AttentionBackend

        backend = get_backend(32, 8, 128, backend_type="mlx")

        # Check it's a valid AttentionBackend
        assert isinstance(backend, AttentionBackend)

        # Check required attributes
        assert hasattr(backend, "name")
        assert backend.name == "mlx"

        # Check required methods exist
        assert hasattr(backend, "forward")

    def test_cpu_backend_interface(self):
        """Test CPU backend implements AttentionBackend protocol."""
        from vdllm.backends import get_backend
        from vdllm.backends.base import AttentionBackend

        backend = get_backend(32, 8, 128, backend_type="cpu")

        # Check it's a valid AttentionBackend
        assert isinstance(backend, AttentionBackend)

        # Check required attributes
        assert hasattr(backend, "name")
        assert backend.name == "cpu"

        # Check required methods exist
        assert hasattr(backend, "forward")
