"""Unified LLM class for block diffusion inference.

Auto-detects hardware and dispatches to the appropriate backend engine.
Works on CUDA (NVIDIA GPU) and MLX (Apple Silicon).

Usage:
    from vdllm import LLM, SamplingParams

    llm = LLM("path/to/sdar-model")
    params = SamplingParams(max_tokens=256, temperature=0.7)
    outputs = llm.generate(["Hello, world!"], params)
    for out in outputs:
        print(out["text"])
"""

from dataclasses import fields

from vdllm.config import Config
from vdllm.sampling_params import SamplingParams


class LLM:
    """Unified LLM class that dispatches to CUDA or MLX engine.

    Uses composition: holds an _engine reference (CUDAEngine or MLXEngine)
    rather than inheriting from either. The user-facing API is identical
    regardless of backend.

    Args:
        model: Path to model directory (safetensors + config.json).
        backend: 'cuda', 'mlx', 'mps', 'cpu', or 'auto' (default).
        **kwargs: Forwarded to Config (max_num_seqs, tensor_parallel_size, etc.)
    """

    def __init__(self, model: str, backend: str = "auto", **kwargs):
        config_field_names = {f.name for f in fields(Config)}
        config_kwargs = {k: v for k, v in kwargs.items()
                         if k in config_field_names}
        config_kwargs["backend"] = backend
        self.config = Config(model, **config_kwargs)

        backend = self.config.backend
        print(f"[vdllm] Using backend: {backend}")

        if backend == "cuda":
            self._engine = self._init_cuda(model, kwargs)
        elif backend == "mlx":
            self._engine = self._init_mlx()
        else:
            raise ValueError(
                f"Unsupported backend: {backend}. "
                f"Install CUDA (torch+cuda) or MLX (mlx) for GPU inference.")

        self.tokenizer = self._engine.tokenizer

    def _init_cuda(self, model, kwargs):
        """Initialize CUDA engine (existing LLMEngine)."""
        from vdllm.engine.llm_engine import LLMEngine
        return LLMEngine(model, **kwargs)

    def _init_mlx(self):
        """Initialize MLX engine."""
        from vdllm.engine.mlx_engine import MLXEngine
        return MLXEngine(self.config)

    def generate(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
        use_tqdm: bool = True,
    ) -> list[dict]:
        """Generate completions for a batch of prompts.

        Returns:
            List of dicts, each with keys:
                text: str -- decoded output
                token_ids: list[int] -- output token IDs
                trajectory: list -- per-token denoising info (CUDA only)
                logprobs: list -- per-token log probabilities (CUDA only)
                entropies: list -- per-token entropies (CUDA only)
        """
        return self._engine.generate(prompts, sampling_params,
                                     use_tqdm=use_tqdm)

    def generate_streaming(self, prompts, sampling_params,
                           max_active=None, use_tqdm=True):
        """Stream prompts with bounded concurrency. CUDA backend only."""
        if not hasattr(self._engine, "generate_streaming"):
            raise NotImplementedError(
                f"generate_streaming is not supported on {self.config.backend} backend")
        return self._engine.generate_streaming(
            prompts, sampling_params,
            max_active=max_active, use_tqdm=use_tqdm)

    def add_request(self, prompt, sampling_params):
        """Add a single generation request. CUDA backend only."""
        if not hasattr(self._engine, "add_request"):
            raise NotImplementedError(
                f"add_request is not supported on {self.config.backend} backend")
        return self._engine.add_request(prompt, sampling_params)

    def step(self):
        """Run one engine step. CUDA backend only."""
        if not hasattr(self._engine, "step"):
            raise NotImplementedError(
                f"step is not supported on {self.config.backend} backend")
        return self._engine.step()

    def is_finished(self):
        return self._engine.is_finished()
