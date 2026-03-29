"""vdllm -- Block Diffusion Language Model inference engine.

Supports CUDA (NVIDIA GPU) and MLX (Apple Silicon) backends.
Auto-detects hardware and dispatches to the appropriate engine.

Quick Start:

    from vdllm import LLM, SamplingParams

    llm = LLM("path/to/sdar-model")
    params = SamplingParams(max_tokens=256, temperature=0.7)
    outputs = llm.generate(["Hello, world!"], params)
    for out in outputs:
        print(out["text"])
"""


def __getattr__(name):
    if name == "LLM":
        from vdllm.llm import LLM
        return LLM
    if name == "SamplingParams":
        from vdllm.sampling_params import SamplingParams
        return SamplingParams
    if name == "Config":
        from vdllm.config import Config
        return Config
    raise AttributeError(f"module 'vdllm' has no attribute {name!r}")


__all__ = ["LLM", "SamplingParams", "Config"]
