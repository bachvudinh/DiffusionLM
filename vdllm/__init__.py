"""vdllm — Block Diffusion Language Model inference engine.

Derived from JetEngine by Yihan Bian et al.
Reference: https://github.com/Labman42/JetEngine

A high-performance inference engine for SDAR and SDAR-MoE block
diffusion language models. Features include:

  - Paged KV cache with prefix caching (xxhash)
  - Staircase block-local attention (Triton kernel)
  - FlashAttention-2 for paged denoise attention
  - CUDA graph capture/replay for denoise steps
  - Tensor parallelism (TP) for multi-GPU inference
  - 5 remasking strategies: sequential, low_confidence_static,
    low_confidence_dynamic, entropy_bounded, random
  - Fused MoE Triton kernel for sparse expert models

Quick Start:

    from vdllm import LLM, SamplingParams

    llm = LLM("path/to/sdar-model")
    params = SamplingParams(max_tokens=256, temperature=0.7)
    outputs = llm.generate(["Hello, world!"], params)
"""

def __getattr__(name):
    if name == "LLM":
        from vdllm.llm import LLM
        return LLM
    if name == "SamplingParams":
        from vdllm.sampling_params import SamplingParams
        return SamplingParams
    raise AttributeError(f"module 'vdllm' has no attribute {name!r}")

__all__ = ["LLM", "SamplingParams"]
