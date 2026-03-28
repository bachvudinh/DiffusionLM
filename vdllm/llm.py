"""Top-level LLM class — user-facing API for vdllm.

Derived from JetEngine by Yihan Bian et al.
Reference: https://github.com/Labman42/JetEngine

Usage
=====

    ┌──────────────────────────────────────────────────────┐
    │  from vdllm import LLM, SamplingParams               │
    │                                                      │
    │  llm = LLM("path/to/sdar-model",                    │
    │            max_num_seqs=32,                           │
    │            tensor_parallel_size=2)                    │
    │                                                      │
    │  params = SamplingParams(                             │
    │      max_tokens=256,                                  │
    │      temperature=0.7,                                 │
    │      block_length=8,                                  │
    │      denoising_steps=10,                              │
    │      remasking_strategy="low_confidence_static")      │
    │                                                      │
    │  outputs = llm.generate(                              │
    │      ["Hello, world!", "Once upon a time"],           │
    │      params)                                          │
    │                                                      │
    │  for out in outputs:                                  │
    │      print(out["text"])                               │
    └──────────────────────────────────────────────────────┘
"""

from vdllm.engine.llm_engine import LLMEngine


class LLM(LLMEngine):
    """User-facing LLM class. Inherits all functionality from LLMEngine.

    Input:
        model:    str — path to model directory
        **kwargs: forwarded to Config (max_num_seqs, tensor_parallel_size, etc.)

    See LLMEngine for full API documentation.
    """
    pass
