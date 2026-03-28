"""Engine module - core inference engine components for vdllm.

Derived from JetEngine by Yihan Bian et al.
Reference: https://github.com/Labman42/JetEngine

Architecture Overview
====================

    The engine module contains the core components that orchestrate
    block diffusion inference:

    ┌────────────────────────────────────────────────────────────────┐
    │                     vdllm Engine Pipeline                      │
    │                                                                │
    │  User API                                                      │
    │  └── LLM (llm.py)                                              │
    │       └── LLMEngine (llm_engine.py)                            │
    │            ├── Scheduler (scheduler.py)                        │
    │            │    ├── schedule()  → prefill/denoise batches      │
    │            │    ├── postprocess() → sampling + remasking       │
    │            │    └── BlockManager (block_manager.py)            │
    │            │         └── KV cache block allocation             │
    │            │                                                   │
    │            ├── ModelRunner (model_runner.py)                   │
    │            │    ├── prepare_prefill/denoise → input tensors    │
    │            │    ├── run() → model forward + logits             │
    │            │    └── CUDA graph capture/replay                  │
    │            │                                                   │
    │            └── DistributedManager (distributed_manager.py)    │
    │                 └── TP/DP process group management             │
    │                                                                │
    │  Data Structures                                               │
    │  └── Sequence (sequence.py)                                    │
    │       └── Tracks one generation request through lifecycle     │
    └────────────────────────────────────────────────────────────────┘
"""
