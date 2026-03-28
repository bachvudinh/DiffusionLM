"""Neural network layer primitives for vdllm.

Derived from JetEngine by Yihan Bian et al.
Reference: https://github.com/Labman42/JetEngine

Layer Components
================

    ┌────────────────────────────────────────────────────────────┐
    │                    vdllm Layer Library                      │
    │                                                            │
    │  Attention                                                 │
    │  ├── BlockAttention        — staircase prefill + paged KV  │
    │  └── store_kvcache         — Triton kernel for KV storage  │
    │                                                            │
    │  Linear (Tensor-Parallel)                                  │
    │  ├── ReplicatedLinear      — no sharding                   │
    │  ├── ColumnParallelLinear  — shard output dim              │
    │  ├── MergedColumnParallel  — fused gate+up projection      │
    │  ├── QKVParallelLinear     — fused Q/K/V with GQA          │
    │  └── RowParallelLinear     — shard input dim + all-reduce  │
    │                                                            │
    │  Normalization                                             │
    │  └── RMSNorm               — compiled fused add+norm       │
    │                                                            │
    │  Activation                                                │
    │  └── SiluAndMul            — Liger fused SiLU gating       │
    │                                                            │
    │  Positional Encoding                                       │
    │  └── RotaryEmbedding       — cached RoPE with factory      │
    │                                                            │
    │  Embedding / Head                                          │
    │  ├── VocabParallelEmbedding — TP-sharded embedding         │
    │  └── ParallelLMHead         — TP-sharded output head       │
    │                                                            │
    │  Sampling                                                  │
    │  └── sample_with_temperature_topk_topp                     │
    └────────────────────────────────────────────────────────────┘
"""
