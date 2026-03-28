"""Model architectures for vdllm block diffusion inference.

Derived from JetEngine by Yihan Bian et al.
Reference: https://github.com/Labman42/JetEngine

Supported Models
================

    ┌──────────────────────────────────────────────────────────────┐
    │                    vdllm Model Registry                      │
    │                                                              │
    │  SDARForCausalLM (sdar.py)                                   │
    │  ├── Dense transformer with block-local attention             │
    │  ├── QKV fusion, gate-up fusion, RoPE, GQA                   │
    │  └── packed_modules_mapping for weight loading                │
    │                                                              │
    │  SDARMoeForCausalLM (sdar_moe.py)                            │
    │  ├── Sparse Mixture-of-Experts variant                        │
    │  ├── Switch-style top-k routing with fused MoE kernel        │
    │  ├── Dense + MoE layers interleaved by decoder_sparse_step   │
    │  └── Same attention as SDAR (shared SDARAttention)            │
    └──────────────────────────────────────────────────────────────┘
"""
