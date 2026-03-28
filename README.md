# vdllm-MLX — Apple Silicon MLX Backend for Block Diffusion Language Models

Port of vdllm to Apple Silicon using MLX/Metal. Derived from [JetEngine](https://github.com/Labman42/JetEngine) by Yihan Bian et al.

**Supported Model**: [JetLM/SDAR-1.7B-Chat](https://huggingface.co/JetLM/SDAR-1.7B-Chat)

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     vdllm / MLX Backend                          │
│                                                                   │
│  User API                                                          │
│  └── LLM (llm.py)                                                 │
│       └── LLMEngine (CUDA/Triton)                                │
│                                                                   │
│  Backend Factory                                                   │
│  └── get_backend() → AttentionBackend Protocol                     │
│                        │                                          │
│                        ├── CUDA (FlashInfer + Triton) ← Primary  │
│                        ├── MLX (Metal) ← Apple Silicon Focus     │
│                        │    ├── mlx_sdar_model.py (28-layer)      │
│                        │    ├── mlx_backend.py (attention)        │
│                        │    ├── mlx_kv_cache.py (paged cache)   │
│                        │    └── tensor_bridge.py (MLX↔PyTorch)   │
│                        ├── MPS (PyTorch)                          │
│                        └── CPU (PyTorch)                          │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                      MLX SDAR Model (mlx_sdar_model.py)             │
│                                                                      │
│  SDARModel                                                          │
│  ├── Embedding (nn.Embedding)                                       │
│  ├── 28 × TransformerBlock                                            │
│  │    ├── Attention (GQA: 8 KV → 16 Q heads)                       │
│  │    │    ├── q_proj, k_proj, v_proj, o_proj (nn.Linear)            │
│  │    │    ├── q_norm, k_norm (RMSNorm per head)                    │
│  │    │    ├── RoPE (nn.RoPE, base=1M)                              │
│  │    │    └── mx.fast.scaled_dot_product_attention                 │
│  │    └── MLP (fused gate+up → SiLU → down_proj)                    │
│  │         ├── gate_up_proj (fused [2048 → 12288])                  │
│  │         └── down_proj ([6144 → 2048])                            │
│  ├── Final RMSNorm                                                  │
│  └── LM Head (nn.Linear [2048 → 151936])                            │
│                                                                      │
│  KV Cache: KVCache class (MLX 0.31.1 compatible)                   │
│  └── update_and_fetch(k, v) → concatenated cache                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Status

### ✅ Working

| Component | Status | Notes |
|-----------|--------|-------|
| CUDA Backend | ✅ Working | FlashInfer + Triton, original backend |
| Backend Protocol | ✅ Complete | `AttentionBackend` Protocol |
| MLX Backend | ✅ Complete | Apple Silicon MLX implementation |
| MLX SDAR Model | ✅ Complete | 28 layers, 2.03B params, full precision |
| Gate+Up Fusion | ✅ Complete | Fused MLP saves memory bandwidth |
| Generation | ✅ Complete | ~22.5 tok/s (M3, full bfloat16) |
| Weight Loading | ✅ Complete | Safetensors → MLX, bfloat16 workaround |
| Tensor Bridge | ✅ Complete | MLX ↔ PyTorch conversion |
| KV Cache | ✅ Complete | Paged KV cache implementation |
| Profiling | ✅ Complete | `OpTimer` with bottleneck detection |
| Triton Kernels | ✅ Working | Staircase attention, paged attention, MoE |
| PyTorch CPU Reference | ✅ Complete | Verified: "Paris" correct |

### ⏳ In Progress

| Component | Status | Notes |
|-----------|--------|-------|
| Phase 4: Metal Kernels | ⏳ Gated | Target: KV cache scatter/read |
| PyTorch MPS Backend | ⚠️ Basic | Implemented, not tested |

---

## Quick Start

### CUDA (Original - NVIDIA GPU)

```python
from vdllm import LLM, SamplingParams

llm = LLM("path/to/sdar-model", tensor_parallel_size=1)

params = SamplingParams(
    max_tokens=256,
    temperature=0.7,
    block_length=8,
    denoising_steps=10)

outputs = llm.generate(["Hello, how are you?"], params)
print(outputs[0]["text"])
```

### MLX (Apple Silicon)

```python
from vdllm.backends.mlx_sdar_model import load_sdar_model
import mlx.core as mx

model, config = load_sdar_model("/tmp/sdar-1.7b-chat")

# Forward pass
input_ids = tokenizer.encode("The capital of France is")
logits = model(mx.array([input_ids]))
next_token = mx.argmax(logits[0, -1])
print(tokenizer.decode([int(next_token)]))  # " Paris"
```

### MLX with KV Cache (Autoregressive Generation)

```python
from vdllm.backends.mlx_sdar_model import Model, KVCache

model, _ = load_sdar_model("/tmp/sdar-1.7b-chat")
caches = [KVCache() for _ in range(28)]

# Prefill
input_ids = tokenizer.encode("The capital of France is")
logits = model(mx.array([input_ids]), cache=caches)

# Generate
for _ in range(50):
    next_token = int(mx.argmax(logits[:, -1]))
    logits = model(mx.array([[next_token]]), cache=caches)
    print(tokenizer.decode([next_token]), end="", flush=True)
```

### Benchmark Generation (MLX)

```bash
cd DiffusionLM
source ../.venv/bin/activate
python tests/test_mlx_sdar_forward.py
```

---

## Project Structure

```
vdllm/
├── __init__.py                  # Package exports: LLM, SamplingParams
├── llm.py                       # User-facing LLM class
│
├── backends/                    # ✅ Multi-backend attention
│   ├── __init__.py             # Factory: get_backend() — auto-detection
│   ├── base.py                  # AttentionBackend Protocol
│   ├── mlx_backend.py          # MLX/Metal backend (Apple Silicon)
│   ├── mlx_sdar_model.py      # ✅ Full 28-layer SDAR model (MLX)
│   ├── mlx_kv_cache.py        # Paged KV cache for MLX
│   ├── mlx_model_loader.py     # Weight loading from safetensors
│   ├── tensor_bridge.py        # MLX ↔ PyTorch conversion
│   ├── cuda_backend.py         # CUDA/FlashInfer backend
│   ├── mps_backend.py          # PyTorch MPS fallback
│   └── cpu_backend.py          # PyTorch CPU fallback
│
├── profiling/                   # ✅ Profiling Infrastructure
│   ├── __init__.py
│   ├── op_timer.py             # OpTimer with mx.synchronize()
│   └── benchmark_e2e.py        # E2E benchmark harness
│
├── engine/                      # Core inference engine
│   ├── llm_engine.py           # Main orchestrator
│   ├── scheduler.py            # Batch scheduling
│   ├── model_runner.py         # Model execution
│   ├── block_manager.py        # Paged KV cache block allocator
│   └── distributed_manager.py   # TP/DP process groups
│
├── layers/                      # Neural network primitives
│   ├── attention.py            # BlockAttention (Triton/CUDA)
│   ├── linear.py              # TP-parallel linear layers
│   ├── layernorm.py           # RMSNorm
│   ├── activation.py          # SiLU activation
│   ├── rotary_embedding.py    # RoPE
│   └── sampler.py             # Sampling
│
├── models/                      # Model architectures
│   ├── sdar.py                # SDARForCausalLM (dense)
│   └── sdar_moe.py            # SDARMoeForCausalLM (sparse MoE)
│
├── inference/                   # Pure PyTorch inference (MPS/CPU)
│   ├── denoiser.py           # Block denoiser
│   ├── generator.py           # Block diffusion generator
│   ├── sdar_inference.py      # CLI inference
│   └── sdar_wrapper.py       # Model wrapper
│
└── kernels/triton/             # Triton GPU kernels
    ├── fused_moe.py           # Fused MoE
    └── attention/
        ├── block_prefill_attention_v2.py  # Staircase attention
        ├── fused_page_attention_v3.py     # Paged KV attention
        └── fused_page_attention_v6.py     # Optimized variant
```

---

## Performance

### MLX Generation Throughput (Full BFloat16, No Quantization)

| Hardware | Throughput |
|----------|------------|
| M3 Max (16") | ~22.5 tok/s |
| M3 Pro | ~15-18 tok/s (estimated) |
| M2 Ultra | ~18-22 tok/s (estimated) |

### MLX Prefill Throughput (Full BFloat16)

| Seq Len | Time | Throughput |
|---------|------|------------|
| 32 | 78 ms | 408 tok/s |
| 128 | 170 ms | 752 tok/s |
| 512 | 639 ms | 801 tok/s |

### Bottleneck Analysis (MLX)

| Phase | Bottleneck | % |
|-------|------------|---|
| Prefill | MLP | 67.6% |
| Generation | Attention (KV cache read) | 60-70% |

**Phase 4 Target**: KV cache operations (`reshape_and_cache.metal`, `pagedattention.metal`)

---

## Block Diffusion Overview

Unlike autoregressive models that generate one token at a time, block diffusion generates **blocks of tokens** via iterative denoising:

```
Step 0: [prompt tokens...] [MASK MASK MASK MASK MASK MASK MASK MASK]
Step 1: [prompt tokens...] [The   MASK MASK MASK MASK MASK MASK MASK]
Step 2: [prompt tokens...] [The   quick MASK fox  MASK MASK MASK MASK]
  ...
Step T: [prompt tokens...] [The   quick brown fox  jumps over the  fence]
                            └──────── block committed ────────────────┘
```

---

## What Left to Do

### Phase 4: Native Metal Kernels (Gated On)
- [ ] `reshape_and_cache.metal` — scatter write K,V to paged cache
- [ ] `pagedattention.metal` — paged attention read during generation
- Reference: `reference_repos/vllm-metal/`

### Phase 5: MPS Backend Testing
- [ ] Verify PyTorch MPS fallback works
- [ ] Compare with MLX performance

### Phase 6: Full Engine Integration (MLX)
- [ ] Connect MLX backend to `LLMEngine`
- [ ] Support streaming generation
- [ ] Support batched requests

### Optional Enhancements
- [ ] 4-bit quantization (would boost to 100+ tok/s, but user requested full precision)

---

## Reference Repos

| Repo | Purpose |
|------|---------|
| [mlx-lm](https://github.com/ml-explore/mlx-lm) | MLX transformer patterns (Qwen2/Llama) |
| [vllm-mlx](https://github.com/waybarrios/vllm-mlx) | MLX serving layer reference |
| [vllm-metal](https://github.com/waybarrios/vllm-metal) | Native Metal kernels reference |
| [JetEngine](https://github.com/Labman42/JetEngine) | Original block diffusion engine |

---

## Citation

```bibtex
@misc{jetengine2025,
    title={JetEngine: Inference Engine for Block Diffusion Language Models},
    author={Yihan Bian et al.},
    year={2025},
    url={https://github.com/Labman42/JetEngine}
}
```

```bibtex
@misc{sdar2025,
    title={SDAR: Synergy of Diffusion and AutoRegression},
    author={JetLM Team},
    year={2025},
    url={https://github.com/Jet-Astra/SDAR}
}
```
