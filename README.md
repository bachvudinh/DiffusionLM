# vdllm — Block Diffusion Language Model Inference Engine

High-performance inference engine for **SDAR** and **SDAR-MoE** block diffusion language models. Derived from [JetEngine](https://github.com/Labman42/JetEngine) by Yihan Bian et al.

**Supported Models:** [JetLM/SDAR-1.7B-Chat](https://huggingface.co/JetLM/SDAR-1.7B-Chat) and SDAR-MoE variants

## Architecture

```
┌────────────────────────────────────────────────────────────────────┐
│                        vdllm Engine Pipeline                       │
│                                                                    │
│  User API                                                          │
│  └── LLM (llm.py)                                                 │
│       └── LLMEngine (engine/llm_engine.py)                         │
│            ├── Scheduler (engine/scheduler.py)                     │
│            │    ├── schedule()    → prefill/denoise batches         │
│            │    ├── postprocess() → sampling + remasking            │
│            │    └── BlockManager (engine/block_manager.py)          │
│            │         └── KV cache block allocation + prefix caching │
│            │                                                       │
│            ├── ModelRunner (engine/model_runner.py)                 │
│            │    ├── prepare_prefill/denoise → input tensors         │
│            │    ├── run() → model forward + logits                  │
│            │    └── CUDA graph capture/replay                       │
│            │                                                       │
│            └── DistributedManager (engine/distributed_manager.py)   │
│                 └── TP/DP process group management                  │
│                                                                    │
│  Models                                                            │
│  ├── SDARForCausalLM (models/sdar.py)      — Dense transformer     │
│  └── SDARMoeForCausalLM (models/sdar_moe.py) — Sparse MoE variant │
│                                                                    │
│  Layers                                                            │
│  ├── BlockAttention  — staircase prefill + paged KV denoise        │
│  ├── QKVParallelLinear, RowParallelLinear, etc. — TP-sharded       │
│  ├── RMSNorm — compiled fused add+norm                             │
│  ├── RotaryEmbedding — cached RoPE                                 │
│  └── ParallelLMHead — TP-sharded output with gather                │
│                                                                    │
│  Kernels                                                           │
│  ├── Staircase block-local attention (Triton)                      │
│  ├── Fused paged KV cache attention (Triton)                       │
│  ├── store_kvcache (Triton)                                        │
│  └── Fused MoE (Triton)                                            │
└────────────────────────────────────────────────────────────────────┘
```

## Quick Start

### Engine API (GPU with Triton + FlashAttention)

```python
from vdllm import LLM, SamplingParams

llm = LLM("path/to/sdar-model",
           max_num_seqs=32,
           tensor_parallel_size=1)

params = SamplingParams(
    max_tokens=256,
    temperature=0.7,
    block_length=8,
    denoising_steps=10,
    remasking_strategy="low_confidence_static")

outputs = llm.generate(
    ["Hello, how are you?", "Once upon a time"],
    params)

for out in outputs:
    print(out["text"])
```

### Streaming Generation

```python
# Process large batches with bounded concurrency
outputs = llm.generate_streaming(
    prompts,
    params,
    max_active=64)
```

### Pure PyTorch Inference (MPS / CPU)

```bash
python -m vdllm.inference.sdar_inference \
    --model_dir /tmp/sdar-1.7b-chat \
    --prompt "Hello, how are you?" \
    --device mps \
    --gen_length 128
```

## Project Structure

```
vdllm/
├── __init__.py                  # Package exports: LLM, SamplingParams
├── llm.py                       # User-facing LLM class
├── config.py                    # Engine configuration dataclass
├── sampling_params.py           # Generation parameters
│
├── engine/                      # Core inference engine
│   ├── llm_engine.py            # Main orchestrator
│   ├── scheduler.py             # Batch scheduling + remasking strategies
│   ├── model_runner.py          # Model execution + CUDA graphs
│   ├── block_manager.py         # Paged KV cache block allocator
│   ├── sequence.py              # Sequence lifecycle management
│   └── distributed_manager.py   # TP/DP process groups
│
├── layers/                      # Neural network primitives
│   ├── attention.py             # BlockAttention + store_kvcache kernel
│   ├── linear.py                # TP-parallel linear layers
│   ├── layernorm.py             # RMSNorm (compiled)
│   ├── activation.py            # SiluAndMul (Liger fused)
│   ├── rotary_embedding.py      # RoPE with cached factory
│   ├── embed_head.py            # VocabParallelEmbedding, ParallelLMHead
│   └── sampler.py               # Temperature + top-k/top-p sampling
│
├── models/                      # Model architectures
│   ├── sdar.py                  # SDARForCausalLM (dense)
│   └── sdar_moe.py              # SDARMoeForCausalLM (sparse MoE)
│
├── kernels/                     # Triton GPU kernels
│   └── triton/
│       ├── fused_moe.py         # Fused MoE expert computation
│       └── attention/
│           ├── block_prefill_attention_v2.py   # Staircase attention
│           ├── fused_page_attention_v3.py      # Paged KV attention
│           └── fused_page_attention_v6.py      # Optimized variant
│
├── utils/                       # Utilities
│   ├── context.py               # Global PREFILL/DENOISE context
│   ├── loader.py                # Weight loading (safetensors + HF)
│   └── statics.py               # KV cache memory estimation
│
└── inference/                   # Pure PyTorch inference (MPS/CPU)
    ├── denoiser.py              # Block denoiser
    ├── generator.py             # Block diffusion generator
    ├── sampler.py               # Gumbel-max sampling
    ├── schedules.py             # Noise schedules
    ├── sdar_inference.py        # CLI inference
    └── sdar_wrapper.py          # Model wrapper
```

## Key Features

| Feature | Description |
|---------|-------------|
| Paged KV Cache | Block-level allocation with prefix caching (xxhash) |
| Staircase Attention | Block-local Triton kernel for prefill |
| FlashAttention-2 | Paged KV cache attention for denoise |
| CUDA Graphs | Captured/replayed for denoise batch sizes |
| Tensor Parallelism | Multi-GPU with TP/DP process groups |
| 5 Remasking Strategies | sequential, low_confidence_static/dynamic, entropy_bounded, random |
| Fused MoE | Triton kernel for sparse expert models |
| Hot Reload | Swap model weights without restarting engine |

## Block Diffusion Overview

Unlike autoregressive models that generate one token at a time, block diffusion generates **blocks of tokens** via iterative denoising:

```
Step 0: [prompt tokens...] [MASK MASK MASK MASK MASK MASK MASK MASK]
Step 1: [prompt tokens...] [The   MASK MASK MASK MASK MASK MASK MASK]
Step 2: [prompt tokens...] [The   quick MASK fox  MASK MASK MASK MASK]
  ...
Step T: [prompt tokens...] [The   quick brown fox  jumps over the  fence]
                            └──────── block committed ────────────────┘

Next block starts → [committed context...] [MASK MASK MASK MASK ...]
```

**Remasking strategies** control which masked positions get unmasked at each step:
- **sequential** — left-to-right unmasking
- **low_confidence_static** — unmask highest-confidence predictions
- **low_confidence_dynamic** — confidence threshold with sequential fallback
- **entropy_bounded** — unmask by cumulative entropy budget
- **random** — randomly select masked positions

## Sequence Lifecycle

```
WAITING ──► PREFILLING ──► DENOISING ◄──► SAVING ──► FINISHED
              │                │              │
              │  encode full   │  iterative   │  commit block,
              │  context       │  block       │  start new block
              │                │  refinement  │  or finish
```

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
