# vdllm

Inference engine for block diffusion language models. Supports CUDA and Apple Silicon (MLX).

## Quick Start

```python
from vdllm import LLM, SamplingParams

llm = LLM("path/to/model")

params = SamplingParams(
    max_tokens=256,
    temperature=0.7,
    block_length=4,
    denoising_steps=4,
    remasking_strategy="low_confidence_dynamic",
)

outputs = llm.generate(["Hello, how are you?"], params)
print(outputs[0]["text"])
```

Backend is auto-detected. Force one with `LLM("model", backend="cuda")` or `backend="mlx"`.

## CLI

```bash
python example.py "What is the capital of France?" --chat
python example.py "Write hello world in Python" --chat --gen-length 512
python example.py "Hello" --backend mlx --temperature 0.7 --remasking-strategy sequential
```

Run `python example.py --help` for all options.

## Install

```bash
pip install -e .
```

**CUDA** requires PyTorch with CUDA, FlashInfer, and Triton.
**MLX** requires `mlx` (Apple Silicon only).

## How It Works

Block diffusion generates text in blocks of tokens via iterative denoising, not one token at a time:

```
Step 0: [prompt...] [MASK MASK MASK MASK]
Step 1: [prompt...] [The  MASK MASK MASK]   <- highest confidence unmasked
Step 2: [prompt...] [The  quick MASK fox ]
Step 3: [prompt...] [The  quick brown fox]   <- block committed, next block starts
```

Each block goes through `denoising_steps` iterations. A remasking strategy selects which tokens to unmask at each step based on model confidence.

## Backends

| Backend | Hardware | Features |
|---------|----------|----------|
| **CUDA** | NVIDIA GPU | Tensor parallelism, paged KV cache, CUDA graphs, Triton kernels, batched scheduling |
| **MLX** | Apple Silicon | Metal GPU, concat KV cache, sequential processing |

## Sampling Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_tokens` | 64 | Max tokens to generate |
| `block_length` | 4 | Tokens per denoising block |
| `denoising_steps` | 4 | Iterations per block |
| `temperature` | 1.0 | Sampling temperature |
| `topk` | 0 | Top-K filtering (0 = off) |
| `topp` | 1.0 | Nucleus sampling (1.0 = off) |
| `remasking_strategy` | `low_confidence_static` | Token selection strategy |
| `dynamic_threshold` | 0.9 | Confidence threshold (for `low_confidence_dynamic`) |
| `eb_threshold` | 0.35 | Entropy budget (for `entropy_bounded`) |

**Remasking strategies:** `sequential`, `low_confidence_static`, `low_confidence_dynamic`, `entropy_bounded`, `random`

## Project Structure

```
vdllm/
  __init__.py              Lazy exports: LLM, SamplingParams, Config
  llm.py                   Unified LLM class (dispatches to CUDA or MLX engine)
  config.py                Backend-agnostic config
  sampling_params.py       Generation parameters
  generation.py            Block diffusion generation loop (MLX)
  sampling.py              Sampling utilities (MLX)

  engine/
    llm_engine.py          CUDA engine (batched scheduler, CUDA graphs)
    mlx_engine.py          MLX engine (wraps generation.py)
    scheduler.py           Batch scheduling + remasking (CUDA)
    model_runner.py        Model execution + KV cache (CUDA)
    block_manager.py       Paged KV cache allocator (CUDA)
    sequence.py            Per-request state tracking

  models/
    mlx_sdar.py            SDAR model in MLX
    sdar.py                SDAR model in PyTorch (TP-parallel)
    sdar_moe.py            SDAR-MoE sparse expert model

  backends/                Attention backend protocol + implementations
  layers/                  TP-parallel layers, RMSNorm, RoPE, BlockAttention
  kernels/triton/          Staircase attention, paged attention, fused MoE
  kernels/metal/           Metal kernel stubs
  utils/                   Hardware detection, weight loading, memory estimation
```

## Tests

```bash
python -m pytest tests/test_unified_llm.py -v          # Unified API tests (44 tests)
python tests/test_mlx_sdar_forward.py                   # MLX vs PyTorch forward pass
python -m pytest tests/unit/ -v                          # Backend unit tests
```

## Supported Models

- [JetLM/SDAR-1.7B-Chat](https://huggingface.co/JetLM/SDAR-1.7B-Chat)

## References
- The codebase is based on [JetEngine](https://github.com/Labman42/JetEngine) by Yihan Bian et al.
- Speical thanks to [vllm-metal](https://github.com/vllm-project/vllm-metal) and [mlx-lm](https://github.com/ml-explore/mlx-lm) for insights on Apple Silicon optimization.

## Citation

```bibtex
@misc{jetengine2025,
    title={vdllm: Block Diffusion Language Models Inference Engine},
    author={Bach Vu},
    year={2026},
    url={https://github.com/bachvudinh/DiffusionLM}
}
```
