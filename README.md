# VDLLM - Block Diffusion Inference Engine

Pure PyTorch implementation of SDAR (Synergy of Diffusion and AutoRegression) block diffusion inference, optimized for Metal (Apple Silicon) and Triton (NVIDIA).

**Model:** [JetLM/SDAR-1.7B-Chat](https://huggingface.co/JetLM/SDAR-1.7B-Chat)
**Based on:** [JetEngine SDAR](https://github.com/Jet-Astra/SDAR) and [vllm-metal](https://github.com/vllm-project/vllm-metal)

================================================================================
                                QUICK START
================================================================================

```bash
# Download SDAR model
huggingface-cli download JetLM/SDAR-1.7B-Chat --local-dir /tmp/sdar-1.7b-chat

# Run inference
python -m vdllm.inference.sdar_inference \
    --model_dir /tmp/sdar-1.7b-chat \
    --prompt "Hello, how are you?" \
    --device mps \
    --gen_length 128
```

================================================================================
                              PROJECT STRUCTURE
================================================================================

```
vdllm/
├── __init__.py                 # Main package exports
│
├── engine/                     # Device backends
│   ├── __init__.py             # Unified backend selection
│   ├── device.py               # Device detection (Triton > Metal > CPU)
│   ├── attention.py            # Block diffusion attention masks
│   ├── diffusion/             # Diffusion components
│   │   ├── noise_schedules.py # Cosine, Linear schedules
│   │   ├── transitions.py     # DDPM transitions
│   │   └── sampler.py        # Semi-AR sampler
│   ├── metal/                 # Metal backend (Apple Silicon)
│   │   ├── backend.py        # MetalBackend (PyTorch MPS)
│   │   ├── attention.py      # Block diffusion masks
│   │   ├── kvcache.py       # BlockKVCache, PagedKVCache
│   │   └── memory.py        # MPS memory management
│   └── triton/               # Triton backend (NVIDIA)
│       ├── backend.py        # TritonBackend
│       └── kernels/          # Triton kernels
│           └── block_attention.py
│
├── layers/                    # Model layers (pure PyTorch)
│   ├── __init__.py
│   ├── attention.py          # SDARAttention (GQA + RoPE)
│   ├── mlp.py               # SiLU/SWiGLU FFN
│   ├── rmsnorm.py           # RMSNorm
│   ├── rotary.py            # RoPE embeddings
│   └── embedding.py         # Token embeddings
│
├── models/                    # Model implementations
│   ├── __init__.py
│   ├── config.py            # SDARConfig
│   └── sdar.py              # SDARModel, SDARForCausalLM
│
├── inference/                 # Inference pipeline
│   ├── __init__.py
│   ├── generator.py          # Block diffusion generator
│   ├── denoiser.py         # Block denoiser
│   ├── sampler.py          # Gumbel-max sampling
│   ├── schedules.py         # Noise schedules
│   ├── sdar_inference.py   # CLI and core inference
│   └── sdar_wrapper.py     # Model wrapper
│
└── utils/                    # Utilities
    ├── __init__.py
    ├── context.py           # Global context management (PREFILL/DENOISE)
    ├── loader.py           # Model weight loading (safetensors, HF)
    └── statics.py          # KV cache size estimation

tests/
└── test_mps_benchmark.py   # TPS benchmark tests
```

================================================================================
                              DEVICE SUPPORT
================================================================================

| Device  | Backend | Notes |
|---------|---------|-------|
| Apple Silicon | Metal (MPS) | Primary - PyTorch MPS backend |
| NVIDIA GPU | Triton | CUDA with Triton kernels |
| CPU | CPU | Fallback |

Auto-detection priority: **Triton > Metal > CPU**

================================================================================
                              WHAT IS BLOCK DIFFUSION?
================================================================================

Unlike standard Autoregressive (AR) models that generate tokens one-by-one,
block diffusion models generate **entire blocks of tokens** in parallel using
a denoising process.

**Key Concepts:**

1. **Block Diffusion**
   - Sequence is divided into fixed-size blocks
   - Each block is denoised over multiple steps
   - Tokens within a block attend bidirectionally
   - Blocks are processed left-to-right

2. **Mask Token (151669)**
   - SDAR uses a special mask token that gets iteratively denoised
   - At each step, high-confidence predictions are "unmasked"

3. **Semi-AR Sampling**
   - Iterative denoising with remasking strategies
   - low_confidence_dynamic (recommended)
   - low_confidence_static
   - sequential

================================================================================
                              USAGE EXAMPLES
================================================================================

### Example 1: Using the Engine Backend

```python
from vdllm.engine import get_backend, build_block_diffusion_mask

backend = get_backend(num_heads=32, head_dim=128)
mask = build_block_diffusion_mask(seq_len=1024, block_size=4)
output = backend.forward(q, k, v, mask=mask)
```

### Example 2: Using SDAR Model

```python
from vdllm.models import SDARForCausalLM, SDARConfig

config = SDARConfig()
model = SDARForCausalLM(config)
logits = model(input_ids, positions)
```

### Example 3: Block Diffusion Generation

```python
from vdllm.inference import BlockDiffusionGenerator

generator = BlockDiffusionGenerator(model, config)
output_ids = generator.generate(prompt_tokens, gen_length=128)
```

================================================================================
                              TESTS & BENCHMARKS
================================================================================

Run TPS benchmarks:

```bash
python -m pytest tests/test_mps_benchmark.py -v -s
```

**Benchmark Results (MPS):**
- Attention Forward: ~290K tokens/sec (seq_len=256)
- Large Attention: ~110K tokens/sec (seq_len=512)
- MPS vs CPU Speedup: 1.39x faster
- End-to-End TPS: ~73K tokens/sec

================================================================================
                              CITATION
================================================================================

```bibtex
@misc{sdar2025,
    title={SDAR: Synergy of Diffusion and AutoRegression},
    author={JetLM Team},
    year={2025},
    url={https://github.com/JetAstra/SDAR}
}
```
