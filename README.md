# Block Diffusion Language Model — Clean Inference

Minimal inference implementation of BD3-LMs (ICLR 2025 Oral).

Reference: [Block Diffusion (Arriola et al.)](https://arxiv.org/abs/2503.09573)

## Project Structure

```
inference/
├── __init__.py       # Exports: generate, BlockDenoiser, GumbelSampler, etc.
├── schedules.py       # LinearSchedule, CosineSchedule — noise schedules
├── sampler.py         # GumbelSampler — discrete token sampling
├── mask.py           # StaircaseMask — block diffusion attention
├── unmask.py         # unmask_top_k — confidence-based unmasking
├── denoiser.py       # BlockDenoiser — per-block denoising orchestration
└── generator.py      # generate() — full generation pipeline

tests/
└── test_block_diffusion_inference.py  # 19 passing tests
```

## Quick Start

```python
from inference import generate

# Your model needs these methods:
#   __call__(x, pos_offset) → (logits, _)
#   set_cache_mode(enabled)
#   reset_kv_cache()
#   parameters()

result = generate(
    model=your_model,
    encode_fn=lambda s: tokenize(s),      # str → list[int]
    decode_fn=lambda ids: detokenize(ids), # list[int] → str
    prompt="Hello",
    max_new_tokens=100,
    block_size=8,
    denoise_steps=10,
    temperature=0.7,
)
```

## Architecture

### Generation Flow

```
1. KV Cache Warmup
   └── Process full prompt blocks with cache_mode=True

2. Block-by-Block Generation (AR, left-to-right)
   └── For each block:
       a. Init: [prompt_rem | MASK MASK ...]
       b. Denoise loop (denoise_steps iterations):
          ├── model(x) → logits
          ├── Gumbel sample → token predictions
          ├── Compute confidence per position
          └── Unmask top-k most confident
       c. Cache block KV
       d. Extract new tokens
```

### Staircase Attention Mask

Input is concatenated: `[x_t || x_0]` (2L tokens)

| Query | Key | Rule |
|-------|-----|------|
| Noisy block i | Noisy block i | ✅ Attend (bidirectional) |
| Noisy block i | Clean block j | ✅ Only if j < i (block-causal, no label leakage) |
| Clean block i | Clean block j | ✅ Only if j ≤ i (block-causal) |

### Noise Schedule

- **Linear**: `mask_prob = t`, `elbo_weight = 1/t`
- **Cosine**: `alpha = cos²(πt/2)`, `mask_prob = 1 - alpha`

## Comparison with kuleshov-group/bd3lms

| | Ours | Reference |
|--|-------|-----------|
| Staircase mask | ✅ Correct | Has indexing bug in clean half |
| Inference | Confidence unmasking | DDPM categorical |
| Schedules | Linear, Cosine | + LogLinear, Square, Log |

## Run Tests

```bash
python -m pytest tests/ -v
# 19 passed
```
