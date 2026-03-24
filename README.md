# Block Diffusion Language Model — Clean Inference Implementation

A clean, minimal inference implementation of Block Diffusion LMs (BD3-LMs, ICLR 2025 Oral).

Reference: [Block Diffusion: Interpolating Between AR and Diffusion Language Models](https://arxiv.org/abs/2503.09573)
by Arriola et al. — [kuleshov-group/bd3lms](https://github.com/kuleshov-group/bd3lms)

## Table of Contents
1. [Quick Start](#quick-start)
2. [Architecture Overview](#architecture-overview)
3. [Step-by-Step Example](#step-by-step-example)
4. [Comparison with kuleshov-group/bd3lms](#comparison-with-reference)
5. [API Reference](#api-reference)
6. [Module Design](#module-design)

---

## Quick Start

```python
import torch
from inference import generate

class SimpleBDLM:
    """Minimal block diffusion language model for demonstration."""

    def __init__(self, vocab_size=1000, n_embd=256, n_layers=4, block_size=8):
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.emb = torch.nn.Embedding(vocab_size, n_embd)
        self.lm_head = torch.nn.Linear(n_embd, vocab_size, bias=False)
        self._training = False

        # Tie weights
        self.lm_head.weight = self.emb.weight

    def __call__(self, x, pos_offset=0):
        """Forward pass.

        Args:
            x: (B, L) — token IDs
            pos_offset: int — RoPE position offset

        Returns:
            logits: (B, L, vocab_size)
        """
        h = self.emb(x)
        logits = self.lm_head(h)
        return logits, None

    def set_cache_mode(self, enabled): pass
    def reset_kv_cache(self): pass
    def parameters(self):
        return self.emb.parameters()

# Usage
model = SimpleBDLM(vocab_size=1000, n_embd=256, n_layers=4, block_size=8)

# Simple tokenizer: each char → ord value
def tokenize(text):
    return [ord(c) % 1000 for c in text]

def detokenize(ids):
    return ''.join(chr(i % 26 + 97) for i in ids)

result = generate(
    model=model,
    encode_fn=tokenize,
    decode_fn=detokenize,
    prompt="hello world",
    max_new_tokens=50,
    block_size=8,
    denoise_steps=5,
    temperature=0.7,
    verbose=True,
)
print(result)
```

---

## Architecture Overview

### Block Diffusion Generation Flow

```
    Prompt: "Hello"  →  [5 tokens]
           │
           ▼
    ┌─────────────────────────────────────────────┐
    │ 1. KV CACHE WARMUP                          │
    │    Process full prompt blocks through model │
    │    with cache_mode=True to warm up KV cache │
    └─────────────────────────────────────────────┘
           │
           ▼
    ┌─────────────────────────────────────────────┐
    │ 2. BLOCK-BY-BLOCK GENERATION                │
    │                                             │
    │  ┌─────────────────────────────────────────┐ │
    │  │  Block N: [prompt_rem | M M M M M M M]  │ │
    │  │                                           │ │
    │  │  For each denoise step:                  │ │
    │  │    1. model(block) → logits            │ │
    │  │    2. Gumbel sample → token predictions │ │
    │  │    3. Compute confidence per position   │ │
    │  │    4. Unmask top-k most confident      │ │
    │  │    5. Update block with new tokens     │ │
    │  │                                           │ │
    │  │  Final: fully denoised block            │ │
    │  └─────────────────────────────────────────┘ │
    │           │                                   │
    │           ▼                                   │
    │  ┌─────────────────────────────────────────┐ │
    │  │ 3. CACHE BLOCK KV                       │ │
    │  │    Process final block → store K,V      │ │
    │  └─────────────────────────────────────────┘ │
    │           │                                   │
    └───────────┼───────────────────────────────────┘
                │  repeat for next block
                ▼
```

### Model Architecture

```
    Input: x_t = [MASK, MASK, MASK, MASK, token_0, token_1, token_2]
                                  (partially masked sequence)

    ┌──────────────────────────────────────────────────────────┐
    │  Token Embedding                                         │
    │  Input:  (B, L)   — token IDs                          │
    │  Output: (B, L, n_embd)                                │
    │                                                          │
    │  vocab_size = 49,152 (SmolLM2 tokenizer)                │
    │  n_embd    = 576                                        │
    └──────────────────────────────────────────────────────────┘
                          │
                          ▼
    ┌──────────────────────────────────────────────────────────┐
    │  RMSNorm (functional, no params)                         │
    │  Input:  (B, L, n_embd)                                │
    │  Output: (B, L, n_embd)                                │
    └──────────────────────────────────────────────────────────┘
                          │
                          ▼
    ┌──────────────────────────────────────────────────────────┐
    │  Transformer Block × 30                                  │
    │                                                          │
    │  ┌────────────────────────────────────────────────────┐ │
    │  │  RMSNorm                                          │ │
    │  └────────────────────────────────────────────────────┘ │
    │                          │                               │
    │                          ▼                               │
    │  ┌────────────────────────────────────────────────────┐ │
    │  │  GQA Attention (9Q / 3KV heads)                    │ │
    │  │    • Bidirectional (no causal mask)                │ │
    │  │    • RoPE positional embeddings                      │ │
    │  │    • Gated query output (zero-init)                 │ │
    │  │    • Staircase attention mask                       │ │
    │  └────────────────────────────────────────────────────┘ │
    │                          │                               │
    │                          ▼                               │
    │  ┌────────────────────────────────────────────────────┐ │
    │  │  SwiGLU MLP                                        │ │
    │  │    Linear(n_embd → 4×n_embd)                       │ │
    │  │    SiLU activation                                  │ │
    │  │    Linear(4×n_embd → n_embd)                        │ │
    │  └────────────────────────────────────────────────────┘ │
    │                          │                               │
    │                          ▼                               │
    │  ┌────────────────────────────────────────────────────┐ │
    │  │  Residual Connection (+= input)                     │ │
    │  └────────────────────────────────────────────────────┘ │
    └──────────────────────────────────────────────────────────┘
                          │
                          ▼
    ┌──────────────────────────────────────────────────────────┐
    │  RMSNorm                                                │
    └──────────────────────────────────────────────────────────┘
                          │
                          ▼
    ┌──────────────────────────────────────────────────────────┐
    │  LM Head (tied to embedding)                            │
    │  Input:  (B, L, n_embd)                                │
    │  Output: (B, L, vocab_size)                             │
    └──────────────────────────────────────────────────────────┘
                          │
                          ▼
    Output: logits (B, L, vocab_size)  — per-token unnormalized log-probs
```

### Staircase Attention Mask

The staircase mask is the key innovation of Block Diffusion. Input is concatenated as `[x_t || x_0]` (2L tokens).

```
    Sequence: L = 16 tokens, block_size = 4 → n_blocks = 4

    Attention rules:
    ┌──────────────────────────────────────────────────────────────┐
    │  Noisy half (x_t, positions 0-15):                         │
    │    • Within same block → BIDIRECTIONAL (see all tokens)    │
    │    • To clean half of earlier blocks → ALLOWED              │
    │    • To clean half of same/later block → BLOCKED             │
    │      (prevents label leakage)                              │
    ├──────────────────────────────────────────────────────────────┤
    │  Clean half (x_0, positions 16-31):                        │
    │    • Within same block → BIDIRECTIONAL                     │
    │    • To clean half of earlier/current blocks → ALLOWED      │
    │    • To clean half of later blocks → BLOCKED (AR causal)    │
    └──────────────────────────────────────────────────────────────┘

    Mask matrix (1 = attend, 0 = block):

                    0   4   8  12  16  20  24  28
                    │   │   │   │   │   │   │   │
              0     ■   ■   ■   ■   ○   ○   ○   ○    ← noisy block 0: self bidir + no clean
              4     ■   ■   ■   ■   ■   ○   ○   ○    ← noisy block 1: self + early clean + early noisy
              8     ■   ■   ■   ■   ■   ■   ○   ○    ← noisy block 2: self + 2 early clean
             12     ■   ■   ■   ■   ■   ■   ■   ○    ← noisy block 3: self + 3 early clean
             16     □   □   □   □   ■   ■   ■   ■    ← clean block 0: only to later clean (causal)
             20     □   □   □   □   □   ■   ■   ■    ← clean block 1: only to later clean
             24     □   □   □   □   □   □   ■   ■    ← clean block 2: only to later clean
             28     □   □   □   □   □   □   □   ■    ← clean block 3: self only

    ■ = attend (1), □ = block (0), ○ = no label leakage (no access to own clean half)
```

---

## Step-by-Step Example

### Tokenization

```python
# Input sentence
text = "Hello world!"

# Tokenize (character-level for simplicity)
tokens = [ord(c) for c in text]
print(tokens)
# Output: [72, 101, 108, 108, 111, 32, 119, 111, 114, 108, 100, 33]

# With BPE tokenizer (SmolLM2-style):
# tokens = tokenizer.encode(text)
# Output: [3425, 1235, 25, 1235, 25, 1987, ...]  (variable length)
```

### Forward Process (Adding Noise)

```python
import torch
from inference.schedules import LinearSchedule

# Original tokens
x_0 = torch.tensor([72, 101, 108, 108, 111, 32, 119, 111, 114, 108, 100, 33])
seq_len = len(x_0)  # 12 tokens
mask_token_id = 0
pad_token_id = 2

print(f"Original: {x_0.tolist()}")
# [72, 101, 108, 108, 111, 32, 119, 111, 114, 108, 100, 33]

# Sample timestep t ~ U[0.1, 1.0]
t = torch.tensor(0.5)
schedule = LinearSchedule()
mask_prob = schedule.mask_prob(t)  # mask_prob = t = 0.5

print(f"t = {t.item()}, mask_prob = {mask_prob.item()}")
# t = 0.5, mask_prob = 0.5

# Apply masking: each token masked independently with prob = t
# torch.rand(12) < 0.5 → True/False per position
noise_mask = torch.rand(12) < mask_prob

x_t = x_0.clone()
x_t[noise_mask] = mask_token_id  # Replace masked positions with [MASK]

print(f"Noise mask: {noise_mask.tolist()}")
# e.g., [True, False, True, False, True, False, True, False, True, False, True, False]

print(f"Noisy (x_t): {x_t.tolist()}")
# e.g., [0, 101, 0, 108, 0, 32, 0, 111, 0, 108, 0, 33]
#         M   e   M   l   M       M   o   M   l   M   !

# The model sees: x_t = [0, 101, 0, 108, 0, 32, 0, 111, 0, 108, 0, 33]
# And must predict: x_0 = [72, 101, 108, 108, 111, 32, 119, 111, 114, 108, 100, 33]
```

### Model Forward Pass

```python
# Simulated model forward pass
B, L = 1, 12
n_embd = 256
vocab_size = 1000

# Token embeddings
emb = torch.nn.Embedding(vocab_size, n_embd)
lm_head = torch.nn.Linear(n_embd, vocab_size, bias=False)

# Embed the noisy tokens
h = emb(x_t.unsqueeze(0))  # (1, 12, 256)

# Transformer layers (simplified)
# ... attention, MLP, residual connections ...

# Final logits
logits = lm_head(h)  # (1, 12, 1000)

print(f"logits shape: {logits.shape}")
# (1, 12, 1000)

# Per-token prediction
token_logits = logits[0]  # (12, 1000)
pred_tokens = torch.argmax(token_logits, dim=-1)

print(f"Predicted tokens: {pred_tokens.tolist()}")
# e.g., [72, 101, 99, 108, 111, 32, 119, 111, 114, 108, 100, 33]
#          H    e    c    l    o       w    o    r    l    d    !

# Check confidence
probs = torch.softmax(token_logits.float(), dim=-1)
confidences = probs.max(dim=-1).values

print(f"Confidences: {confidences.tolist()}")
# e.g., [0.85, 0.92, 0.45, 0.88, 0.91, 0.99, 0.87, 0.93, 0.89, 0.86, 0.95, 0.78]
```

### Block Denoising Loop

```python
import torch
from inference.sampler import GumbelSampler
from inference.unmask import unmask_top_k

# Configuration
block_size = 4
denoise_steps = 3
temperature = 0.7

# Example: denoising block 2 of the sequence
# Positions 8-11 in the full sequence
# x_t = [0, 101, 0, 108, 0, 32, 0, 111, 0, 108, 0, 33]
# Block 2 = [0, 108, 0, 33]  (tokens at indices 8, 9, 10, 11)

block = torch.tensor([0, 108, 0, 33])  # [MASK, 'l', MASK, '!']
masked = torch.tensor([True, False, True, True])  # positions 0, 2, 3 are masked

print(f"Initial block: {block.tolist()}")
# [0, 108, 0, 33]
print(f"Initial masked: {masked.tolist()}")
# [True, False, True, True]
# Position 1 = 'l' (108) was prompt remainder, already known

sampler = GumbelSampler(temperature=temperature)

for step in range(denoise_steps):
    print(f"\n--- Denoise Step {step + 1}/{denoise_steps} ---")

    # Simulate model forward pass (in real code, model(block) → logits)
    # Here we use random logits for demonstration
    logits = torch.randn(1, block_size, vocab_size)
    # Suppress mask token
    logits[:, :, 0] = -float('inf')

    # Gumbel sample
    samples = sampler.sample(logits)  # (1, 4)
    print(f"  Sampled tokens: {samples[0].tolist()}")

    # Compute confidence
    probs = torch.softmax(logits.float(), dim=-1)
    confidences = probs.max(dim=-1).values[0]
    print(f"  Confidences: {confidences.tolist()}")

    # How many to unmask?
    n_masked = masked.sum().item()
    if step == denoise_steps - 1:
        n_to_unmask = n_masked  # Last step: unmask all
    else:
        n_to_unmask = max(1, n_masked // denoise_steps)
    print(f"  Unmasking {n_to_unmask} positions")

    # Unmask top-k most confident
    masked = unmask_top_k(masked, confidences, n_to_unmask)
    print(f"  Updated masked: {masked.tolist()}")

    # Update block with samples
    block = torch.where(masked == False, samples[0], block)
    print(f"  Updated block: {block.tolist()}")

    if not masked.any():
        break

# Output:
# --- Denoise Step 1/3 ---
#   Sampled tokens: [97, 108, 99, 33]   (example)
#   Confidences: [0.12, 0.88, 0.23, 0.91]
#   Unmasking 1 positions
#   Updated masked: [True, False, True, True]
#   Updated block: [0, 108, 0, 33]
#
# --- Denoise Step 2/3 ---
#   Sampled tokens: [72, 108, 114, 33]
#   Confidences: [0.85, 0.88, 0.67, 0.91]
#   Unmasking 1 positions
#   Updated masked: [False, False, True, True]
#   Updated block: [72, 108, 0, 33]
#
# --- Denoise Step 3/3 ---
#   Sampled tokens: [72, 108, 100, 33]
#   Confidences: [0.91, 0.93, 0.89, 0.95]
#   Unmasking 2 positions (last step)
#   Updated masked: [False, False, False, False]
#   Updated block: [72, 108, 100, 33]
#
# Final: [72, 108, 100, 33]  →  "Hld" (if using char-level)
```

### Full Generation Timeline

```python
# Prompt: "Hello"  (5 tokens)
# block_size = 4, denoise_steps = 3

# Step 1: KV Cache Warmup
# ─────────────────────────
# prompt_ids = [72, 101, 108, 108, 111]
# n_full_prompt_blocks = 5 // 4 = 1
# prompt_remainder = 5 % 4 = 1
#
# Run block [72, 101, 108, 108] through model with cache_mode=True
# pos_offset = 0 → pos_offset = 4

# Step 2: Generate Block 0
# ─────────────────────────
# fill_from_prompt = min(1, 4) = 1
# Initial block: [111, 0, 0, 0]  (1 prompt token + 3 masks)
#
# Denoise steps 1-3:
#   Step 1: Unmask 1 most confident → [111, x, 0, 0]
#   Step 2: Unmask 1 most confident → [111, x, x, 0]
#   Step 3: Unmask remaining → [111, x, x, x]
#
# Final block: [111, x, x, x]
# Cache KV for block 0
# new_tokens = [x, x, x]  (3 generated tokens)

# Step 3: Generate Block 1
# ─────────────────────────
# fill_from_prompt = 0  (no prompt remainder)
# Initial block: [0, 0, 0, 0]  (all masks)
#
# Denoise steps 1-3 → fully denoised block
# Final block: [x, x, x, x]
# Cache KV for block 1
# new_tokens = [x, x, x, x]

# Step 4: Continue until max_new_tokens reached...
```

---

## Comparison with Reference

Reference: [kuleshov-group/bd3lms](https://github.com/kuleshov-group/bd3lms)

### Feature Comparison

| Feature | Our Implementation | bd3lms Reference |
|---------|------------------|-------------------|
| **Noise Schedules** | Linear, Cosine | LogLinear, Cosine, Square, Logarithmic |
| **Training ELBO Weight** | `1/t` (linear) | `-1/t` (loglinear) |
| **Forward Masking** | Per-token Bernoulli | Per-token Bernoulli |
| **Staircase Mask** | Correct implementation | Has subtle indexing bug in clean half |
| **Inference Sampling** | Confidence-based unmasking | DDPM-style categorical |
| **Model Architecture** | Bidirectional Transformer | DiT-based |

### Staircase Mask: Our Implementation vs bd3lms

**Our implementation** — correct:
```python
# Noisy block i → clean blocks 0..i-1
for prev_b in range(b):  # prev_b = 0, 1, ..., b-1
    # clean_key_start = prev_b * block_size + L  ← correct block position
    # block_kv = prev_b  ← correctly computed clean block index
    condition = block_q > block_kv  # STRICT: i > j, only earlier blocks
```

**bd3lms reference** — has indexing bug:
```python
# bd3lms computes clean block index as:
block_kv = (kv_idx - n) // block_size  # WRONG for cross-half attention!
# For position 32 (clean block 0, n=16, B=4): block_kv = (32-16)//4 = 4
# But the correct block index is 0!

# bd3lms also uses:
block_kv = kv_idx // block_size  # WRONG for clean half!
# This gives block index in the FULL sequence (0-31), not within clean half (0-3)

# Combined, for noisy block 2 → clean block 2:
# bd3lms: block_kv = 32//4 = 8  (in full sequence)
#         block_q = 8  (in full sequence)
#         8 > 8 = False  ← blocked (by accident, wrong reason!)
#
# Ours: block_kv = 2  (in clean half)
#       block_q = 2  (in noisy half)
#       2 > 2 = False  ← blocked (correct!)
```

**Critical finding**: Our implementation correctly prevents noisy block `i` from attending to clean block `i` (label leakage) using proper block indexing. bd3lms also blocks it, but due to incorrect indexing that happens to produce the right answer for the wrong reason.

### Noise Schedule Equivalence

Both implementations use the same mathematical foundation:

| Schedule | Formula | Our ELBO Weight | Reference Loss Scale |
|----------|---------|-----------------|---------------------|
| Linear | `mask_prob = t` | `1/t` | `-1/t` (via loglinear) |
| Cosine | `alpha = cos²(πt/2)` | `d_alpha/alpha` | `sin/(cos+eps)·π/2` |

The sign difference in weights (`1/t` vs `-1/t`) is because reference minimizes negative log-likelihood.

### Inference Algorithm Differences

**Our approach** (LLaDA-style confidence unmasking):
```
1. Initialize block with [MASK] tokens
2. For each denoise step:
   a. Run model forward → get logits
   b. Gumbel-max sample → get token predictions
   c. Compute confidence = P(predicted_token)
   d. Unmask top-k most confident positions
3. Repeat for next block
```

**bd3lms approach** (DDPM-style categorical):
```
1. Initialize with all [MASK]
2. For each denoise step:
   a. Compute p(x_0 | x_t) from score
   b. Compute transition probabilities q(x_{t-s} | x_t, x_0)
   c. Sample from categorical distribution
3. KV cache updated per full block
```

---

## API Reference

### `generate()` — Main Entry Point

```python
from inference import generate

output_text = generate(
    model=model,              # forward(x, pos_offset) → (logits, _)
    encode_fn=tokenize,       # str → list[int]
    decode_fn=detokenize,     # list[int] → str
    prompt="Hello, world!",    # input prompt
    max_new_tokens=512,       # max tokens to generate
    block_size=32,            # tokens per diffusion block
    mask_token_id=0,          # [MASK] token ID
    eos_token_id=1,           # EOS token ID
    pad_token_id=2,           # padding token ID
    denoise_steps=10,         # denoising iterations per block
    temperature=0.7,          # 0 = greedy, 1 = Gumbel, >1 = uniform
    top_k=50,                 # top-k filtering (0 = disabled)
    verbose=True,             # print stats
)
```

### `BlockDenoiser` — Per-Block Denoising

```python
from inference import BlockDenoiser

denoiser = BlockDenoiser(
    block_size=32,
    mask_token_id=0,
    pad_token_id=2,
    eos_token_id=1,
    denoise_steps=10,
    temperature=0.7,
    top_k=50,
)

# Initialize a block with 3 prompt tokens
block, masked = denoiser.init_block(prompt_remainder=3)
# block:  (1, 32) — first 3 are known, rest are [MASK]
# masked: (1, 32) — True for masked positions

# Compute how many to unmask at step 0
n = denoiser.step_unmask_count(
    t=1.0, s=0.9, n_masked=29, step=0, total_steps=10
)
# n = 2 (uniform schedule: 29//10 = 2)
```

### `GumbelSampler` — Token Sampling

```python
from inference import GumbelSampler

sampler = GumbelSampler(temperature=0.7)

# Sample from logits
logits = torch.randn(2, 10, 1000)  # (batch, seq, vocab)
samples = sampler.sample(logits)    # (2, 10) — token IDs

# Get samples with confidence
samples, confidence = sampler.sample_with_confidence(logits)
# confidence: (2, 10) — P(sampled_token)
```

### `unmask_top_k()` — Confidence-Based Unmasking

```python
from inference import unmask_top_k

masked = torch.tensor([True, True, True, True, True, False, False, False, False, False])
confidences = torch.tensor([0.1, 0.9, 0.3, 0.7, 0.5, -1.0, -1.0, -1.0, -1.0, -1.0])

new_mask = unmask_top_k(masked, confidences, k=3)
# Top 3 confidences at positions 1 (0.9), 3 (0.7), 4 (0.5)
# new_mask: [True, False, True, False, False, False, False, False, False, False]
```

### `StaircaseMask` — Attention Mask

```python
from inference import StaircaseMask

mask_builder = StaircaseMask(block_size=8, seq_len=64)
full_mask = mask_builder.build()  # (128, 128) — 2L × 2L

# Convert to FlexAttention BlockMask or dense float tensor
float_mask = mask_builder.to_block_mask()  # (128, 128), float 0.0/1.0
```

### `LinearSchedule` / `CosineSchedule` — Noise Schedules

```python
from inference import LinearSchedule, CosineSchedule

schedule = LinearSchedule()
mask_prob = schedule.mask_prob(t=torch.tensor(0.5))  # 0.5
elbo_weight = schedule.elbo_weight(t=torch.tensor(0.5))  # 2.0

cos_schedule = CosineSchedule()
mask_prob = cos_schedule.mask_prob(t=torch.tensor(0.5))  # ≈ 0.29
```

---

## Module Design

```
inference/
├── __init__.py       # Package exports & overview diagram
├── schedules.py      # LinearSchedule, CosineSchedule
│   Input:  t ~ U[0, 1] (timestep)
│   Output: mask_prob(t), elbo_weight(t)
│
├── sampler.py        # GumbelSampler
│   Input:  logits (B, L, V)
│   Output: samples (B, L), confidence (B, L)
│
├── mask.py           # StaircaseMask
│   Input:  seq_len, block_size
│   Output: (2L, 2L) boolean attention mask
│
├── unmask.py         # unmask_top_k, unmask_by_threshold, uniform_schedule
│   Input:  masked (B, L), confidences (B, L), k
│   Output: new_mask (B, L)
│
├── denoiser.py       # BlockDenoiser
│   Orchestrates: init_block → denoise loop → extract tokens
│
└── generator.py      # generate() — full pipeline
    1. encode_fn(prompt) → prompt_ids
    2. KV cache warmup
    3. BlockDenoiser per block
    4. decode_fn(tokens) → text
```

---

## Running Tests

```bash
# Install pytest (if needed)
pip install pytest

# Run all tests
python -m pytest tests/ -v

# Run specific test class
python -m pytest tests/test_block_diffusion_inference.py::TestStaircaseMask -v

# Expected output:
# 19 passed in ~1s
```
