"""Block Diffusion Language Model — Clean Inference Package.

Architecture Overview
====================

    ┌─────────────────────────────────────────────────────────────────────────────┐
    │                           External Interface                                  │
    │                                                                              │
    │    from inference import generate                                             │
    │                                                                              │
    │    output = generate(                                                        │
    │        model=my_model,          # forward(x, pos_offset) → (logits, _)       │
    │        encode_fn=tokenize,      # str → list[int]                            │
    │        decode_fn=detokenize,    # list[int] → str                           │
    │        prompt="Hello",                                                    │
    │        max_new_tokens=512,                                                  │
    │    )                                                                        │
    └─────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
    ┌─────────────────────────────────────────────────────────────────────────────┐
    │  generator.py                                                                │
    │                                                                              │
    │  generate() — Full generation pipeline                                       │
    │    1. encode_fn(prompt) → prompt_ids                                         │
    │    2. KV cache warmup with prompt blocks                                     │
    │    3. BlockDenoiser per block:                                              │
    │         a. init_block(prompt_remainder) → (block, masked)                    │
    │         b. denoise loop: forward → Gumbel sample → unmask top-k             │
    │         c. cache KV for finalized block                                      │
    │         d. extract new tokens                                                │
    │    4. decode_fn(tokens) → string                                             │
    └─────────────────────────────────────────────────────────────────────────────┘
                                        │
                    ┌───────────────────┼───────────────────┐
                    ▼                   ▼                   ▼
    ┌───────────────────┐ ┌───────────────────┐ ┌───────────────────┐
    │    denoiser.py    │ │    sampler.py     │ │    unmask.py      │
    │                    │ │                   │ │                   │
    │ BlockDenoiser:     │ │ GumbelSampler:    │ │ unmask_top_k():   │
    │  • init_block()    │ │  • sample()       │ │  unmask highest   │
    │  • step_unmask()   │ │  • prob()          │ │  confidence first │
    │  • denoise_block() │ │  • sample+conf()   │ │                   │
    └───────────────────┘ └───────────────────┘ └───────────────────┘
                                        │
                                        ▼
    ┌─────────────────────────────────────────────────────────────────────────────┐
    │    mask.py                           │    schedules.py                      │
    │                                      │                                      │
    │ StaircaseMask:                       │ NoiseSchedule:                        │
    │  • build() → (2L, 2L) bool mask      │  • mask_prob(t) → how many masked     │
    │  • to_block_mask() → FlexAttention   │  • elbo_weight(t) → loss weighting    │
    │                                      │                                       │
    │  Constraints:                        │ Supported:                            │
    │  • Within-block: bidirectional        │  • LinearSchedule (mask_prob = t)    │
    │  • Between blocks: block-causal       │  • CosineSchedule (SEDD paper)        │
    │  • No label leakage:                  │                                       │
    │    noisy_i cannot see clean_i         │                                       │
    └─────────────────────────────────────────────────────────────────────────────┘


Module Dependencies
═══════════════════

    generator.py
        └── denoiser.py
        │     ├── sampler.py (GumbelSampler)
        │     └── unmask.py (unmask_top_k)
        │
        ├── mask.py (StaircaseMask)      [used by model, not this package]
        └── schedules.py (LinearSchedule, CosineSchedule)  [for training, not inference]


Usage Example
═════════════

    import torch
    from inference import generate

    class SimpleModel:
        def __init__(self, vocab_size=1000, n_embd=256, n_layers=4):
            self.vocab_size = vocab_size
            self.emb = torch.nn.Embedding(vocab_size, n_embd)
            self.lm_head = torch.nn.Linear(n_embd, vocab_size)
            self._training = False

        def __call__(self, x, pos_offset=0):
            h = self.emb(x)
            logits = self.lm_head(h)
            return logits, None

        def set_cache_mode(self, enabled): pass
        def reset_kv_cache(self): pass
        def parameters(self):
            return list(self.parameters())

    model = SimpleModel()
    tokens = generate(
        model=model,
        encode_fn=lambda s: [ord(c) % 1000 for c in s],
        decode_fn=lambda ids: ''.join(chr(i % 26 + 97) for i in ids),
        prompt="hello",
        max_new_tokens=20,
        block_size=4,
        denoise_steps=5,
    )
    print(tokens)
"""

from .schedules import LinearSchedule, CosineSchedule
from .sampler import GumbelSampler
from .mask import StaircaseMask
from .unmask import unmask_top_k, unmask_by_threshold, uniform_schedule
from .denoiser import BlockDenoiser
from .generator import generate

__all__ = [
    # Schedules
    "LinearSchedule",
    "CosineSchedule",
    # Sampling
    "GumbelSampler",
    # Masking
    "StaircaseMask",
    # Unmasking strategies
    "unmask_top_k",
    "unmask_by_threshold",
    "uniform_schedule",
    # Block denoiser
    "BlockDenoiser",
    # Main entry point
    "generate",
]
