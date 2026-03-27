"""Block Diffusion Inference Package — Clean MPS-Optimized Implementation.

================================================================================
                           ARCHITECTURE OVERVIEW
================================================================================

    ┌─────────────────────────────────────────────────────────────────────────────┐
    │                           External Interface                                  │
    │                                                                              │
    │    from inference import generate_sdar                                         │
    │                                                                              │
    │    output = generate_sdar(                                                  │
    │        model_dir="/path/to/SDAR-1.7B-Chat",                                │
    │        prompt="Hello, how are you?",                                        │
    │        device="mps",                                                        │
    │    )                                                                        │
    └─────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
    ┌─────────────────────────────────────────────────────────────────────────────┐
    │  generation pipeline                                                        │
    │                                                                              │
    │    1. Load SDAR model with MPS-optimized attention                          │
    │    2. Prepare prompt with chat template                                     │
    │    3. KV cache warmup with prompt blocks                                   │
    │    4. Block-by-block diffusion generation:                                  │
    │         a. Initialize block with prompt remainder + [MASK]                 │
    │         b. Denoise loop (4-8 steps per block):                           │
    │              - Forward pass with block attention mask                       │
    │              - Sample x0 predictions                                        │
    │              - Remask based on confidence (low_confidence_dynamic)         │
    │         c. Cache KV for finalized block                                     │
    │         d. Extract new tokens                                               │
    │    5. Decode and return generated text                                      │
    └─────────────────────────────────────────────────────────────────────────────┘

================================================================================
                              SDAR MODEL WRAPPER
================================================================================

    SDARWrapper (sdar_wrapper.py)
    ├── Wraps SDARForCausalLM for block diffusion inference
    ├── Unified interface: __call__(x, pos_offset) → (logits, _)
    ├── Supports MPS (Apple Silicon) and CPU devices
    ├── Uses SDPA for attention (no CUDA dependencies)
    └── KV cache via DynamicCache

================================================================================
                              CORE COMPONENTS
================================================================================

    denoiser.py
    ├── BlockDenoiser: Per-block denoising orchestration
    │   ├── init_block(): Create block with prompt + masks
    │   └── step_unmask_count(): Compute unmasking schedule
    │
    sampler.py
    ├── GumbelSampler: Gumbel-max sampling with temperature
    │   ├── sample(): Gumbel-max sampling
    │   └── sample_with_confidence(): Sample + return confidence
    │
    unmask.py
    ├── unmask_top_k(): Unmask highest confidence positions
    ├── unmask_by_threshold(): Unmask positions exceeding threshold
    └── uniform_schedule(): Uniform unmasking across steps

    mask.py
    ├── StaircaseMask: Attention mask for block diffusion
    │   ├── Within-block: bidirectional
    │   ├── Across blocks: block-causal
    │   └── No label leakage: noisy block can't see own clean tokens
    │
    schedules.py
    ├── LinearSchedule: mask_prob(t) = t
    └── CosineSchedule: mask_prob(t) = 1 - cos²(πt/2)

================================================================================
                              USAGE EXAMPLES
================================================================================

    Example 1: Basic Generation
    ───────────────────────────
    from inference import generate_sdar

    output = generate_sdar(
        model_dir="/tmp/sdar-1.7b-chat",
        prompt="Hello, how are you?",
        device="mps",
        gen_length=128,
        block_length=4,
        denoising_steps=4,
    )
    print(output)


    Example 2: Using SDARWrapper directly
    ──────────────────────────────────────
    from inference.sdar_wrapper import SDARWrapper
    from inference.generator import generate

    # Create wrapper
    wrapper = SDARWrapper(
        model_path="/tmp/sdar-1.7b-chat",
        device="mps",
        dtype="bfloat16"
    )

    # Use with generator
    output = generate(
        model=wrapper,
        encode_fn=lambda s: tokenizer.encode(s),
        decode_fn=lambda ids: tokenizer.decode(ids),
        prompt="Hello",
        max_new_tokens=100,
        block_size=4,
        denoise_steps=4,
    )


    Example 3: Custom model with block diffusion interface
    ───────────────────────────────────────────────────────
    class MyBlockDiffusionModel:
        def __call__(self, x, pos_offset=0):
            # x: (B, L) token IDs
            # Returns: (logits, _) — logits: (B, L, vocab_size)
            ...

        def set_cache_mode(self, enabled): ...
        def reset_kv_cache(self): ...

    from inference import generate
    output = generate(
        model=MyBlockDiffusionModel(),
        encode_fn=tokenizer.encode,
        decode_fn=tokenizer.decode,
        prompt="Hello",
        max_new_tokens=100,
    )

================================================================================
                            MODULE STRUCTURE
================================================================================

    inference/
    ├── __init__.py           # Main exports
    ├── generator.py          # Full generation pipeline
    ├── denoiser.py           # Block denoising orchestration
    ├── sampler.py            # Gumbel-max sampling
    ├── unmask.py             # Confidence-based unmasking
    ├── mask.py               # Staircase attention masks
    ├── schedules.py          # Noise schedules
    ├── sdar_wrapper.py       # SDAR model wrapper
    └── sdar_inference.py     # Standalone SDAR CLI

================================================================================
"""

from .schedules import LinearSchedule, CosineSchedule
from .sampler import GumbelSampler
from .mask import StaircaseMask
from .unmask import unmask_top_k, unmask_by_threshold, uniform_schedule
from .denoiser import BlockDenoiser
from .generator import generate
from .sdar_wrapper import SDARWrapper, create_sdar_wrapper

# SDAR-specific generation function
from .sdar_inference import main as generate_sdar

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
    # SDAR wrapper
    "SDARWrapper",
    "create_sdar_wrapper",
    # SDAR generation
    "generate_sdar",
]
