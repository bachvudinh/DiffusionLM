# SDAR Block Diffusion Inference — MPS-Optimized

A clean implementation of SDAR (Synergy of Diffusion and AutoRegression) block diffusion inference for Apple Silicon Macs.

**Model:** [JetLM/SDAR-1.7B-Chat](https://huggingface.co/JetLM/SDAR-1.7B-Chat)
**Paper:** [SDAR: Synergy of Diffusion and AutoRegression](https://github.com/JetAstra/SDAR)

================================================================================
                                QUICK START
================================================================================

```bash
# Clone the repository
cd DiffusionLM

# Download SDAR model (if not already)
huggingface-cli download JetLM/SDAR-1.7B-Chat --local-dir /tmp/sdar-1.7b-chat

# Run inference
python -m inference.sdar_inference \
    --model_dir /tmp/sdar-1.7b-chat \
    --prompt "Hello, how are you?" \
    --device mps \
    --gen_length 128
```

**Output:**
```
Prompt: <|im_start|>user
Hello, how are you?<|im_end|>
<|im_start|>assistant

Generating with block_length=4, denoising_steps=4...

================================================================
GENERATED OUTPUT:
================================================================
Hello! I am an AI language model and I am ready to assist you. How may I help you today?<|im_end|>
<|endoftext|>
================================================================
```

================================================================================
                              PROJECT STRUCTURE
================================================================================

```
DiffusionLM/
├── README.md                    # This file
├── .gitignore                  # Git ignore patterns
│
├── inference/                   # Block diffusion inference module
│   ├── __init__.py             # Main exports
│   ├── generator.py             # Generic block diffusion pipeline
│   ├── denoiser.py             # Block denoising orchestration
│   ├── sampler.py               # Gumbel-max sampling
│   ├── unmask.py               # Confidence-based unmasking
│   ├── mask.py                 # Staircase attention masks
│   ├── schedules.py             # Noise schedules (Linear, Cosine)
│   ├── sdar_wrapper.py         # SDAR model wrapper (MPS-compatible)
│   └── sdar_inference.py       # SDAR CLI and core inference
│
├── sdar_model/                  # SDAR model implementation (MPS-compatible)
│   ├── __init__.py
│   ├── modeling_sdar.py         # SDAR Transformer model
│   ├── configuration_sdar.py   # Model configuration
│   └── fused_linear_diffusion_cross_entropy.py  # Training loss (unused in inference)
│
└── tests/                     # Unit tests
```

================================================================================
                           WHAT IS BLOCK DIFFUSION?
================================================================================

Unlike standard Autoregressive (AR) models that generate tokens one-by-one
(left-to-right), block diffusion models generate **entire blocks of tokens**
in parallel using a denoising process.

**Key Concepts:**

1. **Block Diffusion**
   - Sequence is divided into fixed-size blocks
   - Each block is denoised over multiple steps
   - Tokens within a block attend bidirectionally (like diffusion)
   - Blocks are processed left-to-right (like AR)

2. **Mask Token (151669)**
   - SDAR uses a special mask token that gets iteratively denoised
   - At each step, high-confidence predictions are "unmasked"
   - Process continues until all masks are resolved

3. **Remasking Strategies**
   - `low_confidence_dynamic`: Unmask high confidence first (recommended)
   - `low_confidence_static`: Unmask lowest confidence first
   - `sequential`: Left-to-right unmasking
   - `entropy_bounded`: Based on prediction entropy

================================================================================
                           USAGE EXAMPLES
================================================================================

### Example 1: CLI Usage

```bash
python -m inference.sdar_inference \
    --model_dir /tmp/sdar-1.7b-chat \
    --prompt "What is the capital of France?" \
    --device mps \
    --gen_length 128 \
    --block_length 4 \
    --denoising_steps 8 \
    --temperature 0.8
```

### Example 2: Module Usage (Programmatic)

```python
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
```

### Example 3: Using SDARWrapper Directly

```python
from inference.sdar_wrapper import SDARWrapper
from transformers import AutoTokenizer

# Load model
wrapper = SDARWrapper(
    model_path="/tmp/sdar-1.7b-chat",
    device="mps",
    dtype="bfloat16"
)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("/tmp/sdar-1.7b-chat")

# Prepare prompt
messages = [{"role": "user", "content": "Hello!"}]
prompt_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
tokens = tokenizer(prompt_text, return_tensors='pt')

# Generate using core inference function
from inference.sdar_inference import block_diffusion_generate

output_ids = block_diffusion_generate(
    model=wrapper,
    prompt=tokens,
    mask_id=151669,
    gen_length=128,
    block_length=4,
    denoising_steps=4,
)

# Decode
output_text = tokenizer.decode(output_ids[0])
print(output_text)
```

### Example 4: Custom Block Diffusion Model

```python
from inference import generate, BlockDenoiser, GumbelSampler

class MyBlockDiffusionModel:
    """Your model needs these methods:"""
    def __call__(self, x, pos_offset=0):
        # x: (B, L) token IDs
        # Returns: (logits, _) — logits: (B, L, vocab_size)
        ...

    def set_cache_mode(self, enabled): pass
    def reset_kv_cache(self): pass
    def parameters(self): ...

model = MyBlockDiffusionModel()
result = generate(
    model=model,
    encode_fn=lambda s: tokenizer.encode(s),
    decode_fn=lambda ids: tokenizer.decode(ids),
    prompt="Hello",
    max_new_tokens=100,
    block_size=4,
    denoise_steps=4,
)
```

================================================================================
                           INPUT/OUTPUT SPECIFICATIONS
================================================================================

### `block_diffusion_generate()`

```python
block_diffusion_generate(
    model,                  # SDARWrapper or compatible model
    prompt,                # dict: {'input_ids': Tensor (1, prompt_len)}
    mask_id,               # int: 151669 (SDAR mask token)
    gen_length=128,        # Maximum tokens to generate
    block_length=4,         # Tokens per block
    denoising_steps=4,      # Denoising iterations per block
    temperature=1.0,      # Sampling temperature
    top_k=0,              # Top-k filtering (0=disabled)
    top_p=1.0,            # Top-p filtering (1.0=disabled)
    remasking_strategy='low_confidence_dynamic',
    confidence_threshold=0.85,
    eb_threshold=0.35,
    stopping_criteria_idx=None,
    echo=False,
) -> Tensor  # (1, gen_length) token IDs
```

### `generate_sdar()` (High-Level)

```python
generate_sdar(
    model_dir,             # Path to SDAR model
    prompt,                # str: input text
    device='mps',         # 'mps' or 'cpu'
    gen_length=256,       # Maximum generation length
    block_length=4,        # Block size
    denoising_steps=4,     # Denoising steps per block
    temperature=1.0,
    top_k=0,
    top_p=1.0,
    remasking_strategy='low_confidence_dynamic',
) -> str  # Generated text
```

================================================================================
                              DEVICE SUPPORT
================================================================================

| Device | Status | Notes |
|--------|--------|-------|
| **MPS** | ✅ Primary | Apple Silicon GPU via Metal |
| **CPU** | ✅ Fallback | For Intel Macs |

**Auto-detection:**
```python
import torch

device = 'mps' if torch.backends.mps.is_available() else 'cpu'
print(f"Using device: {device}")
```

================================================================================
                           CORE ALGORITHM
================================================================================

### Block Diffusion Generation Flow

```
Prompt: "Hello" (3 tokens), block_size=4, denoising_steps=3

Step 1: KV Cache Warmup
───────────────────────
prompt_ids = [10, 20, 30]
n_full_prompt_blocks = 0
prompt_remainder = 3

Step 2: Generate Block 0
────────────────────────
Initial block = [token_10, token_20, token_30, MASK]

Denoise Step 0:
    logits → sample → unmask top-k(k=1)
    block = [token_10, token_20, token_30, token_42]

Denoise Step 1:
    logits → sample → unmask top-k(k=1)
    block = [token_10, token_20, token_30, token_88]

Denoise Step 2 (final):
    logits → sample → unmask ALL
    block = [token_10, token_20, token_30, token_99]

Cache KV for block 0
new_tokens = [token_99]

Continue until max_new_tokens reached
```

### Attention Mask Pattern

For block_size=4, num_blocks=3:

```
         Block 0   Block 1   Block 2
         ───────   ───────   ───────
Block 0  [1 1 1 1 | 0 0 0 0 | 0 0 0 0]  ← Block 0 only (within block)
Block 1  [1 1 1 1 | 1 1 1 1 | 0 0 0 0]  ← Block 0,1 (block-causal)
Block 2  [1 1 1 1 | 1 1 1 1 | 1 1 1 1]  ← All blocks (full causal)

1 = attention allowed, 0 = blocked
```

================================================================================
                           REQUIREMENTS
================================================================================

- Python 3.10+
- PyTorch 2.0+
- Transformers 4.30+
- safetensors
- Apple Silicon Mac (MPS) or Intel Mac (CPU)

================================================================================
                              TESTING
================================================================================

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test
python -m pytest tests/test_block_diffusion_inference.py -v
```

================================================================================
                              CITATION
================================================================================

If you use this code, please cite:

```bibtex
@misc{sdar2025,
    title={SDAR: Synergy of Diffusion and AutoRegression},
    author={JetLM Team},
    year={2025},
    url={https://github.com/JetAstra/SDAR}
}
```
