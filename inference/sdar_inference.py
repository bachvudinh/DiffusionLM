#!/usr/bin/env python3
"""
================================================================================
SDAR Block Diffusion Inference — MPS-Optimized Implementation
================================================================================

Implements the block diffusion generation algorithm from:
    https://github.com/JetAstra/SDAR
    https://huggingface.co/JetLM/SDAR-1.7B-Chat

This module provides complete inference for SDAR models which use block
diffusion (not standard autoregressive generation).

================================================================================
                            KEY CONCEPTS
================================================================================

1. BLOCK DIFFUSION
   - Unlike AR models (left-to-right), block diffusion generates entire blocks
   - Each block goes through multiple denoising steps
   - Tokens within a block attend bidirectionally (like diffusion)
   - Blocks are processed left-to-right (like AR)

2. MASK TOKEN (151669)
   - SDAR uses a special mask token that gets iteratively denoised
   - At each step, high-confidence predictions are "unmasked"
   - Process continues until all masks are resolved

3. REMASKING STRATEGIES
   - sequential: Unmask left-to-right
   - low_confidence_static: Unmask lowest confidence first
   - low_confidence_dynamic: Unmask high confidence first, fallback to static
   - entropy_bounded: Unmask based on entropy

4. ATTENTION MASK
   - Within block: Bidirectional (tokens see all other tokens in block)
   - Across blocks: Causal (token only sees previous blocks)

================================================================================
                           USAGE EXAMPLES
================================================================================

    # CLI Usage:
    python -m inference.sdar_inference \
        --model_dir /tmp/sdar-1.7b-chat \
        --prompt "Hello, how are you?" \
        --device mps \
        --gen_length 128

    # Module usage:
    from inference import generate_sdar
    output = generate_sdar(
        model_dir="/tmp/sdar-1.7b-chat",
        prompt="Hello",
        device="mps",
    )

    # Programmatic usage:
    from inference.sdar_wrapper import SDARWrapper
    from inference.sdar_inference import block_diffusion_generate

    wrapper = SDARWrapper(model_dir, device="mps")
    output_ids = block_diffusion_generate(
        model=wrapper,
        prompt=tokens,
        mask_id=151669,
        gen_length=128,
        block_length=4,
        denoising_steps=4,
    )

================================================================================
                         INPUT/OUTPUT SPECIFICATIONS
================================================================================

    block_diffusion_generate()
    ─────────────────────────
    Input:
        model: SDARWrapper — SDAR model with forward(x) → logits
        prompt: dict — {'input_ids': Tensor (1, prompt_len)}
        mask_id: int — 151669 (SDAR mask token)
        gen_length: int — Maximum tokens to generate (default: 128)
        block_length: int — Tokens per block (default: 4)
        denoising_steps: int — Denoising iterations per block (default: 4)
        temperature: float — Sampling temperature (default: 1.0)
        top_k: int — Top-k filtering (default: 0 = disabled)
        top_p: float — Top-p filtering (default: 1.0 = disabled)
        remasking_strategy: str — 'low_confidence_dynamic' recommended
        confidence_threshold: float — For dynamic strategy (default: 0.85)
        eb_threshold: float — For entropy_bounded strategy (default: 0.35)
        stopping_criteria_idx: list — EOS token IDs to stop at
        echo: bool — Include prompt in output (default: False)

    Output:
        Tensor — Generated token IDs of shape (1, gen_length)

================================================================================
                         DATA FLOW DIAGRAM
================================================================================

    Prompt: "Hello" → [151643, 9925, ...] (tokenized)

    ┌─────────────────────────────────────────────────────────────────────────┐
    │ 1. PREFILL STAGE                                                         │
    │    Process full prompt blocks with block-causal attention                │
    │    Store KV cache for each block                                        │
    └─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
    ┌─────────────────────────────────────────────────────────────────────────┐
    │ 2. DECODE STAGE (per block)                                             │
    │                                                                          │
    │    Block 0 (first generated block):                                      │
    │    ┌──────────────────────────────────────────────────────────────┐     │
    │    │ Initial: [token_H, token_e, token_l, MASK]                    │     │
    │    │                prompt        mask                              │     │
    │    └──────────────────────────────────────────────────────────────┘     │
    │    │                                                                   │
    │    │  For step in denoising_steps:                                     │
    │    │  ┌──────────────────────────────────────────────────────────┐   │     │
    │    │  │ 1. Forward pass with block attention                    │   │     │
    │    │  │ 2. Get logits for all positions                        │   │     │
    │    │  │ 3. Sample x0 predictions                               │   │     │
    │    │  │ 4. Compute confidence per position                      │   │     │
    │    │  │ 5. Remask: keep highest confidence, reset others       │   │     │
    │    │  └──────────────────────────────────────────────────────────┘   │     │
    │    └─────────────────────────────────────────────────────────────────┘     │
    │    │                                                                   │
    │    │  Final: [token_H, token_e, token_l, token_o]                   │
    │    │  Cache KV for this block                                         │
    │    │  Extract: [token_o] → output                                     │
    └─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
    ┌─────────────────────────────────────────────────────────────────────────┐
    │ 3. REPEAT for remaining blocks until gen_length reached                 │
    └─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
    Output: "Hello, how are you? I am doing well..."

================================================================================
"""
import sys
from pathlib import Path

import argparse
import time

import torch
from torch.nn import functional as F
from transformers import AutoTokenizer
from transformers.cache_utils import DynamicCache

# Add parent to path for sdar_model import
sys.path.insert(0, str(Path(__file__).parent.parent))

from sdar_model import SDARForCausalLM, SDARConfig
from safetensors.torch import load_file


# =============================================================================
# SAMPLING FUNCTIONS
# =============================================================================

def top_k_logits(logits: torch.Tensor, k: int) -> torch.Tensor:
    """Apply top-k filtering to logits.

    Args:
        logits: Tensor of any shape
        k: Number of top logits to keep

    Returns:
        Filtered logits with -inf for non-top-k values
    """
    if k <= 0:
        return logits
    values, _ = torch.topk(logits, k)
    min_values = values[..., -1, None]
    return torch.where(logits < min_values, torch.full_like(logits, float('-inf')), logits)


def top_p_logits(logits: torch.Tensor, p: float) -> torch.Tensor:
    """Apply nucleus (top-p) filtering to logits.

    Args:
        logits: Tensor of shape (..., vocab_size)
        p: Cumulative probability threshold

    Returns:
        Filtered logits with -inf for tokens outside top-p
    """
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

    # Create mask: keep tokens until cumsum exceeds p
    sorted_mask = cumulative_probs > p
    sorted_mask[..., 1:] = sorted_mask[..., :-1].clone()  # Shift right
    sorted_mask[..., 0] = False  # Always include top token

    # Scatter mask back to original order
    mask_indices = torch.scatter(
        torch.full_like(logits, False, dtype=torch.bool),
        -1, sorted_indices, sorted_mask
    )
    return logits.masked_fill(mask_indices, float('-inf'))


def sample_with_temperature_topk_topp(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample tokens with temperature, top-k, and top-p filtering.

    Args:
        logits: Tensor of shape (batch, seq_len, vocab_size)
        temperature: Sampling temperature (1.0 = no scaling)
        top_k: Top-k filtering (0 = disabled)
        top_p: Top-p filtering (1.0 = disabled)

    Returns:
        Tuple of (sampled_tokens (batch, seq_len), token_probabilities (batch, seq_len))
    """
    orig_shape = logits.shape[:-1]  # (batch, seq_len)
    vocab_size = logits.shape[-1]

    # Flatten for sampling
    logits_flat = logits.reshape(-1, vocab_size)  # (batch * seq_len, vocab)

    # Apply temperature
    if temperature != 1.0:
        logits_flat = logits_flat / temperature

    # Apply filters
    if top_k > 0:
        logits_flat = top_k_logits(logits_flat, top_k)
    if top_p < 1.0:
        logits_flat = top_p_logits(logits_flat, top_p)

    # Sample
    probs = F.softmax(logits_flat, dim=-1)
    tokens = torch.multinomial(probs, num_samples=1).squeeze(-1)  # (batch * seq_len,)
    token_probs = torch.gather(probs, -1, tokens.unsqueeze(-1)).squeeze(-1)

    # Reshape back
    return tokens.view(*orig_shape), token_probs.view(*orig_shape)


# =============================================================================
# TOKEN TRANSFER SCHEDULE
# =============================================================================

def get_num_transfer_tokens(block_length: int, steps: int) -> torch.Tensor:
    """Calculate how many tokens to unmask at each denoising step.

    Args:
        block_length: Size of the block
        steps: Number of denoising steps

    Returns:
        Tensor of shape (steps,) with tokens to unmask at each step

    Example:
        block_length=4, steps=4 → [1, 1, 1, 1]
        block_length=8, steps=4 → [2, 2, 2, 2]
        block_length=5, steps=4 → [2, 1, 1, 1]
    """
    base = block_length // steps
    remainder = block_length % steps
    num_transfer = torch.zeros(steps, dtype=torch.int64) + base
    num_transfer[:remainder] += 1
    return num_transfer


# =============================================================================
# ATTENTION MASK
# =============================================================================

def block_diffusion_attention_mask(
    num_blocks: int,
    block_length: int,
    device: torch.device,
) -> torch.Tensor:
    """Create block causal attention mask for inference.

    Creates a mask where:
    - Within a block: tokens can attend to each other (bidirectional)
    - Between blocks: later blocks attend to earlier blocks (causal)

    Args:
        num_blocks: Number of blocks
        block_length: Size of each block
        device: Device to create tensor on

    Returns:
        Attention mask of shape [1, num_blocks*block_length, num_blocks*block_length]
        True = attention allowed, False = blocked

    Example:
        num_blocks=3, block_length=4, seq_len=12

        Block 0 (pos 0-3):   Can attend to block 0 only (within block bidirectional)
        Block 1 (pos 4-7):   Can attend to blocks 0,1 (within + block-causal)
        Block 2 (pos 8-11):  Can attend to blocks 0,1,2 (within + block-causal)

        Attention pattern:
        [1 1 1 1 | 0 0 0 0 | 0 0 0 0 ]  ← Block 0
        [1 1 1 1 | 1 1 1 1 | 0 0 0 0 ]  ← Block 1
        [1 1 1 1 | 1 1 1 1 | 1 1 1 1 ]  ← Block 2
    """
    # Create lower triangular block matrix
    block_mask = torch.tril(torch.ones(num_blocks, num_blocks, device=device, dtype=torch.bool))

    # Expand: repeat block_mask along both dimensions
    mask = block_mask.repeat_interleave(block_length, dim=0).repeat_interleave(block_length, dim=1)

    return mask.unsqueeze(0)  # Add batch dimension


# =============================================================================
# CORE BLOCK DIFFUSION GENERATION
# =============================================================================

@torch.no_grad()
def block_diffusion_generate(
    model,
    prompt: dict,
    mask_id: int,
    gen_length: int = 128,
    block_length: int = 4,
    denoising_steps: int = 4,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
    remasking_strategy: str = 'low_confidence_dynamic',
    confidence_threshold: float = 0.85,
    eb_threshold: float = 0.35,
    stopping_criteria_idx: list = None,
    echo: bool = False,
) -> torch.Tensor:
    """Generate text using SDAR block diffusion algorithm.

    This is the core algorithm from the SDAR paper. It processes the sequence
    block-by-block, denoising each block through multiple steps.

    Args:
        model: SDAR model with forward(x) → logits
        prompt: Dict with 'input_ids' tensor of shape (1, prompt_length)
        mask_id: Token ID for mask token (151669 for SDAR)
        gen_length: Maximum length to generate (default: 128)
        block_length: Size of token blocks for diffusion (default: 4)
        denoising_steps: Number of denoising iterations per block (default: 4)
        temperature: Sampling temperature (default: 1.0)
        top_k: Top-k filtering (0=disabled)
        top_p: Top-p filtering (1.0=disabled)
        remasking_strategy: 'sequential', 'low_confidence_static',
                          'low_confidence_dynamic', 'entropy_bounded'
        confidence_threshold: For low_confidence_dynamic (default: 0.85)
        eb_threshold: For entropy_bounded (default: 0.35)
        stopping_criteria_idx: List of token IDs that stop generation
        echo: If True, include prompt in output

    Returns:
        Generated token IDs tensor of shape (1, prompt_length + gen_length)

    Example:
        prompt = {'input_ids': tensor([[9925, 15359,  ...]])}  # "Hello"
        output = block_diffusion_generate(model, prompt, mask_id=151669)
        # output shape: (1, 128)
    """
    model.eval()
    input_ids = prompt['input_ids']
    prompt_length = input_ids.shape[1]

    # Initialize KV cache
    past_key_values = DynamicCache()

    # Calculate blocks needed
    total_length = prompt_length + gen_length
    num_blocks = (total_length + block_length - 1) // block_length
    total_length = num_blocks * block_length

    # Create attention mask for all blocks
    block_mask = block_diffusion_attention_mask(num_blocks, block_length, model.device)

    # Position IDs for all tokens
    position_ids = torch.arange(total_length, device=model.device).unsqueeze(0)

    # Initialize sequence with mask tokens
    x = torch.full((1, total_length), mask_id, dtype=torch.long, device=model.device)
    x[:, :prompt_length] = input_ids  # Fill prompt

    # Calculate prompt blocks
    prefill_blocks = prompt_length // block_length
    prefill_length = prefill_blocks * block_length

    # Prefill stage: process prompt with block causal mask
    if prefill_length > 0:
        cur_x = x[:, :prefill_length]
        cur_attn_mask = block_mask[:, :prefill_length, :prefill_length]
        cur_position_ids = position_ids[:, :prefill_length]

        model(
            cur_x,
            attention_mask=cur_attn_mask,
            position_ids=cur_position_ids,
            past_key_values=past_key_values,
            use_cache=True,
            store_kv=True,
        )

    # Get token transfer schedule
    num_transfer_tokens = get_num_transfer_tokens(block_length, denoising_steps)

    # Decode stage: generate blocks one by one
    for num_block in range(prefill_blocks, num_blocks):
        # Current block boundaries
        block_start = num_block * block_length
        block_end = block_start + block_length

        # Extract current block state
        cur_x = x[:, block_start:block_end].clone()
        # For attention mask: block attends to all previous blocks (up to block_end)
        cur_attn_mask = block_mask[:, block_start:block_end, :block_end]
        # For position_ids: only the current block's positions
        cur_position_ids = position_ids[:, block_start:block_end]

        # Denoising loop
        for step in range(denoising_steps + 1):
            # Check if all tokens are filled
            mask_index = (cur_x == mask_id)
            if mask_index.sum() == 0:
                # All tokens filled, just cache KV
                model(
                    cur_x,
                    attention_mask=cur_attn_mask,
                    position_ids=cur_position_ids,
                    past_key_values=past_key_values,
                    use_cache=True,
                    store_kv=True,
                )
                break

            # Forward pass - during denoising, don't store KV (they should only attend to context, not extend it)
            output = model(
                cur_x,
                attention_mask=cur_attn_mask,
                position_ids=cur_position_ids,
                past_key_values=past_key_values,
                use_cache=True,
                store_kv=False,
            )
            logits = output.logits if hasattr(output, 'logits') else output[0]

            # Sample x0 predictions
            x0, x0_p = sample_with_temperature_topk_topp(
                logits,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )

            # Apply remasking strategy
            transfer_index = compute_transfer_index(
                cur_x=cur_x,
                x0=x0,
                x0_p=x0_p,
                mask_index=mask_index,
                step=step,
                num_transfer_tokens=num_transfer_tokens,
                remasking_strategy=remasking_strategy,
                confidence_threshold=confidence_threshold,
                eb_threshold=eb_threshold,
            )

            # Update masked positions with sampled tokens
            cur_x[transfer_index] = x0[transfer_index]

        # Store completed block
        x[:, block_start:block_end] = cur_x

        # Check stopping criteria
        if stopping_criteria_idx is not None:
            if any(stop_idx in x[:, prompt_length:] for stop_idx in stopping_criteria_idx):
                break

    # Trim to actual generated length
    if not echo:
        x = x[:, prompt_length:]

    return x


def compute_transfer_index(
    cur_x: torch.Tensor,
    x0: torch.Tensor,
    x0_p: torch.Tensor,
    mask_index: torch.Tensor,
    step: int,
    num_transfer_tokens: torch.Tensor,
    remasking_strategy: str,
    confidence_threshold: float,
    eb_threshold: float,
) -> torch.Tensor:
    """Compute which positions to unmask based on remasking strategy.

    Args:
        cur_x: Current block state (1, block_length)
        x0: Sampled x0 predictions (1, block_length)
        x0_p: Token probabilities (1, block_length)
        mask_index: Boolean mask of masked positions (1, block_length)
        step: Current denoising step
        num_transfer_tokens: Tokens to unmask at each step
        remasking_strategy: Strategy name
        confidence_threshold: For dynamic strategy
        eb_threshold: For entropy strategy

    Returns:
        Boolean tensor (1, block_length) — True = transfer (unmask)
    """
    batch_size = cur_x.shape[0]
    transfer_index = torch.zeros_like(x0, dtype=torch.bool)

    n_to_transfer = num_transfer_tokens[step].item()

    for b in range(batch_size):
        if remasking_strategy == 'sequential':
            # Left-to-right unmasking
            if mask_index[b].any():
                first_mask = mask_index[b].nonzero(as_tuple=True)[0].min().item()
                transfer_index[b, first_mask:first_mask + n_to_transfer] = True

        elif remasking_strategy == 'low_confidence_static':
            # Unmask lowest confidence positions
            confidence = torch.where(mask_index[b], x0_p[b], -torch.inf)
            _, idx = torch.topk(confidence, n_to_transfer)
            transfer_index[b, idx] = True

        elif remasking_strategy == 'low_confidence_dynamic':
            # Unmask high confidence first, fallback to static
            confidence = torch.where(mask_index[b], x0_p[b], -torch.inf)
            high_conf_mask = confidence > confidence_threshold
            n_high_conf = high_conf_mask.sum().item()

            if n_high_conf >= n_to_transfer:
                # Enough high confidence, use them
                transfer_index[b] = high_conf_mask
            else:
                # Use all high confidence + fill with top-k of rest
                if n_high_conf > 0:
                    transfer_index[b] = high_conf_mask
                _, idx = torch.topk(confidence, n_to_transfer)
                transfer_index[b, idx] = True

        elif remasking_strategy == 'entropy_bounded':
            # Unmask based on entropy (lower = more certain)
            eps = 1e-12
            entropies = -(x0_p[b].clamp_min(eps) * x0_p[b].clamp_min(eps).log()).sum(dim=-1)
            entropies = torch.where(mask_index[b], entropies, torch.inf)
            ent_sorted, order = torch.sort(entropies, dim=0, descending=False)
            cumsum = torch.cumsum(ent_sorted, dim=0)

            k = torch.searchsorted(cumsum, torch.tensor(eb_threshold, device=x0.device))
            k = max(1, min(k.item(), mask_index[b].sum().item()))
            selected_token_indices = order[:k]
            transfer_index[b, selected_token_indices] = True
        else:
            raise ValueError(f"Unknown remasking strategy: {remasking_strategy}")

    return transfer_index


# =============================================================================
# MAIN CLI ENTRY POINT
# =============================================================================

def main():
    """CLI entry point for SDAR inference."""
    parser = argparse.ArgumentParser(description='SDAR Block Diffusion Inference (MPS-Optimized)')

    # Model arguments
    parser.add_argument('--model_dir', type=str,
                        help='Path to SDAR model directory')
    parser.add_argument('--model_name', type=str, default='JetLM/SDAR-1.7B-Chat',
                        help='HuggingFace model name')
    parser.add_argument('--dtype', type=str, default='bfloat16',
                        choices=['float16', 'bfloat16'],
                        help='Model dtype')

    # Generation arguments
    parser.add_argument('--prompt', type=str, default='Hello, how are you?',
                        help='Input prompt')
    parser.add_argument('--gen_length', type=int, default=256,
                        help='Maximum generation length')
    parser.add_argument('--block_length', type=int, default=4,
                        help='Block length for diffusion')
    parser.add_argument('--denoising_steps', type=int, default=4,
                        help='Number of denoising steps')
    parser.add_argument('--temperature', type=float, default=1.0,
                        help='Sampling temperature')
    parser.add_argument('--top_k', type=int, default=0,
                        help='Top-k filtering (0=disabled)')
    parser.add_argument('--top_p', type=float, default=1.0,
                        help='Top-p filtering (1.0=disabled)')
    parser.add_argument('--max_tokens', type=int, default=None,
                        help='Alias for gen_length')

    # Remasking strategy
    parser.add_argument('--remasking_strategy', type=str,
                        default='low_confidence_dynamic',
                        choices=['low_confidence_dynamic',
                                 'low_confidence_static',
                                 'sequential',
                                 'entropy_bounded'],
                        help='Remasking strategy')
    parser.add_argument('--confidence_threshold', type=float, default=0.9,
                        help='Confidence threshold for low_confidence_dynamic')
    parser.add_argument('--eb_threshold', type=float, default=0.35,
                        help='Entropy threshold for entropy_bounded')

    # System
    parser.add_argument('--device', type=str, default='mps',
                        help='Device to use (mps/cpu)')
    parser.add_argument('--echo', action='store_true',
                        help='Echo prompt in output')

    args = parser.parse_args()

    # Handle gen_length alias
    if args.max_tokens is not None:
        args.gen_length = args.max_tokens

    # Auto-detect device
    if args.device == 'auto':
        if torch.backends.mps.is_available():
            args.device = 'mps'
        else:
            args.device = 'cpu'

    print(f"Loading SDAR model from {args.model_name}...")
    print(f"Device: {args.device}, Dtype: {args.dtype}")

    # Determine model path
    if args.model_dir:
        model_path = args.model_dir
    else:
        model_path = args.model_name

    # Load model using our local sdar_model (MPS-compatible)
    dtype = torch.bfloat16 if args.dtype == 'bfloat16' else torch.float16

    print(f"Loading SDAR model using local sdar_model (MPS-compatible)...")
    config = SDARConfig.from_pretrained(model_path)
    model = SDARForCausalLM(config)

    # Load weights from safetensors
    safetensors_path = f'{model_path}/model.safetensors'
    print(f"Loading weights from {safetensors_path}...")
    state_dict = load_file(safetensors_path)
    model.load_state_dict(state_dict, strict=False)

    model = model.to(device=args.device)
    model = model.to(dtype=dtype)
    model.eval()

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=False)

    # Get mask token ID
    if hasattr(tokenizer, 'mask_token') and tokenizer.mask_token is not None:
        mask_id = tokenizer(tokenizer.mask_token)['input_ids'][0]
    else:
        mask_id = 151669  # SDAR default mask token
    print(f"Mask token ID: {mask_id}")

    # Get stopping criteria
    stopping_criteria_idx = [tokenizer.eos_token_id] if tokenizer.eos_token_id else []

    # Prepare prompt with chat template
    messages = [{"role": "user", "content": args.prompt}]
    prompt_text = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False
    )

    print(f"\nPrompt: {prompt_text}")

    # Tokenize
    tokens = tokenizer(
        prompt_text,
        return_tensors='pt',
        padding=True,
        truncation=True,
        add_special_tokens=False,
        max_length=2048
    )
    tokens = {k: v.to(model.device) for k, v in tokens.items()}

    print(f"Prompt tokens: {tokens['input_ids'].shape[1]}")

    # Generate
    print(f"\nGenerating with block_length={args.block_length}, "
          f"denoising_steps={args.denoising_steps}...")

    start_time = time.time()

    output_ids = block_diffusion_generate(
        model,
        prompt=tokens,
        mask_id=mask_id,
        gen_length=args.gen_length,
        block_length=args.block_length,
        denoising_steps=args.denoising_steps,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        remasking_strategy=args.remasking_strategy,
        confidence_threshold=args.confidence_threshold,
        eb_threshold=args.eb_threshold,
        stopping_criteria_idx=stopping_criteria_idx,
        echo=args.echo
    )

    elapsed = time.time() - start_time

    # Decode output
    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=False)
    cleaned_text = output_text.replace('<|MASK|>', '').replace('<|end_of_text|>', '')

    print(f"\nGenerated in {elapsed:.2f}s")
    print(f"{'='*60}")
    print("GENERATED OUTPUT:")
    print(f"{'='*60}")
    print(cleaned_text)
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
