"""Block diffusion generation example using SDAR model on MLX."""

import argparse
import time
import sys

import mlx.core as mx

from vdllm.models.mlx_sdar import load_sdar_model
from vdllm.generation import block_diffusion_generate


def main():
    parser = argparse.ArgumentParser(description="Block Diffusion Generation with SDAR on MLX")
    parser.add_argument("prompt", type=str, help="Input prompt text")
    parser.add_argument("--model-path", type=str, default="/tmp/sdar-1.7b-chat",
                        help="Path to SDAR model directory")
    parser.add_argument("--mask-id", type=int, default=None,
                        help="Mask token id (auto-detected from tokenizer if not set)")
    parser.add_argument("--prompt-length", type=int, default=4096,
                        help="Maximum prompt length in tokens")
    parser.add_argument("--gen-length", type=int, default=20480,
                        help="Maximum generation length in tokens")
    parser.add_argument("--block-length", type=int, default=4,
                        help="Length of token block to replace each denoising step")
    parser.add_argument("--denoising-steps", type=int, default=4,
                        help="Number of denoising steps (iterations)")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Sampling temperature")
    parser.add_argument("--top-k", type=int, default=0,
                        help="Top-K sampling (0 to disable)")
    parser.add_argument("--top-p", type=float, default=1.0,
                        help="Top-P sampling probability threshold (1.0 to disable)")
    parser.add_argument("--remasking-strategy", type=str, default="low_confidence_dynamic",
                        choices=["low_confidence_dynamic", "low_confidence_static",
                                 "sequential", "entropy_bounded"],
                        help="Strategy for remasking tokens")
    parser.add_argument("--confidence-threshold", type=float, default=0.85,
                        help="Confidence threshold for low-confidence remasking")
    parser.add_argument("--eb-threshold", type=float, default=0.35,
                        help="Entropy threshold for entropy bounded sampling")
    parser.add_argument("--stopping-criteria-idx", type=int, nargs="+", default=None,
                        help="List of token IDs that stop generation (auto-detected if not set)")
    parser.add_argument("--chat", action="store_true",
                        help="Use chat template formatting")
    args = parser.parse_args()

    if args.remasking_strategy == "low_confidence_dynamic" and args.confidence_threshold is None:
        parser.error("--confidence-threshold is required when --remasking-strategy=low_confidence_dynamic")
    if args.remasking_strategy == "entropy_bounded" and args.eb_threshold is None:
        parser.error("--eb-threshold is required when --remasking-strategy=entropy_bounded")

    print(f"Loading model from {args.model_path}...")
    t0 = time.time()
    model, config = load_sdar_model(args.model_path)
    t1 = time.time()
    print(f"Model loaded in {t1 - t0:.1f}s")

    from transformers import AutoTokenizer, GenerationConfig
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    # Resolve mask_id
    if args.mask_id is not None:
        mask_id = args.mask_id
    else:
        mask_id = tokenizer(tokenizer.mask_token)['input_ids'][0]
    print(f"Mask token id: {mask_id}")

    # Resolve stopping criteria
    if args.stopping_criteria_idx is not None:
        eos_ids = args.stopping_criteria_idx
    else:
        gen_cfg = GenerationConfig.from_pretrained(args.model_path)
        eos_ids = gen_cfg.eos_token_id
    if isinstance(eos_ids, int):
        eos_ids = [eos_ids]
    elif eos_ids is None:
        eos_ids = []
    stop_words = tokenizer.convert_ids_to_tokens(eos_ids)
    print(f"Stop tokens: {list(zip(eos_ids, stop_words))}")

    # Prepare input
    if args.chat:
        messages = [{"role": "user", "content": args.prompt}]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        input_ids = tokenizer.encode(text, return_tensors="np",
                                     truncation=True, max_length=args.prompt_length,
                                     add_special_tokens=False)
    else:
        input_ids = tokenizer.encode(args.prompt, return_tensors="np",
                                     truncation=True, max_length=args.prompt_length)

    input_ids_mx = mx.array(input_ids)
    prompt_length = input_ids_mx.shape[1]

    print(f"\nPrompt ({prompt_length} tokens): {args.prompt}")
    print(f"Generation config: gen_length={args.gen_length}, block_length={args.block_length}, "
          f"steps={args.denoising_steps}")
    print(f"Sampling: temp={args.temperature}, top_k={args.top_k}, top_p={args.top_p}")
    print(f"Strategy: {args.remasking_strategy}")

    print(f"\nGenerating...")
    output, timing = block_diffusion_generate(
        model=model,
        input_ids=input_ids_mx,
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
        stopping_criteria_idx=eos_ids if eos_ids else None,
    )
    mx.eval(output)

    # Decode output
    output_ids = output[0].tolist()
    generated_ids = output_ids[prompt_length:]

    # Truncate at EOS if present
    for eos_id in eos_ids:
        if eos_id in generated_ids:
            generated_ids = generated_ids[:generated_ids.index(eos_id)]
            break

    # Also remove any remaining mask tokens
    generated_ids = [t for t in generated_ids if t != mask_id]

    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    full_text = tokenizer.decode(output_ids[:prompt_length], skip_special_tokens=True) + generated_text

    # Speed metrics (Qwen-style: separate prefill and decode)
    prefill_t = timing["prefill_time"]
    decode_t = timing["decode_time"]
    prompt_toks = timing["prompt_tokens"]
    gen_toks = timing["generated_tokens"]

    prefill_tps = prompt_toks / prefill_t if prefill_t > 0 else 0
    decode_tps = gen_toks / decode_t if decode_t > 0 else 0

    print(f"\n{'=' * 60}")
    print(f"Speed:")
    print(f"  Prefill : {prompt_toks} tokens in {prefill_t*1000:.0f} ms ({prefill_tps:.1f} tok/s)")
    print(f"  Decode  : {gen_toks} tokens in {decode_t*1000:.0f} ms ({decode_tps:.1f} tok/s)")
    print(f"  Total   : {prefill_t+decode_t:.2f}s")
    print(f"{'=' * 60}")
    print(f"Output:\n{full_text}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
