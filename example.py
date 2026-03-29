"""Block diffusion generation using vdllm.

Works on both CUDA (NVIDIA GPU) and Apple Silicon (MLX).
Auto-detects hardware and uses the appropriate backend.

Usage:
    python example.py "What is the capital of France?"
    python example.py "Write a Python function" --chat --gen-length 512
    python example.py "Hello" --backend mlx --temperature 0.7
"""

import argparse

from vdllm import LLM, SamplingParams


def main():
    parser = argparse.ArgumentParser(
        description="Block Diffusion Generation with vdllm")
    parser.add_argument("prompt", type=str, help="Input prompt text")
    parser.add_argument(
        "--model-path", type=str, default="/tmp/sdar-1.7b-chat",
        help="Path to SDAR model directory")
    parser.add_argument(
        "--backend", type=str, default="auto",
        choices=["auto", "cuda", "mlx", "mps", "cpu"],
        help="Backend to use (default: auto-detect)")
    parser.add_argument(
        "--mask-id", type=int, default=-1,
        help="Mask token id (-1 for auto-detect from tokenizer)")
    parser.add_argument(
        "--gen-length", type=int, default=256,
        help="Maximum generation length in tokens")
    parser.add_argument(
        "--block-length", type=int, default=4,
        help="Length of token block to replace each denoising step")
    parser.add_argument(
        "--denoising-steps", type=int, default=4,
        help="Number of denoising steps per block")
    parser.add_argument(
        "--temperature", type=float, default=1.0,
        help="Sampling temperature")
    parser.add_argument(
        "--top-k", type=int, default=0,
        help="Top-K sampling (0 to disable)")
    parser.add_argument(
        "--top-p", type=float, default=1.0,
        help="Top-P sampling threshold (1.0 to disable)")
    parser.add_argument(
        "--remasking-strategy", type=str, default="low_confidence_dynamic",
        choices=["low_confidence_dynamic", "low_confidence_static",
                 "sequential", "entropy_bounded", "random"],
        help="Strategy for remasking tokens")
    parser.add_argument(
        "--confidence-threshold", type=float, default=0.85,
        help="Confidence threshold for low-confidence remasking")
    parser.add_argument(
        "--eb-threshold", type=float, default=0.35,
        help="Entropy threshold for entropy bounded sampling")
    parser.add_argument(
        "--chat", action="store_true",
        help="Use chat template formatting")
    args = parser.parse_args()

    # Initialize LLM (auto-detects backend)
    llm = LLM(
        args.model_path,
        backend=args.backend,
        mask_token_id=args.mask_id,
    )

    # Prepare prompt
    if args.chat:
        messages = [{"role": "user", "content": args.prompt}]
        prompt_text = llm.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
    else:
        prompt_text = args.prompt

    # Build sampling params
    params = SamplingParams(
        max_tokens=args.gen_length,
        block_length=args.block_length,
        denoising_steps=args.denoising_steps,
        temperature=args.temperature,
        topk=args.top_k,
        topp=args.top_p,
        remasking_strategy=args.remasking_strategy,
        dynamic_threshold=args.confidence_threshold,
        eb_threshold=args.eb_threshold,
    )

    print(f"\nPrompt: {args.prompt}")
    print(f"Config: gen_length={args.gen_length}, block_length={args.block_length}, "
          f"steps={args.denoising_steps}")
    print(f"Sampling: temp={args.temperature}, top_k={args.top_k}, "
          f"top_p={args.top_p}")
    print(f"Strategy: {args.remasking_strategy}")
    print()

    # Generate
    outputs = llm.generate([prompt_text], params, use_tqdm=False)

    print(f"{'=' * 60}")
    print(f"Output:\n{outputs[0]['text']}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
