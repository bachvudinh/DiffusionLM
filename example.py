#!/usr/bin/env python3
"""
Example script: Run SDAR model with MLX on Apple Silicon

Usage:
    python example.py "Your prompt here"
    python example.py "The capital of France is" --gen-length 50
"""

import sys
import argparse
import time
from pathlib import Path

MODEL_PATH = Path("/tmp/sdar-1.7b-chat")


def load_model():
    """Load MLX SDAR model and tokenizer."""
    import mlx.core as mx
    from vdllm.models.mlx_sdar import SDARForCausalLM, SDARModelArgs, KVCache
    import json

    with open(MODEL_PATH / "config.json") as f:
        config = json.load(f)

    args = SDARModelArgs.from_dict(config)
    model = SDARForCausalLM(args)

    weights = {}
    for wf in sorted(MODEL_PATH.glob("*.safetensors")):
        weights.update(mx.load(str(wf)))
    weights = model.sanitize(weights)
    model.load_weights(list(weights.items()), strict=False)
    mx.eval(model.parameters())

    import os
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["_attn_implementation"] = "eager"

    import types
    sys.modules["flash_attn"] = types.ModuleType("flash_attn")
    sys.modules["flash_attn"].__spec__ = type("Spec", (), {"name": "flash_attn"})()
    sys.modules["fused_linear_diffusion_cross_entropy"] = types.ModuleType("fused_linear_diffusion_cross_entropy")
    sys.modules["fused_linear_diffusion_cross_entropy"].__spec__ = type("Spec", (), {"name": "fused_linear_diffusion_cross_entropy"})()

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_PATH), trust_remote_code=True)

    return model, tokenizer


def generate(model, tokenizer, prompt: str, gen_length: int = 50):
    """Generate text from prompt using MLX model."""
    import mlx.core as mx
    from vdllm.models.mlx_sdar import KVCache

    print(f"\nPrompt: \"{prompt}\"")
    print("-" * 50)

    input_ids = tokenizer.encode(prompt)
    input_mlx = mx.array([input_ids])

    caches = [KVCache() for _ in range(28)]

    start = time.perf_counter()
    logits = model(input_mlx, cache=caches)
    mx.eval(logits)
    prefill_time = (time.perf_counter() - start) * 1000

    generated = input_ids[:]
    gen_start = time.perf_counter()

    for _ in range(gen_length):
        next_token = int(mx.argmax(logits[:, -1]).item())
        generated.append(next_token)
        logits = model(mx.array([[next_token]]), cache=caches)
        mx.eval(logits)

    gen_time = (time.perf_counter() - gen_start) * 1000

    output = tokenizer.decode(generated)
    continuation = tokenizer.decode(generated[len(input_ids):])

    print(f"Prefill: {prefill_time:.0f} ms ({len(input_ids)} tokens)")
    print(f"Generated {gen_length} tokens in {gen_time:.0f} ms")
    print(f"Speed: {gen_length / (gen_time / 1000):.1f} tokens/sec")
    print()
    print(f"Output: \"{output}\"")
    print()
    print(f"Continuation: \"{continuation[:100]}{'...' if len(continuation) > 100 else ''}\"")

    return output


def main():
    parser = argparse.ArgumentParser(description="Run SDAR model with MLX")
    parser.add_argument("prompt", help="Prompt text")
    parser.add_argument("--gen-length", type=int, default=50, help="Number of tokens to generate")
    args = parser.parse_args()

    print("Loading model...")
    model, tokenizer = load_model()
    print("Model loaded.")

    generate(model, tokenizer, args.prompt, args.gen_length)


if __name__ == "__main__":
    main()
