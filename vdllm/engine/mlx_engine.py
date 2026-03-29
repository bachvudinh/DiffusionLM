"""MLX backend engine for block diffusion inference on Apple Silicon.

Wraps the MLX generation pipeline into the same API surface as LLMEngine,
so the unified LLM class can dispatch to either backend transparently.
"""

import time

import mlx.core as mx
import numpy as np
from transformers import AutoTokenizer, GenerationConfig

from vdllm.config import Config
from vdllm.sampling_params import SamplingParams
from vdllm.models.mlx_sdar import load_sdar_model
from vdllm.generation import block_diffusion_generate


class MLXEngine:
    """MLX inference engine for block diffusion models.

    Processes prompts sequentially through block_diffusion_generate().
    Designed for Apple Silicon — no batching, no paged KV cache, no TP.

    Output format matches LLMEngine.generate():
        [{"text": str, "token_ids": list, "trajectory": [], "logprobs": [], "entropies": []}]
    """

    def __init__(self, config: Config):
        self.config = config

        print(f"Loading MLX model from {config.model}...")
        t0 = time.time()
        self.model, self.model_config = load_sdar_model(
            config.model, dtype=config.mlx_dtype)
        print(f"Model loaded in {time.time() - t0:.1f}s")

        self.tokenizer = AutoTokenizer.from_pretrained(
            config.model, use_fast=True, trust_remote_code=True)

        # Resolve mask_token_id
        if config.mask_token_id == -1:
            try:
                config.mask_token_id = self.tokenizer(
                    self.tokenizer.mask_token)['input_ids'][0]
            except Exception:
                config.mask_token_id = (
                    self.tokenizer.mask_token_id
                    if self.tokenizer.mask_token_id is not None
                    else self.tokenizer.pad_token_id)
        assert config.mask_token_id is not None and config.mask_token_id != -1, \
            "Could not resolve mask_token_id from tokenizer"

        # Resolve EOS tokens
        if config.eos == -1:
            config.eos = self.tokenizer.eos_token_id
        self._resolve_eos_ids()

        print(f"Mask token id: {config.mask_token_id}")
        print(f"EOS tokens: {self.eos_ids}")

    def _resolve_eos_ids(self):
        """Build list of EOS token IDs from GenerationConfig or tokenizer."""
        try:
            gen_cfg = GenerationConfig.from_pretrained(self.config.model)
            eos = gen_cfg.eos_token_id
        except Exception:
            eos = self.config.eos

        if isinstance(eos, int):
            self.eos_ids = [eos]
        elif isinstance(eos, list):
            self.eos_ids = eos
        else:
            self.eos_ids = [self.tokenizer.eos_token_id] if self.tokenizer.eos_token_id else []

    def generate(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
        use_tqdm: bool = True,
    ) -> list[dict]:
        """Generate completions for a batch of prompts.

        Processes prompts sequentially (MLX doesn't benefit from
        the complex batched scheduling that CUDA needs).
        """
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)

        if use_tqdm:
            try:
                from tqdm.auto import tqdm
                pbar = tqdm(total=len(prompts), desc="Generating",
                            dynamic_ncols=True)
            except ImportError:
                use_tqdm = False

        outputs = []
        for prompt, sp in zip(prompts, sampling_params):
            result = self._generate_one(prompt, sp)
            outputs.append(result)
            if use_tqdm:
                pbar.update(1)

        if use_tqdm:
            pbar.close()
        return outputs

    def _generate_one(self, prompt: str | list[int],
                      sp: SamplingParams) -> dict:
        """Generate for a single prompt."""
        config = self.config
        mask_id = config.mask_token_id

        # Tokenize
        if isinstance(prompt, str):
            input_ids = self.tokenizer.encode(
                prompt, return_tensors="np", truncation=True,
                max_length=config.max_position_embeddings)
        else:
            input_ids = np.array([prompt], dtype=np.int32)

        input_ids_mx = mx.array(input_ids)
        prompt_length = input_ids_mx.shape[1]

        # Resolve stop words
        stop_ids = sp.stop_words if sp.stop_words else self.eos_ids

        # Run block diffusion generation
        output, timing = block_diffusion_generate(
            model=self.model,
            input_ids=input_ids_mx,
            mask_id=mask_id,
            gen_length=sp.max_tokens,
            block_length=sp.block_length,
            denoising_steps=sp.denoising_steps,
            temperature=sp.temperature,
            top_k=sp.topk,
            top_p=sp.topp,
            remasking_strategy=sp.remasking_strategy,
            confidence_threshold=sp.dynamic_threshold,
            eb_threshold=sp.eb_threshold,
            stopping_criteria_idx=stop_ids if stop_ids else None,
        )
        mx.eval(output)

        # Decode
        output_ids = output[0].tolist()
        generated_ids = output_ids[prompt_length:]

        # Truncate at EOS
        for eos_id in (stop_ids or []):
            if eos_id in generated_ids:
                generated_ids = generated_ids[:generated_ids.index(eos_id)]
                break

        # Remove remaining mask tokens
        generated_ids = [t for t in generated_ids if t != mask_id]

        token_ids = output_ids[:prompt_length] + generated_ids
        try:
            text = self.tokenizer.decode(token_ids, skip_special_tokens=True)
        except Exception:
            text = self.tokenizer.decode(
                [self.tokenizer.eos_token_id], skip_special_tokens=True)

        # Print speed metrics
        prefill_t = timing["prefill_time"]
        decode_t = timing["decode_time"]
        prompt_toks = timing["prompt_tokens"]
        gen_toks = timing["generated_tokens"]
        prefill_tps = prompt_toks / prefill_t if prefill_t > 0 else 0
        decode_tps = gen_toks / decode_t if decode_t > 0 else 0
        print(f"  Prefill: {prompt_toks} tokens, {prefill_tps:.1f} tok/s | "
              f"Decode: {gen_toks} tokens, {decode_tps:.1f} tok/s")

        return {
            "text": text,
            "token_ids": token_ids,
            "trajectory": [],
            "logprobs": [],
            "entropies": [],
            "timing": timing,
        }

    def is_finished(self):
        return True
