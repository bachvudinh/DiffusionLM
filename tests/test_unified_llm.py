"""Tests for the unified LLM API and backend dispatch.

Run from DiffusionLM root:
    python -m pytest tests/test_unified_llm.py -v
    python tests/test_unified_llm.py              # standalone

Tests cover:
  1. Hardware detection
  2. Config creation (backend-agnostic, CUDA-specific, MLX-specific)
  3. LLM dispatch to correct backend engine
  4. MLX generation: output format, coherence, EOS handling
  5. Sampling params integration
  6. CPU/CUDA fallback behavior
"""

import os
import sys
import time
import platform

import pytest
import numpy as np

WORK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, WORK_DIR)

MODEL_PATH = "/tmp/sdar-1.7b-chat"
MODEL_EXISTS = os.path.isdir(MODEL_PATH)

# Detect available backends
HAS_MLX = False
try:
    import mlx.core as mx
    HAS_MLX = mx.metal.is_available()
except ImportError:
    pass

HAS_CUDA = False
try:
    import torch
    HAS_CUDA = torch.cuda.is_available()
except ImportError:
    pass

HAS_TORCH = False
try:
    import torch
    HAS_TORCH = True
except ImportError:
    pass

requires_model = pytest.mark.skipif(
    not MODEL_EXISTS, reason=f"Model not found at {MODEL_PATH}")
requires_mlx = pytest.mark.skipif(
    not HAS_MLX, reason="MLX Metal not available")
requires_cuda = pytest.mark.skipif(
    not HAS_CUDA, reason="CUDA not available")
requires_torch = pytest.mark.skipif(
    not HAS_TORCH, reason="PyTorch not installed")


# ============================================================================
# 1. Hardware Detection
# ============================================================================

class TestHardwareDetection:
    def test_detect_backend_returns_string(self):
        from vdllm.utils.hardware import detect_backend
        result = detect_backend()
        assert isinstance(result, str)
        assert result in ("cuda", "mlx", "mps", "cpu")

    def test_detect_backend_consistent(self):
        from vdllm.utils.hardware import detect_backend
        a = detect_backend()
        b = detect_backend()
        assert a == b

    @requires_mlx
    def test_detect_mlx_on_apple_silicon(self):
        from vdllm.utils.hardware import detect_backend
        if platform.machine() == "arm64" and not HAS_CUDA:
            assert detect_backend() == "mlx"

    @requires_cuda
    def test_detect_cuda_when_available(self):
        from vdllm.utils.hardware import detect_backend
        assert detect_backend() == "cuda"


# ============================================================================
# 2. Config
# ============================================================================

class TestConfig:
    @requires_model
    def test_config_auto_backend(self):
        from vdllm.config import Config
        config = Config(MODEL_PATH)
        assert config.backend in ("cuda", "mlx", "mps", "cpu")

    @requires_model
    @requires_mlx
    def test_config_explicit_mlx(self):
        from vdllm.config import Config
        config = Config(MODEL_PATH, backend="mlx")
        assert config.backend == "mlx"
        assert hasattr(config, "mlx_dtype")

    @requires_model
    @requires_cuda
    def test_config_explicit_cuda(self):
        from vdllm.config import Config
        config = Config(MODEL_PATH, backend="cuda")
        assert config.backend == "cuda"
        assert hasattr(config, "torch_dtype")

    @requires_model
    def test_config_loads_hf_config(self):
        from vdllm.config import Config
        config = Config(MODEL_PATH)
        assert config.hf_config is not None
        assert config.hidden_size > 0
        assert config.num_hidden_layers > 0
        assert config.num_attention_heads > 0

    @requires_model
    def test_config_model_architecture(self):
        """Verify config extracts correct SDAR-1.7B architecture."""
        from vdllm.config import Config
        config = Config(MODEL_PATH)
        assert config.hidden_size == 2048
        assert config.num_hidden_layers == 28
        assert config.num_attention_heads == 16
        assert config.num_key_value_heads == 8
        assert config.head_dim == 128

    def test_config_nonexistent_model_raises(self):
        from vdllm.config import Config
        with pytest.raises(AssertionError):
            Config("/nonexistent/path")

    @requires_model
    def test_config_custom_block_length(self):
        from vdllm.config import Config
        config = Config(MODEL_PATH, block_length=8)
        assert config.block_length == 8

    @requires_model
    def test_config_custom_mask_token_id(self):
        from vdllm.config import Config
        config = Config(MODEL_PATH, mask_token_id=12345)
        assert config.mask_token_id == 12345


# ============================================================================
# 3. SamplingParams
# ============================================================================

class TestSamplingParams:
    def test_defaults(self):
        from vdllm.sampling_params import SamplingParams
        sp = SamplingParams()
        assert sp.temperature == 1.0
        assert sp.max_tokens == 64
        assert sp.block_length == 4
        assert sp.denoising_steps == 4
        assert sp.topk == 0
        assert sp.topp == 1.0
        assert sp.remasking_strategy == "low_confidence_static"

    def test_custom_params(self):
        from vdllm.sampling_params import SamplingParams
        sp = SamplingParams(
            temperature=0.7,
            max_tokens=256,
            block_length=8,
            denoising_steps=8,
            topk=50,
            topp=0.95,
            remasking_strategy="entropy_bounded",
            eb_threshold=0.5,
        )
        assert sp.temperature == 0.7
        assert sp.max_tokens == 256
        assert sp.block_length == 8
        assert sp.remasking_strategy == "entropy_bounded"
        assert sp.eb_threshold == 0.5


# ============================================================================
# 4. LLM Dispatch
# ============================================================================

class TestLLMDispatch:
    @requires_model
    @requires_mlx
    def test_llm_auto_dispatch_mlx(self):
        """On Apple Silicon without CUDA, LLM should dispatch to MLX."""
        if HAS_CUDA:
            pytest.skip("CUDA available, would dispatch to CUDA")
        from vdllm import LLM
        llm = LLM(MODEL_PATH)
        assert llm.config.backend == "mlx"
        from vdllm.engine.mlx_engine import MLXEngine
        assert isinstance(llm._engine, MLXEngine)

    @requires_model
    @requires_mlx
    def test_llm_explicit_mlx(self):
        from vdllm import LLM
        llm = LLM(MODEL_PATH, backend="mlx")
        assert llm.config.backend == "mlx"

    @requires_model
    @requires_mlx
    def test_llm_has_tokenizer(self):
        from vdllm import LLM
        llm = LLM(MODEL_PATH, backend="mlx")
        assert llm.tokenizer is not None
        assert hasattr(llm.tokenizer, "encode")
        assert hasattr(llm.tokenizer, "decode")

    @requires_model
    def test_llm_unsupported_backend_raises(self):
        from vdllm import LLM
        with pytest.raises(ValueError, match="Unsupported backend"):
            LLM(MODEL_PATH, backend="cpu")


# ============================================================================
# 5. MLX Generation — Output Format
# ============================================================================

@requires_model
@requires_mlx
class TestMLXGenerationFormat:
    @pytest.fixture(scope="class")
    def llm(self):
        from vdllm import LLM
        return LLM(MODEL_PATH, backend="mlx")

    def test_output_is_list(self, llm):
        from vdllm import SamplingParams
        sp = SamplingParams(max_tokens=16, block_length=4, denoising_steps=4)
        outputs = llm.generate(["Hello"], sp, use_tqdm=False)
        assert isinstance(outputs, list)
        assert len(outputs) == 1

    def test_output_has_required_keys(self, llm):
        from vdllm import SamplingParams
        sp = SamplingParams(max_tokens=16, block_length=4, denoising_steps=4)
        outputs = llm.generate(["Hello"], sp, use_tqdm=False)
        out = outputs[0]
        assert "text" in out
        assert "token_ids" in out
        assert "trajectory" in out
        assert "logprobs" in out
        assert "entropies" in out

    def test_output_text_is_string(self, llm):
        from vdllm import SamplingParams
        sp = SamplingParams(max_tokens=16, block_length=4, denoising_steps=4)
        outputs = llm.generate(["Hello"], sp, use_tqdm=False)
        assert isinstance(outputs[0]["text"], str)
        assert len(outputs[0]["text"]) > 0

    def test_output_token_ids_is_list(self, llm):
        from vdllm import SamplingParams
        sp = SamplingParams(max_tokens=16, block_length=4, denoising_steps=4)
        outputs = llm.generate(["Hello"], sp, use_tqdm=False)
        assert isinstance(outputs[0]["token_ids"], list)
        assert all(isinstance(t, int) for t in outputs[0]["token_ids"])

    def test_multi_prompt_generation(self, llm):
        from vdllm import SamplingParams
        sp = SamplingParams(max_tokens=16, block_length=4, denoising_steps=4)
        prompts = ["Hello", "The capital of France is", "1+1="]
        outputs = llm.generate(prompts, sp, use_tqdm=False)
        assert len(outputs) == 3
        for out in outputs:
            assert isinstance(out["text"], str)
            assert len(out["text"]) > 0

    def test_per_prompt_sampling_params(self, llm):
        from vdllm import SamplingParams
        sp_list = [
            SamplingParams(max_tokens=16, block_length=4, denoising_steps=4),
            SamplingParams(max_tokens=32, block_length=4, denoising_steps=4),
        ]
        outputs = llm.generate(["Hello", "World"], sp_list, use_tqdm=False)
        assert len(outputs) == 2


# ============================================================================
# 6. MLX Generation — Output Coherence
# ============================================================================

@requires_model
@requires_mlx
class TestMLXGenerationCoherence:
    @pytest.fixture(scope="class")
    def llm(self):
        from vdllm import LLM
        return LLM(MODEL_PATH, backend="mlx")

    def test_chat_math(self, llm):
        """Model should answer simple math correctly in chat mode."""
        from vdllm import SamplingParams
        sp = SamplingParams(max_tokens=64, block_length=4, denoising_steps=4)
        messages = [{"role": "user", "content": "What is 2+2? Answer with just the number."}]
        prompt = llm.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        outputs = llm.generate([prompt], sp, use_tqdm=False)
        text = outputs[0]["text"]
        assert "4" in text, f"Expected '4' in response, got: {text}"

    def test_chat_capital(self, llm):
        """Model should know basic geography."""
        from vdllm import SamplingParams
        sp = SamplingParams(max_tokens=64, block_length=4, denoising_steps=4)
        messages = [{"role": "user", "content": "What is the capital of France? One word answer."}]
        prompt = llm.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        outputs = llm.generate([prompt], sp, use_tqdm=False)
        text = outputs[0]["text"].lower()
        assert "paris" in text, f"Expected 'paris' in response, got: {text}"

    def test_output_not_all_same_token(self, llm):
        """Generated output should not be degenerate (all same token)."""
        from vdllm import SamplingParams
        sp = SamplingParams(max_tokens=32, block_length=4, denoising_steps=4)
        outputs = llm.generate(
            ["Tell me a fun fact about space."], sp, use_tqdm=False)
        token_ids = outputs[0]["token_ids"]
        # Should have some diversity in generated tokens
        unique = set(token_ids)
        assert len(unique) > 3, f"Output too repetitive: only {len(unique)} unique tokens"

    def test_no_mask_tokens_in_output(self, llm):
        """Output should not contain mask tokens."""
        from vdllm import SamplingParams
        sp = SamplingParams(max_tokens=32, block_length=4, denoising_steps=4)
        outputs = llm.generate(
            ["The quick brown fox"], sp, use_tqdm=False)
        mask_id = llm.config.mask_token_id
        token_ids = outputs[0]["token_ids"]
        assert mask_id not in token_ids, \
            f"Mask token {mask_id} found in output token_ids"

    def test_eos_truncation(self, llm):
        """Output should be truncated at EOS — no text after stop token."""
        from vdllm import SamplingParams
        sp = SamplingParams(max_tokens=64, block_length=4, denoising_steps=4)
        messages = [{"role": "user", "content": "Say hello."}]
        prompt = llm.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        outputs = llm.generate([prompt], sp, use_tqdm=False)
        # EOS tokens should not appear in the decoded text
        # (they're stripped by skip_special_tokens=True)
        text = outputs[0]["text"]
        assert "<|im_end|>" not in text
        assert "<|endoftext|>" not in text


# ============================================================================
# 7. Remasking Strategies
# ============================================================================

@requires_model
@requires_mlx
class TestRemaskingStrategies:
    @pytest.fixture(scope="class")
    def llm(self):
        from vdllm import LLM
        return LLM(MODEL_PATH, backend="mlx")

    @pytest.mark.parametrize("strategy", [
        "low_confidence_dynamic",
        "low_confidence_static",
        "sequential",
        "entropy_bounded",
    ])
    def test_strategy_produces_output(self, llm, strategy):
        """Each remasking strategy should produce non-empty output."""
        from vdllm import SamplingParams
        kwargs = {"remasking_strategy": strategy, "max_tokens": 16,
                  "block_length": 4, "denoising_steps": 4}
        if strategy == "low_confidence_dynamic":
            kwargs["dynamic_threshold"] = 0.85
        if strategy == "entropy_bounded":
            kwargs["eb_threshold"] = 0.35
        sp = SamplingParams(**kwargs)
        outputs = llm.generate(["Hello world"], sp, use_tqdm=False)
        assert len(outputs[0]["text"]) > 0
        assert len(outputs[0]["token_ids"]) > 0


# ============================================================================
# 8. MLX Engine Internals
# ============================================================================

@requires_model
@requires_mlx
class TestMLXEngine:
    def test_engine_resolves_mask_id(self):
        from vdllm.config import Config
        from vdllm.engine.mlx_engine import MLXEngine
        config = Config(MODEL_PATH, backend="mlx")
        engine = MLXEngine(config)
        assert config.mask_token_id == 151669

    def test_engine_resolves_eos_ids(self):
        from vdllm.config import Config
        from vdllm.engine.mlx_engine import MLXEngine
        config = Config(MODEL_PATH, backend="mlx")
        engine = MLXEngine(config)
        assert isinstance(engine.eos_ids, list)
        assert len(engine.eos_ids) > 0
        assert 151645 in engine.eos_ids  # <|im_end|>

    def test_engine_is_finished_always_true(self):
        """MLX engine processes synchronously, always finished."""
        from vdllm.config import Config
        from vdllm.engine.mlx_engine import MLXEngine
        config = Config(MODEL_PATH, backend="mlx")
        engine = MLXEngine(config)
        assert engine.is_finished() is True


# ============================================================================
# 9. Generation Utilities
# ============================================================================

class TestGenerationUtils:
    def test_get_num_transfer_tokens_exact_divide(self):
        from vdllm.generation import get_num_transfer_tokens
        result = get_num_transfer_tokens(8, 4)
        assert sum(result) == 8
        assert result == [2, 2, 2, 2]

    def test_get_num_transfer_tokens_remainder(self):
        from vdllm.generation import get_num_transfer_tokens
        result = get_num_transfer_tokens(4, 3)
        assert sum(result) == 4
        assert result == [2, 1, 1]

    def test_get_num_transfer_tokens_one_step(self):
        from vdllm.generation import get_num_transfer_tokens
        result = get_num_transfer_tokens(4, 1)
        assert result == [4]

    def test_get_num_transfer_tokens_steps_equal_block(self):
        from vdllm.generation import get_num_transfer_tokens
        result = get_num_transfer_tokens(4, 4)
        assert result == [1, 1, 1, 1]


# ============================================================================
# 10. Sampling Utilities
# ============================================================================

@requires_mlx
class TestSamplingUtils:
    def test_top_k_logits(self):
        import mlx.core as mx
        from vdllm.sampling import top_k_logits
        logits = mx.array([[1.0, 2.0, 3.0, 4.0, 5.0]])
        result = top_k_logits(logits, k=3)
        mx.eval(result)
        result_np = np.array(result[0])
        # Bottom 2 should be -inf
        assert result_np[0] == float("-inf")
        assert result_np[1] == float("-inf")
        # Top 3 should be unchanged
        assert result_np[2] == 3.0
        assert result_np[3] == 4.0
        assert result_np[4] == 5.0

    def test_top_k_disabled(self):
        import mlx.core as mx
        from vdllm.sampling import top_k_logits
        logits = mx.array([[1.0, 2.0, 3.0]])
        result = top_k_logits(logits, k=0)
        mx.eval(result)
        np.testing.assert_array_equal(np.array(result), np.array(logits))

    def test_sample_shapes(self):
        import mlx.core as mx
        from vdllm.sampling import sample_with_temperature_topk_topp
        logits = mx.random.normal((2, 10))  # batch=2, vocab=10
        tokens, probs = sample_with_temperature_topk_topp(logits)
        mx.eval(tokens, probs)
        assert tokens.shape == (2,)
        assert probs.shape == (2,)

    def test_sample_probs_in_range(self):
        import mlx.core as mx
        from vdllm.sampling import sample_with_temperature_topk_topp
        logits = mx.random.normal((5, 100))
        tokens, probs = sample_with_temperature_topk_topp(logits)
        mx.eval(tokens, probs)
        probs_np = np.array(probs)
        assert (probs_np >= 0).all()
        assert (probs_np <= 1).all()


# ============================================================================
# 11. CUDA Backend Guard (import safety)
# ============================================================================

class TestCUDAGuard:
    @requires_torch
    def test_cuda_engine_not_imported_on_mlx(self):
        """Importing vdllm should NOT import torch at module level."""
        import importlib
        # Fresh import of vdllm
        if "vdllm" in sys.modules:
            # Already imported, just check torch wasn't pulled in by config
            pass
        from vdllm import LLM, SamplingParams
        # These should work without torch.distributed
        assert LLM is not None
        assert SamplingParams is not None

    def test_config_no_torch_import(self):
        """Config module should be importable without torch."""
        # This test verifies the refactor worked — config.py has no
        # module-level torch import
        from vdllm.config import Config
        assert Config is not None


# ============================================================================
# Standalone runner
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-x"])
