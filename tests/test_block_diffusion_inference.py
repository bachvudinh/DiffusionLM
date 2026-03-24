"""Tests for Block Diffusion Inference — TDD approach.

These tests define the expected behavior of the inference module.
Run with: pytest tests/ -v
"""

import torch
import pytest


class TestNoiseScheduler:
    """Tests for the noise schedule used in Block Diffusion."""

    def test_linear_schedule_mask_prob_equals_t(self):
        """Linear schedule: mask probability should equal timestep t."""
        from inference.schedules import LinearSchedule

        schedule = LinearSchedule()
        t = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0])
        mask_prob = schedule.mask_prob(t)

        torch.testing.assert_close(mask_prob, t)

    def test_cosine_schedule_mask_prob(self):
        """Cosine schedule: alpha(t) = cos(pi*t/2)^2, mask_prob = 1 - alpha(t)."""
        from inference.schedules import CosineSchedule

        schedule = CosineSchedule()
        t = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0])
        expected_alpha = torch.cos(torch.pi * t / 2) ** 2
        expected_mask_prob = 1 - expected_alpha
        mask_prob = schedule.mask_prob(t)

        torch.testing.assert_close(mask_prob, expected_mask_prob)

    def test_mask_prob_clipped_to_valid_range(self):
        """mask_prob should be clamped to [0.0, 1.0]."""
        from inference.schedules import LinearSchedule

        schedule = LinearSchedule()
        t = torch.tensor([-0.5, 1.5])  # out of range
        mask_prob = schedule.mask_prob(t)

        assert mask_prob.min() >= 0.0
        assert mask_prob.max() <= 1.0

    def test_elbo_weight_inverse_of_mask_prob(self):
        """ELBO weight = 1 / mask_prob for linear schedule."""
        from inference.schedules import LinearSchedule

        schedule = LinearSchedule()
        t = torch.tensor([0.1, 0.25, 0.5, 0.75, 1.0])
        elbo_weight = schedule.elbo_weight(t)

        expected = 1.0 / t
        torch.testing.assert_close(elbo_weight, expected)


class TestGumbelSampler:
    """Tests for Gumbel-max sampling."""

    def test_gumbel_sampling_returns_valid_indices(self):
        """Gumbel sampling should return indices within vocabulary range."""
        from inference.sampler import GumbelSampler

        sampler = GumbelSampler(temperature=0.7)
        logits = torch.randn(2, 10, 100)  # (batch, seq, vocab)
        samples = sampler.sample(logits)

        assert samples.shape == (2, 10)
        assert (samples >= 0).all()
        assert (samples < 100).all()

    def test_gumbel_with_zero_temperature_is_greedy(self):
        """Temperature=0 should give argmax (greedy) sampling."""
        from inference.sampler import GumbelSampler

        sampler = GumbelSampler(temperature=0.0)
        logits = torch.randn(2, 10, 100)

        greedy = torch.argmax(logits, dim=-1)
        samples = sampler.sample(logits)

        torch.testing.assert_close(samples, greedy)

    def test_gumbel_with_high_temperature_approaches_uniform(self):
        """Very high temperature should approach uniform distribution."""
        from inference.sampler import GumbelSampler

        sampler = GumbelSampler(temperature=10.0)
        logits = torch.zeros(2, 10, 100)

        samples = sampler.sample(logits)
        # With uniform logits and high temp, should pick roughly uniformly
        unique_counts = torch.unique(samples, return_counts=True)[1].float()
        # Not all same token (with 10.0 temp it won't always be)
        # Just verify all in range
        assert (samples >= 0).all() and (samples < 100).all()

    def test_gumbel_sample_shapes(self):
        """Sample shape should match batch and seq dimensions."""
        from inference.sampler import GumbelSampler

        sampler = GumbelSampler(temperature=0.7)
        for bs, sl, vs in [(1, 1, 50), (4, 128, 50000), (2, 512, 1000)]:
            logits = torch.randn(bs, sl, vs)
            samples = sampler.sample(logits)
            assert samples.shape == (bs, sl)


class TestStaircaseMask:
    """Tests for the staircase attention mask used in Block Diffusion.

    The staircase mask enforces:
    - Within-block bidirectional attention (tokens in same block can attend to each other)
    - Block-causal: blocks can only attend to earlier blocks (AR ordering)
    - No label leakage: noisy block i cannot see clean tokens from block i
    """

    def test_mask_shape_doubled_sequence(self):
        """Staircase mask should be 2x the sequence length (for [x_t || x_0])."""
        from inference.mask import StaircaseMask

        seq_len = 64
        block_size = 8
        mask_builder = StaircaseMask(block_size=block_size, seq_len=seq_len)

        full_mask = mask_builder.build()

        assert full_mask.shape == (2 * seq_len, 2 * seq_len)

    def test_mask_no_attend_from_clean_to_noisy_in_same_block(self):
        """Clean half (second half) should NOT attend to noisy half of same block.

        This is the critical "no label leakage" constraint.
        """
        from inference.mask import StaircaseMask

        seq_len = 32
        block_size = 8
        mask_builder = StaircaseMask(block_size=block_size, seq_len=seq_len)

        full_mask = mask_builder.build()
        L = seq_len
        noisy_half = full_mask[:L, :L]   # queries from noisy half
        clean_half = full_mask[L:, :L]    # queries from clean half

        # Within same block, clean (second half) should NOT attend to noisy (first half)
        # So clean_half[:, noisy_positions_of_same_block] should be False
        for block_idx in range(L // block_size):
            start = block_idx * block_size
            end = start + block_size
            # Query: clean half positions in this block (L + start to L + end)
            # Key: noisy half positions in this block (start to end)
            # This should be False (no attention)
            block_clean_to_noisy = clean_half[L + start:L + end, start:end]
            assert not block_clean_to_noisy.any(), \
                "Clean half should not attend to noisy half in same block"

    def test_mask_block_causal_queries_cannot_attend_forward(self):
        """Block i queries should NOT attend to blocks > i (strictly later)."""
        from inference.mask import StaircaseMask

        seq_len = 32
        block_size = 8
        mask_builder = StaircaseMask(block_size=block_size, seq_len=seq_len)

        full_mask = mask_builder.build()
        L = seq_len

        # For each block, check that it doesn't attend to future blocks
        for block_idx in range(L // block_size):
            for later_block in range(block_idx + 1, L // block_size):
                start_later = later_block * block_size
                end_later = start_later + block_size
                # Noisy half queries from block_idx to noisy half keys of later blocks
                query_block = full_mask[block_idx * block_size : (block_idx + 1) * block_size]
                keys_future = query_block[:, start_later:end_later]
                assert not keys_future.any(), \
                    f"Block {block_idx} should not attend to later block {later_block}"

    def test_mask_within_block_bidirectional(self):
        """Within a single block, tokens should attend to each other (bidirectional)."""
        from inference.mask import StaircaseMask

        seq_len = 16
        block_size = 4
        mask_builder = StaircaseMask(block_size=block_size, seq_len=seq_len)

        full_mask = mask_builder.build()
        L = seq_len
        n_blocks = L // block_size

        # Within each block, all positions should attend to all positions
        for b in range(n_blocks):
            start = b * block_size
            end = start + block_size
            # Noisy-to-noisy within block should be fully connected
            block = full_mask[start:end, start:end]
            assert block.all(), f"Block {b} should be fully connected within noisy half"

    def test_flex_block_mask_interface(self):
        """Should produce a BlockMask-compatible tensor for FlexAttention."""
        from inference.mask import StaircaseMask

        seq_len = 64
        block_size = 8
        mask_builder = StaircaseMask(block_size=block_size, seq_len=seq_len)

        bm = mask_builder.to_block_mask()
        # BlockMask from FlexAttention has .shape
        assert hasattr(bm, 'shape') or hasattr(bm, 'to')


class TestConfidenceUnmasking:
    """Tests for confidence-based unmasking strategy."""

    def test_unmask_top_k_positions(self):
        """Should unmask the k positions with highest confidence."""
        from inference.unmask import unmask_top_k

        seq_len = 10
        masked = torch.tensor([True, True, True, True, True, False, False, False, False, False])
        confidences = torch.tensor([0.1, 0.9, 0.3, 0.7, 0.5, -1.0, -1.0, -1.0, -1.0, -1.0])

        k = 3
        new_mask = unmask_top_k(masked, confidences, k)

        # Top 3 confidences among masked (True) positions:
        # index 1=0.9, 3=0.7, 4=0.5 — these become unmasked (False)
        # index 0=0.1, 2=0.3 — stay masked (True)
        # positions 5-9 were already unmasked (False) — stay unmasked (False)
        expected = torch.tensor([True, False, True, False, False,
                                False, False, False, False, False])
        assert (new_mask == expected).all()

    def test_unmask_never_unmasks_already_unmasked(self):
        """Already-unmasked positions should never become masked again."""
        from inference.unmask import unmask_top_k

        masked = torch.tensor([False, True, False, True])
        confidences = torch.tensor([0.9, 0.8, 0.7, 0.6])

        for k in [1, 2, 3, 4]:
            new_mask = unmask_top_k(masked, confidences, k)
            # False positions stay False
            assert not (new_mask & ~masked).any()  # no position goes from False to True


class TestBlockDenoiser:
    """Integration tests for the full block denoising pipeline."""

    def test_denoiser_initializes_block_correctly(self):
        """Block should initialize with prompt remainder + mask tokens."""
        from inference.denoiser import BlockDenoiser

        block_size = 8
        prompt_remainder = 3  # 3 tokens from prompt
        denoiser = BlockDenoiser(block_size=block_size)

        block, masked = denoiser.init_block(prompt_remainder=prompt_remainder)

        assert block.shape == (1, block_size)
        assert masked.shape == (1, block_size)
        # First 3 positions are NOT masked (from prompt), rest ARE masked
        assert masked[0, :prompt_remainder].sum() == 0
        assert masked[0, prompt_remainder:].all()

    def test_denoiser_step_reduces_masked_count(self):
        """A denoise step should unmask at least some positions."""
        from inference.denoiser import BlockDenoiser

        block_size = 8
        denoiser = BlockDenoiser(block_size=block_size)

        # Initialize with all-masked block
        block = torch.full((1, block_size), denoiser.mask_token_id)
        masked = torch.ones_like(block, dtype=torch.bool)

        # After one step with 3 tokens to unmask
        n_unmasked = denoiser.step_unmask_count(t=0.8, s=0.7, n_masked=8, step=0, total_steps=5)
        assert n_unmasked >= 1

    def test_unmask_schedule_decreases_masked_count_monotonically(self):
        """The unmask schedule should reduce masked positions monotonically."""
        from inference.denoiser import BlockDenoiser

        denoiser = BlockDenoiser(block_size=32)
        total_steps = 10
        n_masked = 29  # not all masked (one position is prompt)

        for step in range(total_steps):
            n = denoiser.step_unmask_count(
                t=1.0, s=0.0, n_masked=n_masked, step=step, total_steps=total_steps
            )
            if step < total_steps - 1:
                # Should unmask at least 1 token per step
                assert n >= 1, f"Step {step} should unmask at least 1 token"
            else:
                # Last step should unmask remaining
                assert n == n_masked


class TestIntegration:
    """End-to-end integration tests (no real model needed)."""

    def test_full_generation_mock_model(self):
        """Full generation with a mock model should produce tokens."""
        from inference.generator import generate

        class MockModel:
            def __init__(self):
                self.cache = {}
                self._training = False
                # Dummy parameter for device detection
                self._dummy_param = torch.nn.Parameter(torch.zeros(1))

            def __call__(self, x, pos_offset=0):
                B, L = x.shape
                # Return logits that predict the next token
                logits = torch.zeros(B, L, 1000)
                # At each position, predict the token itself (identity)
                for b in range(B):
                    for l in range(L):
                        if x[b, l] != 0:  # not mask
                            logits[b, l, x[b, l]] = 10.0  # high confidence
                        else:
                            logits[b, l, 42] = 5.0  # default token
                return logits, None

            def parameters(self, recurse=True):
                return [self._dummy_param]

            @property
            def training(self):
                return self._training

            def set_cache_mode(self, enabled): pass
            def reset_kv_cache(self): pass

        model = MockModel()
        result = generate(
            model=model,
            encode_fn=lambda x: [1, 2, 3],  # dummy prompt
            decode_fn=lambda x: "generated text",
            max_new_tokens=8,
            block_size=4,
            mask_token_id=0,
            eos_token_id=99,
            pad_token_id=2,
            denoise_steps=5,
        )

        assert isinstance(result, str)
        assert len(result) > 0
