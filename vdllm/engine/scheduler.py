"""Scheduler for block diffusion inference with batched remasking strategies.

Derived from JetEngine by Yihan Bian et al.
Reference: https://github.com/Labman42/JetEngine

Architecture Overview
====================

    The Scheduler orchestrates the block diffusion inference loop:
    deciding which sequences to prefill, which to denoise, and
    applying remasking strategies after each denoise step.

    ┌──────────────────────────────────────────────────────────────────┐
    │                      Scheduler Pipeline                          │
    │                                                                  │
    │  schedule() — called each engine step                            │
    │    1. Release finished sequences, free KV cache blocks           │
    │    2. Promote waiting sequences to running (allocate blocks)     │
    │    3. Build prefill batch (newly added sequences)                │
    │    4. Build denoise batch (sequences in DENOISING/SAVING state)  │
    │    5. Return ScheduleResult(prefill=[], denoise=[])              │
    │                                                                  │
    │  postprocess() / postprocess_unify() — after model forward       │
    │    ┌──────────────────────────────────────────────────────────┐  │
    │    │  For PREFILL sequences:                                  │  │
    │    │    Mark as DENOISING, record cached token count          │  │
    │    │                                                          │  │
    │    │  For DENOISE sequences (batched tensor operations):      │  │
    │    │    1. Compute probs from logits (FlashInfer LogitsPipe)  │  │
    │    │    2. Sample tokens: top_k_top_p_sampling                │  │
    │    │    3. Apply remasking strategy to select transfer_index  │  │
    │    │    4. Update intermediate_block_tokens where transferred │  │
    │    │    5. Track trajectory, logprobs, entropies              │  │
    │    │    6. If block fully denoised → SAVING → commit_block    │  │
    │    │    7. If SAVING: commit block, start new block           │  │
    │    └──────────────────────────────────────────────────────────┘  │
    │                                                                  │
    │  Remasking Strategies (all batched over sequences):              │
    │  ┌──────────────────────────────────────────────────────────┐    │
    │  │  sequential:           unmask left-to-right              │    │
    │  │  low_confidence_static: unmask highest-confidence masks  │    │
    │  │  low_confidence_dynamic: confidence > threshold, fallback│    │
    │  │  entropy_bounded:      unmask by cumulative entropy      │    │
    │  │  random:               unmask random masked positions    │    │
    │  │                                                          │    │
    │  │  Visualized (block_length=8, step transfers 3 tokens):   │    │
    │  │                                                          │    │
    │  │  Before: [tok M M tok M M M M]  (M = mask_token)        │    │
    │  │                                                          │    │
    │  │  sequential:    [tok X X tok M M M M]  (left-to-right)   │    │
    │  │  confidence:    [tok M M tok X M X X]  (highest conf)    │    │
    │  │  random:        [tok M X tok M X M X]  (random pick)     │    │
    │  │                                                          │    │
    │  │  X = newly unmasked (transferred from x0 prediction)     │    │
    │  └──────────────────────────────────────────────────────────┘    │
    └──────────────────────────────────────────────────────────────────┘
"""

from collections import deque
from dataclasses import dataclass
import torch
from torch.nn import functional as F
import numpy as np
import random

from vdllm.config import Config
from vdllm.engine.sequence import Sequence, SequenceStatus, RunType
from vdllm.engine.block_manager import BlockManager
from vdllm.layers.sampler import sample_with_temperature_topk_topp
from flashinfer.logits_processor import LogitsPipe, Temperature, Softmax, TopP, TopK, Sample
from flashinfer.sampling import top_p_sampling_from_probs, top_k_top_p_sampling_from_probs
from torch.distributions import Categorical

EPS = 1e-12


@dataclass
class ScheduleResult:
    """Result of a scheduling step.

    Attributes:
        prefill: list of sequences to prefill (encode full context)
        denoise: list of sequences to denoise (refine current block)
    """
    prefill: list[Sequence]
    denoise: list[Sequence]

    @property
    def has_work(self) -> bool:
        return bool(self.prefill or self.denoise)


class Scheduler:
    """Manages sequence lifecycle and batched postprocessing for block diffusion.

    Input:
        config: Config — engine configuration with max_num_seqs,
                max_num_batched_tokens, mask_token_id, etc.

    Key State:
        running:        list — currently active sequences
        waiting_prefill: deque — sequences waiting for capacity
        prefill_ready:   deque — sequences ready to prefill this step
        block_manager:   BlockManager — KV cache block allocator
    """

    def __init__(self, config: Config):
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.eos = config.eos
        self.mask_token_id = config.mask_token_id
        self.diversity_enforce = config.diversity_enforce
        self.epsilon_greedy = config.epsilon_greedy
        self.epsilon = config.epsilon
        self.barrier = config.diversity_enforce_barrier
        self.block_manager = BlockManager(
            config.num_kvcache_blocks, config.kvcache_block_size)
        self.running: list[Sequence] = []
        self.waiting_prefill: deque[Sequence] = deque()
        self.prefill_ready: deque[Sequence] = deque()
        self.sample_pipe = LogitsPipe([
            Temperature(),
            Softmax(),
        ])

    def apply_repetition_penalty(self, probs: torch.Tensor,
                                  seqs: list[Sequence]):
        """Apply repetition penalty to token probabilities.

        Input:
            probs: (batch_size, block_len, vocab_size) — token probabilities
            seqs:  list of Sequence — for token_counts and penalty values

        Output:
            probs modified in-place
        """
        for i, seq in enumerate(seqs):
            if seq.repetition_penalty == 1.0 or seq.num_tokens <= seq.max_tokens // 2:
                continue
            penalty = seq.repetition_penalty
            seen_tokens = list(seq.token_counts.keys())
            if seen_tokens:
                probs[i, :, seen_tokens] /= penalty
            probs[i] /= probs[i].sum(dim=-1, keepdim=True)

    def add(self, seq: Sequence):
        """Add a sequence to the scheduler.

        Input:
            seq: Sequence — new generation request

        Tries to add directly to running; falls back to waiting queue.
        """
        if not self._try_add_to_running(seq):
            self.waiting_prefill.append(seq)

    def _try_add_to_running(self, seq: Sequence) -> bool:
        if len(self.running) >= self.max_num_seqs:
            return False
        if seq.is_finished:
            return False
        if not self.block_manager.can_allocate(seq):
            return False
        self.block_manager.allocate(seq)
        if seq.status == SequenceStatus.WAITING:
            seq.status = SequenceStatus.PREFILLING
            self.prefill_ready.append(seq)
        self.running.append(seq)
        return True

    def _release_finished_sequences(self):
        if not self.running:
            return
        finished = [seq for seq in self.running if seq.is_finished]
        if not finished:
            return
        for seq in finished:
            self.block_manager.deallocate(seq)
        self.running = [seq for seq in self.running if not seq.is_finished]
        self.prefill_ready = deque(
            seq for seq in self.prefill_ready if not seq.is_finished)

    def _fill_slots_from_waiting(self):
        while self.waiting_prefill and len(self.running) < self.max_num_seqs:
            seq = self.waiting_prefill[0]
            if self._try_add_to_running(seq):
                self.waiting_prefill.popleft()
            else:
                break

    def is_finished(self):
        return (not self.running and not self.waiting_prefill
                and not self.prefill_ready)

    def _prepare_prefill_batch(self) -> list[Sequence]:
        batch: list[Sequence] = []
        while self.prefill_ready:
            seq = self.prefill_ready.popleft()
            if seq.is_finished:
                if seq.block_table:
                    self.block_manager.deallocate(seq)
                continue
            if seq.status not in (SequenceStatus.PREFILLING, SequenceStatus.WAITING):
                continue
            seq.status = SequenceStatus.PREFILLING
            batch.append(seq)
        return batch

    def _prepare_denoise_batch(self, prefill_batch: list[Sequence]) -> list[Sequence]:
        batch: list[Sequence] = []
        prefill_ids = {seq.seq_id for seq in prefill_batch}

        requests: list[tuple[Sequence, int]] = []
        total_needed = 0
        available_blocks = len(self.block_manager.free_block_ids)

        for seq in self.running:
            if seq.seq_id in prefill_ids:
                continue
            if seq.status not in (SequenceStatus.DENOISING, SequenceStatus.SAVING):
                continue

            num_new_blocks = seq.num_new_blocks_needed(self.block_manager.block_size)
            if num_new_blocks > 0:
                if total_needed + num_new_blocks > available_blocks:
                    print(f"[Warning] Cannot append {num_new_blocks} blocks "
                          f"for seq {seq.seq_id}. Not enough memory.")
                    continue
                requests.append((seq, num_new_blocks))
                total_needed += num_new_blocks

            batch.append(seq)

        if requests:
            self.block_manager.append_blocks_batch(requests)

        return batch

    def schedule(self) -> ScheduleResult:
        """Run one scheduling step.

        Output:
            ScheduleResult with prefill and denoise batches
        """
        self._release_finished_sequences()
        self._fill_slots_from_waiting()

        prefill_batch = self._prepare_prefill_batch()
        denoise_batch = self._prepare_denoise_batch(prefill_batch)

        if not prefill_batch and not denoise_batch:
            if self.running or self.waiting_prefill:
                print("[Warning] Scheduler idle: no batches prepared "
                      "despite pending sequences.")
            return ScheduleResult(prefill=[], denoise=[])

        return ScheduleResult(prefill=prefill_batch, denoise=denoise_batch)

    def postprocess_unify(self, seqs: list[Sequence], logits: torch.Tensor,
                          run_type: RunType) -> list[Sequence]:
        """Unified batched postprocessing (consistent sampling params).

        Used when all sequences share the same sampling parameters.
        All remasking strategies are applied in batched tensor operations.

        Input:
            seqs:     list of Sequence
            logits:   (batch * block_len, vocab_size) — model output
            run_type: RunType.PREFILL or RunType.DENOISE

        Output:
            list of finished Sequence objects
        """
        if run_type == RunType.PREFILL:
            for seq in seqs:
                seq.num_cached_tokens = seq.num_prefill_tokens
                seq.status = SequenceStatus.DENOISING
            return []

        if run_type != RunType.DENOISE or not seqs:
            return []

        device = logits.device
        batch_size = len(seqs)
        block_len = seqs[0].block_length

        probs = self.sample_pipe(
            logits, temperature=seqs[0].temperature
        ).view(batch_size, block_len, -1)
        self.apply_repetition_penalty(probs, seqs)
        entropies_all = -(probs.clamp_min(EPS) * (probs.clamp_min(EPS)).log()).sum(dim=-1)

        flat_probs = probs.view(-1, probs.shape[-1])
        flat_samples = top_k_top_p_sampling_from_probs(
            flat_probs, top_k=seqs[0].top_k, top_p=seqs[0].top_p
        ).to(torch.int64)
        batch_seq_x0 = flat_samples.view(batch_size, block_len)
        batch_seq_x0_p = torch.gather(
            flat_probs, 1, flat_samples.unsqueeze(-1)
        ).view(batch_size, block_len)
        batch_seq_x0_logp = torch.log(batch_seq_x0_p.clamp_min(EPS))

        batch_current_tokens = torch.stack(
            [seq.intermediate_block_tokens.to(device) for seq in seqs])
        batch_logprobs = torch.stack(
            [seq.block_logprobs.to(device) for seq in seqs])
        batch_entropies = torch.stack(
            [seq.block_entropies.to(device) for seq in seqs])
        batch_trajectory = torch.stack(
            [seq.block_trajectory.to(device) for seq in seqs])

        batch_global_step_plus_1 = torch.tensor(
            [seq.global_denoising_step + 1 for seq in seqs],
            device=device, dtype=torch.long).unsqueeze(1)

        denoising_mask_bool = torch.tensor(
            [seq.status == SequenceStatus.DENOISING for seq in seqs], device=device)
        saving_mask_bool = torch.tensor(
            [seq.status == SequenceStatus.SAVING for seq in seqs], device=device)
        denoising_mask = denoising_mask_bool.unsqueeze(1)

        num_to_transfer_list = [
            seq.num_transfer_tokens_per_step[seq.current_denoising_step]
            if seq.status == SequenceStatus.DENOISING else 0
            for seq in seqs
        ]
        batch_num_to_transfer = torch.tensor(
            num_to_transfer_list, device=device, dtype=torch.long)

        mask_token_mask = (batch_current_tokens == self.mask_token_id) & denoising_mask
        mask_available = mask_token_mask.any(dim=1)
        effective_num_to_transfer = torch.where(
            mask_available, batch_num_to_transfer,
            torch.zeros_like(batch_num_to_transfer))

        strategy = seqs[0].remasking_strategy
        transfer_index = torch.zeros(
            (batch_size, block_len), dtype=torch.bool, device=device)

        if strategy == "sequential":
            range_tensor = torch.arange(block_len, device=device).unsqueeze(0)
            first_mask_pos = torch.where(
                mask_available,
                torch.argmax(mask_token_mask.int(), dim=1),
                torch.full_like(effective_num_to_transfer, block_len))
            start = first_mask_pos.unsqueeze(1)
            end = (start + effective_num_to_transfer.unsqueeze(1)).clamp_max(block_len)
            seq_transfer_index = (range_tensor >= start) & (range_tensor < end)
            transfer_index = seq_transfer_index & mask_token_mask

        elif strategy == "low_confidence_static":
            confidence = torch.where(
                mask_token_mask, batch_seq_x0_p,
                torch.full_like(batch_seq_x0_p, -torch.inf))
            max_k = int(effective_num_to_transfer.max().item())
            if max_k > 0:
                _, top_indices = torch.topk(confidence, k=max_k, dim=1)
                k_mask = (torch.arange(max_k, device=device).unsqueeze(0)
                          < effective_num_to_transfer.unsqueeze(1))
                transfer_index = torch.zeros_like(confidence, dtype=torch.bool)
                transfer_index.scatter_(1, top_indices, k_mask)
                transfer_index &= mask_token_mask

        elif strategy == "low_confidence_dynamic":
            confidence = torch.where(
                mask_token_mask, batch_seq_x0_p,
                torch.full_like(batch_seq_x0_p, -torch.inf))
            dyn_transfer_index = confidence > seqs[0].dynamic_threshold
            dyn_transfer_index &= mask_token_mask
            num_transferred_dyn = dyn_transfer_index.sum(dim=1)
            needs_fallback = ((num_transferred_dyn < effective_num_to_transfer)
                              & mask_available & denoising_mask_bool)
            if needs_fallback.any():
                fallback_mask = mask_token_mask[needs_fallback]
                fallback_num = effective_num_to_transfer[needs_fallback]
                range_tensor = torch.arange(block_len, device=device).unsqueeze(0)
                first_mask_pos = torch.argmax(
                    fallback_mask.int(), dim=1).unsqueeze(1)
                end = (first_mask_pos + fallback_num.unsqueeze(1)).clamp_max(block_len)
                fallback_index = ((range_tensor >= first_mask_pos)
                                  & (range_tensor < end))
                fallback_index &= fallback_mask
                dyn_transfer_index[needs_fallback] = fallback_index
            transfer_index = dyn_transfer_index

        elif strategy == "entropy_bounded":
            masked_entropies = torch.where(
                mask_token_mask, entropies_all,
                torch.full_like(entropies_all, torch.inf))
            ent_sorted, order = torch.sort(masked_entropies, dim=1)
            ent_sorted_masked = torch.where(
                torch.isfinite(ent_sorted), ent_sorted,
                torch.zeros_like(ent_sorted))
            cumsum = torch.cumsum(ent_sorted_masked, dim=1)
            thresholds = torch.full(
                (batch_size, 1), seqs[0].eb_threshold, device=device)
            k_tensor = torch.searchsorted(cumsum, thresholds, right=False)
            k_tensor = torch.where(
                (mask_available & denoising_mask_bool).unsqueeze(1),
                k_tensor.clamp_min(1),
                torch.zeros_like(k_tensor)).squeeze(1)
            max_k = int(k_tensor.max().item())
            if max_k > 0:
                k_mask = (torch.arange(max_k, device=device).unsqueeze(0)
                          < k_tensor.unsqueeze(1))
                transfer_index = torch.zeros_like(mask_token_mask)
                transfer_index.scatter_(1, order[:, :max_k], k_mask)
                transfer_index &= mask_token_mask

        elif strategy == "random":
            B, L = mask_token_mask.shape
            scores = torch.rand((B, L), device=device)
            scores = scores.masked_fill(~mask_token_mask, -1.0)
            max_k = int(effective_num_to_transfer.max().item())
            if max_k > 0:
                _, top_indices = scores.topk(max_k, dim=-1)
                k_mask = (torch.arange(max_k, device=device).unsqueeze(0)
                          < effective_num_to_transfer.unsqueeze(1))
                transfer_index = torch.zeros_like(
                    mask_token_mask, dtype=torch.bool)
                transfer_index.scatter_(1, top_indices, k_mask)
                transfer_index &= mask_token_mask
        else:
            raise ValueError(f"Unsupported remasking strategy: {strategy}")

        final_transfer_index = transfer_index & mask_token_mask

        batch_new_tokens = torch.where(
            final_transfer_index, batch_seq_x0, batch_current_tokens)
        batch_new_trajectory = torch.where(
            final_transfer_index & (batch_trajectory == 0),
            batch_global_step_plus_1, batch_trajectory)
        batch_new_logprobs = torch.where(
            final_transfer_index, batch_seq_x0_logp, batch_logprobs)
        batch_new_entropies = torch.where(
            final_transfer_index, entropies_all, batch_entropies)

        denoise_increment = denoising_mask_bool.int()
        new_denoising_steps = torch.tensor(
            [seq.current_denoising_step for seq in seqs], device=device
        ) + denoise_increment
        new_global_steps = torch.tensor(
            [seq.global_denoising_step for seq in seqs], device=device
        ) + denoise_increment

        remaining_masks = (batch_new_tokens == self.mask_token_id).any(dim=1)
        step_limits = torch.tensor(
            [seq.denoising_steps for seq in seqs], device=device)
        is_fully_denoised = (~remaining_masks) | (new_denoising_steps >= step_limits)
        is_fully_denoised &= denoising_mask_bool

        for i, seq in enumerate(seqs):
            if seq.status == SequenceStatus.DENOISING:
                seq.intermediate_block_tokens = batch_new_tokens[i]
                seq.intermediate_block_tokens_entropy = batch_new_entropies[i]
                seq.block_trajectory = batch_new_trajectory[i]
                seq.block_logprobs = batch_new_logprobs[i]
                seq.block_entropies = batch_new_entropies[i]
                seq.current_denoising_step = new_denoising_steps[i].item()
                seq.global_denoising_step = new_global_steps[i].item()
                seq.num_to_transfer = final_transfer_index[i].sum().item()
                if is_fully_denoised[i]:
                    seq.status = SequenceStatus.SAVING

            elif seq.status == SequenceStatus.SAVING:
                seq.commit_block(seq.intermediate_block_tokens)
                seq.num_to_transfer = 0
                if not seq.is_finished:
                    seq.start_new_block()
                else:
                    self.block_manager.deallocate(seq)

        finished_seqs = [seq for seq in self.running if seq.is_finished]
        if finished_seqs:
            for seq in finished_seqs:
                self.block_manager.deallocate(seq)
            self.running = [seq for seq in self.running if not seq.is_finished]
        if self.prefill_ready:
            self.prefill_ready = deque(
                seq for seq in self.prefill_ready if not seq.is_finished)
        return finished_seqs

    def postprocess(self, seqs: list[Sequence], logits: torch.Tensor,
                    run_type: RunType) -> list[Sequence]:
        """Full postprocessing with per-sequence sampling params and mixed strategies.

        Supports heterogeneous temperatures, top-k/top-p values, and
        different remasking strategies within the same batch.

        Input:
            seqs:     list of Sequence
            logits:   (batch * block_len, vocab_size) — model output
            run_type: RunType.PREFILL or RunType.DENOISE

        Output:
            list of finished Sequence objects
        """
        if run_type == RunType.PREFILL:
            for seq in seqs:
                seq.num_cached_tokens = seq.num_prefill_tokens
                seq.status = SequenceStatus.DENOISING
        elif run_type == RunType.DENOISE:
            device = logits.device
            batch_size = len(seqs)
            block_len = seqs[0].block_length

            # --- 1. Batched Sampling ---
            if self.consistent_sampling_params:
                probs = self.sample_pipe(
                    logits, temperature=seqs[0].temperature
                ).view(batch_size, block_len, -1)
                self.apply_repetition_penalty(probs, seqs)
            else:
                if logits.dim() == 3:
                    logits = logits.view(-1, logits.shape[-1])
                temps = torch.tensor(
                    [seq.temperature for seq in seqs],
                    device=device, dtype=torch.float)
                temps = temps.repeat_interleave(block_len).unsqueeze(1)
                logits = logits / temps
                probs = F.softmax(logits, dim=-1).view(batch_size, block_len, -1)
                self.apply_repetition_penalty(probs, seqs)

            entropies_all = -(probs.clamp_min(EPS) * (probs.clamp_min(EPS)).log()).sum(dim=-1)

            if self.consistent_sampling_params:
                batch_top_k = seqs[0].top_k
                batch_top_p = seqs[0].top_p
            else:
                batch_top_p = torch.tensor(
                    [seq.top_p for seq in seqs], device=device, dtype=torch.float)
                batch_top_k = torch.tensor(
                    [seq.top_k for seq in seqs], device=device, dtype=torch.long)

            batch_seq_x0 = top_k_top_p_sampling_from_probs(
                probs.view(-1, probs.shape[-1]),
                top_k=batch_top_k, top_p=batch_top_p
            ).to(torch.int64).view(batch_size, block_len)

            batch_seq_x0_p = torch.gather(
                probs, -1, batch_seq_x0.unsqueeze(-1)).squeeze(-1)
            batch_seq_x0_logp = torch.log(batch_seq_x0_p.clamp_min(EPS))

            batch_current_tokens = torch.stack(
                [seq.intermediate_block_tokens.to(device) for seq in seqs])
            batch_logprobs = torch.stack(
                [seq.block_logprobs.to(device) for seq in seqs])
            batch_entropies = torch.stack(
                [seq.block_entropies.to(device) for seq in seqs])
            batch_trajectory = torch.stack(
                [seq.block_trajectory.to(device) for seq in seqs])

            num_to_transfer_list = [
                seq.num_transfer_tokens_per_step[seq.current_denoising_step]
                if seq.status == SequenceStatus.DENOISING else 0
                for seq in seqs
            ]
            batch_num_to_transfer = torch.tensor(
                num_to_transfer_list, device=device, dtype=torch.long)

            batch_global_step_plus_1 = torch.tensor(
                [seq.global_denoising_step + 1 for seq in seqs],
                device=device, dtype=torch.long).unsqueeze(1)

            all_statuses = [seq.status for seq in seqs]
            denoising_mask_bool = torch.tensor(
                [s == SequenceStatus.DENOISING for s in all_statuses], device=device)
            saving_mask_bool = torch.tensor(
                [s == SequenceStatus.SAVING for s in all_statuses], device=device)

            denoising_mask = denoising_mask_bool.unsqueeze(1)

            mask_token_mask = (batch_current_tokens == self.mask_token_id) & denoising_mask

            strategies = [
                seq.remasking_strategy if status == SequenceStatus.DENOISING else ''
                for seq, status in zip(seqs, all_statuses)]
            if self.diversity_enforce:
                strategies = [
                    strategy if (seq.num_generated_tokens > self.barrier)
                    else 'sequential'
                    for strategy, seq in zip(strategies, seqs)]
            elif self.epsilon_greedy:
                strategies = [
                    'random' if random.random() < self.epsilon else strategy
                    for strategy in strategies]

            seq_mask = torch.tensor(
                [s == 'sequential' for s in strategies], device=device).unsqueeze(1)
            low_conf_static_mask = torch.tensor(
                ['low_confidence_static' in s for s in strategies], device=device).unsqueeze(1)
            low_conf_dynamic_mask = torch.tensor(
                ['low_confidence_dynamic' in s for s in strategies], device=device).unsqueeze(1)
            entropy_bounded_mask = torch.tensor(
                ['entropy_bounded' in s for s in strategies], device=device).unsqueeze(1)
            random_mask = torch.tensor(
                [s == 'random' for s in strategies], device=device).unsqueeze(1)

            transfer_index = torch.zeros(
                (batch_size, block_len), dtype=torch.bool, device=device)

            # --- Sequential strategy ---
            if seq_mask.any():
                first_mask_pos = torch.argmax(mask_token_mask.int(), dim=1, keepdim=True)
                range_tensor = torch.arange(block_len, device=device).unsqueeze(0)
                start_pos_b = first_mask_pos
                end_pos_b = (start_pos_b + batch_num_to_transfer.unsqueeze(1)).clamp_max(block_len)
                seq_transfer_index = (
                    (range_tensor >= start_pos_b) & (range_tensor < end_pos_b)
                    & mask_token_mask)
                transfer_index = torch.where(seq_mask, seq_transfer_index, transfer_index)

            # --- Low confidence static ---
            if low_conf_static_mask.any():
                confidence = torch.where(mask_token_mask, batch_seq_x0_p, -torch.inf)
                max_k = batch_num_to_transfer.max().item()
                _, top_indices = torch.topk(confidence, k=max_k, dim=1)
                k_mask = (torch.arange(max_k, device=device).unsqueeze(0)
                          < batch_num_to_transfer.unsqueeze(1))
                static_transfer_index = torch.zeros_like(
                    confidence, dtype=torch.bool).scatter_(1, top_indices, k_mask)
                transfer_index = torch.where(
                    low_conf_static_mask, static_transfer_index, transfer_index)

            # --- Low confidence dynamic ---
            if low_conf_dynamic_mask.any():
                dyn_thresholds = torch.tensor(
                    [seq.dynamic_threshold for seq in seqs],
                    device=device).unsqueeze(1)
                confidence = torch.where(mask_token_mask, batch_seq_x0_p, -torch.inf)
                dyn_transfer_index = (confidence > dyn_thresholds)
                num_transferred_dyn = dyn_transfer_index.sum(dim=1)
                needs_fallback = ((num_transferred_dyn < batch_num_to_transfer)
                                  & low_conf_dynamic_mask.squeeze())
                if needs_fallback.any():
                    fallback_mask_token_mask = mask_token_mask[needs_fallback]
                    fallback_num_to_transfer = batch_num_to_transfer[needs_fallback].unsqueeze(1)
                    first_mask_pos = torch.argmax(
                        fallback_mask_token_mask.int(), dim=1, keepdim=True)
                    range_tensor = torch.arange(block_len, device=device).unsqueeze(0)
                    end_pos_b = (first_mask_pos + fallback_num_to_transfer).clamp_max(block_len)
                    fallback_indices = (
                        (range_tensor >= first_mask_pos) & (range_tensor < end_pos_b)
                        & fallback_mask_token_mask)
                    dyn_transfer_index[needs_fallback] = fallback_indices
                batch_num_to_transfer = torch.where(
                    low_conf_dynamic_mask.squeeze(),
                    dyn_transfer_index.sum(dim=1),
                    batch_num_to_transfer)
                transfer_index = torch.where(
                    low_conf_dynamic_mask, dyn_transfer_index, transfer_index)

            # --- Entropy bounded ---
            if entropy_bounded_mask.any():
                masked_entropies = torch.where(mask_token_mask, entropies_all, torch.inf)
                ent_sorted, order = torch.sort(masked_entropies, dim=1, descending=False)
                ent_sorted_masked = torch.where(ent_sorted == torch.inf, 0.0, ent_sorted)
                cumsum = torch.cumsum(ent_sorted_masked, dim=1)
                eb_thresholds = torch.tensor(
                    [seq.eb_threshold for seq in seqs], device=device).unsqueeze(1)
                k_tensor = torch.searchsorted(cumsum, eb_thresholds, right=False)
                k_tensor.clamp_min_(1)
                k_mask_eb = (torch.arange(block_len, device=device).unsqueeze(0)
                             < k_tensor)
                eb_transfer_index = torch.zeros_like(
                    mask_token_mask, dtype=torch.bool).scatter_(1, order, k_mask_eb)
                batch_num_to_transfer = torch.where(
                    entropy_bounded_mask.squeeze(),
                    k_tensor.squeeze(1),
                    batch_num_to_transfer)
                transfer_index = torch.where(
                    entropy_bounded_mask, eb_transfer_index, transfer_index)

            # --- Random ---
            if random_mask.any():
                B, L = mask_token_mask.shape
                scores = torch.rand((B, L), device=device)
                scores = scores.masked_fill(~mask_token_mask, -1.0)
                max_k = batch_num_to_transfer.max().item()
                _, top_indices = scores.topk(max_k, dim=-1)
                k_mask = (torch.arange(max_k, device=device).unsqueeze(0)
                          < batch_num_to_transfer.unsqueeze(1))
                random_transfer_index = torch.zeros_like(
                    mask_token_mask, dtype=torch.bool).scatter_(1, top_indices, k_mask)
                transfer_index = torch.where(
                    random_mask, random_transfer_index, transfer_index)

            final_transfer_index = transfer_index & mask_token_mask

            batch_new_tokens = torch.where(
                final_transfer_index, batch_seq_x0, batch_current_tokens)
            batch_new_trajectory = torch.where(
                final_transfer_index & (batch_trajectory == 0),
                batch_global_step_plus_1, batch_trajectory)
            batch_new_logprobs = torch.where(
                final_transfer_index, batch_seq_x0_logp, batch_logprobs)
            batch_new_entropies = torch.where(
                final_transfer_index, entropies_all, batch_entropies)

            new_denoising_steps = torch.tensor(
                [seq.current_denoising_step for seq in seqs], device=device
            ) + denoising_mask_bool.int()
            new_global_steps = torch.tensor(
                [seq.global_denoising_step for seq in seqs], device=device
            ) + denoising_mask_bool.int()

            is_fully_denoised = (
                ~(batch_new_tokens == self.mask_token_id).any(dim=1)
                | (new_denoising_steps >= torch.tensor(
                    [seq.denoising_steps for seq in seqs], device=device)))

            new_denoising_steps_cpu = new_denoising_steps.tolist()
            new_global_steps_cpu = new_global_steps.tolist()
            num_to_transfer_cpu = final_transfer_index.sum(dim=1).tolist()
            is_fully_denoised_cpu = is_fully_denoised.tolist()
            denoising_mask_cpu = denoising_mask_bool.tolist()
            saving_mask_cpu = saving_mask_bool.tolist()

            for i, seq in enumerate(seqs):
                if denoising_mask_cpu[i]:
                    seq.intermediate_block_tokens = batch_new_tokens[i].clone()
                    seq.intermediate_block_tokens_entropy = batch_new_entropies[i].clone()
                    seq.block_trajectory = batch_new_trajectory[i].clone()
                    seq.block_logprobs = batch_new_logprobs[i].clone()
                    seq.block_entropies = batch_new_entropies[i].clone()

                    seq.current_denoising_step = new_denoising_steps_cpu[i]
                    seq.global_denoising_step = new_global_steps_cpu[i]
                    seq.num_to_transfer = num_to_transfer_cpu[i]

                    if is_fully_denoised_cpu[i]:
                        seq.status = SequenceStatus.SAVING

                elif saving_mask_cpu[i]:
                    seq.commit_block(seq.intermediate_block_tokens)
                    seq.num_to_transfer = 0
                    if not seq.is_finished:
                        seq.start_new_block()
                    else:
                        self.block_manager.deallocate(seq)

        finished_seqs = [seq for seq in self.running if seq.is_finished]
        self.running = [seq for seq in self.running if not seq.is_finished]
        for seq in finished_seqs:
            self.block_manager.deallocate(seq)

        if self.prefill_ready:
            self.prefill_ready = deque(
                seq for seq in self.prefill_ready if not seq.is_finished)
        return finished_seqs
