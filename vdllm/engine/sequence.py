"""Sequence state management for block diffusion inference.

Derived from JetEngine by Yihan Bian et al.
Reference: https://github.com/Labman42/JetEngine

Architecture Overview
====================

    Each Sequence tracks one generation request through its lifecycle:

    ┌──────────────────────────────────────────────────────────────────┐
    │                    Sequence Lifecycle                             │
    │                                                                   │
    │  WAITING ──► PREFILLING ──► DENOISING ◄──► SAVING ──► FINISHED   │
    │                                  │            │                    │
    │                                  │            │                    │
    │                                  └── loop ────┘                   │
    │                                                                   │
    │  WAITING:     Prompt queued, not yet prefilled                    │
    │  PREFILLING:  Prompt tokens being processed into KV cache        │
    │  DENOISING:   Current block being iteratively denoised           │
    │  SAVING:      Block fully denoised, ready to commit              │
    │  FINISHED:    Generation complete (max_tokens or stop word)      │
    └──────────────────────────────────────────────────────────────────┘

    Token Layout:

        prompt = "Hello world how are you doing today?"
        prompt_token_ids = [101, 7592, 2088, 2129, 2024, 2017, 2725, 2651, 102]
        block_length = 4

        num_prefill_tokens = 8  (largest multiple of 4 ≤ 9)

        token_ids (prefill part):
            [101, 7592, 2088, 2129, 2024, 2017, 2725, 2651]
             └──────────── prefilled into KV cache ───────┘

        intermediate_block_tokens (first denoise block):
            [102, MASK, MASK, MASK]
             └─┘  └──── to be denoised ────┘
             prompt remainder

    Denoise Loop (per block):

        Step 0: [102,  MASK, MASK, MASK]  mask_count=3
        Step 1: [102,  tok1, MASK, MASK]  mask_count=2  (1 unmasked)
        Step 2: [102,  tok1, tok2, MASK]  mask_count=1  (1 unmasked)
        Step 3: [102,  tok1, tok2, tok3]  mask_count=0  → SAVING

        commit_block() → token_ids extended, start_new_block()

    Input:
        prompt_token_ids: list[int] — tokenized prompt
        mask_token_id:    int       — special [MASK] token ID
        sampling_params:  SamplingParams — generation config

    Output:
        Sequence object tracking all state for one generation request
"""

from copy import copy
from enum import Enum, auto
from collections import Counter
from itertools import count
import torch
from vdllm.sampling_params import SamplingParams


class SequenceStatus(Enum):
    WAITING = auto()
    PREFILLING = auto()
    DENOISING = auto()
    SAVING = auto()
    FINISHED = auto()

class RunType(Enum):
    PREFILL = auto()
    DENOISE = auto()


class Sequence:
    default_block_size = 256
    block_size = default_block_size
    counter = count()

    def __init__(self, prompt_token_ids: list[int], mask_token_id: int, sampling_params=SamplingParams(), block_size: int | None = None):
        self.seq_id = next(Sequence.counter)
        if block_size is not None:
            self.block_size = block_size
        self.block_length = sampling_params.block_length
        self.prompt_token_ids = prompt_token_ids
        prompt_len = len(self.prompt_token_ids)

        self.num_prefill_tokens = (prompt_len // self.block_length) * self.block_length
        prefill_part = self.prompt_token_ids[:self.num_prefill_tokens]

        first_denoise_part = self.prompt_token_ids[self.num_prefill_tokens:]
        self.generation_start_index = len(first_denoise_part)

        self.token_ids = prefill_part
        self.num_tokens = len(self.token_ids)
        self.num_prompt_tokens = prompt_len
        self.num_generated_tokens = self.num_tokens - self.num_prompt_tokens

        self.intermediate_block_tokens = torch.tensor(
            first_denoise_part + [mask_token_id] * (self.block_length - len(first_denoise_part)),
            dtype=torch.long
        )
        self.intermediate_block_tokens_entropy = torch.zeros(self.block_length, dtype=torch.float)

        self.num_to_transfer = 0
        self.current_denoising_step = 0

        self.trajectory: list[int] = []
        self.block_trajectory = torch.zeros(self.block_length, dtype=torch.long)
        self.global_denoising_step = 0

        self.logprobs: list[float] = []
        self.block_logprobs = torch.zeros(self.block_length, dtype=torch.float)

        self.entropies: list[float] = []
        self.block_entropies = torch.zeros(self.block_length, dtype=torch.float)

        if self.num_prefill_tokens > 0:
            self.status = SequenceStatus.WAITING
        else:
            self.status = SequenceStatus.DENOISING

        self.temperature = sampling_params.temperature
        self.stop_words = sampling_params.stop_words if sampling_params.stop_words is not None else []
        self.stop_words = set(self.stop_words)
        self.top_k = sampling_params.topk
        self.top_p = sampling_params.topp
        self.max_tokens = sampling_params.max_tokens
        self.ignore_eos = sampling_params.ignore_eos
        self.denoising_steps = sampling_params.denoising_steps
        self.remasking_strategy = sampling_params.remasking_strategy
        self.dynamic_threshold = sampling_params.dynamic_threshold
        self.eb_threshold = sampling_params.eb_threshold
        self.mask_token_id = mask_token_id
        self.num_transfer_tokens_per_step = self._get_num_transfer_tokens()

        self.repetition_penalty = sampling_params.repetition_penalty
        self.token_counts = Counter(self.prompt_token_ids)

        self.num_cached_tokens = 0
        self.block_table = []

    def __len__(self):
        return self.num_tokens

    def __getitem__(self, key):
        return self.token_ids[key]

    def _get_num_transfer_tokens(self):
        base = self.block_length // self.denoising_steps
        remainder = self.block_length % self.denoising_steps
        num_tokens = [base] * self.denoising_steps
        for i in range(remainder):
            num_tokens[i] += 1
        return num_tokens

    def start_new_block(self):
        self.current_denoising_step = 0
        device = self.intermediate_block_tokens.device
        self.intermediate_block_tokens = torch.full((self.block_length,), self.mask_token_id, dtype=torch.long, device=device)
        self.intermediate_block_tokens_entropy = torch.zeros(self.block_length, dtype=torch.float, device=device)
        self.block_trajectory = torch.zeros(self.block_length, dtype=torch.long, device=device)
        self.block_logprobs = torch.zeros(self.block_length, dtype=torch.float, device=device)
        self.block_entropies = torch.zeros(self.block_length, dtype=torch.float, device=device)
        self.status = SequenceStatus.DENOISING

    def commit_block(self, block_tokens):
        if isinstance(block_tokens, torch.Tensor):
            block_tokens = block_tokens.tolist()

        start = self.generation_start_index
        remaining = self.max_tokens - self.num_completion_tokens
        if remaining <= 0:
            self.status = SequenceStatus.FINISHED
            return []

        stop_pos = None
        if (not self.ignore_eos) and self.stop_words:
            stop_pos = next(
                (j for j, tok in enumerate(block_tokens[start:], start)
                if tok in self.stop_words),
                None
            )

        end_by_tokens = start + remaining
        end_by_stop = (stop_pos + 1) if stop_pos is not None else len(block_tokens)
        end = min(len(block_tokens), end_by_tokens, end_by_stop)

        if end == end_by_tokens or (stop_pos is not None and end == stop_pos + 1):
            self.status = SequenceStatus.FINISHED
        self.generation_start_index = 0

        before_ntok = self.num_tokens
        self.token_ids.extend(block_tokens)
        self.num_tokens = len(self.token_ids)
        self.num_generated_tokens = self.num_tokens - self.num_prompt_tokens

        self.token_counts.update(block_tokens)

        block_len = len(block_tokens)
        if self.block_trajectory is not None:
            if isinstance(self.block_trajectory, torch.Tensor):
                b_traj = self.block_trajectory.tolist()
            else:
                b_traj = self.block_trajectory

            block_start = max(0, self.num_prompt_tokens - before_ntok)
            start_idx = min(block_start, block_len)
            if start_idx < block_len:
                self.trajectory.extend(b_traj[start_idx:block_len])

                if self.block_logprobs is not None:
                    if isinstance(self.block_logprobs, torch.Tensor):
                        b_logprobs = self.block_logprobs.tolist()
                    else:
                        b_logprobs = self.block_logprobs
                    self.logprobs.extend(b_logprobs[start_idx:block_len])

                if self.block_entropies is not None:
                    if isinstance(self.block_entropies, torch.Tensor):
                        b_entropies = self.block_entropies.tolist()
                    else:
                        b_entropies = self.block_entropies
                    self.entropies.extend(b_entropies[start_idx:block_len])

        if self.num_tokens >= self.num_prompt_tokens + self.max_tokens:
             self.status = SequenceStatus.FINISHED

    def get_len_for_next_step(self):
        return self.num_tokens + self.block_length

    def num_new_blocks_needed(self, block_size):
        needed_total_blocks = (self.num_tokens + self.block_length + block_size - 1) // block_size
        return max(0, needed_total_blocks - len(self.block_table))

    @property
    def is_finished(self):
        return self.status == SequenceStatus.FINISHED

    @property
    def num_completion_tokens(self):
        return self.num_tokens - self.num_prompt_tokens

    @property
    def completion_token_ids(self):
        return self.token_ids[self.num_prompt_tokens:]

    @property
    def num_cached_blocks(self):
        return self.num_cached_tokens // self.block_size

    @property
    def num_blocks(self):
        return (self.num_tokens + self.block_size - 1) // self.block_size

    @property
    def last_block_num_tokens(self):
        return self.num_tokens - (self.num_blocks - 1) * self.block_size

    def block(self, i):
        assert 0 <= i < self.num_blocks
        return self.token_ids[i*self.block_size: (i+1)*self.block_size]

    def append_token(self, token_id):
        self.token_ids.append(token_id)
        self.last_token = token_id
        self.num_tokens += 1
        self.token_counts[token_id] += 1

    def __getstate__(self):
        return (
            self.seq_id, self.status, self.token_ids, self.num_tokens,
            self.num_prompt_tokens, self.num_cached_tokens, self.block_table,
            self.intermediate_block_tokens, self.current_denoising_step,
            self.block_size,
        )

    def __setstate__(self, state):
        if len(state) == 9:
            (self.seq_id, self.status, self.token_ids, self.num_tokens,
             self.num_prompt_tokens, self.num_cached_tokens, self.block_table,
             self.intermediate_block_tokens, self.current_denoising_step) = state
            self.block_size = Sequence.default_block_size
        else:
            (self.seq_id, self.status, self.token_ids, self.num_tokens,
             self.num_prompt_tokens, self.num_cached_tokens, self.block_table,
             self.intermediate_block_tokens, self.current_denoising_step,
             self.block_size) = state
