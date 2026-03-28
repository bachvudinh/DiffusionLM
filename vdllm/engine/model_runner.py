"""Model execution engine with KV cache management and CUDA graph support.

Derived from JetEngine by Yihan Bian et al.
Reference: https://github.com/Labman42/JetEngine

Architecture Overview
====================

    ModelRunner is the bridge between the Scheduler and the Model.
    It handles input preparation, KV cache allocation, model execution,
    and CUDA graph capture/replay for optimized denoise steps.

    ┌──────────────────────────────────────────────────────────────────┐
    │                       ModelRunner Pipeline                       │
    │                                                                  │
    │  Initialization:                                                 │
    │    1. Instantiate model (SDAR or SDAR-MoE based on config)       │
    │    2. Load weights from safetensors                               │
    │    3. Warmup with dummy prefill batch                             │
    │    4. Allocate KV cache based on remaining GPU memory             │
    │    5. Capture CUDA graphs for common denoise batch sizes          │
    │                                                                  │
    │  Run Flow:                                                       │
    │    run(seqs, run_type)                                            │
    │      │                                                           │
    │      ├── PREFILL: prepare_prefill(seqs)                          │
    │      │    ├── Flatten all token_ids across sequences              │
    │      │    ├── Compute positions, cu_seqlens, slot_mapping         │
    │      │    └── set_context(PREFILL, ...)                           │
    │      │                                                           │
    │      ├── DENOISE: prepare_denoise(seqs)                          │
    │      │    ├── Stack intermediate_block_tokens                     │
    │      │    ├── Compute positions from cached_lens                  │
    │      │    ├── Prepare block_tables for paged KV access            │
    │      │    └── set_context(DENOISE, ...)                           │
    │      │                                                           │
    │      ├── Try CUDA graph replay (for DENOISE)                     │
    │      │    ├── Match batch_size to captured graph                  │
    │      │    ├── Copy inputs to graph buffers                        │
    │      │    └── graph.replay() → compute_logits()                  │
    │      │                                                           │
    │      ├── Fallback: eager model forward + compute_logits          │
    │      └── reset_context()                                         │
    │                                                                  │
    │  KV Cache Layout:                                                │
    │    Shape: (2, num_layers, num_blocks, block_size, kv_heads, dim) │
    │    Index:  [K/V][layer_id][block_id][offset][head][dim]          │
    │    Allocated to fill remaining GPU memory after model loading.    │
    │                                                                  │
    │  CUDA Graph Capture:                                             │
    │    Captured for batch sizes: [1, 2, 4, 8, 16, 32, ..., max_bs]  │
    │    Shares a single graph memory pool across all batch sizes.      │
    └──────────────────────────────────────────────────────────────────┘
"""

import math
import torch
import torch.distributed as dist

from vdllm.config import Config
from vdllm.engine.sequence import Sequence, RunType, SequenceStatus
from vdllm.models.sdar import SDARForCausalLM
from vdllm.models.sdar_moe import SDARMoeForCausalLM
from vdllm.utils.context import set_context, get_context, reset_context
from vdllm.utils.loader import load_model
from vdllm.engine.distributed_manager import DistributedManager


class ModelRunner:
    """Manages model execution, KV cache, and CUDA graph capture.

    Input:
        config:       Config — engine configuration
        dist_manager: DistributedManager — TP/DP process groups

    Lifecycle:
        __init__ → load weights → warmup → allocate KV → capture graphs
    """

    def __init__(self, config: Config, dist_manager: DistributedManager):
        self.config = config
        self.dist_manager = dist_manager

        self.block_size = config.kvcache_block_size
        self.enforce_eager = config.enforce_eager
        self.world_size = dist_manager.tp_size
        self.rank = dist_manager.tp_rank

        torch.cuda.set_device(dist_manager.device)
        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(config.torch_dtype)
        torch.set_default_device("cuda")

        model_kwargs = {"config": config.hf_config,
                        "process_group": self.dist_manager.tp_group}
        if "sdar" in config.hf_config.model_type and "moe" in config.hf_config.model_type:
            raise ValueError("MoE not supported for dp tp hybrid yet")
            self.ModelClass = SDARMoeForCausalLM
        elif "sdar" in config.hf_config.model_type:
            self.ModelClass = SDARForCausalLM
        else:
            raise ValueError(f"Unsupported model type: {config.hf_config.model_type}")
        self.model = self.ModelClass(**model_kwargs)
        load_model(self.model, config.model)
        self.warmup_model()
        self.allocate_kv_cache()
        if not self.enforce_eager:
            self.capture_cudagraph()

        torch.set_default_device("cpu")
        torch.set_default_dtype(default_dtype)

    def reinit_model(self):
        """Re-create model shell (for weight reload)."""
        self.model = self.ModelClass(
            self.config.hf_config, self.dist_manager.tp_group)

    def exit(self):
        if not self.enforce_eager:
            del self.graphs, self.graph_pool
        torch.cuda.synchronize()

    def warmup_model(self):
        """Run a dummy prefill to trigger compilation and measure peak memory."""
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        max_num_batched_tokens = self.config.max_num_batched_tokens
        max_model_len = self.config.max_model_len
        num_seqs = min(max_num_batched_tokens // max_model_len,
                       self.config.max_num_seqs)
        seqs = [Sequence([0] * max_model_len, self.config.mask_token_id)
                for _ in range(num_seqs)]
        self.run(seqs, RunType.PREFILL)
        torch.cuda.empty_cache()

    def allocate_kv_cache(self):
        """Allocate KV cache blocks to fill remaining GPU memory.

        Computes available memory after model loading and warmup,
        then allocates as many KV cache blocks as possible within
        the gpu_memory_utilization budget.

        KV Cache Shape:
            (2, num_layers, num_blocks, block_size, kv_heads, head_dim)
        """
        config = self.config
        free, total = torch.cuda.mem_get_info()
        used = total - free
        peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
        current = torch.cuda.memory_stats()["allocated_bytes.all.current"]
        num_kv_heads = config.num_key_value_heads // self.world_size
        block_bytes = (2 * config.num_hidden_layers * self.block_size
                       * num_kv_heads * config.head_dim
                       * config.torch_dtype.itemsize)
        config.num_kvcache_blocks = int(
            total * config.gpu_memory_utilization - used - peak + current
        ) // block_bytes
        assert config.num_kvcache_blocks > 0
        print(
            f"[rank {dist.get_rank()}][KVCache] Allocated "
            f"{config.num_kvcache_blocks:,} blocks "
            f"({config.num_kvcache_blocks * block_bytes / (1024**3):.2f} GiB) "
            f"based on peak memory usage.")
        self.kv_cache = torch.zeros(
            2, config.num_hidden_layers, config.num_kvcache_blocks,
            self.block_size, num_kv_heads, config.head_dim)
        layer_id = 0
        for module in self.model.modules():
            if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                module.k_cache = self.kv_cache[0, layer_id]
                module.v_cache = self.kv_cache[1, layer_id]
                layer_id += 1

    def prepare_block_tables(self, seqs: list[Sequence]):
        """Pad and tensorize block tables for paged KV cache access.

        Input:
            seqs: list of Sequence with .block_table

        Output:
            (batch, max_blocks) int32 tensor on CUDA, or None if empty
        """
        max_len = max(len(seq.block_table) for seq in seqs)
        if max_len == 0:
            return None
        block_tables = [
            seq.block_table + [-1] * (max_len - len(seq.block_table))
            for seq in seqs
        ]
        block_tables = torch.tensor(
            block_tables, dtype=torch.int32, pin_memory=True
        ).cuda(non_blocking=True)
        return block_tables

    def prepare_prefill(self, seqs: list[Sequence]):
        """Prepare input tensors for prefill (full context encoding).

        Input:
            seqs: list of Sequence in PREFILLING state

        Output:
            (input_ids, positions) — both 1D tensors on CUDA

        Side Effect:
            Sets global Context with PREFILL metadata (cu_seqlens, slot_mapping)
        """
        device = torch.device("cuda")
        all_token_ids_flat = []
        all_block_tables_flat = []
        seqlens_list = []
        block_table_lens_list = []
        is_last_step_list = []

        for seq in seqs:
            seq_len = len(seq.token_ids)
            seqlens_list.append(seq_len)
            all_token_ids_flat.extend(seq.token_ids)
            is_last_step_list.append(False)

            if seq.block_table:
                block_table_lens_list.append(len(seq.block_table))
                all_block_tables_flat.extend(seq.block_table)
            else:
                block_table_lens_list.append(0)

        input_ids_cpu = torch.tensor(all_token_ids_flat, dtype=torch.int64, pin_memory=True)
        input_ids = input_ids_cpu.to(device=device, non_blocking=True)

        flat_block_tables_cpu = torch.tensor(
            all_block_tables_flat, dtype=torch.int32, pin_memory=True)
        flat_block_tables = flat_block_tables_cpu.to(device=device, non_blocking=True)

        seqlens_q_cpu = torch.tensor(seqlens_list, dtype=torch.int32, pin_memory=True)
        block_table_lens_cpu = torch.tensor(
            block_table_lens_list, dtype=torch.int32, pin_memory=True)
        is_last_step_cpu = torch.tensor(is_last_step_list, dtype=torch.bool, pin_memory=True)

        seqlens_q = seqlens_q_cpu.to(device=device, non_blocking=True)
        block_table_lens = block_table_lens_cpu.to(device=device, non_blocking=True)
        is_last_step = is_last_step_cpu.to(device=device, non_blocking=True)

        batch_size = len(seqs)
        if batch_size == 0:
            cu_seqlens_q_cpu = torch.tensor([0], dtype=torch.int32, pin_memory=True)
            cu_seqlens_q = cu_seqlens_q_cpu.to(device=device, non_blocking=True)
            max_seqlen_q = 0
            positions = torch.empty(0, dtype=torch.int64, device=device)
            slot_mapping = torch.empty(0, dtype=torch.int32, device=device)
        else:
            cu_seqlens_q = torch.nn.functional.pad(
                seqlens_q.cumsum(dim=0, dtype=torch.int32), (1, 0))
            cu_block_table_lens = torch.nn.functional.pad(
                block_table_lens.cumsum(dim=0, dtype=torch.int32), (1, 0))
            max_seqlen_q = seqlens_q.max().item()
            total_tokens = input_ids.shape[0]
            token_indices_global = torch.arange(total_tokens, dtype=torch.int64, device=device)
            seq_start_offsets = torch.repeat_interleave(
                cu_seqlens_q[:-1], repeats=seqlens_q)
            positions = token_indices_global - seq_start_offsets

            has_block_table_mask_per_seq = (block_table_lens > 0)
            has_block_table_mask_per_token = torch.repeat_interleave(
                has_block_table_mask_per_seq, repeats=seqlens_q)
            i = positions[has_block_table_mask_per_token]
            seq_idx_for_each_token = torch.repeat_interleave(
                torch.arange(batch_size, device=device), repeats=seqlens_q)
            seq_idx_with_slot = seq_idx_for_each_token[has_block_table_mask_per_token]
            block_idx = i // self.block_size
            block_offset = i % self.block_size
            block_table_start_offsets = cu_block_table_lens[seq_idx_with_slot]
            block_table_global_idx = block_table_start_offsets + block_idx
            physical_block_id = flat_block_tables[block_table_global_idx]
            slot_mapping = physical_block_id * self.block_size + block_offset

        set_context(
            run_type=RunType.PREFILL,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_q,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_q,
            slot_mapping=slot_mapping.to(torch.int32),
            is_last_denoise_step=is_last_step,
            block_length=self.config.block_length)
        return input_ids, positions

    def prepare_denoise(self, seqs: list[Sequence]):
        """Prepare input tensors for denoise (block refinement) step.

        Input:
            seqs: list of Sequence in DENOISING state

        Output:
            (input_ids, positions) — both 1D tensors on CUDA

        Side Effect:
            Sets global Context with DENOISE metadata (context_lens, block_tables)

        Visualized Example (batch=2, block_length=4):
            Seq 0: cached_len=12, block=[MASK, MASK, tok_a, MASK]
            Seq 1: cached_len=8,  block=[tok_b, MASK, MASK, MASK]

            input_ids: [MASK, MASK, tok_a, MASK, tok_b, MASK, MASK, MASK]
            positions: [  12,   13,    14,   15,    8,    9,   10,   11]
            context_lens: [12, 8]
        """
        device = torch.device("cuda")

        block_tokens_list = []
        for seq in seqs:
            t = seq.intermediate_block_tokens
            if t.device != device:
                t = t.to(device, non_blocking=True)
                seq.intermediate_block_tokens = t
            block_tokens_list.append(t)

        input_ids = torch.stack(block_tokens_list).view(-1)

        cached_lens = torch.tensor(
            [len(seq) for seq in seqs], dtype=torch.int32, device=device)

        block_len = seqs[0].block_length
        start_positions = cached_lens.unsqueeze(1)
        offsets = torch.arange(block_len, dtype=torch.int64, device=device).unsqueeze(0)
        positions = (start_positions + offsets).view(-1)
        block_tables = self.prepare_block_tables(seqs)

        set_context(
            run_type=RunType.DENOISE,
            context_lens=cached_lens,
            block_tables=block_tables,
            block_length=self.config.block_length)

        return input_ids, positions

    @torch.inference_mode()
    def run_model(self, input_ids: torch.Tensor, positions: torch.Tensor):
        """Eager model forward pass.

        Input:
            input_ids: (total_tokens,) int64
            positions: (total_tokens,) int64

        Output:
            (batch_size, vocab_size) — logits
        """
        return self.model.compute_logits(self.model(input_ids, positions))

    @torch.inference_mode()
    def _run_denoise_with_cudagraph(
        self, seqs: list[Sequence],
        input_ids: torch.Tensor, positions: torch.Tensor
    ) -> tuple[torch.Tensor | None, bool]:
        """Try to replay a captured CUDA graph for denoise.

        Returns (logits, True) if graph was used, (None, False) otherwise.
        Falls back to eager if batch size doesn't match any captured graph
        or block table exceeds captured capacity.
        """
        if (self.enforce_eager
                or not hasattr(self, "graphs")
                or not getattr(self, "graphs", None)):
            return None, False

        context = get_context()
        if context.run_type != RunType.DENOISE:
            return None, False

        batch_size = len(seqs)
        if batch_size == 0:
            return None, False

        graph = self.graphs.get(batch_size)
        if graph is None:
            return None, False

        graph_vars = self.graph_vars
        block_len = self.config.block_length
        global_bs = batch_size * block_len

        if (global_bs > graph_vars["input_ids"].shape[0]
                or context.context_lens is None):
            return None, False

        graph_vars["input_ids"][:global_bs].copy_(input_ids)
        graph_vars["positions"][:global_bs].copy_(positions)
        graph_context_lens = graph_vars["context_lens"][:batch_size]
        graph_context_lens.copy_(context.context_lens)

        graph_block_tables = graph_vars["block_tables"][:batch_size]
        graph_block_tables.fill_(-1)
        if context.block_tables is not None:
            required_blocks = context.block_tables.shape[1]
            if required_blocks > graph_block_tables.shape[1]:
                return None, False
            graph_block_tables[:, :required_blocks].copy_(context.block_tables)

        set_context(
            run_type=RunType.DENOISE,
            context_lens=graph_context_lens,
            block_tables=graph_block_tables,
            block_length=self.config.block_length,
            is_last_denoise_step=context.is_last_denoise_step)

        graph.replay()
        hidden_states = graph_vars["outputs"][:global_bs]
        logits = self.model.compute_logits(hidden_states)
        return logits, True

    def run(self, seqs: list[Sequence], run_type: RunType) -> torch.Tensor:
        """Execute model forward pass for a batch of sequences.

        Input:
            seqs:     list of Sequence — batch to process
            run_type: RunType.PREFILL or RunType.DENOISE

        Output:
            (batch_size, vocab_size) logits on rank 0, None on other ranks
        """
        if run_type == RunType.PREFILL:
            input_ids, positions = self.prepare_prefill(seqs)
        elif run_type == RunType.DENOISE:
            input_ids, positions = self.prepare_denoise(seqs)
        else:
            return None

        if run_type == RunType.DENOISE and not self.enforce_eager:
            logits, used_graph = self._run_denoise_with_cudagraph(
                seqs, input_ids, positions)
            if not used_graph:
                logits = self.run_model(input_ids, positions)
        else:
            logits = self.run_model(input_ids, positions)
        reset_context()
        return logits if self.rank == 0 else None

    @torch.inference_mode()
    def capture_cudagraph(self):
        """Capture CUDA graphs for common denoise batch sizes.

        Captures graphs for batch sizes [1, 2, 4, 8, 16, 32, ..., max_bs].
        All graphs share a single memory pool for efficiency.

        Graph Variables (persistent buffers):
            input_ids:    (max_bs * block_length,) int64
            positions:    (max_bs * block_length,) int64
            context_lens: (max_bs,) int32
            block_tables: (max_bs, max_num_blocks) int32
            outputs:      (max_bs * block_length, hidden_size)
        """
        config = self.config
        max_bs = min(self.config.max_num_seqs, 256)
        max_global_bs = max_bs * self.config.block_length
        max_num_blocks = math.ceil(
            (config.max_model_len + self.config.block_length) / self.block_size
        ) + 1
        input_ids = torch.zeros(max_global_bs, dtype=torch.int64)
        positions = torch.zeros(max_global_bs, dtype=torch.int64)
        context_lens = torch.zeros(max_bs, dtype=torch.int32)
        block_tables = torch.zeros(max_bs, max_num_blocks, dtype=torch.int32)
        outputs = torch.zeros(
            max_global_bs, config.hidden_size, dtype=config.torch_dtype)
        self.graph_bs = (
            [bs for bs in [1, 2, 4, 8] if bs <= max_bs]
            + list(range(16, max_bs + 1, 16))
        )
        self.graphs = {}
        self.graph_pool = None

        for bs in reversed(self.graph_bs):
            graph = torch.cuda.CUDAGraph()
            set_context(
                run_type=RunType.DENOISE,
                context_lens=context_lens[:bs],
                block_tables=block_tables[:bs],
                block_length=self.config.block_length)
            global_bs = bs * self.config.block_length
            outputs[:global_bs] = self.model(
                input_ids[:global_bs], positions[:global_bs])
            with torch.cuda.graph(graph, self.graph_pool):
                outputs[:global_bs] = self.model(
                    input_ids[:global_bs], positions[:global_bs])
            if self.graph_pool is None:
                self.graph_pool = graph.pool()
            self.graphs[bs] = graph
            torch.cuda.synchronize()
            reset_context()

        self.graph_vars = dict(
            input_ids=input_ids,
            positions=positions,
            context_lens=context_lens,
            block_tables=block_tables,
            outputs=outputs)
