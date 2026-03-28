"""Tensor-parallel linear layers for distributed inference.

Derived from JetEngine by Yihan Bian et al.
Reference: https://github.com/Labman42/JetEngine

Architecture Overview
====================

    All linear layers support tensor parallelism (TP) by sharding
    weights across GPUs. Each variant handles a different sharding
    pattern:

    ┌────────────────────────────────────────────────────────────────┐
    │                  TP Linear Layer Variants                      │
    │                                                                │
    │  ReplicatedLinear                                              │
    │  ├── Full weight on every GPU (no sharding)                    │
    │  └── Used for: MoE gating, small projections                   │
    │                                                                │
    │  ColumnParallelLinear                                          │
    │  ├── Shards output dimension: W[out/tp, in]                    │
    │  ├── Each GPU computes a slice of the output                   │
    │  └── No communication needed in forward                        │
    │                                                                │
    │  MergedColumnParallelLinear                                    │
    │  ├── Fuses multiple column-parallel projections                │
    │  ├── Example: gate_proj + up_proj → gate_up_proj               │
    │  └── weight_loader handles per-shard loading                   │
    │                                                                │
    │  QKVParallelLinear                                             │
    │  ├── Fuses Q, K, V projections with GQA support                │
    │  ├── Layout: [Q_heads | K_heads | V_heads] per GPU             │
    │  └── Handles different num_heads vs num_kv_heads               │
    │                                                                │
    │  RowParallelLinear                                              │
    │  ├── Shards input dimension: W[out, in/tp]                     │
    │  ├── Each GPU computes partial output                          │
    │  └── all_reduce aggregates across TP group                     │
    │                                                                │
    │  Data Flow (Column + Row pair):                                │
    │    x ──► ColumnParallel ──► [local_out] ──► RowParallel ──► y  │
    │          (no comm)          (activation)    (all_reduce)       │
    └────────────────────────────────────────────────────────────────┘

    Weight Loading:
        Each layer has a weight_loader() method called by utils/loader.py
        that handles TP-aware slicing from full checkpoint tensors.


ColumnParallelLinear — Tensor Flow
===================================

    Example: hidden_size=4096, intermediate_size=11008, tp_size=2

    Full checkpoint weight: (11008, 4096) float16
                                │
                                ▼  weight_loader (shard on dim=0)
    Rank 0 weight: (5504, 4096) float16     ← rows [0:5504]
    Rank 1 weight: (5504, 4096) float16     ← rows [5504:11008]

    Forward:
        x: (N, 4096) float16
                │
                ▼  F.linear(x, weight)
        out: (N, 5504) float16              ← local shard, no communication


QKVParallelLinear — Tensor Flow
================================

    Example: hidden_size=4096, num_heads=32, num_kv_heads=8,
             head_dim=128, tp_size=2

    Per-rank sizes:
        Q: 32/2=16 heads × 128 = 2048
        K: 8/2=4 heads  × 128 = 512
        V: 8/2=4 heads  × 128 = 512
        Total per rank: 3072

    Fused weight layout on each rank:
        weight: (3072, 4096) float16
                 ├─ Q ─┤├ K ┤├ V ┤
                 [2048   512  512 ]

    Loading from checkpoint (e.g., q_proj.weight):
        Full q_weight: (4096, 4096) float16
                          │
                          ▼  chunk by tp_size, take rank's slice
        Rank 0: (2048, 4096) → copied into weight[0:2048, :]
        Rank 1: (2048, 4096) → copied into weight[0:2048, :]

    Forward:
        x: (N, 4096) float16
                │
                ▼  F.linear(x, weight)
        qkv: (N, 3072) float16
                │
                ▼  split([2048, 512, 512], dim=-1)
        q: (N, 2048) float16    k: (N, 512) float16    v: (N, 512) float16


MergedColumnParallelLinear — Tensor Flow
==========================================

    Example: gate_up_proj for MLP, hidden=4096, intermediate=11008, tp_size=2

    output_sizes = [11008, 11008]  → total output = 22016
    Per-rank output = 22016/2 = 11008

    Fused weight layout on rank 0:
        weight: (11008, 4096) float16
                 ├── gate ──┤├── up ──┤
                 [5504        5504    ]

    Loading gate_proj.weight (shard_id=0):
        Full: (11008, 4096) → chunk(2, dim=0) → rank 0 gets (5504, 4096)
        Copied into weight[0:5504, :]

    Loading up_proj.weight (shard_id=1):
        Full: (11008, 4096) → chunk(2, dim=0) → rank 0 gets (5504, 4096)
        Copied into weight[5504:11008, :]

    Forward:
        x: (N, 4096) float16
                │
                ▼  F.linear(x, weight)
        gate_up: (N, 11008) float16       ← concatenated gate + up shard


RowParallelLinear — Tensor Flow
================================

    Example: down_proj for MLP, input=11008, output=4096, tp_size=2

    Per-rank input = 11008/2 = 5504

    weight: (4096, 5504) float16          ← sharded on input dim

    Forward:
        x: (N, 5504) float16             ← local shard from preceding Column
                │
                ▼  F.linear(x, weight, bias)
        y_local: (N, 4096) float16
                │
                ▼  dist.all_reduce(y_local, group=tp_group)
        y: (N, 4096) float16             ← sum across all ranks
"""

import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist


def divide(numerator, denominator):
    assert numerator % denominator == 0
    return numerator // denominator


class LinearBase(nn.Module):
    """Base class for all TP-aware linear layers.

    Input:
        input_size:    int — input feature dimension
        output_size:   int — output feature dimension (before TP split)
        process_group: ProcessGroup — TP communication group
        tp_dim:        int | None — dimension to shard (0=output, 1=input)
    """

    def __init__(self, input_size: int, output_size: int,
                 process_group: dist.ProcessGroup, tp_dim: int | None = None):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.process_group = process_group
        self.tp_dim = tp_dim
        self.tp_rank = dist.get_rank(group=self.process_group)
        self.tp_size = dist.get_world_size(group=self.process_group)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class ReplicatedLinear(LinearBase):
    """Linear layer replicated across all TP ranks (no sharding).

    Input:
        x: (..., input_size)

    Output:
        (..., output_size) — same on all ranks
    """

    def __init__(self, input_size: int, output_size: int,
                 process_group: dist.ProcessGroup, bias: bool = False):
        super().__init__(input_size, output_size, process_group)
        self.weight = nn.Parameter(torch.empty(self.output_size, self.input_size))
        self.weight.weight_loader = self.weight_loader
        if bias:
            self.bias = nn.Parameter(torch.empty(self.output_size))
            self.bias.weight_loader = self.weight_loader
        else:
            self.register_parameter("bias", None)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param.data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)


class ColumnParallelLinear(LinearBase):
    """Column-parallel: shards output dimension across TP ranks.

    Input:
        x: (..., input_size)

    Output:
        (..., output_size // tp_size) — each rank gets a different slice
    """

    def __init__(self, input_size: int, output_size: int,
                 process_group: dist.ProcessGroup, bias: bool = False):
        super().__init__(input_size, output_size, process_group, 0)
        self.input_size_per_partition = input_size
        self.output_size_per_partition = divide(output_size, self.tp_size)

        self.weight = nn.Parameter(
            torch.empty(self.output_size_per_partition, self.input_size))
        self.weight.weight_loader = self.weight_loader
        if bias:
            self.bias = nn.Parameter(torch.empty(self.output_size_per_partition))
            self.bias.weight_loader = self.weight_loader
        else:
            self.register_parameter("bias", None)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param_data = param.data
        shard_size = param_data.size(self.tp_dim)
        start_idx = self.tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(self.tp_dim, start_idx, shard_size)
        param_data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)


class MergedColumnParallelLinear(ColumnParallelLinear):
    """Fused column-parallel for multiple projections (e.g., gate + up).

    Input:
        x: (..., input_size)

    Output:
        (..., sum(output_sizes) // tp_size) — concatenated sharded outputs

    Visualized Example (gate_up_proj, tp_size=2, rank=0):
        Full weight: [gate_weight | up_weight]  shape: (2*I, H)
        Rank 0 shard: [gate_weight[:I/2] | up_weight[:I/2]]  shape: (I, H)
    """

    def __init__(self, input_size: int, output_sizes: list[int],
                 process_group: dist.ProcessGroup, bias: bool = False):
        self.output_sizes = output_sizes
        super().__init__(input_size, sum(output_sizes), process_group, bias=bias)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor,
                      loaded_shard_id: int):
        param_data = param.data
        shard_offset = sum(self.output_sizes[:loaded_shard_id]) // self.tp_size
        shard_size = self.output_sizes[loaded_shard_id] // self.tp_size
        param_data = param_data.narrow(self.tp_dim, shard_offset, shard_size)
        loaded_weight = loaded_weight.chunk(self.tp_size, self.tp_dim)[self.tp_rank]
        param_data.copy_(loaded_weight)


class QKVParallelLinear(ColumnParallelLinear):
    """Fused Q/K/V projection with grouped-query attention (GQA) support.

    Input:
        x: (..., hidden_size)

    Output:
        (..., (num_heads + 2*num_kv_heads) * head_size // tp_size)

    Weight Layout per TP rank:
        ┌──────────────────────────────────────────────────┐
        │  Q heads (num_heads/tp)  │  K heads  │  V heads  │
        │  num_heads/tp * head_dim │ num_kv/tp │ num_kv/tp │
        └──────────────────────────────────────────────────┘
    """

    def __init__(self, hidden_size: int, head_size: int,
                 process_group: dist.ProcessGroup,
                 total_num_heads: int, total_num_kv_heads: int | None = None,
                 bias: bool = False):
        self.head_size = head_size
        self.total_num_heads = total_num_heads
        self.total_num_kv_heads = total_num_kv_heads or total_num_heads
        tp_size = dist.get_world_size(group=process_group)
        self.num_heads = divide(self.total_num_heads, tp_size)
        self.num_kv_heads = divide(self.total_num_kv_heads, tp_size)
        input_size = hidden_size
        output_size = (self.total_num_heads + 2 * self.total_num_kv_heads) * self.head_size
        super().__init__(input_size, output_size, process_group, bias)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor,
                      loaded_shard_id: str):
        """Load Q, K, or V shard into the fused QKV weight.

        Input:
            param:           nn.Parameter — fused QKV weight
            loaded_weight:   torch.Tensor — full Q, K, or V weight from checkpoint
            loaded_shard_id: str — one of "q", "k", "v"
        """
        param_data = param.data
        assert loaded_shard_id in ["q", "k", "v"]
        if loaded_shard_id == "q":
            shard_size = self.num_heads * self.head_size
            shard_offset = 0
        elif loaded_shard_id == "k":
            shard_size = self.num_kv_heads * self.head_size
            shard_offset = self.num_heads * self.head_size
        else:
            shard_size = self.num_kv_heads * self.head_size
            shard_offset = self.num_heads * self.head_size + self.num_kv_heads * self.head_size
        param_data = param_data.narrow(self.tp_dim, shard_offset, shard_size)
        loaded_weight = loaded_weight.chunk(self.tp_size, self.tp_dim)[self.tp_rank]
        param_data.copy_(loaded_weight)


class RowParallelLinear(LinearBase):
    """Row-parallel: shards input dimension, all-reduces output.

    Input:
        x: (..., input_size // tp_size) — each rank gets different input slice

    Output:
        (..., output_size) — aggregated via all_reduce across TP group

    Data Flow:
        rank 0: x_0 @ W_0 ──┐
        rank 1: x_1 @ W_1 ──┼── all_reduce ──► y (same on all ranks)
        rank 2: x_2 @ W_2 ──┘
    """

    def __init__(self, input_size: int, output_size: int,
                 process_group: dist.ProcessGroup, bias: bool = False):
        super().__init__(input_size, output_size, process_group, 1)
        self.input_size_per_partition = divide(input_size, self.tp_size)
        self.output_size_per_partition = output_size

        self.weight = nn.Parameter(
            torch.empty(self.output_size, self.input_size_per_partition))
        self.weight.weight_loader = self.weight_loader
        if bias:
            self.bias = nn.Parameter(torch.empty(self.output_size))
            self.bias.weight_loader = self.weight_loader
        else:
            self.register_parameter("bias", None)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param_data = param.data
        shard_size = param_data.size(self.tp_dim)
        start_idx = self.tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(self.tp_dim, start_idx, shard_size)
        param_data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.linear(x, self.weight, self.bias if self.tp_rank == 0 else None)
        if self.tp_size > 1:
            dist.all_reduce(y, group=self.process_group)
        return y
