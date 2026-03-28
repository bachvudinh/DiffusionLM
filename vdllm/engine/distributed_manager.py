"""Distributed process group manager for tensor/data parallelism.

Derived from JetEngine by Yihan Bian et al.
Reference: https://github.com/Labman42/JetEngine

Architecture Overview
====================

    Manages multi-GPU parallelism using accelerate.PartialState.
    Creates two orthogonal process groups for hybrid parallelism:

    TP (Tensor Parallel) — splits model parameters across GPUs
    DP (Data Parallel)   — replicates model, splits data across GPUs

    Example: 8 GPUs with tp_size=4
    ┌─────────────────────────────────────────────────────┐
    │                  8 GPU processes                     │
    │                                                     │
    │  TP Group 0: [GPU0, GPU1, GPU2, GPU3]              │
    │  TP Group 1: [GPU4, GPU5, GPU6, GPU7]              │
    │                                                     │
    │  DP Group 0: [GPU0, GPU4]                          │
    │  DP Group 1: [GPU1, GPU5]                          │
    │  DP Group 2: [GPU2, GPU6]                          │
    │  DP Group 3: [GPU3, GPU7]                          │
    │                                                     │
    │  Properties:                                        │
    │    .device    → torch device for this process       │
    │    .tp_rank   → rank within TP group (0..tp_size-1) │
    │    .dp_rank   → rank within DP group (0..dp_size-1) │
    │    .tp_group  → ProcessGroup for TP collectives     │
    │    .dp_group  → ProcessGroup for DP collectives     │
    └─────────────────────────────────────────────────────┘
"""

from accelerate import PartialState
import torch.distributed as dist


class DistributedManager:
    """Manages TP and DP process groups for multi-GPU inference.

    Input:
        tp_size: int — number of GPUs per tensor-parallel group

    Output:
        Manager instance with .tp_group, .dp_group, .device, etc.
    """

    def __init__(self, tp_size: int):
        self.state = PartialState()
        self.tp_size = tp_size
        self.dp_size = self.state.num_processes // self.tp_size
        self.tp_group = None
        self.dp_group = None

        # Create tensor parallel (TP) process groups
        for i in range(self.dp_size):
            ranks = list(range(i * self.tp_size, (i + 1) * self.tp_size))
            group = dist.new_group(ranks)
            if self.state.process_index in ranks:
                self.tp_group = group
                self.tp_rank = self.state.process_index % self.tp_size

        # Create data parallel (DP) process groups
        for i in range(self.tp_size):
            ranks = list(range(i, self.state.num_processes, self.tp_size))
            group = dist.new_group(ranks)
            if self.state.process_index in ranks:
                self.dp_group = group
                self.dp_rank = self.state.process_index // self.tp_size

    @property
    def device(self):
        return self.state.device

    def wait_for_everyone(self):
        self.state.wait_for_everyone()
