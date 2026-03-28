"""KV cache block allocation with prefix caching.

Derived from JetEngine by Yihan Bian et al.
Reference: https://github.com/Labman42/JetEngine

Architecture Overview
====================

    BlockManager maps logical sequence positions to physical KV cache blocks.
    Each block holds `block_size` token KV pairs.

    Physical KV Cache Layout:
    ┌──────────────────────────────────────────────────────────────────┐
    │  kv_cache: [2, num_layers, num_blocks, block_size, heads, dim]  │
    │                                                                   │
    │  Block 0:  [slot_0, slot_1, ..., slot_255]                       │
    │  Block 1:  [slot_256, slot_257, ..., slot_511]                   │
    │  Block 2:  [slot_512, slot_513, ..., slot_767]                   │
    │  ...                                                              │
    └──────────────────────────────────────────────────────────────────┘

    Prefix Caching:
    ┌──────────────────────────────────────────────────────────────────┐
    │                                                                   │
    │  Sequence A: "The quick brown fox" → blocks [0, 1]               │
    │  Sequence B: "The quick brown dog" → blocks [0, 2]               │
    │                                         ▲                         │
    │                                         │                         │
    │                              Block 0 shared (same prefix hash)    │
    │                              ref_count = 2                        │
    │                                                                   │
    │  When Sequence A finishes:                                        │
    │    Block 0: ref_count = 1 (still used by B)                      │
    │    Block 1: ref_count = 0 → freed                                │
    └──────────────────────────────────────────────────────────────────┘

    Input:
        num_blocks: int  — total physical blocks available
        block_size: int  — tokens per physical block

    Key Methods:
        allocate(seq)   — assign physical blocks to sequence
        deallocate(seq) — release blocks when sequence finishes
        append_blocks() — add more blocks as sequence grows
"""

from collections import deque
import xxhash
import numpy as np

from vdllm.engine.sequence import Sequence


class Block:

    def __init__(self, block_id):
        self.block_id = block_id
        self.ref_count = 0
        self.hash = -1
        self.token_ids = []

    def update(self, hash: int, token_ids: list[int]):
        self.hash = hash
        self.token_ids = token_ids

    def reset(self):
        self.ref_count = 1
        self.hash = -1
        self.token_ids = []


class BlockManager:

    def __init__(self, num_blocks: int, block_size: int):
        assert num_blocks > 0
        self.block_size = block_size
        self.blocks: list[Block] = [Block(i) for i in range(num_blocks)]
        self.hash_to_block_id: dict[int, int] = dict()
        self.free_block_ids: set[int] = set(range(num_blocks))
        self.used_block_ids: set[int] = set()

    @classmethod
    def compute_hash(cls, token_ids: list[int], prefix: int = -1):
        h = xxhash.xxh64()
        if prefix != -1:
            h.update(prefix.to_bytes(8, "little"))
        h.update(np.array(token_ids).tobytes())
        return h.intdigest()

    def _allocate_block(self, block_id: int) -> Block:
        block = self.blocks[block_id]
        assert block.ref_count == 0
        if block.hash != -1:
            if self.hash_to_block_id.get(block.hash) == block_id:
                del self.hash_to_block_id[block.hash]
        block.reset()
        if block_id in self.free_block_ids:
            self.free_block_ids.remove(block_id)
        self.used_block_ids.add(block_id)
        return self.blocks[block_id]

    def _deallocate_block(self, block_id: int) -> Block:
        assert self.blocks[block_id].ref_count == 0
        self.used_block_ids.remove(block_id)
        self.free_block_ids.add(block_id)

    def can_allocate(self, seq: Sequence) -> bool:
        return len(self.free_block_ids) >= seq.num_blocks

    def allocate(self, seq: Sequence):
        assert not seq.block_table
        h = -1
        cache_miss = False
        num_blocks = seq.num_blocks

        for i in range(num_blocks):
            token_ids = seq.block(i)
            if len(token_ids) == self.block_size:
                h = self.compute_hash(token_ids, h)
            else:
                h = -1

            block_id = self.hash_to_block_id.get(h, -1)
            if block_id == -1 or self.blocks[block_id].token_ids != token_ids:
                cache_miss = True

            if cache_miss:
                if not self.free_block_ids:
                    raise ValueError("No free blocks available")
                block_id = self.free_block_ids.pop()
                block = self._allocate_block(block_id)
            else:
                seq.num_cached_tokens += self.block_size
                if block_id in self.used_block_ids:
                    block = self.blocks[block_id]
                    block.ref_count += 1
                else:
                    block = self._allocate_block(block_id)

            if h != -1:
                if block.hash != -1 and block.hash != h:
                     if self.hash_to_block_id.get(block.hash) == block_id:
                         del self.hash_to_block_id[block.hash]
                block.update(h, token_ids)
                self.hash_to_block_id[h] = block_id
            else:
                block.token_ids = token_ids

            seq.block_table.append(block_id)

    def allocate_batch(self, seqs: list[Sequence]):
        for seq in seqs:
            self.allocate(seq)

    def deallocate(self, seq: Sequence):
        for block_id in reversed(seq.block_table):
            block = self.blocks[block_id]
            block.ref_count -= 1
            if block.ref_count == 0:
                self._deallocate_block(block_id)
        seq.num_cached_tokens = 0
        seq.block_table.clear()

    def can_append_blocks(self, num_blocks: int) -> bool:
        return len(self.free_block_ids) >= num_blocks

    def append_blocks(self, seq: Sequence, num_blocks: int):
        for _ in range(num_blocks):
            if not self.free_block_ids:
                 raise ValueError("No free blocks available")
            block_id = self.free_block_ids.pop()
            block = self.blocks[block_id]
            assert block.ref_count == 0
            if block.hash != -1:
                if self.hash_to_block_id.get(block.hash) == block_id:
                    del self.hash_to_block_id[block.hash]
            block.reset()
            self.used_block_ids.add(block_id)
            seq.block_table.append(block_id)

    def append_blocks_batch(self, seqs_and_counts: list[tuple[Sequence, int]]):
        for seq, count in seqs_and_counts:
            self.append_blocks(seq, count)

    def can_append(self, seq: Sequence) -> bool:
        return len(self.free_block_ids) >= (len(seq) % self.block_size == 1)

    def may_append(self, seq: Sequence):
        block_table = seq.block_table
        last_block = self.blocks[block_table[-1]]
        if len(seq) % self.block_size == 1:
            assert last_block.hash != -1
            block_id = next(iter(self.free_block_ids))
            self._allocate_block(block_id)
            block_table.append(block_id)
        elif len(seq) % self.block_size == 0:
            assert last_block.hash == -1
            token_ids = seq.block(seq.num_blocks-1)
            prefix = self.blocks[block_table[-2]].hash if len(block_table) > 1 else -1
            h = self.compute_hash(token_ids, prefix)
            last_block.update(h, token_ids)
            self.hash_to_block_id[h] = last_block.block_id
        else:
            assert last_block.hash == -1
