"""Paged KV cache using MLX arrays on unified memory.

Layout:
  key_cache:   [num_blocks, num_kv_heads, block_size, head_dim]
  value_cache: [num_blocks, num_kv_heads, block_size, head_dim]

Block table:
  Maps (sequence_id, logical_block_idx) -> physical_block_idx
"""

import mlx.core as mx
from typing import Optional


class MLXPagedKVCache:
    """Paged KV cache using MLX arrays.

    This implements a paged KV cache similar to vLLM's design, but using
    MLX arrays for computation on Apple Silicon.

    Memory Layout:
        key_cache:   (num_blocks, num_kv_heads, block_size, head_dim)
        value_cache: (num_blocks, num_kv_heads, block_size, head_dim)

    Block Table:
        Maps sequence_id -> list of physical block indices
    """

    def __init__(
        self,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        num_blocks: int = 1024,
        block_size: int = 64,
        dtype: mx.Dtype = mx.bfloat16,
    ):
        """Initialize KV cache for all layers.

        Args:
            num_layers: Number of attention layers
            num_kv_heads: Number of key/value heads
            head_dim: Dimension per head
            num_blocks: Total number of cache blocks
            block_size: Tokens per block
            dtype: MLX dtype for cache
        """
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.dtype = dtype

        # Allocate KV caches for all layers
        # Shape: (num_layers, num_blocks, num_kv_heads, block_size, head_dim)
        self.key_caches: list[mx.array] = []
        self.value_caches: list[mx.array] = []

        for _ in range(num_layers):
            k_cache = mx.zeros(
                (num_blocks, num_kv_heads, block_size, head_dim),
                dtype=dtype
            )
            v_cache = mx.zeros(
                (num_blocks, num_kv_heads, block_size, head_dim),
                dtype=dtype
            )
            self.key_caches.append(k_cache)
            self.value_caches.append(v_cache)

        # Free block list (stack of available physical block indices)
        self._free_blocks: list[int] = list(range(num_blocks))

        # Block table: sequence_id -> list of physical block indices
        self._block_tables: dict[int, list[int]] = {}

        # Track which sequence owns which blocks
        self._sequence_blocks: dict[int, list[int]] = {}

    def allocate(self, seq_id: int, num_blocks: int) -> list[int]:
        """Allocate physical blocks for a sequence.

        Args:
            seq_id: Unique sequence identifier
            num_blocks: Number of blocks needed

        Returns:
            List of physical block indices allocated to the sequence
        """
        if num_blocks > len(self._free_blocks):
            raise RuntimeError(
                f"Cannot allocate {num_blocks} blocks - only "
                f"{len(self._free_blocks)} free blocks available"
            )

        # Allocate from free list (pop from end for O(1))
        allocated = []
        for _ in range(num_blocks):
            block_idx = self._free_blocks.pop()
            allocated.append(block_idx)

        # Record allocation
        self._block_tables[seq_id] = allocated
        self._sequence_blocks[seq_id] = allocated

        return allocated

    def free(self, seq_id: int) -> None:
        """Return blocks to free list.

        Args:
            seq_id: Sequence whose blocks should be freed
        """
        if seq_id not in self._sequence_blocks:
            return

        # Return blocks to free list
        for block_idx in self._sequence_blocks[seq_id]:
            self._free_blocks.append(block_idx)

        # Clean up tracking
        del self._block_tables[seq_id]
        del self._sequence_blocks[seq_id]

    def get_block_table(self, seq_id: int) -> list[int]:
        """Get block table for a sequence.

        Args:
            seq_id: Sequence identifier

        Returns:
            List of physical block indices
        """
        return self._block_tables.get(seq_id, [])

    def get_cache_arrays(
        self, layer_idx: int
    ) -> tuple[mx.array, mx.array]:
        """Get the KV cache arrays for a specific layer.

        Args:
            layer_idx: Layer index

        Returns:
            Tuple of (key_cache, value_cache) for the layer
        """
        return self.key_caches[layer_idx], self.value_caches[layer_idx]

    def get_kv_tensors(
        self,
        seq_id: int,
        layer_idx: int,
        num_tokens: int,
    ) -> tuple[mx.array, mx.array]:
        """Gather KV tensors for a sequence from a specific layer.

        Args:
            seq_id: Sequence identifier
            layer_idx: Layer index
            num_tokens: Number of tokens to gather

        Returns:
            Tuple of (k, v) tensors with shape (num_tokens, num_kv_heads, head_dim)
        """
        block_table = self.get_block_table(seq_id)
        if not block_table:
            return (
                mx.zeros((0, self.num_kv_heads, self.head_dim), dtype=self.dtype),
                mx.zeros((0, self.num_kv_heads, self.head_dim), dtype=self.dtype),
            )

        k_cache = self.key_caches[layer_idx]
        v_cache = self.value_caches[layer_idx]

        num_cached_blocks = len(block_table)
        num_cached_tokens = num_cached_blocks * self.block_size

        # Clamp to actual tokens needed
        actual_tokens = min(num_tokens, num_cached_tokens)

        # Allocate output
        k_out = mx.zeros((actual_tokens, self.num_kv_heads, self.head_dim), dtype=self.dtype)
        v_out = mx.zeros((actual_tokens, self.num_kv_heads, self.head_dim), dtype=self.dtype)

        tokens_copied = 0
        for block_i, block_global_idx in enumerate(block_table):
            block_end_in_cache = min(self.block_size, actual_tokens - tokens_copied)
            if block_end_in_cache <= 0:
                break

            dst_start = tokens_copied
            dst_end = tokens_copied + block_end_in_cache

            # Extract from cache
            # k_cache[block_idx, :, :block_end_in_cache, :]
            k_block = k_cache[block_global_idx, :, :block_end_in_cache, :]
            v_block = v_cache[block_global_idx, :, :block_end_in_cache, :]

            # Transpose: (num_kv_heads, tokens, head_dim) -> (tokens, num_kv_heads, head_dim)
            k_block = k_block.transpose(1, 0, 2)
            v_block = v_block.transpose(1, 0, 2)

            # Copy to output using at[] syntax for proper accumulation
            k_out = k_out.at[dst_start:dst_end].add(k_block - k_out[dst_start:dst_end])
            v_out = v_out.at[dst_start:dst_end].add(v_block - v_out[dst_start:dst_end])

            tokens_copied += block_end_in_cache

        return k_out, v_out

    def store_kv_tensors(
        self,
        seq_id: int,
        layer_idx: int,
        k: mx.array,
        v: mx.array,
        start_offset: int = 0,
    ) -> None:
        """Store KV tensors into the cache for a sequence.

        Args:
            seq_id: Sequence identifier
            layer_idx: Layer index
            k: Key tensor (num_tokens, num_kv_heads, head_dim)
            v: Value tensor (num_tokens, num_kv_heads, head_dim)
            start_offset: Starting token offset in the sequence's cache
        """
        block_table = self.get_block_table(seq_id)
        if not block_table:
            return

        k_cache = self.key_caches[layer_idx]
        v_cache = self.value_caches[layer_idx]

        num_tokens = k.shape[0]
        current_offset = start_offset

        for block_i, block_global_idx in enumerate(block_table):
            # Compute which tokens go in this block
            block_start = block_i * self.block_size
            block_end = block_start + self.block_size

            # Determine token range for this block
            token_start = max(0, block_start - start_offset)
            token_end = min(num_tokens, block_start + self.block_size - start_offset)

            if token_start >= token_end:
                break

            # Offset within this block
            cache_offset = current_offset % self.block_size
            tokens_to_store = token_end - token_start

            # Extract tokens to store
            k_tokens = k[token_start:token_end]
            v_tokens = v[token_start:token_end]

            # Scatter into cache
            # k_cache[block_idx, :, offset:offset+tokens, :] = k_tokens
            for t in range(tokens_to_store):
                offset = (cache_offset + t) % self.block_size
                k_cache_block = k_cache[block_global_idx, :, offset, :]
                v_cache_block = v_cache[block_global_idx, :, offset, :]

                # Use at[] for proper update
                k_cache = k_cache.at[block_global_idx, :, offset, :].add(
                    k_tokens[t] - k_cache_block
                )
                v_cache = v_cache.at[block_global_idx, :, offset, :].add(
                    v_tokens[t] - v_cache_block
                )

            current_offset += tokens_to_store

        # Update cache arrays
        self.key_caches[layer_idx] = k_cache
        self.value_caches[layer_idx] = v_cache

        # Force evaluation
        mx.eval(self.key_caches[layer_idx], self.value_caches[layer_idx])

    def get_num_free_blocks(self) -> int:
        """Get number of free blocks available."""
        return len(self._free_blocks)

    def reset(self) -> None:
        """Reset all caches and free list."""
        # Reallocate caches
        self.key_caches = []
        self.value_caches = []

        for _ in range(self.num_layers):
            k_cache = mx.zeros(
                (self.num_blocks, self.num_kv_heads, self.block_size, self.head_dim),
                dtype=self.dtype
            )
            v_cache = mx.zeros(
                (self.num_blocks, self.num_kv_heads, self.block_size, self.head_dim),
                dtype=self.dtype
            )
            self.key_caches.append(k_cache)
            self.value_caches.append(v_cache)

        # Reset free list
        self._free_blocks = list(range(self.num_blocks))

        # Clear tracking
        self._block_tables.clear()
        self._sequence_blocks.clear()
