"""
Metal KV Cache - Block-wise KV Cache for Diffusion.

This module provides KV cache management optimized for block diffusion
inference. Unlike autoregressive KV cache (stores all past tokens),
block diffusion caches per-block and reuses across denoising steps.

Architecture patterns adapted from vllm-metal:
- vllm_metal/metal_kernel_backend/cache.py - MetalPagedKVCache

================================================================================
                              KEY INSIGHT
================================================================================

In block diffusion:
- Block-external attention is STABLE across denoising steps
- Only block-internal attention fluctuates during denoising

So we cache per-block K/V and only recompute the current block.

================================================================================
                              USAGE
================================================================================

    from engine.metal.kvcache import BlockKVCache

    cache = BlockKVCache(
        num_layers=32,
        num_kv_heads=8,
        head_dim=128,
        max_blocks=256,
        block_size=4,
    )

    # Store block
    cache.store(block_idx=0, layer_idx=0, k=k_block, v=v_block)

    # Retrieve block
    k, v = cache.get(block_idx=0, layer_idx=0)

"""

import torch
from typing import Optional, Tuple


class BlockKVCache:
    """
    Block-wise KV cache for semi-AR inference.

    This cache stores K/V tensors per block rather than per token,
    which is more efficient for block diffusion where the same
    block is processed multiple times during denoising.

    Attributes:
        num_layers: Number of transformer layers
        num_kv_heads: Number of key/value attention heads
        head_dim: Dimension of each attention head
        max_blocks: Maximum number of blocks that can be cached
        block_size: Size of each block in tokens
    """

    def __init__(
        self,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        max_blocks: int,
        block_size: int,
        dtype: torch.dtype = torch.float16,
        device: torch.device = torch.device("cpu"),
    ):
        """
        Initialize block KV cache.

        Args:
            num_layers: Number of transformer layers
            num_kv_heads: Number of key/value attention heads
            head_dim: Dimension of each attention head
            max_blocks: Maximum number of blocks that can be cached
            block_size: Size of each block in tokens
            dtype: Data type for cache tensors
            device: Device to allocate cache on
        """
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.max_blocks = max_blocks
        self.block_size = block_size
        self.dtype = dtype
        self.device = device

        # Allocate cache tensors
        # Layout: [num_layers, max_blocks, num_kv_heads, block_size, head_dim]
        self.key_cache = torch.zeros(
            num_layers, max_blocks, num_kv_heads, block_size, head_dim,
            dtype=dtype, device=device
        )
        self.value_cache = torch.zeros(
            num_layers, max_blocks, num_kv_heads, block_size, head_dim,
            dtype=dtype, device=device
        )

        # Track which blocks have been filled
        self._filled_blocks = torch.zeros(max_blocks, dtype=torch.bool, device=device)

    def store(
        self,
        block_idx: int,
        layer_idx: int,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> None:
        """
        Store K/V tensors for a specific block and layer.

        Args:
            block_idx: Index of the block
            layer_idx: Index of the layer
            k: Key tensor [num_kv_heads, block_size, head_dim]
            v: Value tensor [num_kv_heads, block_size, head_dim]
        """
        self.key_cache[layer_idx, block_idx] = k.to(self.dtype)
        self.value_cache[layer_idx, block_idx] = v.to(self.dtype)
        self._filled_blocks[block_idx] = True

    def get(
        self,
        block_idx: int,
        layer_idx: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieve K/V tensors for a specific block and layer.

        Args:
            block_idx: Index of the block
            layer_idx: Index of the layer

        Returns:
            Tuple of (key_tensor, value_tensor)
        """
        return (
            self.key_cache[layer_idx, block_idx],
            self.value_cache[layer_idx, block_idx],
        )

    def get_packed(
        self,
        block_indices: list[int],
        layer_idx: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get K/V tensors packed together for multiple blocks.

        Args:
            block_indices: List of block indices
            layer_idx: Index of the layer

        Returns:
            Tuple of (packed_keys, packed_values)
                packed_keys: [len(block_indices), num_kv_heads, block_size * len(block_indices), head_dim]
                packed_values: [len(block_indices), num_kv_heads, block_size * len(block_indices), head_dim]
        """
        keys = [self.key_cache[layer_idx, i] for i in block_indices]
        values = [self.value_cache[layer_idx, i] for i in block_indices]

        # Concatenate along sequence dimension
        packed_keys = torch.cat(keys, dim=1)  # [num_kv_heads, total_seq, head_dim]
        packed_values = torch.cat(values, dim=1)

        return packed_keys, packed_values

    def is_filled(self, block_idx: int) -> bool:
        """Check if a block has been filled."""
        return self._filled_blocks[block_idx].item()

    def num_filled_blocks(self) -> int:
        """Get number of filled blocks."""
        return self._filled_blocks.sum().item()

    def reset(self) -> None:
        """Reset the cache (zero out all tensors)."""
        self.key_cache.zero_()
        self.value_cache.zero_()
        self._filled_blocks.zero_()

    def reset_block(self, block_idx: int) -> None:
        """Reset a specific block."""
        self.key_cache[:, block_idx] = 0
        self.value_cache[:, block_idx] = 0
        self._filled_blocks[block_idx] = False


class PagedKVCache:
    """
    Paged KV cache with physical block allocation.

    This is similar to vllm's paged attention but optimized for
    block diffusion. It maps logical token positions to physical
    cache blocks.

    Architecture patterns from vllm-metal:
    - vllm_metal/metal_kernel_backend/cache.py - MetalPagedKVCache
    """

    def __init__(
        self,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        num_blocks: int,
        block_size: int = 16,
        dtype: torch.dtype = torch.float16,
        device: torch.device = torch.device("cpu"),
    ):
        """
        Initialize paged KV cache.

        Args:
            num_layers: Number of transformer layers
            num_kv_heads: Number of key/value attention heads
            head_dim: Dimension of each attention head
            num_blocks: Number of physical cache blocks
            block_size: Size of each block in tokens
            dtype: Data type for cache tensors
            device: Device to allocate cache on
        """
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.dtype = dtype
        self.device = device

        # Physical cache: [num_layers, num_blocks, num_kv_heads, block_size, head_dim]
        self.key_cache = torch.zeros(
            num_layers, num_blocks, num_kv_heads, block_size, head_dim,
            dtype=dtype, device=device
        )
        self.value_cache = torch.zeros(
            num_layers, num_blocks, num_kv_heads, block_size, head_dim,
            dtype=dtype, device=device
        )

        # Block allocation tracking
        self._free_blocks: set[int] = set(range(num_blocks))
        self._block_allocations: dict[int, list[int]] = {}  # seq_pos -> block_idx

    def allocate(self, num_tokens: int) -> list[int]:
        """
        Allocate cache blocks for a sequence of tokens.

        Args:
            num_tokens: Number of tokens to allocate

        Returns:
            List of block indices
        """
        num_blocks_needed = (num_tokens + self.block_size - 1) // self.block_size

        if len(self._free_blocks) < num_blocks_needed:
            raise RuntimeError(f"Out of cache blocks: need {num_blocks_needed}, have {len(self._free_blocks)}")

        allocated = []
        for _ in range(num_blocks_needed):
            block_idx = self._free_blocks.pop()
            allocated.append(block_idx)

        return allocated

    def store(
        self,
        token_start: int,
        layer_idx: int,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> None:
        """
        Store K/V tensors starting at a token position.

        Args:
            token_start: Starting token position
            layer_idx: Index of the layer
            k: Key tensor [num_kv_heads, seq_len, head_dim]
            v: Value tensor [num_kv_heads, seq_len, head_dim]
        """
        seq_len = k.shape[1]
        num_blocks_needed = (token_start + seq_len + self.block_size - 1) // self.block_size - token_start // self.block_size

        block_idx = token_start // self.block_size
        offset = token_start % self.block_size

        for i in range(seq_len):
            current_block = block_idx + i // self.block_size
            current_offset = (offset + i) % self.block_size

            self.key_cache[layer_idx, current_block, :, current_offset, :] = k[:, i, :]
            self.value_cache[layer_idx, current_block, :, current_offset, :] = v[:, i, :]

    def get(
        self,
        token_start: int,
        seq_len: int,
        layer_idx: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieve K/V tensors for a sequence range.

        Args:
            token_start: Starting token position
            seq_len: Number of tokens
            layer_idx: Index of the layer

        Returns:
            Tuple of (keys, values)
        """
        k_out = torch.zeros(
            self.num_kv_heads, seq_len, self.head_dim,
            dtype=self.dtype, device=self.device
        )
        v_out = torch.zeros(
            self.num_kv_heads, seq_len, self.head_dim,
            dtype=self.dtype, device=self.device
        )

        for i in range(seq_len):
            block_idx = (token_start + i) // self.block_size
            offset = (token_start + i) % self.block_size
            k_out[:, i, :] = self.key_cache[layer_idx, block_idx, :, offset, :]
            v_out[:, i, :] = self.value_cache[layer_idx, block_idx, :, offset, :]

        return k_out, v_out

    def free(self, block_indices: list[int]) -> None:
        """Free allocated blocks."""
        self._free_blocks.update(block_indices)

    def reset(self) -> None:
        """Reset the cache."""
        self.key_cache.zero_()
        self.value_cache.zero_()
        self._free_blocks = set(range(self.num_blocks))
        self._block_allocations.clear()
