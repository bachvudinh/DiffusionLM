"""
Token Embeddings for SDAR.

This module provides token embedding layers for SDAR models.

Based on JetEngine's embedding implementation:
https://github.com/Jet-Astra/SDAR/blob/main/jetengine/layers/embed_head.py

================================================================================
                              TOKEN EMBEDDINGS
================================================================================

Token embeddings map vocabulary indices to dense vectors.
SDAR uses standard learned embeddings with a vocabulary size of ~152K.

================================================================================
                              USAGE
================================================================================

    from sdar_model.layers import TokenEmbedding

    embed = TokenEmbedding(vocab_size=151936, hidden_size=4096)
    hidden_states = embed(token_ids)

"""

from typing import Optional

import torch
import torch.nn as nn


class TokenEmbedding(nn.Module):
    """
    Token embedding layer.

    Maps token indices to dense vectors for use in transformer models.

    Attributes:
        embedding: The embedding table
        padding_idx: Token index used for padding (weights not updated)
    """

    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        padding_idx: Optional[int] = None,
        device: Optional[torch.device] = None,
    ):
        """
        Initialize token embeddings.

        Args:
            vocab_size: Size of vocabulary
            hidden_size: Dimension of embedding vectors
            padding_idx: Optional index for padding token
            device: Device for layer creation
        """
        super().__init__()

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.padding_idx = padding_idx

        self.embedding = nn.Embedding(
            vocab_size,
            hidden_size,
            padding_idx=padding_idx,
            device=device,
        )

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Look up embeddings for token IDs.

        Args:
            token_ids: Token indices [batch_size, seq_len]

        Returns:
            Embeddings [batch_size, seq_len, hidden_size]
        """
        return self.embedding(token_ids)

    def extra_repr(self) -> str:
        """String representation."""
        return f"vocab_size={self.vocab_size}, hidden_size={self.hidden_size}, padding_idx={self.padding_idx}"


class LMHead(nn.Module):
    """
    Language model head.

    Projects hidden states to vocabulary logits for next-token prediction.

    Attributes:
        linear: Linear projection layer
    """

    def __init__(
        self,
        hidden_size: int,
        vocab_size: int,
        bias: bool = False,
        device: Optional[torch.device] = None,
    ):
        """
        Initialize LM head.

        Args:
            hidden_size: Hidden dimension size
            vocab_size: Size of vocabulary
            bias: Whether to use bias
            device: Device for layer creation
        """
        super().__init__()

        self.linear = nn.Linear(hidden_size, vocab_size, bias=bias, device=device)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Project hidden states to logits.

        Args:
            hidden_states: Hidden states [batch_size, seq_len, hidden_size]

        Returns:
            Logits [batch_size, seq_len, vocab_size]
        """
        return self.linear(hidden_states)
