"""
SDAR Model - Synergy of Diffusion and AutoRegression.

This module provides the complete SDAR model implementation in pure PyTorch,
including the transformer backbone and language model head.

Based on JetEngine's SDAR implementation:
https://github.com/Jet-Astra/SDAR

================================================================================
                              ARCHITECTURE
================================================================================

SDAR uses a standard decoder-only transformer architecture with:
- VocabParallelEmbedding for token embeddings
- SDARDecoderLayer (xN) with:
  - SDARAttention (GQA) with RMSNorm and RoPE
  - SDARMLP (SiLU/SWiGLU)
- Final RMSNorm
- ParallelLMHead for logits

Key differences from standard GPT:
- GQA (Grouped Query Attention) for memory efficiency
- Per-head Q/K RMSNorm before RoPE
- SiLU/SWiGLU activation

================================================================================
                              USAGE
================================================================================

    from vdllm.models import SDARForCausalLM, SDARConfig

    config = SDARConfig()
    model = SDARForCausalLM(config)

    # Forward pass
    logits = model(input_ids, positions)

"""

from typing import Optional, Tuple

import torch
import torch.nn as nn

from .config import SDARConfig
from vdllm.layers import (
    RMSNorm,
    SDARAttention,
    SDARMLP,
    TokenEmbedding,
    LMHead,
    precompute_freqs_cis,
    apply_rotary_pos_emb,
)


class SDARDecoderLayer(nn.Module):
    """
    Single SDAR decoder layer.

    A decoder layer consists of:
    1. Input RMSNorm
    2. Self-attention (SDARAttention)
    3. Post-attention RMSNorm
    4. MLP (SDARMLP)

    The layer uses residual connections and RMSNorm's streaming design.
    """

    def __init__(self, config: SDARConfig, device: Optional[torch.device] = None):
        """
        Initialize decoder layer.

        Args:
            config: SDAR configuration
            device: Device for layer creation
        """
        super().__init__()

        self.hidden_size = config.hidden_size

        # Self-attention with GQA
        self.self_attn = SDARAttention(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            head_dim=config.head_dim,
            rope_theta=config.rope_theta,
            rope_scaling=config.rope_scaling,
            rms_norm_eps=config.rms_norm_eps,
            bias=config.attention_bias,
            device=device,
        )

        # MLP
        self.mlp = SDARMLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            device=device,
        )

        # RMSNorm layers
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for decoder layer.

        Args:
            positions: Position indices [batch_size, seq_len]
            hidden_states: Input hidden states
            attention_mask: Optional attention mask

        Returns:
            Tuple of (output hidden_states, residual)
        """
        # Self-attention with RMSNorm (streaming residual)
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        # SDAR attention uses RoPE internally
        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
            attention_mask=attention_mask,
        )

        # Add residual connection
        hidden_states = residual + hidden_states

        # MLP with RMSNorm (streaming residual)
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states, residual


class SDARModel(nn.Module):
    """
    SDAR Transformer backbone.

    This is the core transformer model without the language model head.
    """

    def __init__(self, config: SDARConfig, device: Optional[torch.device] = None):
        """
        Initialize SDAR model.

        Args:
            config: SDAR configuration
            device: Device for model creation
        """
        super().__init__()

        self.config = config

        # Token embeddings
        self.embed_tokens = TokenEmbedding(
            vocab_size=config.vocab_size,
            hidden_size=config.hidden_size,
            padding_idx=None,
            device=device,
        )

        # Transformer layers
        self.layers = nn.ModuleList([
            SDARDecoderLayer(config, device=device)
            for _ in range(config.num_hidden_layers)
        ])

        # Final layer norm
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):
        """Initialize model weights."""
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass for SDAR model.

        Args:
            input_ids: Token indices [batch_size, seq_len]
            positions: Position indices (if None, computed from input_ids)
            attention_mask: Optional attention mask

        Returns:
            Hidden states [batch_size, seq_len, hidden_size]
        """
        batch_size, seq_len = input_ids.shape

        # Token embeddings
        hidden_states = self.embed_tokens(input_ids)

        # Compute positions if not provided
        if positions is None:
            positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)

        # Transformer layers
        for layer in self.layers:
            hidden_states, _ = layer(
                positions=positions,
                hidden_states=hidden_states,
                attention_mask=attention_mask,
            )

        # Final layer norm
        hidden_states, _ = self.norm(hidden_states, None)

        return hidden_states


class SDARForCausalLM(nn.Module):
    """
    SDAR model for causal language modeling.

    This combines the SDAR transformer backbone with a language model head
    for next-token prediction.

    This is the main entry point for SDAR inference.
    """

    def __init__(self, config: SDARConfig, device: Optional[torch.device] = None):
        """
        Initialize SDAR causal LM.

        Args:
            config: SDAR configuration
            device: Device for model creation
        """
        super().__init__()

        self.config = config

        # Transformer backbone
        self.model = SDARModel(config, device=device)

        # Language model head
        self.lm_head = LMHead(
            hidden_size=config.hidden_size,
            vocab_size=config.vocab_size,
            device=device,
        )

        # Tie weights between embeddings and LM head (optional)
        if config.tie_word_embeddings:
            self.lm_head.linear.weight = self.model.embed_tokens.embedding.weight

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):
        """Initialize model weights."""
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass for causal LM.

        Args:
            input_ids: Token indices [batch_size, seq_len]
            positions: Position indices
            attention_mask: Optional attention mask

        Returns:
            Logits [batch_size, seq_len, vocab_size]
        """
        # Get hidden states from transformer
        hidden_states = self.model(
            input_ids=input_ids,
            positions=positions,
            attention_mask=attention_mask,
        )

        # Compute logits
        logits = self.lm_head(hidden_states)

        return logits

    def compute_loss(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute cross-entropy loss.

        Args:
            input_ids: Input token indices
            labels: Target token indices

        Returns:
            Loss scalar
        """
        logits = self.forward(input_ids)

        # Shift for causal LM: predict next token
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        # Flatten and compute loss
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(
            shift_logits.view(-1, self.config.vocab_size),
            shift_labels.view(-1),
        )

        return loss

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
    ) -> torch.Tensor:
        """
        Generate tokens autoregressively.

        Args:
            input_ids: Input token indices [batch_size, seq_len]
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0 = greedy)
            top_k: Top-k filtering
            top_p: Top-p (nucleus) filtering

        Returns:
            Generated token indices
        """
        self.eval()

        batch_size, seq_len = input_ids.shape
        generated = input_ids.clone()

        for _ in range(max_new_tokens):
            # Forward pass
            logits = self.forward(generated)

            # Get logits for last position
            next_token_logits = logits[:, -1, :]

            # Apply temperature
            if temperature > 0:
                next_token_logits = next_token_logits / temperature

            # Apply top-k filtering
            if top_k > 0:
                v, _ = torch.topk(next_token_logits, top_k)
                next_token_logits[next_token_logits < v[:, [-1]]] = float('-inf')

            # Apply top-p filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cum_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)

                sorted_indices_to_remove = cum_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0

                for i in range(batch_size):
                    indices_to_remove = sorted_indices[i][sorted_indices_to_remove[i]]
                    next_token_logits[i, indices_to_remove] = float('-inf')

            # Sample or greedy
            if temperature > 0:
                probs = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

            # Append to generated
            generated = torch.cat([generated, next_token], dim=-1)

        return generated
