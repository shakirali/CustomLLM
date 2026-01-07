"""
Attention module for GPT model.

This module provides a MultiHeadAttention wrapper that uses PyTorch's built-in
nn.MultiheadAttention for GPT-style models.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple


class MultiHeadAttention(nn.Module):
    """
    Multi-head attention wrapper using PyTorch's built-in nn.MultiheadAttention.
    
    This class wraps PyTorch's MultiheadAttention and adds causal masking
    to prevent attention to future tokens, enabling GPT-style autoregressive models.
    
    Attributes:
        embed_dim: Total dimension of the model
        num_heads: Number of parallel attention heads
        max_seq_len: Maximum sequence length for causal mask
        attn: PyTorch's MultiheadAttention module
        causal_mask: Upper triangular mask for causal attention
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.1,
        bias: bool = True,
        max_seq_len: int = 1024
    ):
        """
        Initialize MultiHeadAttention.
        
        Args:
            embed_dim: Total dimension of the model (must be divisible by num_heads)
            num_heads: Number of parallel attention heads
            dropout: Dropout probability on attention weights
            bias: Whether to use bias in Q/K/V projections
            max_seq_len: Maximum sequence length for causal mask
        
        Raises:
            AssertionError: If embed_dim is not divisible by num_heads
        
        Note:
            This implementation uses batch_first=True (standard practice).
            Input/output tensors should have shape (batch, seq_len, embed_dim).
        """
        super().__init__()
        
        # Validate embed_dim is divisible by num_heads
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        # Store configuration
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.max_seq_len = max_seq_len
        
        # Initialize PyTorch's built-in MultiheadAttention
        # Always use batch_first=True (modern standard practice)
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            bias=bias,
            batch_first=True  # Standard practice: batch dimension first
        )
        
        # Create causal mask (upper triangular matrix)
        # Shape: (max_seq_len, max_seq_len)
        # True = mask out (prevent attention), False = allow attention
        mask = torch.triu(torch.ones(max_seq_len, max_seq_len), diagonal=1)
        # Register as buffer (not a trainable parameter)
        self.register_buffer("causal_mask", mask.bool())
    
    def _get_causal_mask(self, seq_len: int) -> torch.Tensor:
        """
        Get causal mask for the given sequence length.
        
        Args:
            seq_len: Current sequence length
        
        Returns:
            Causal mask tensor of shape (seq_len, seq_len)
            True values indicate positions to mask out (prevent attention)
        """
        # Return the mask truncated to the current sequence length
        return self.causal_mask[:seq_len, :seq_len]
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        need_weights: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass with causal masking.
        
        Args:
            query: Query tensor of shape (batch, seq_len, embed_dim)
            key: Key tensor of shape (batch, seq_len, embed_dim)
            value: Value tensor of shape (batch, seq_len, embed_dim)
            need_weights: Whether to return attention weights
        
        Returns:
            Tuple of (output, attention_weights)
            - output: Output tensor of shape (batch, seq_len, embed_dim)
            - attention_weights: Optional attention weights if need_weights=True
        """
        # Get sequence length (batch_first=True, so seq_len is at index 1)
        seq_len = query.size(1)
        
        # Get causal mask for current sequence length
        causal_mask = self._get_causal_mask(seq_len)
        
        # Convert mask to float format for PyTorch's attention
        # PyTorch expects: -inf for masked positions, 0.0 for allowed positions
        attn_mask = causal_mask.float().masked_fill(causal_mask, float('-inf'))
        
        # Call PyTorch's MultiheadAttention with causal mask
        output, attn_weights = self.attn(
            query=query,
            key=key,
            value=value,
            attn_mask=attn_mask,
            need_weights=need_weights
        )
        
        return output, attn_weights

