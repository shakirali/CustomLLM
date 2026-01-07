"""
Transformer block module for GPT models.
"""

import torch
import torch.nn as nn

# Import MultiHeadAttention - use relative import since we're in the same package
try:
    from ..attention import MultiHeadAttention
except ImportError:
    # Fallback for when running as script
    from src.attention import MultiHeadAttention

from .layer_norm import LayerNorm
from .feedforward import FeedForward


class TransformerBlock(nn.Module):
    """
    Transformer block used in GPT models.
    
    This block combines MultiHeadAttention and FeedForward with:
    - Pre-norm architecture (LayerNorm before each component)
    - Residual connections (skip connections)
    - Dropout for regularization
    
    The Pre-Norm architecture applies LayerNorm before attention and feedforward,
    which has been shown to improve training stability compared to Post-Norm.
    
    Architecture:
        Input
          ↓
        LayerNorm1 → MultiHeadAttention → Dropout → Add (residual)
          ↓
        LayerNorm2 → FeedForward → Dropout → Add (residual)
          ↓
        Output
    
    Attributes:
        attention: MultiHeadAttention layer
        feedforward: FeedForward layer
        norm1: LayerNorm before attention
        norm2: LayerNorm before feedforward
        dropout1: Dropout after attention
        dropout2: Dropout after feedforward
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.1,
        max_seq_len: int = 1024,
        bias: bool = True
    ):
        """
        Initialize TransformerBlock.
        
        Args:
            embed_dim: Embedding dimension (must be divisible by num_heads)
            num_heads: Number of attention heads
            dropout: Dropout probability (applied after attention and feedforward)
            max_seq_len: Maximum sequence length for causal mask
            bias: Whether to use bias in attention projections
        """
        super().__init__()
        
        # Multi-head attention layer
        self.attention = MultiHeadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            bias=bias,
            max_seq_len=max_seq_len
        )
        
        # Feed-forward network
        self.feedforward = FeedForward(emb_dim=embed_dim)
        
        # Layer normalization layers (Pre-Norm architecture)
        self.norm1 = LayerNorm(emb_dim=embed_dim)
        self.norm2 = LayerNorm(emb_dim=embed_dim)
        
        # Dropout layers for regularization
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through transformer block.
        
        Args:
            x: Input tensor of shape (batch, seq_len, embed_dim)
        
        Returns:
            Output tensor of same shape (batch, seq_len, embed_dim)
        
        Flow:
            (batch, seq_len, embed_dim)
            → LayerNorm1 → Attention → Dropout → Add residual
            → LayerNorm2 → FeedForward → Dropout → Add residual
            → (batch, seq_len, embed_dim)
        """
        # Pre-norm attention with residual connection
        # Apply LayerNorm before attention
        norm_x = self.norm1(x)
        
        # Self-attention (query, key, value all from normalized input)
        attn_output, _ = self.attention(norm_x, norm_x, norm_x, need_weights=False)
        
        # Apply dropout and add residual
        x = x + self.dropout1(attn_output)
        
        # Pre-norm feedforward with residual connection
        # Apply LayerNorm before feedforward
        norm_x = self.norm2(x)
        
        # Feed-forward network
        ff_output = self.feedforward(norm_x)
        
        # Apply dropout and add residual
        x = x + self.dropout2(ff_output)
        
        return x

