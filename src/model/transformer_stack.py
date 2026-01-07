"""
Transformer stack module for GPT models.

This module stacks multiple TransformerBlocks to create a deep transformer network.
"""

import torch
import torch.nn as nn
from .transformer_block import TransformerBlock


class TransformerStack(nn.Module):
    """
    Stack of TransformerBlocks for GPT models.
    
    This module creates a configurable number of TransformerBlocks and processes
    input sequentially through all of them. Each block refines the representation
    from the previous block, allowing the model to learn increasingly complex patterns.
    
    Architecture:
        Input (batch, seq_len, embed_dim)
          ↓
        TransformerBlock 0
          ↓
        TransformerBlock 1
          ↓
        ...
          ↓
        TransformerBlock N-1
          ↓
        Output (batch, seq_len, embed_dim)
    
    Attributes:
        layers: ModuleList containing num_layers TransformerBlock instances
        num_layers: Number of transformer blocks in the stack
        embed_dim: Embedding dimension (same across all blocks)
    """
    
    def __init__(
        self,
        num_layers: int,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.1,
        max_seq_len: int = 1024,
        bias: bool = True
    ):
        """
        Initialize TransformerStack.
        
        Args:
            num_layers: Number of transformer blocks to stack
                       (e.g., 12 for GPT-2 small, 24 for GPT-2 medium)
            embed_dim: Embedding dimension (must be divisible by num_heads)
                      (e.g., 768 for GPT-2 small)
            num_heads: Number of attention heads per block
                      (e.g., 12 for GPT-2 small)
            dropout: Dropout probability (applied in each block)
            max_seq_len: Maximum sequence length for causal masks
            bias: Whether to use bias in attention projections
        
        Note:
            All blocks share the same configuration (embed_dim, num_heads, etc.).
            Each block has its own independent parameters (weights, biases).
        """
        super().__init__()
        
        self.num_layers = num_layers
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        
        # Create a list of transformer blocks
        # Using ModuleList to properly register all blocks as submodules
        self.layers = nn.ModuleList([
            TransformerBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                dropout=dropout,
                max_seq_len=max_seq_len,
                bias=bias
            )
            for _ in range(num_layers)
        ])
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through all transformer blocks.
        
        Args:
            x: Input tensor of shape (batch, seq_len, embed_dim)
               Typically the output from token + position embeddings
        
        Returns:
            Output tensor of same shape (batch, seq_len, embed_dim)
            Processed through all transformer blocks
        
        Flow:
            (batch, seq_len, embed_dim)
            → TransformerBlock 0 → (batch, seq_len, embed_dim)
            → TransformerBlock 1 → (batch, seq_len, embed_dim)
            → ...
            → TransformerBlock N-1 → (batch, seq_len, embed_dim)
        
        Example:
            >>> stack = TransformerStack(num_layers=12, embed_dim=768, num_heads=12)
            >>> x = torch.randn(2, 10, 768)  # (batch=2, seq_len=10, embed_dim=768)
            >>> output = stack(x)
            >>> output.shape
            torch.Size([2, 10, 768])
        """
        # Process input through each transformer block sequentially
        # Each block refines the representation from the previous block
        for layer in self.layers:
            x = layer(x)
        
        return x
    
    def __len__(self) -> int:
        """
        Return the number of transformer blocks in the stack.
        
        Returns:
            Number of layers (transformer blocks)
        """
        return len(self.layers)
    
    def __repr__(self) -> str:
        """
        String representation of the TransformerStack.
        
        Returns:
            String describing the stack configuration
        """
        return (
            f"TransformerStack(\n"
            f"  num_layers={self.num_layers},\n"
            f"  embed_dim={self.embed_dim},\n"
            f"  num_heads={self.num_heads}\n"
            f")"
        )

