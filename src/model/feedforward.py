"""
Feed-forward network module for GPT models.
"""

import torch
import torch.nn as nn
from .gelu import GELU


class FeedForward(nn.Module):
    """
    Feed-forward network (2-layer MLP) used in Transformer blocks.
    
    This is a simple 2-layer multi-layer perceptron that:
    1. Expands the embedding dimension (emb_dim → 4×emb_dim)
    2. Applies GELU activation
    3. Compresses back to original dimension (4×emb_dim → emb_dim)
    
    The expansion allows the model to learn more complex patterns before
    compressing back to the original embedding dimension.
    
    Attributes:
        layers: Sequential container with linear layers and GELU activation
    """
    
    def __init__(self, emb_dim: int):
        """
        Initialize FeedForward network.
        
        Args:
            emb_dim: Embedding dimension (input and output size)
                    Intermediate layer will be 4×emb_dim
        """
        super().__init__()
        self.layers = nn.Sequential(
            # First linear layer: expand to 4×embedding dimension
            nn.Linear(emb_dim, 4 * emb_dim),
            # GELU activation (adds non-linearity)
            GELU(),
            # Second linear layer: compress back to embedding dimension
            nn.Linear(4 * emb_dim, emb_dim),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through feed-forward network.
        
        Args:
            x: Input tensor of shape (batch, seq_len, emb_dim)
        
        Returns:
            Output tensor of same shape (batch, seq_len, emb_dim)
        
        Flow:
            (batch, seq_len, emb_dim)
            → Linear → (batch, seq_len, 4×emb_dim)
            → GELU → (batch, seq_len, 4×emb_dim)
            → Linear → (batch, seq_len, emb_dim)
        """
        return self.layers(x)

