"""
Layer normalization module for GPT models.
"""

import torch
import torch.nn as nn


class LayerNorm(nn.Module):
    """
    Custom layer normalization module.
    
    Normalizes activations across the embedding dimension to stabilize training.
    Uses learnable scale and shift parameters.
    
    Attributes:
        eps: Small value for numerical stability
        scale: Learnable scale parameter (initialized to 1.0)
        shift: Learnable shift parameter (initialized to 0.0)
    """
    
    def __init__(self, emb_dim: int, eps: float = 1e-5):
        """
        Initialize LayerNorm.
        
        Args:
            emb_dim: Embedding dimension (size of the last dimension to normalize)
            eps: Small epsilon value for numerical stability (default: 1e-5)
        """
        super().__init__()
        self.eps = eps
        # Learnable parameters
        self.scale = nn.Parameter(torch.ones(emb_dim))  # Initialize to 1.0
        self.shift = nn.Parameter(torch.zeros(emb_dim))  # Initialize to 0.0
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: normalize across embedding dimension.
        
        Args:
            x: Input tensor of shape (batch, seq_len, emb_dim) or (batch, emb_dim)
        
        Returns:
            Normalized tensor of same shape as input
        
        Formula:
            normalized = (x - mean) / sqrt(variance + eps)
            output = scale * normalized + shift
        """
        # Compute mean and variance across the last dimension (embedding dimension)
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        
        # Normalize
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        
        # Apply learnable scale and shift
        return self.scale * norm_x + self.shift

