"""
GELU activation function module for GPT models.
"""

import torch
import torch.nn as nn


class GELU(nn.Module):
    """
    Gaussian Error Linear Unit activation function.
    
    GELU is a smooth, non-linear activation function used in GPT models.
    It's smoother than ReLU and has been shown to work better for language models.
    
    Formula:
        GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
    """
    
    def __init__(self):
        """Initialize GELU activation (no parameters needed)."""
        super().__init__()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply GELU activation element-wise.
        
        Args:
            x: Input tensor of any shape
        
        Returns:
            Tensor of same shape with GELU activation applied
        
        Formula:
            GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
        """
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi, device=x.device, dtype=x.dtype)) *
            (x + 0.044715 * torch.pow(x, 3))
        ))

