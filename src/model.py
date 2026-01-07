"""
Model components for GPT model (Compatibility Layer).

This module is kept for backward compatibility. All classes have been moved
to the modular structure in src/model/ package.

New code should import from src.model directly:
    from src.model import LayerNorm, GELU, FeedForward, TransformerBlock, TransformerStack, TokenEmbedding, PositionEmbedding, OutputProjection, GPTModel

This file re-exports all classes from the new modular structure.
"""

# Re-export all classes from the modular structure for backward compatibility
from .model import (
    LayerNorm,
    GELU,
    FeedForward,
    TransformerBlock,
    TransformerStack,
    TokenEmbedding,
    PositionEmbedding,
    OutputProjection,
    GPTModel,
)

__all__ = [
    "LayerNorm",
    "GELU",
    "FeedForward",
    "TransformerBlock",
    "TransformerStack",
    "TokenEmbedding",
    "PositionEmbedding",
    "OutputProjection",
    "GPTModel",
]
