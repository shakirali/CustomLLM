"""
Model components for GPT model.

This package provides modular components for building GPT models:
- LayerNorm: Custom layer normalization
- GELU: Gaussian Error Linear Unit activation
- FeedForward: 2-layer MLP with GELU activation
- TransformerBlock: Complete transformer block with attention and feedforward
- TransformerStack: Stack of multiple TransformerBlocks
- TokenEmbedding: Token ID to embedding vector mapping
- PositionEmbedding: Position index to embedding vector mapping
- OutputProjection: Transformer output to vocabulary logits mapping
- GPTModel: Complete GPT model with forward pass and text generation
"""

from .layer_norm import LayerNorm
from .gelu import GELU
from .feedforward import FeedForward
from .transformer_block import TransformerBlock
from .transformer_stack import TransformerStack
from .token_embedding import TokenEmbedding
from .position_embedding import PositionEmbedding
from .output_projection import OutputProjection
from .gpt_model import GPTModel

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

