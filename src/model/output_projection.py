"""
Output projection module for GPT models.
"""

import torch
import torch.nn as nn
from typing import Optional


class OutputProjection(nn.Module):
    """
    Output projection layer for GPT models.
    
    Converts transformer output embeddings to vocabulary logits (raw scores)
    for next-token prediction. This is the final step in the GPT forward pass
    that produces predictions over the vocabulary.
    
    This is the last step in the GPT model pipeline:
    Transformer Output → Output Projection → Vocabulary Logits
    
    The layer can optionally use weight tying, where the projection weights
    are shared with the token embedding layer. This reduces parameters and
    has been shown to improve generalization in language models.
    
    Attributes:
        projection: Linear layer that maps embed_dim to vocab_size
        embed_dim: Embedding dimension (input size)
        vocab_size: Vocabulary size (output size)
        tie_weights: Whether weights are tied with embedding layer
    """
    
    def __init__(
        self,
        embed_dim: int,
        vocab_size: int,
        tie_weights: bool = False,
        embedding_layer: Optional[nn.Embedding] = None,
        bias: bool = False
    ):
        """
        Initialize OutputProjection layer.
        
        Args:
            embed_dim: Embedding dimension (must match transformer output)
                      (e.g., 768 for GPT-2 small)
            vocab_size: Size of the vocabulary (number of unique tokens)
                       (e.g., 50257 for GPT-2)
            tie_weights: If True, share weights with embedding_layer
                        This reduces parameters and improves generalization
            embedding_layer: Token embedding layer to tie weights with
                           Required if tie_weights=True
            bias: Whether to use bias in the linear layer
                 Typically False for output projection in GPT models
        
        Note:
            If tie_weights=True, the projection weight matrix shares the same
            weights as the embedding weight matrix. This works because:
            - nn.Embedding weight shape: (vocab_size, embed_dim)
            - nn.Linear weight shape: (vocab_size, embed_dim) - same shape!
            - PyTorch's nn.Linear stores weights as (out_features, in_features)
            - For nn.Linear(embed_dim, vocab_size), weight is (vocab_size, embed_dim)
            - So we can directly share: linear_weight = embedding_weight
            
            Weight tying is a common technique in GPT models that:
            - Reduces model parameters (one matrix instead of two)
            - Improves generalization by sharing representations
            - Is used in GPT-2, GPT-3, and other modern language models
        """
        super().__init__()
        
        self.embed_dim = embed_dim
        self.vocab_size = vocab_size
        self.tie_weights = tie_weights
        
        if tie_weights:
            if embedding_layer is None:
                raise ValueError(
                    "embedding_layer must be provided when tie_weights=True"
                )
            if embedding_layer.num_embeddings != vocab_size:
                raise ValueError(
                    f"embedding_layer vocab_size ({embedding_layer.num_embeddings}) "
                    f"must match vocab_size ({vocab_size})"
                )
            if embedding_layer.embedding_dim != embed_dim:
                raise ValueError(
                    f"embedding_layer embed_dim ({embedding_layer.embedding_dim}) "
                    f"must match embed_dim ({embed_dim})"
                )
            
            # Create linear layer without initializing weights
            # We'll tie the weights manually
            self.projection = nn.Linear(embed_dim, vocab_size, bias=bias)
            
            # Tie weights: linear.weight = embedding.weight
            # nn.Embedding weight: (vocab_size, embed_dim)
            # nn.Linear weight: (vocab_size, embed_dim) - same shape!
            # PyTorch's nn.Linear stores weights as (out_features, in_features)
            # So for nn.Linear(embed_dim, vocab_size), weight is (vocab_size, embed_dim)
            # This matches the embedding weight shape exactly
            # 
            # IMPORTANT: We directly share the same weight tensor (not a copy!)
            # This ensures:
            # 1. True weight sharing (same memory)
            # 2. Correct parameter counting (counted only once in embedding)
            # 3. No need to sync weights in forward pass
            self.projection.weight = embedding_layer.weight
        else:
            # Create standard linear layer with independent weights
            self.projection = nn.Linear(embed_dim, vocab_size, bias=bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: convert embeddings to vocabulary logits.
        
        Args:
            x: Input tensor from transformer stack
               Shape: (batch, seq_len, embed_dim)
               Values: Contextual embeddings from transformer layers
        
        Returns:
            Logits (raw scores) for each token in vocabulary
            Shape: (batch, seq_len, vocab_size)
            Values: Unnormalized scores (use softmax for probabilities)
        
        Example:
            >>> # Without weight tying
            >>> output_proj = OutputProjection(embed_dim=768, vocab_size=50257)
            >>> x = torch.randn(2, 10, 768)  # (batch=2, seq_len=10, embed_dim=768)
            >>> logits = output_proj(x)
            >>> logits.shape
            torch.Size([2, 10, 50257])
            >>> 
            >>> # With weight tying
            >>> token_emb = TokenEmbedding(vocab_size=50257, embed_dim=768)
            >>> output_proj = OutputProjection(
            ...     embed_dim=768, 
            ...     vocab_size=50257,
            ...     tie_weights=True,
            ...     embedding_layer=token_emb.embedding
            ... )
            >>> logits = output_proj(x)
            >>> logits.shape
            torch.Size([2, 10, 50257])
        
        Flow:
            (batch, seq_len, embed_dim)
            → Linear projection → (batch, seq_len, vocab_size)
        
        Note:
            The output logits are unnormalized. To get probabilities:
            - Use F.softmax(logits, dim=-1) for probabilities
            - Use F.log_softmax(logits, dim=-1) for log probabilities
            - For training: F.cross_entropy expects logits directly
        """
        # Apply linear transformation
        # Note: If weights are tied, they point to the same tensor as embedding
        # so no synchronization is needed - they're always in sync
        # Input: (batch, seq_len, embed_dim)
        # Output: (batch, seq_len, vocab_size)
        return self.projection(x)
    
    def __repr__(self) -> str:
        """
        String representation of the OutputProjection layer.
        
        Returns:
            String describing the projection configuration
        """
        tie_str = "tied" if self.tie_weights else "untied"
        bias_str = "with bias" if self.projection.bias is not None else "no bias"
        return (
            f"OutputProjection(\n"
            f"  embed_dim={self.embed_dim},\n"
            f"  vocab_size={self.vocab_size},\n"
            f"  weights={tie_str},\n"
            f"  {bias_str}\n"
            f")"
        )

