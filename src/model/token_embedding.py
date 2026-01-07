"""
Token embedding module for GPT models.
"""

import torch
import torch.nn as nn


class TokenEmbedding(nn.Module):
    """
    Token embedding layer for GPT models.
    
    Converts token IDs (integers) into dense embedding vectors that the model
    can process. Each token in the vocabulary is mapped to a learnable vector
    of a specified dimension.
    
    This is the first step in the GPT model pipeline:
    Token IDs → Embedding Vectors → Position Embeddings → Transformer Blocks
    
    Attributes:
        embedding: PyTorch embedding layer that maps token IDs to vectors
        vocab_size: Size of the vocabulary (number of unique tokens)
        embed_dim: Dimension of the embedding vectors
    """
    
    def __init__(self, vocab_size: int, embed_dim: int):
        """
        Initialize TokenEmbedding layer.
        
        Args:
            vocab_size: Number of unique tokens in the vocabulary
                       (e.g., 50257 for GPT-2)
            embed_dim: Dimension of the embedding vectors
                      (e.g., 768 for GPT-2 small)
        
        Note:
            The embedding weights are randomly initialized and learned during
            training. Each token ID is mapped to a unique embedding vector.
        """
        super().__init__()
        
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        
        # Create embedding layer
        # Shape: (vocab_size, embed_dim)
        # Each row is the embedding vector for a token ID
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embed_dim
        )
    
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: convert token IDs to embedding vectors.
        
        Args:
            token_ids: Input tensor of token IDs
                      Shape: (batch, seq_len) or (seq_len,)
                      Values: Integers in range [0, vocab_size)
        
        Returns:
            Embedding vectors
            Shape: (batch, seq_len, embed_dim) or (seq_len, embed_dim)
        
        Example:
            >>> token_embedding = TokenEmbedding(vocab_size=50257, embed_dim=768)
            >>> token_ids = torch.tensor([[101, 202, 303], [404, 505, 606]])
            >>> embeddings = token_embedding(token_ids)
            >>> embeddings.shape
            torch.Size([2, 3, 768])
        
        Flow:
            (batch, seq_len) → Embedding lookup → (batch, seq_len, embed_dim)
        """
        # Embedding layer performs lookup: token_id → embedding vector
        return self.embedding(token_ids)

