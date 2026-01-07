"""
Position embedding module for GPT models.
"""

import torch
import torch.nn as nn


class PositionEmbedding(nn.Module):
    """
    Position embedding layer for GPT models.
    
    Adds positional information to token embeddings by learning position-specific
    embedding vectors. Each position in the sequence gets a unique learnable vector
    that encodes its position in the sequence.
    
    This is the second step in the GPT model pipeline:
    Token IDs → Token Embeddings → Position Embeddings → Combined Embeddings → Transformer Blocks
    
    Key Design Decision:
        Position embeddings have shape (seq_len, embed_dim), NOT (batch, seq_len, embed_dim).
        This is because position embeddings are the SAME for all sequences in a batch.
        PyTorch broadcasting automatically handles adding them to token embeddings.
    
    Attributes:
        embedding: PyTorch embedding layer that maps position indices to vectors
        max_seq_len: Maximum sequence length supported
        embed_dim: Dimension of the embedding vectors
    """
    
    def __init__(self, max_seq_len: int, embed_dim: int):
        """
        Initialize PositionEmbedding layer.
        
        Args:
            max_seq_len: Maximum sequence length that can be processed
                        (e.g., 1024 for GPT-2)
            embed_dim: Dimension of the embedding vectors
                      (e.g., 768 for GPT-2 small)
                      Must match the embed_dim of TokenEmbedding
        
        Note:
            The embedding weights are randomly initialized and learned during
            training. Each position index (0, 1, 2, ...) is mapped to a unique
            embedding vector.
        """
        super().__init__()
        
        self.max_seq_len = max_seq_len
        self.embed_dim = embed_dim
        
        # Create embedding layer
        # Shape: (max_seq_len, embed_dim)
        # Each row is the embedding vector for a position index
        self.embedding = nn.Embedding(
            num_embeddings=max_seq_len,
            embedding_dim=embed_dim
        )
    
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: generate position embeddings for the sequence.
        
        Args:
            token_ids: Input tensor of token IDs (used to determine sequence length)
                      Shape: (batch, seq_len) or (seq_len,)
                      Values: Integers in range [0, vocab_size)
        
        Returns:
            Position embeddings
            Shape: (seq_len, embed_dim) - NO batch dimension!
            
        Note:
            The output has shape (seq_len, embed_dim), not (batch, seq_len, embed_dim).
            This is intentional! Position embeddings are the same for all sequences
            in a batch. When added to token embeddings with shape (batch, seq_len, embed_dim),
            PyTorch broadcasting automatically handles the batch dimension.
        
        Example:
            >>> pos_embedding = PositionEmbedding(max_seq_len=1024, embed_dim=768)
            >>> token_ids = torch.tensor([[101, 202, 303], [404, 505, 606]])
            >>> pos_embeds = pos_embedding(token_ids)
            >>> pos_embeds.shape
            torch.Size([3, 768])  # (seq_len, embed_dim)
            >>> 
            >>> # When combined with token embeddings:
            >>> token_embeds = token_embedding(token_ids)  # (2, 3, 768)
            >>> combined = token_embeds + pos_embeds  # Broadcasting: (2, 3, 768)
        
        Flow:
            (batch, seq_len) → Extract seq_len → Create positions [0, 1, ..., seq_len-1]
            → Embedding lookup → (seq_len, embed_dim)
        """
        # Handle both batched and unbatched inputs
        if token_ids.dim() == 1:
            seq_len = token_ids.shape[0]
        else:
            seq_len = token_ids.shape[1]
        
        # Ensure we don't exceed max_seq_len
        if seq_len > self.max_seq_len:
            raise ValueError(
                f"Sequence length {seq_len} exceeds max_seq_len {self.max_seq_len}. "
                f"Increase max_seq_len or truncate the sequence."
            )
        
        # Create position indices [0, 1, 2, ..., seq_len-1]
        # Use the same device as token_ids for proper device placement
        positions = torch.arange(seq_len, device=token_ids.device, dtype=torch.long)
        
        # Embedding layer performs lookup: position_index → embedding vector
        # Output shape: (seq_len, embed_dim)
        return self.embedding(positions)

