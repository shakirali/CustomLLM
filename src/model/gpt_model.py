"""
Complete GPT model implementation.

This module provides the GPTModel class that combines all components into a single,
trainable language model with text generation capabilities.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union

from .token_embedding import TokenEmbedding
from .position_embedding import PositionEmbedding
from .transformer_stack import TransformerStack
from .output_projection import OutputProjection


class GPTModel(nn.Module):
    """
    Complete GPT (Generative Pre-trained Transformer) model.
    
    This class combines all components into a single model:
    - Token embeddings: Convert token IDs to dense vectors
    - Position embeddings: Add positional information
    - Transformer stack: Process through multiple transformer blocks
    - Output projection: Convert to vocabulary logits
    
    The model supports both training (forward pass) and inference (text generation).
    
    Architecture:
        Input: token_ids (batch, seq_len)
            ↓
        TokenEmbedding → (batch, seq_len, embed_dim)
            ↓
        PositionEmbedding → (seq_len, embed_dim) [broadcasts]
            ↓
        Combined: token_emb + pos_emb → (batch, seq_len, embed_dim)
            ↓
        TransformerStack (N layers) → (batch, seq_len, embed_dim)
            ↓
        OutputProjection → (batch, seq_len, vocab_size)
            ↓
        Output: logits
    
    Attributes:
        vocab_size: Size of the vocabulary
        embed_dim: Embedding dimension
        num_layers: Number of transformer blocks
        num_heads: Number of attention heads
        max_seq_len: Maximum sequence length
        dropout: Dropout probability
        tie_weights: Whether output projection shares weights with token embedding
    """
    
    def __init__(
        self,
        vocab_size: int = 50257,  # GPT-2 vocab size
        embed_dim: int = 768,     # GPT-2 small embed dim
        num_layers: int = 12,      # GPT-2 small num layers
        num_heads: int = 12,       # GPT-2 small num heads
        max_seq_len: int = 1024,   # GPT-2 max sequence length
        dropout: float = 0.1,
        bias: bool = True,
        tie_weights: bool = True   # Standard for GPT models
    ):
        """
        Initialize GPTModel.
        
        Args:
            vocab_size: Size of the vocabulary (e.g., 50257 for GPT-2)
            embed_dim: Embedding dimension (e.g., 768 for GPT-2 small)
            num_layers: Number of transformer blocks (e.g., 12 for GPT-2 small)
            num_heads: Number of attention heads (e.g., 12 for GPT-2 small)
            max_seq_len: Maximum sequence length (e.g., 1024 for GPT-2)
            dropout: Dropout probability (default: 0.1)
            bias: Whether to use bias in attention projections (default: True)
            tie_weights: Whether to tie output projection with token embedding
                        (default: True, standard for GPT models)
        
        Note:
            Default values match GPT-2 small configuration. For other sizes:
            - GPT-2 medium: num_layers=24, embed_dim=1024, num_heads=16
            - GPT-2 large: num_layers=36, embed_dim=1280, num_heads=20
            - GPT-2 XL: num_layers=48, embed_dim=1600, num_heads=25
        """
        super().__init__()
        
        # Store configuration
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.max_seq_len = max_seq_len
        self.dropout = dropout
        self.tie_weights = tie_weights
        
        # Token embedding layer
        self.token_embedding = TokenEmbedding(
            vocab_size=vocab_size,
            embed_dim=embed_dim
        )
        
        # Position embedding layer
        self.position_embedding = PositionEmbedding(
            max_seq_len=max_seq_len,
            embed_dim=embed_dim
        )
        
        # Transformer stack
        self.transformer_stack = TransformerStack(
            num_layers=num_layers,
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            max_seq_len=max_seq_len,
            bias=bias
        )
        
        # Output projection (with optional weight tying)
        self.output_projection = OutputProjection(
            embed_dim=embed_dim,
            vocab_size=vocab_size,
            tie_weights=tie_weights,
            embedding_layer=self.token_embedding.embedding if tie_weights else None,
            bias=False  # Standard: no bias in output projection
        )
    
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the GPT model.
        
        Args:
            token_ids: Input tensor of token IDs
                      Shape: (batch, seq_len) or (seq_len,)
                      Values: Integers in range [0, vocab_size)
        
        Returns:
            Logits (raw scores) for each token in vocabulary
            Shape: (batch, seq_len, vocab_size) or (seq_len, vocab_size)
            Values: Unnormalized scores (use softmax for probabilities)
        
        Example:
            >>> model = GPTModel(vocab_size=50257, embed_dim=768, num_layers=12)
            >>> token_ids = torch.tensor([[101, 202, 303], [404, 505, 606]])
            >>> logits = model(token_ids)
            >>> logits.shape
            torch.Size([2, 3, 50257])
        
        Flow:
            (batch, seq_len) → TokenEmbedding → (batch, seq_len, embed_dim)
            → PositionEmbedding → (seq_len, embed_dim) [broadcasts]
            → Combined → (batch, seq_len, embed_dim)
            → TransformerStack → (batch, seq_len, embed_dim)
            → OutputProjection → (batch, seq_len, vocab_size)
        """
        # Handle unbatched input
        was_unbatched = False
        if token_ids.dim() == 1:
            was_unbatched = True
            token_ids = token_ids.unsqueeze(0)  # Add batch dimension
        
        # Token embeddings: (batch, seq_len) → (batch, seq_len, embed_dim)
        token_emb = self.token_embedding(token_ids)
        
        # Position embeddings: (batch, seq_len) → (seq_len, embed_dim)
        # Broadcasting automatically handles batch dimension
        pos_emb = self.position_embedding(token_ids)
        
        # Combine embeddings: (batch, seq_len, embed_dim)
        x = token_emb + pos_emb
        
        # Transformer stack: (batch, seq_len, embed_dim) → (batch, seq_len, embed_dim)
        x = self.transformer_stack(x)
        
        # Output projection: (batch, seq_len, embed_dim) → (batch, seq_len, vocab_size)
        logits = self.output_projection(x)
        
        # Remove batch dimension if input was unbatched
        if was_unbatched:
            logits = logits.squeeze(0)
        
        return logits
    
    def generate(
        self,
        token_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        stop_token_id: Optional[int] = None
    ) -> torch.Tensor:
        """
        Generate text autoregressively.
        
        This method generates new tokens one at a time, using the model's predictions
        to extend the sequence. The model is set to evaluation mode during generation.
        
        Args:
            token_ids: Initial token sequence
                      Shape: (batch, seq_len) or (seq_len,)
                      Values: Integers in range [0, vocab_size)
            max_new_tokens: Maximum number of new tokens to generate (default: 100)
            temperature: Sampling temperature (default: 1.0)
                         Higher = more random, Lower = more deterministic
            top_k: If set, only sample from top-k most likely tokens (default: None)
            top_p: If set, use nucleus sampling with this probability (default: None)
            stop_token_id: If set, stop generation when this token is generated
                          (default: None)
        
        Returns:
            Generated token sequence (including initial tokens)
            Shape: (batch, original_seq_len + num_generated) or (original_seq_len + num_generated)
        
        Example:
            >>> model = GPTModel(vocab_size=50257, embed_dim=768, num_layers=12)
            >>> prompt = torch.tensor([[101, 202, 303]])  # (1, 3)
            >>> generated = model.generate(prompt, max_new_tokens=50, temperature=0.8)
            >>> generated.shape
            torch.Size([1, 53])  # 3 original + 50 new
        
        Note:
            - Model is set to eval mode during generation (dropout disabled)
            - Uses torch.no_grad() for efficiency (no gradients computed)
            - Generation is deterministic if temperature=0.0 (greedy decoding)
        """
        # Set model to evaluation mode
        self.eval()
        
        # Handle unbatched input
        was_unbatched = False
        if token_ids.dim() == 1:
            was_unbatched = True
            token_ids = token_ids.unsqueeze(0)  # Add batch dimension
        
        # Start with initial sequence
        generated = token_ids.clone()
        
        # Generate tokens autoregressively
        with torch.no_grad():
            for _ in range(max_new_tokens):
                # Forward pass: get logits for last position
                # We only need the last position's logits for next-token prediction
                logits = self.forward(generated)  # (batch, current_seq_len, vocab_size)
                
                # Extract logits for last position
                next_token_logits = logits[:, -1, :]  # (batch, vocab_size)
                
                # Apply temperature
                if temperature != 1.0:
                    next_token_logits = next_token_logits / temperature
                
                # Apply top-k filtering
                if top_k is not None:
                    # Get top-k values and indices
                    top_k_values, top_k_indices = torch.topk(next_token_logits, top_k, dim=-1)
                    # Create mask: set non-top-k positions to -inf
                    mask = torch.full_like(next_token_logits, float('-inf'))
                    mask.scatter_(-1, top_k_indices, top_k_values)
                    next_token_logits = mask
                
                # Apply top-p (nucleus) sampling
                if top_p is not None:
                    # Sort logits in descending order
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True, dim=-1)
                    # Compute cumulative probabilities
                    probs = F.softmax(sorted_logits, dim=-1)
                    cumprobs = torch.cumsum(probs, dim=-1)
                    # Find cutoff point where cumprobs > top_p
                    sorted_indices_to_remove = cumprobs > top_p
                    # Keep at least one token
                    sorted_indices_to_remove[..., 0] = False
                    # Create mask
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    next_token_logits[indices_to_remove] = float('-inf')
                
                # Sample next token
                if temperature == 0.0:
                    # Greedy decoding: take most likely token
                    next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)  # (batch, 1)
                else:
                    # Sample from probability distribution
                    probs = F.softmax(next_token_logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)  # (batch, 1)
                
                # Append to generated sequence
                generated = torch.cat([generated, next_token], dim=1)  # (batch, seq_len+1)
                
                # Check for stop token
                if stop_token_id is not None:
                    # Check if any sequence in batch generated stop token
                    if (next_token == stop_token_id).any():
                        break
        
        # Remove batch dimension if input was unbatched
        if was_unbatched:
            generated = generated.squeeze(0)
        
        return generated
    
    def get_num_params(self, trainable_only: bool = True) -> int:
        """
        Count the number of parameters in the model.
        
        Args:
            trainable_only: If True, count only trainable parameters (default: True)
        
        Returns:
            Number of parameters (as integer)
        
        Example:
            >>> model = GPTModel(vocab_size=50257, embed_dim=768, num_layers=12)
            >>> num_params = model.get_num_params()
            >>> print(f"Model has {num_params:,} parameters")
            Model has 117,000,000 parameters
        """
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        else:
            return sum(p.numel() for p in self.parameters())
    
    def __repr__(self) -> str:
        """
        String representation of the GPTModel.
        
        Returns:
            String describing the model configuration
        """
        num_params = self.get_num_params()
        return (
            f"GPTModel(\n"
            f"  vocab_size={self.vocab_size},\n"
            f"  embed_dim={self.embed_dim},\n"
            f"  num_layers={self.num_layers},\n"
            f"  num_heads={self.num_heads},\n"
            f"  max_seq_len={self.max_seq_len},\n"
            f"  dropout={self.dropout},\n"
            f"  tie_weights={self.tie_weights},\n"
            f"  num_params={num_params:,}\n"
            f")"
        )

