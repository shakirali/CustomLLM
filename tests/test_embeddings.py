"""
Unit tests for TokenEmbedding and PositionEmbedding.
"""

import pytest
import torch
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.model import TokenEmbedding, PositionEmbedding


class TestTokenEmbedding:
    """Tests for TokenEmbedding class."""
    
    def test_initialization(self):
        """Test basic initialization."""
        emb = TokenEmbedding(vocab_size=1000, embed_dim=64)
        assert emb.vocab_size == 1000
        assert emb.embed_dim == 64
    
    def test_embedding_weight_shape(self):
        """Test that embedding weight has correct shape."""
        vocab_size = 1000
        embed_dim = 64
        emb = TokenEmbedding(vocab_size=vocab_size, embed_dim=embed_dim)
        
        assert emb.embedding.weight.shape == (vocab_size, embed_dim)
    
    def test_output_shape_batched(self):
        """Test output shape with batched input."""
        emb = TokenEmbedding(vocab_size=1000, embed_dim=64)
        token_ids = torch.randint(0, 1000, (2, 10))  # (batch=2, seq_len=10)
        
        output = emb(token_ids)
        assert output.shape == (2, 10, 64)
    
    def test_output_shape_unbatched(self):
        """Test output shape with unbatched input."""
        emb = TokenEmbedding(vocab_size=1000, embed_dim=64)
        token_ids = torch.randint(0, 1000, (10,))  # (seq_len=10)
        
        output = emb(token_ids)
        assert output.shape == (10, 64)
    
    def test_same_token_same_embedding(self):
        """Test that the same token always produces the same embedding."""
        emb = TokenEmbedding(vocab_size=1000, embed_dim=64)
        
        # Create two inputs with the same token at different positions
        token_ids = torch.tensor([[5, 5, 5], [5, 10, 5]])
        
        output = emb(token_ids)
        
        # All embeddings for token 5 should be identical
        assert torch.equal(output[0, 0], output[0, 1])
        assert torch.equal(output[0, 0], output[0, 2])
        assert torch.equal(output[0, 0], output[1, 0])
        assert torch.equal(output[0, 0], output[1, 2])
    
    def test_different_tokens_different_embeddings(self):
        """Test that different tokens produce different embeddings."""
        emb = TokenEmbedding(vocab_size=1000, embed_dim=64)
        
        token_ids = torch.tensor([[0, 1]])
        output = emb(token_ids)
        
        # Different tokens should have different embeddings
        assert not torch.equal(output[0, 0], output[0, 1])
    
    def test_parameter_count(self):
        """Test that parameter count is correct."""
        vocab_size = 1000
        embed_dim = 64
        emb = TokenEmbedding(vocab_size=vocab_size, embed_dim=embed_dim)
        
        expected_params = vocab_size * embed_dim
        actual_params = sum(p.numel() for p in emb.parameters())
        
        assert actual_params == expected_params
    
    def test_gradients_flow(self):
        """Test that gradients flow through the embedding."""
        emb = TokenEmbedding(vocab_size=1000, embed_dim=64)
        token_ids = torch.randint(0, 1000, (2, 10))
        
        output = emb(token_ids)
        loss = output.sum()
        loss.backward()
        
        # Check that embedding weights have gradients
        assert emb.embedding.weight.grad is not None
    
    def test_gpt2_vocab_size(self):
        """Test with GPT-2 vocabulary size."""
        emb = TokenEmbedding(vocab_size=50257, embed_dim=768)
        token_ids = torch.randint(0, 50257, (2, 10))
        
        output = emb(token_ids)
        assert output.shape == (2, 10, 768)


class TestPositionEmbedding:
    """Tests for PositionEmbedding class."""
    
    def test_initialization(self):
        """Test basic initialization."""
        emb = PositionEmbedding(max_seq_len=128, embed_dim=64)
        assert emb.max_seq_len == 128
        assert emb.embed_dim == 64
    
    def test_embedding_weight_shape(self):
        """Test that embedding weight has correct shape."""
        max_seq_len = 128
        embed_dim = 64
        emb = PositionEmbedding(max_seq_len=max_seq_len, embed_dim=embed_dim)
        
        assert emb.embedding.weight.shape == (max_seq_len, embed_dim)
    
    def test_output_shape_no_batch_dim(self):
        """Test that output has no batch dimension."""
        emb = PositionEmbedding(max_seq_len=128, embed_dim=64)
        token_ids = torch.randint(0, 1000, (2, 10))  # (batch=2, seq_len=10)
        
        output = emb(token_ids)
        # Output should be (seq_len, embed_dim), NOT (batch, seq_len, embed_dim)
        assert output.shape == (10, 64)
    
    def test_output_same_for_all_batches(self):
        """Test that position embeddings are the same for all batch items."""
        emb = PositionEmbedding(max_seq_len=128, embed_dim=64)
        
        # Two different batches with same sequence length
        token_ids1 = torch.randint(0, 1000, (2, 10))
        token_ids2 = torch.randint(0, 1000, (3, 10))
        
        output1 = emb(token_ids1)
        output2 = emb(token_ids2)
        
        # Should be identical regardless of batch content
        assert torch.equal(output1, output2)
    
    def test_position_0_same_across_sequences(self):
        """Test that position 0 always gets the same embedding."""
        emb = PositionEmbedding(max_seq_len=128, embed_dim=64)
        
        # Different sequence lengths
        token_ids_5 = torch.randint(0, 1000, (1, 5))
        token_ids_10 = torch.randint(0, 1000, (1, 10))
        
        output_5 = emb(token_ids_5)   # (5, 64)
        output_10 = emb(token_ids_10)  # (10, 64)
        
        # Position 0 embedding should be the same
        assert torch.equal(output_5[0], output_10[0])
    
    def test_different_positions_different_embeddings(self):
        """Test that different positions have different embeddings."""
        emb = PositionEmbedding(max_seq_len=128, embed_dim=64)
        token_ids = torch.randint(0, 1000, (1, 10))
        
        output = emb(token_ids)
        
        # Position 0 and position 1 should have different embeddings
        assert not torch.equal(output[0], output[1])
    
    def test_unbatched_input(self):
        """Test with unbatched (1D) input."""
        emb = PositionEmbedding(max_seq_len=128, embed_dim=64)
        token_ids = torch.randint(0, 1000, (10,))  # (seq_len=10)
        
        output = emb(token_ids)
        assert output.shape == (10, 64)
    
    def test_exceeds_max_seq_len_raises_error(self):
        """Test that exceeding max_seq_len raises an error."""
        emb = PositionEmbedding(max_seq_len=10, embed_dim=64)
        token_ids = torch.randint(0, 1000, (1, 20))  # seq_len=20 > max_seq_len=10
        
        with pytest.raises(ValueError):
            emb(token_ids)
    
    def test_broadcasting_with_token_embeddings(self):
        """Test that position embeddings broadcast correctly with token embeddings."""
        vocab_size = 1000
        embed_dim = 64
        max_seq_len = 128
        batch_size = 2
        seq_len = 10
        
        token_emb = TokenEmbedding(vocab_size=vocab_size, embed_dim=embed_dim)
        pos_emb = PositionEmbedding(max_seq_len=max_seq_len, embed_dim=embed_dim)
        
        token_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
        
        token_output = token_emb(token_ids)  # (batch, seq_len, embed_dim)
        pos_output = pos_emb(token_ids)      # (seq_len, embed_dim)
        
        # Broadcasting should work
        combined = token_output + pos_output  # (batch, seq_len, embed_dim)
        
        assert combined.shape == (batch_size, seq_len, embed_dim)
    
    def test_gradients_flow(self):
        """Test that gradients flow through the embedding."""
        emb = PositionEmbedding(max_seq_len=128, embed_dim=64)
        token_ids = torch.randint(0, 1000, (2, 10))
        
        output = emb(token_ids)
        loss = output.sum()
        loss.backward()
        
        # Check that embedding weights have gradients
        assert emb.embedding.weight.grad is not None
    
    def test_device_placement(self):
        """Test that positions are created on the correct device."""
        emb = PositionEmbedding(max_seq_len=128, embed_dim=64)
        token_ids = torch.randint(0, 1000, (1, 10))
        
        output = emb(token_ids)
        
        # Output should be on the same device as input
        assert output.device == token_ids.device


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

