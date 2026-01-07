"""
Unit tests for OutputProjection.
"""

import pytest
import torch
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.model import OutputProjection, TokenEmbedding


class TestOutputProjection:
    """Tests for OutputProjection class."""
    
    def test_initialization(self):
        """Test basic initialization without weight tying."""
        proj = OutputProjection(embed_dim=64, vocab_size=1000)
        assert proj.embed_dim == 64
        assert proj.vocab_size == 1000
        assert proj.tie_weights == False
    
    def test_initialization_with_weight_tying(self):
        """Test initialization with weight tying."""
        token_emb = TokenEmbedding(vocab_size=1000, embed_dim=64)
        proj = OutputProjection(
            embed_dim=64,
            vocab_size=1000,
            tie_weights=True,
            embedding_layer=token_emb.embedding
        )
        assert proj.tie_weights == True
    
    def test_weight_tying_requires_embedding_layer(self):
        """Test that weight tying requires embedding_layer."""
        with pytest.raises(ValueError):
            OutputProjection(
                embed_dim=64,
                vocab_size=1000,
                tie_weights=True,
                embedding_layer=None
            )
    
    def test_weight_tying_vocab_size_mismatch(self):
        """Test that vocab_size must match embedding layer."""
        token_emb = TokenEmbedding(vocab_size=500, embed_dim=64)  # Different vocab_size
        
        with pytest.raises(ValueError):
            OutputProjection(
                embed_dim=64,
                vocab_size=1000,  # Mismatch!
                tie_weights=True,
                embedding_layer=token_emb.embedding
            )
    
    def test_weight_tying_embed_dim_mismatch(self):
        """Test that embed_dim must match embedding layer."""
        token_emb = TokenEmbedding(vocab_size=1000, embed_dim=128)  # Different embed_dim
        
        with pytest.raises(ValueError):
            OutputProjection(
                embed_dim=64,  # Mismatch!
                vocab_size=1000,
                tie_weights=True,
                embedding_layer=token_emb.embedding
            )
    
    def test_output_shape(self):
        """Test that output has correct shape."""
        proj = OutputProjection(embed_dim=64, vocab_size=1000)
        x = torch.randn(2, 10, 64)
        
        output = proj(x)
        assert output.shape == (2, 10, 1000)
    
    def test_output_shape_with_weight_tying(self):
        """Test output shape with weight tying."""
        token_emb = TokenEmbedding(vocab_size=1000, embed_dim=64)
        proj = OutputProjection(
            embed_dim=64,
            vocab_size=1000,
            tie_weights=True,
            embedding_layer=token_emb.embedding
        )
        x = torch.randn(2, 10, 64)
        
        output = proj(x)
        assert output.shape == (2, 10, 1000)
    
    def test_weight_tying_shares_same_tensor(self):
        """Test that weight tying shares the exact same tensor."""
        token_emb = TokenEmbedding(vocab_size=1000, embed_dim=64)
        proj = OutputProjection(
            embed_dim=64,
            vocab_size=1000,
            tie_weights=True,
            embedding_layer=token_emb.embedding
        )
        
        # Should be the exact same tensor object (not just equal values)
        assert proj.projection.weight is token_emb.embedding.weight
    
    def test_weight_shapes_match(self):
        """Test that tied weights have the same shape."""
        token_emb = TokenEmbedding(vocab_size=1000, embed_dim=64)
        proj = OutputProjection(
            embed_dim=64,
            vocab_size=1000,
            tie_weights=True,
            embedding_layer=token_emb.embedding
        )
        
        assert proj.projection.weight.shape == token_emb.embedding.weight.shape
        assert proj.projection.weight.shape == (1000, 64)
    
    def test_no_bias_by_default(self):
        """Test that there's no bias by default."""
        proj = OutputProjection(embed_dim=64, vocab_size=1000)
        assert proj.projection.bias is None
    
    def test_with_bias(self):
        """Test initialization with bias."""
        proj = OutputProjection(embed_dim=64, vocab_size=1000, bias=True)
        assert proj.projection.bias is not None
        assert proj.projection.bias.shape == (1000,)
    
    def test_parameter_count_without_tying(self):
        """Test parameter count without weight tying."""
        embed_dim = 64
        vocab_size = 1000
        proj = OutputProjection(embed_dim=embed_dim, vocab_size=vocab_size, bias=False)
        
        # Without bias: vocab_size * embed_dim
        expected_params = vocab_size * embed_dim
        actual_params = sum(p.numel() for p in proj.parameters())
        
        assert actual_params == expected_params
    
    def test_parameter_count_with_tying(self):
        """Test that tied weights are not counted as separate parameters."""
        token_emb = TokenEmbedding(vocab_size=1000, embed_dim=64)
        proj = OutputProjection(
            embed_dim=64,
            vocab_size=1000,
            tie_weights=True,
            embedding_layer=token_emb.embedding
        )
        
        # With weight tying, the projection's weight is shared with embedding
        # So projection should NOT have separate parameters
        proj_params = sum(p.numel() for p in proj.parameters())
        
        # The weight is the same tensor as embedding.weight, so it will be counted
        # But it should be the SAME memory, not a copy
        assert proj_params == 1000 * 64  # It's counted but shared
    
    def test_gradients_flow_without_tying(self):
        """Test that gradients flow through projection without tying."""
        proj = OutputProjection(embed_dim=64, vocab_size=1000)
        x = torch.randn(2, 10, 64, requires_grad=True)
        
        output = proj(x)
        loss = output.sum()
        loss.backward()
        
        assert x.grad is not None
        assert proj.projection.weight.grad is not None
    
    def test_gradients_flow_with_tying(self):
        """Test that gradients flow through to embedding with tying."""
        token_emb = TokenEmbedding(vocab_size=1000, embed_dim=64)
        proj = OutputProjection(
            embed_dim=64,
            vocab_size=1000,
            tie_weights=True,
            embedding_layer=token_emb.embedding
        )
        
        x = torch.randn(2, 10, 64, requires_grad=True)
        
        output = proj(x)
        loss = output.sum()
        loss.backward()
        
        # Gradients should flow to the embedding weights
        assert token_emb.embedding.weight.grad is not None
    
    def test_repr(self):
        """Test string representation."""
        proj = OutputProjection(embed_dim=64, vocab_size=1000)
        
        repr_str = repr(proj)
        assert "embed_dim=64" in repr_str
        assert "vocab_size=1000" in repr_str
        assert "untied" in repr_str
        assert "no bias" in repr_str
    
    def test_repr_with_tying(self):
        """Test string representation with weight tying."""
        token_emb = TokenEmbedding(vocab_size=1000, embed_dim=64)
        proj = OutputProjection(
            embed_dim=64,
            vocab_size=1000,
            tie_weights=True,
            embedding_layer=token_emb.embedding
        )
        
        repr_str = repr(proj)
        assert "tied" in repr_str
    
    def test_different_batch_sizes(self):
        """Test with different batch sizes."""
        proj = OutputProjection(embed_dim=64, vocab_size=1000)
        
        for batch_size in [1, 2, 8, 32]:
            x = torch.randn(batch_size, 10, 64)
            output = proj(x)
            assert output.shape == (batch_size, 10, 1000)
    
    def test_different_sequence_lengths(self):
        """Test with different sequence lengths."""
        proj = OutputProjection(embed_dim=64, vocab_size=1000)
        
        for seq_len in [1, 5, 10, 50, 100]:
            x = torch.randn(2, seq_len, 64)
            output = proj(x)
            assert output.shape == (2, seq_len, 1000)
    
    def test_logits_are_unnormalized(self):
        """Test that output logits are unnormalized (not probabilities)."""
        proj = OutputProjection(embed_dim=64, vocab_size=1000)
        x = torch.randn(2, 10, 64)
        
        logits = proj(x)
        
        # Logits should NOT sum to 1 (they're unnormalized)
        sums = logits.sum(dim=-1)
        assert not torch.allclose(sums, torch.ones_like(sums))
        
        # Logits can be negative
        assert (logits < 0).any()


class TestOutputProjectionIntegration:
    """Integration tests for OutputProjection with other components."""
    
    def test_complete_pipeline(self):
        """Test complete pipeline from embeddings to logits."""
        from src.model import TransformerStack, PositionEmbedding
        
        vocab_size = 1000
        embed_dim = 64
        num_heads = 4
        num_layers = 2
        batch_size = 2
        seq_len = 10
        
        # Create all components
        token_emb = TokenEmbedding(vocab_size=vocab_size, embed_dim=embed_dim)
        pos_emb = PositionEmbedding(max_seq_len=128, embed_dim=embed_dim)
        stack = TransformerStack(
            num_layers=num_layers,
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=0.0
        )
        proj = OutputProjection(
            embed_dim=embed_dim,
            vocab_size=vocab_size,
            tie_weights=True,
            embedding_layer=token_emb.embedding
        )
        
        # Forward pass
        token_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
        
        x = token_emb(token_ids) + pos_emb(token_ids)
        x = stack(x)
        logits = proj(x)
        
        assert logits.shape == (batch_size, seq_len, vocab_size)
    
    def test_weight_tying_updates_both(self):
        """Test that weight updates affect both embedding and projection."""
        token_emb = TokenEmbedding(vocab_size=100, embed_dim=32)
        proj = OutputProjection(
            embed_dim=32,
            vocab_size=100,
            tie_weights=True,
            embedding_layer=token_emb.embedding
        )
        
        # Get original weights
        original_weights = token_emb.embedding.weight.clone()
        
        # Simulate training: forward pass and backward
        token_ids = torch.randint(0, 100, (2, 5))
        x = token_emb(token_ids)  # (2, 5, 32)
        logits = proj(x)  # (2, 5, 100)
        
        loss = logits.sum()
        loss.backward()
        
        # Update weights manually
        with torch.no_grad():
            token_emb.embedding.weight -= 0.01 * token_emb.embedding.weight.grad
        
        # Both should be updated (since they share the same tensor)
        assert not torch.equal(token_emb.embedding.weight, original_weights)
        assert torch.equal(proj.projection.weight, token_emb.embedding.weight)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

