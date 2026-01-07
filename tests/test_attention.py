"""
Unit tests for MultiHeadAttention.
"""

import pytest
import torch
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.attention import MultiHeadAttention


class TestMultiHeadAttention:
    """Tests for MultiHeadAttention class."""
    
    def test_initialization(self):
        """Test basic initialization."""
        attn = MultiHeadAttention(embed_dim=64, num_heads=4)
        assert attn.embed_dim == 64
        assert attn.num_heads == 4
        assert attn.max_seq_len == 1024
    
    def test_initialization_with_custom_params(self):
        """Test initialization with custom parameters."""
        attn = MultiHeadAttention(
            embed_dim=128,
            num_heads=8,
            dropout=0.2,
            bias=False,
            max_seq_len=512
        )
        assert attn.embed_dim == 128
        assert attn.num_heads == 8
        assert attn.max_seq_len == 512
    
    def test_invalid_embed_dim(self):
        """Test that embed_dim must be divisible by num_heads."""
        with pytest.raises(AssertionError):
            MultiHeadAttention(embed_dim=65, num_heads=4)  # 65 not divisible by 4
    
    def test_output_shape(self):
        """Test that output has correct shape."""
        batch_size = 2
        seq_len = 10
        embed_dim = 64
        num_heads = 4
        
        attn = MultiHeadAttention(embed_dim=embed_dim, num_heads=num_heads)
        x = torch.randn(batch_size, seq_len, embed_dim)
        
        output, _ = attn(x, x, x)
        
        assert output.shape == (batch_size, seq_len, embed_dim)
    
    def test_self_attention(self):
        """Test self-attention (query=key=value)."""
        attn = MultiHeadAttention(embed_dim=64, num_heads=4, dropout=0.0)
        x = torch.randn(2, 10, 64)
        
        output, _ = attn(x, x, x)
        
        # Output should be different from input (transformation applied)
        assert not torch.allclose(output, x)
    
    def test_causal_mask_shape(self):
        """Test that causal mask has correct shape."""
        max_seq_len = 128
        attn = MultiHeadAttention(embed_dim=64, num_heads=4, max_seq_len=max_seq_len)
        
        assert attn.causal_mask.shape == (max_seq_len, max_seq_len)
    
    def test_causal_mask_is_upper_triangular(self):
        """Test that causal mask is upper triangular (True above diagonal)."""
        attn = MultiHeadAttention(embed_dim=64, num_heads=4, max_seq_len=5)
        
        # Causal mask should be True above diagonal (prevent attention to future)
        expected = torch.tensor([
            [False, True, True, True, True],
            [False, False, True, True, True],
            [False, False, False, True, True],
            [False, False, False, False, True],
            [False, False, False, False, False],
        ])
        
        assert torch.equal(attn.causal_mask, expected)
    
    def test_get_causal_mask(self):
        """Test _get_causal_mask method."""
        attn = MultiHeadAttention(embed_dim=64, num_heads=4, max_seq_len=10)
        
        # Get mask for smaller sequence
        mask = attn._get_causal_mask(5)
        
        assert mask.shape == (5, 5)
        # First row should only have True values after position 0
        assert mask[0, 0] == False  # Can attend to self
        assert mask[0, 1] == True   # Cannot attend to future
    
    def test_attention_weights_output(self):
        """Test that attention weights are returned when requested."""
        attn = MultiHeadAttention(embed_dim=64, num_heads=4, dropout=0.0)
        x = torch.randn(2, 5, 64)
        
        output, weights = attn(x, x, x, need_weights=True)
        
        assert output is not None
        assert weights is not None
        # Weights shape should be (batch, seq_len, seq_len)
        assert weights.shape == (2, 5, 5)
    
    def test_no_attention_weights_by_default(self):
        """Test that attention weights are None by default."""
        attn = MultiHeadAttention(embed_dim=64, num_heads=4)
        x = torch.randn(2, 5, 64)
        
        output, weights = attn(x, x, x, need_weights=False)
        
        assert output is not None
        # Weights should be None when not requested
        assert weights is None
    
    def test_different_sequence_lengths(self):
        """Test with different sequence lengths."""
        attn = MultiHeadAttention(embed_dim=64, num_heads=4, dropout=0.0)
        
        for seq_len in [1, 5, 10, 50]:
            x = torch.randn(2, seq_len, 64)
            output, _ = attn(x, x, x)
            assert output.shape == (2, seq_len, 64)
    
    def test_single_batch(self):
        """Test with batch size of 1."""
        attn = MultiHeadAttention(embed_dim=64, num_heads=4, dropout=0.0)
        x = torch.randn(1, 10, 64)
        
        output, _ = attn(x, x, x)
        assert output.shape == (1, 10, 64)
    
    def test_deterministic_with_no_dropout(self):
        """Test that output is deterministic when dropout=0."""
        attn = MultiHeadAttention(embed_dim=64, num_heads=4, dropout=0.0)
        attn.eval()  # Set to eval mode
        
        x = torch.randn(2, 10, 64)
        
        output1, _ = attn(x, x, x)
        output2, _ = attn(x, x, x)
        
        assert torch.allclose(output1, output2)
    
    def test_causal_mask_prevents_future_attention(self):
        """Test that causal mask prevents attention to future positions."""
        # This is a key property: position i should not attend to positions > i
        attn = MultiHeadAttention(embed_dim=64, num_heads=4, dropout=0.0)
        attn.eval()
        
        x = torch.randn(1, 5, 64)
        
        # Get attention weights
        _, weights = attn(x, x, x, need_weights=True)
        
        if weights is not None:
            # Check that attention weights for future positions are ~0
            # Due to softmax after -inf masking
            for i in range(5):
                for j in range(i+1, 5):
                    # Position i should not attend to position j (j > i)
                    assert weights[0, i, j] < 1e-5


class TestMultiHeadAttentionGradients:
    """Tests for gradient flow through MultiHeadAttention."""
    
    def test_gradients_flow(self):
        """Test that gradients flow through the attention layer."""
        attn = MultiHeadAttention(embed_dim=64, num_heads=4, dropout=0.0)
        x = torch.randn(2, 10, 64, requires_grad=True)
        
        output, _ = attn(x, x, x)
        loss = output.sum()
        loss.backward()
        
        assert x.grad is not None
        assert x.grad.shape == x.shape
    
    def test_parameters_have_gradients(self):
        """Test that model parameters receive gradients."""
        attn = MultiHeadAttention(embed_dim=64, num_heads=4, dropout=0.0)
        x = torch.randn(2, 10, 64)
        
        output, _ = attn(x, x, x)
        loss = output.sum()
        loss.backward()
        
        # Check that at least some parameters have gradients
        has_grad = False
        for param in attn.parameters():
            if param.grad is not None:
                has_grad = True
                break
        
        assert has_grad


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

