"""
Unit tests for LayerNorm, GELU, and FeedForward.
"""

import pytest
import torch
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.model import LayerNorm, GELU, FeedForward


class TestLayerNorm:
    """Tests for LayerNorm class."""
    
    def test_initialization(self):
        """Test basic initialization."""
        norm = LayerNorm(emb_dim=64)
        assert norm.scale.shape == (64,)
        assert norm.shift.shape == (64,)
    
    def test_scale_initialized_to_ones(self):
        """Test that scale is initialized to ones."""
        norm = LayerNorm(emb_dim=64)
        assert torch.allclose(norm.scale, torch.ones(64))
    
    def test_shift_initialized_to_zeros(self):
        """Test that shift is initialized to zeros."""
        norm = LayerNorm(emb_dim=64)
        assert torch.allclose(norm.shift, torch.zeros(64))
    
    def test_output_shape(self):
        """Test that output has same shape as input."""
        norm = LayerNorm(emb_dim=64)
        x = torch.randn(2, 10, 64)
        
        output = norm(x)
        assert output.shape == x.shape
    
    def test_normalization_mean_near_zero(self):
        """Test that normalized output has mean near zero."""
        norm = LayerNorm(emb_dim=64)
        x = torch.randn(2, 10, 64)
        
        output = norm(x)
        
        # Mean across embedding dimension should be close to shift (0)
        mean = output.mean(dim=-1)
        assert torch.allclose(mean, torch.zeros_like(mean), atol=1e-5)
    
    def test_normalization_variance_near_one(self):
        """Test that normalized output has variance near one."""
        norm = LayerNorm(emb_dim=64)
        x = torch.randn(2, 10, 64)
        
        output = norm(x)
        
        # Variance across embedding dimension should be close to scale (1)
        var = output.var(dim=-1, unbiased=False)
        assert torch.allclose(var, torch.ones_like(var), atol=1e-3)
    
    def test_learnable_parameters(self):
        """Test that scale and shift are learnable parameters."""
        norm = LayerNorm(emb_dim=64)
        
        # Should have exactly 2 parameters: scale and shift
        params = list(norm.parameters())
        assert len(params) == 2
    
    def test_2d_input(self):
        """Test with 2D input (batch, embed_dim)."""
        norm = LayerNorm(emb_dim=64)
        x = torch.randn(4, 64)
        
        output = norm(x)
        assert output.shape == (4, 64)
    
    def test_gradients_flow(self):
        """Test that gradients flow through LayerNorm."""
        norm = LayerNorm(emb_dim=64)
        x = torch.randn(2, 10, 64, requires_grad=True)
        
        output = norm(x)
        loss = output.sum()
        loss.backward()
        
        assert x.grad is not None


class TestGELU:
    """Tests for GELU activation class."""
    
    def test_initialization(self):
        """Test basic initialization."""
        gelu = GELU()
        # GELU has no parameters
        assert len(list(gelu.parameters())) == 0
    
    def test_output_shape(self):
        """Test that output has same shape as input."""
        gelu = GELU()
        x = torch.randn(2, 10, 64)
        
        output = gelu(x)
        assert output.shape == x.shape
    
    def test_negative_values_reduced(self):
        """Test that negative values are reduced (not completely zeroed like ReLU)."""
        gelu = GELU()
        x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
        
        output = gelu(x)
        
        # GELU of negative values should be small but not exactly 0
        assert output[0] < 0  # GELU(-2) is slightly negative
        assert output[1] < 0  # GELU(-1) is slightly negative
        assert torch.abs(output[2]) < 1e-6  # GELU(0) ≈ 0
    
    def test_positive_values_preserved(self):
        """Test that positive values are mostly preserved."""
        gelu = GELU()
        x = torch.tensor([1.0, 2.0, 3.0])
        
        output = gelu(x)
        
        # GELU(x) ≈ x for large positive x
        assert output[0] > 0.8  # GELU(1) > 0.8
        assert output[1] > 1.9  # GELU(2) > 1.9
        assert output[2] > 2.9  # GELU(3) > 2.9
    
    def test_smooth_at_zero(self):
        """Test that GELU is smooth at zero (unlike ReLU)."""
        gelu = GELU()
        
        # Check values around zero
        x = torch.linspace(-0.1, 0.1, 100)
        output = gelu(x)
        
        # Output should be smooth (no sharp corners)
        # Check that the derivative is continuous by looking at differences
        diffs = output[1:] - output[:-1]
        
        # All differences should have the same sign (monotonic around 0)
        assert torch.all(diffs > 0)  # GELU is monotonically increasing
    
    def test_matches_pytorch_gelu(self):
        """Test that our GELU approximates PyTorch's GELU."""
        import torch.nn.functional as F
        
        gelu = GELU()
        x = torch.randn(100)
        
        our_output = gelu(x)
        pytorch_output = F.gelu(x, approximate='tanh')
        
        # Should be very close (same approximation formula)
        assert torch.allclose(our_output, pytorch_output, atol=1e-4)
    
    def test_gradients_flow(self):
        """Test that gradients flow through GELU."""
        gelu = GELU()
        x = torch.randn(2, 10, 64, requires_grad=True)
        
        output = gelu(x)
        loss = output.sum()
        loss.backward()
        
        assert x.grad is not None


class TestFeedForward:
    """Tests for FeedForward class."""
    
    def test_initialization(self):
        """Test basic initialization."""
        ff = FeedForward(emb_dim=64)
        assert ff is not None
    
    def test_output_shape(self):
        """Test that output has same shape as input."""
        ff = FeedForward(emb_dim=64)
        x = torch.randn(2, 10, 64)
        
        output = ff(x)
        assert output.shape == x.shape
    
    def test_expansion_factor(self):
        """Test that intermediate dimension is 4x embedding dimension."""
        emb_dim = 64
        ff = FeedForward(emb_dim=emb_dim)
        
        # First layer should be (emb_dim, 4*emb_dim)
        first_linear = ff.layers[0]
        assert first_linear.in_features == emb_dim
        assert first_linear.out_features == 4 * emb_dim
        
        # Last layer should be (4*emb_dim, emb_dim)
        last_linear = ff.layers[2]
        assert last_linear.in_features == 4 * emb_dim
        assert last_linear.out_features == emb_dim
    
    def test_has_gelu_activation(self):
        """Test that FeedForward contains GELU activation."""
        ff = FeedForward(emb_dim=64)
        
        has_gelu = any(isinstance(layer, GELU) for layer in ff.layers)
        assert has_gelu
    
    def test_parameter_count(self):
        """Test that parameter count is correct."""
        emb_dim = 64
        ff = FeedForward(emb_dim=emb_dim)
        
        # First linear: emb_dim * 4*emb_dim + 4*emb_dim (weights + bias)
        # Second linear: 4*emb_dim * emb_dim + emb_dim (weights + bias)
        expected_params = (emb_dim * 4 * emb_dim + 4 * emb_dim) + \
                         (4 * emb_dim * emb_dim + emb_dim)
        
        actual_params = sum(p.numel() for p in ff.parameters())
        assert actual_params == expected_params
    
    def test_different_embedding_dimensions(self):
        """Test with different embedding dimensions."""
        for emb_dim in [32, 64, 128, 256]:
            ff = FeedForward(emb_dim=emb_dim)
            x = torch.randn(2, 10, emb_dim)
            
            output = ff(x)
            assert output.shape == (2, 10, emb_dim)
    
    def test_gradients_flow(self):
        """Test that gradients flow through FeedForward."""
        ff = FeedForward(emb_dim=64)
        x = torch.randn(2, 10, 64, requires_grad=True)
        
        output = ff(x)
        loss = output.sum()
        loss.backward()
        
        assert x.grad is not None
    
    def test_deterministic_output(self):
        """Test that output is deterministic (no dropout in FeedForward itself)."""
        ff = FeedForward(emb_dim=64)
        ff.eval()
        
        x = torch.randn(2, 10, 64)
        
        output1 = ff(x)
        output2 = ff(x)
        
        assert torch.allclose(output1, output2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

