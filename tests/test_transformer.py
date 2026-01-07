"""
Unit tests for TransformerBlock and TransformerStack.
"""

import pytest
import torch
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.model import TransformerBlock, TransformerStack


class TestTransformerBlock:
    """Tests for TransformerBlock class."""
    
    def test_initialization(self):
        """Test basic initialization."""
        block = TransformerBlock(embed_dim=64, num_heads=4)
        assert block is not None
    
    def test_initialization_with_custom_params(self):
        """Test initialization with custom parameters."""
        block = TransformerBlock(
            embed_dim=128,
            num_heads=8,
            dropout=0.2,
            max_seq_len=512,
            bias=False
        )
        assert block is not None
    
    def test_output_shape(self):
        """Test that output has same shape as input."""
        block = TransformerBlock(embed_dim=64, num_heads=4, dropout=0.0)
        x = torch.randn(2, 10, 64)
        
        output = block(x)
        assert output.shape == x.shape
    
    def test_has_attention_layer(self):
        """Test that block has attention layer."""
        block = TransformerBlock(embed_dim=64, num_heads=4)
        assert hasattr(block, 'attention')
    
    def test_has_feedforward_layer(self):
        """Test that block has feedforward layer."""
        block = TransformerBlock(embed_dim=64, num_heads=4)
        assert hasattr(block, 'feedforward')
    
    def test_has_layer_norms(self):
        """Test that block has layer normalization layers."""
        block = TransformerBlock(embed_dim=64, num_heads=4)
        assert hasattr(block, 'norm1')
        assert hasattr(block, 'norm2')
    
    def test_has_dropout_layers(self):
        """Test that block has dropout layers."""
        block = TransformerBlock(embed_dim=64, num_heads=4)
        assert hasattr(block, 'dropout1')
        assert hasattr(block, 'dropout2')
    
    def test_residual_connection(self):
        """Test that residual connections are present (output differs from pure transformation)."""
        block = TransformerBlock(embed_dim=64, num_heads=4, dropout=0.0)
        block.eval()
        
        x = torch.randn(2, 10, 64)
        output = block(x)
        
        # Due to residual connections, output should contain some info from input
        # but not be identical to input (transformation is applied)
        assert not torch.equal(output, x)
    
    def test_different_sequence_lengths(self):
        """Test with different sequence lengths."""
        block = TransformerBlock(embed_dim=64, num_heads=4, dropout=0.0)
        
        for seq_len in [1, 5, 10, 50]:
            x = torch.randn(2, seq_len, 64)
            output = block(x)
            assert output.shape == (2, seq_len, 64)
    
    def test_single_batch(self):
        """Test with batch size of 1."""
        block = TransformerBlock(embed_dim=64, num_heads=4, dropout=0.0)
        x = torch.randn(1, 10, 64)
        
        output = block(x)
        assert output.shape == (1, 10, 64)
    
    def test_gradients_flow(self):
        """Test that gradients flow through the block."""
        block = TransformerBlock(embed_dim=64, num_heads=4, dropout=0.0)
        x = torch.randn(2, 10, 64, requires_grad=True)
        
        output = block(x)
        loss = output.sum()
        loss.backward()
        
        assert x.grad is not None
    
    def test_deterministic_in_eval_mode(self):
        """Test that output is deterministic in eval mode."""
        block = TransformerBlock(embed_dim=64, num_heads=4, dropout=0.0)
        block.eval()
        
        x = torch.randn(2, 10, 64)
        
        output1 = block(x)
        output2 = block(x)
        
        assert torch.allclose(output1, output2)
    
    def test_dropout_different_in_train_mode(self):
        """Test that dropout creates variation in train mode."""
        block = TransformerBlock(embed_dim=64, num_heads=4, dropout=0.5)
        block.train()
        
        x = torch.randn(2, 10, 64)
        
        # With high dropout, outputs should differ
        output1 = block(x)
        output2 = block(x)
        
        # Very unlikely to be exactly equal with dropout=0.5
        assert not torch.equal(output1, output2)


class TestTransformerStack:
    """Tests for TransformerStack class."""
    
    def test_initialization(self):
        """Test basic initialization."""
        stack = TransformerStack(num_layers=2, embed_dim=64, num_heads=4)
        assert stack.num_layers == 2
        assert stack.embed_dim == 64
        assert stack.num_heads == 4
    
    def test_correct_number_of_layers(self):
        """Test that stack has correct number of layers."""
        for num_layers in [1, 3, 6, 12]:
            stack = TransformerStack(num_layers=num_layers, embed_dim=64, num_heads=4)
            assert len(stack.layers) == num_layers
            assert len(stack) == num_layers
    
    def test_output_shape(self):
        """Test that output has same shape as input."""
        stack = TransformerStack(num_layers=3, embed_dim=64, num_heads=4, dropout=0.0)
        x = torch.randn(2, 10, 64)
        
        output = stack(x)
        assert output.shape == x.shape
    
    def test_sequential_processing(self):
        """Test that layers process sequentially (same as manual iteration)."""
        stack = TransformerStack(num_layers=3, embed_dim=64, num_heads=4, dropout=0.0)
        stack.eval()
        
        x = torch.randn(2, 10, 64)
        
        # Manual processing through each layer
        manual_output = x
        for layer in stack.layers:
            manual_output = layer(manual_output)
        
        # Stack forward pass
        stack_output = stack(x)
        
        assert torch.allclose(manual_output, stack_output)
    
    def test_each_layer_is_transformer_block(self):
        """Test that each layer is a TransformerBlock."""
        stack = TransformerStack(num_layers=3, embed_dim=64, num_heads=4)
        
        for layer in stack.layers:
            assert isinstance(layer, TransformerBlock)
    
    def test_layers_have_independent_parameters(self):
        """Test that each layer has independent parameters."""
        stack = TransformerStack(num_layers=3, embed_dim=64, num_heads=4)
        
        # Get attention weights from each layer
        weights = [layer.attention.attn.in_proj_weight for layer in stack.layers]
        
        # Each layer should have different weights (not shared)
        assert not torch.equal(weights[0], weights[1])
        assert not torch.equal(weights[1], weights[2])
    
    def test_repr(self):
        """Test string representation."""
        stack = TransformerStack(num_layers=12, embed_dim=768, num_heads=12)
        
        repr_str = repr(stack)
        assert "num_layers=12" in repr_str
        assert "embed_dim=768" in repr_str
        assert "num_heads=12" in repr_str
    
    def test_gpt2_small_config(self):
        """Test with GPT-2 small configuration."""
        stack = TransformerStack(
            num_layers=12,
            embed_dim=768,
            num_heads=12,
            dropout=0.1,
            max_seq_len=1024
        )
        
        x = torch.randn(2, 10, 768)
        output = stack(x)
        
        assert output.shape == (2, 10, 768)
        assert len(stack) == 12
    
    def test_gradients_flow_through_all_layers(self):
        """Test that gradients flow through all layers."""
        stack = TransformerStack(num_layers=3, embed_dim=64, num_heads=4, dropout=0.0)
        x = torch.randn(2, 10, 64, requires_grad=True)
        
        output = stack(x)
        loss = output.sum()
        loss.backward()
        
        # Check that each layer has gradients
        for layer in stack.layers:
            has_grad = any(p.grad is not None for p in layer.parameters())
            assert has_grad
    
    def test_parameter_count_scales_with_layers(self):
        """Test that parameter count scales linearly with number of layers."""
        stack_2 = TransformerStack(num_layers=2, embed_dim=64, num_heads=4)
        stack_4 = TransformerStack(num_layers=4, embed_dim=64, num_heads=4)
        
        params_2 = sum(p.numel() for p in stack_2.parameters())
        params_4 = sum(p.numel() for p in stack_4.parameters())
        
        # 4 layers should have ~2x parameters of 2 layers
        assert abs(params_4 - 2 * params_2) < 10  # Allow small rounding difference
    
    def test_different_sequence_lengths(self):
        """Test with different sequence lengths."""
        stack = TransformerStack(num_layers=2, embed_dim=64, num_heads=4, dropout=0.0)
        
        for seq_len in [1, 5, 10, 50, 100]:
            x = torch.randn(2, seq_len, 64)
            output = stack(x)
            assert output.shape == (2, seq_len, 64)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

