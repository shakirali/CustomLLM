"""
Example: Using TransformerStack

This example demonstrates how to:
1. Create a stack of TransformerBlocks
2. Process input through multiple layers
3. Understand how depth affects the model
4. See the sequential processing flow
"""

import sys
import os

# Add parent directory to path to import src modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
from src.model import TransformerStack, TokenEmbedding, PositionEmbedding


def example_basic_usage():
    """Basic example: Create and use a transformer stack."""
    print("=" * 70)
    print("Example 1: Basic TransformerStack Usage")
    print("=" * 70)
    
    # Configuration (GPT-2 small style)
    vocab_size = 50257
    embed_dim = 768
    num_heads = 12
    num_layers = 12
    max_seq_len = 1024
    dropout = 0.1
    
    # Create components
    token_embedding = TokenEmbedding(vocab_size, embed_dim)
    pos_embedding = PositionEmbedding(max_seq_len, embed_dim)
    transformer_stack = TransformerStack(
        num_layers=num_layers,
        embed_dim=embed_dim,
        num_heads=num_heads,
        dropout=dropout,
        max_seq_len=max_seq_len
    )
    
    print(f"Configuration:")
    print(f"  Vocabulary size: {vocab_size}")
    print(f"  Embedding dimension: {embed_dim}")
    print(f"  Number of heads: {num_heads}")
    print(f"  Number of layers: {num_layers}")
    print(f"  Max sequence length: {max_seq_len}")
    print()
    
    # Create some token IDs
    batch_size = 2
    seq_len = 10
    token_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    print(f"Input token_ids shape: {token_ids.shape}")
    print()
    
    # Forward pass through the model
    # Step 1: Token embeddings
    token_embeds = token_embedding(token_ids)
    print(f"After token embedding: {token_embeds.shape}")
    
    # Step 2: Position embeddings
    pos_embeds = pos_embedding(token_ids)
    print(f"Position embeddings: {pos_embeds.shape}")
    
    # Step 3: Combine token + position
    x = token_embeds + pos_embeds
    print(f"After combining embeddings: {x.shape}")
    
    # Step 4: Process through transformer stack
    output = transformer_stack(x)
    print(f"After transformer stack: {output.shape}")
    print()
    
    print("✅ Transformer stack processes input correctly!")
    print(f"   Processed through {len(transformer_stack)} transformer blocks")


def example_different_depths():
    """Show how different numbers of layers work."""
    print("\n" * 2)
    print("=" * 70)
    print("Example 2: Different Stack Depths")
    print("=" * 70)
    
    embed_dim = 64
    num_heads = 4
    batch_size = 1
    seq_len = 5
    
    # Create input
    x = torch.randn(batch_size, seq_len, embed_dim)
    print(f"Input shape: {x.shape}")
    print()
    
    # Test different depths
    depths = [1, 3, 6, 12]
    
    for num_layers in depths:
        stack = TransformerStack(
            num_layers=num_layers,
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=0.0  # No dropout for this example
        )
        
        output = stack(x)
        print(f"Stack with {num_layers} layers:")
        print(f"  Input:  {x.shape}")
        print(f"  Output: {output.shape}")
        print(f"  Number of blocks: {len(stack)}")
        print()
    
    print("Key Insight:")
    print("  • More layers = deeper network = more capacity")
    print("  • Each layer refines the representation")
    print("  • Output shape stays the same: (batch, seq_len, embed_dim)")


def example_sequential_processing():
    """Demonstrate that blocks process sequentially."""
    print("\n" * 2)
    print("=" * 70)
    print("Example 3: Sequential Processing")
    print("=" * 70)
    
    embed_dim = 32
    num_heads = 4
    num_layers = 3
    
    # Create a simple stack
    stack = TransformerStack(
        num_layers=num_layers,
        embed_dim=embed_dim,
        num_heads=num_heads,
        dropout=0.0
    )
    
    x = torch.randn(1, 4, embed_dim)
    print(f"Initial input shape: {x.shape}")
    print(f"Initial input mean: {x.mean().item():.4f}")
    print(f"Initial input std:  {x.std().item():.4f}")
    print()
    
    # Process through each block manually to show sequential nature
    print("Processing through each block:")
    current = x
    for i, block in enumerate(stack.layers):
        current = block(current)
        print(f"  After block {i}: mean={current.mean().item():.4f}, std={current.std().item():.4f}")
    
    print()
    print("Using stack.forward() (same result):")
    output = stack(x)
    print(f"  Final output: mean={output.mean().item():.4f}, std={output.std().item():.4f}")
    print()
    
    # Verify they're the same
    assert torch.allclose(current, output), "Manual and automatic processing should match!"
    print("✅ Sequential processing verified!")


def example_parameter_sharing():
    """Show that each block has independent parameters."""
    print("\n" * 2)
    print("=" * 70)
    print("Example 4: Independent Parameters per Block")
    print("=" * 70)
    
    embed_dim = 16
    num_heads = 2
    num_layers = 3
    
    stack = TransformerStack(
        num_layers=num_layers,
        embed_dim=embed_dim,
        num_heads=num_heads,
        dropout=0.0
    )
    
    print(f"Stack with {num_layers} layers")
    print()
    
    # Count parameters in each block
    total_params = 0
    for i, block in enumerate(stack.layers):
        block_params = sum(p.numel() for p in block.parameters())
        total_params += block_params
        print(f"Block {i} parameters: {block_params:,}")
    
    print(f"Total parameters: {total_params:,}")
    print()
    
    # Verify blocks have different parameters
    block0_attn_weight = stack.layers[0].attention.attention.in_proj_weight
    block1_attn_weight = stack.layers[1].attention.attention.in_proj_weight
    
    print("Verifying blocks have independent parameters:")
    print(f"  Block 0 and Block 1 attention weights are different: "
          f"{not torch.allclose(block0_attn_weight, block1_attn_weight)}")
    print()
    
    print("Key Point:")
    print("  • Each block has its own independent parameters")
    print("  • Blocks learn different transformations")
    print("  • Parameters are NOT shared between blocks")


def example_gpt2_configurations():
    """Show GPT-2 style configurations."""
    print("\n" * 2)
    print("=" * 70)
    print("Example 5: GPT-2 Style Configurations")
    print("=" * 70)
    
    configs = {
        "GPT-2 Small": {
            "num_layers": 12,
            "embed_dim": 768,
            "num_heads": 12,
        },
        "GPT-2 Medium": {
            "num_layers": 24,
            "embed_dim": 1024,
            "num_heads": 16,
        },
        "GPT-2 Large": {
            "num_layers": 36,
            "embed_dim": 1280,
            "num_heads": 20,
        },
    }
    
    for name, config in configs.items():
        stack = TransformerStack(
            num_layers=config["num_layers"],
            embed_dim=config["embed_dim"],
            num_heads=config["num_heads"],
            dropout=0.1,
            max_seq_len=1024
        )
        
        print(f"{name}:")
        print(f"  Layers: {config['num_layers']}")
        print(f"  Embed dim: {config['embed_dim']}")
        print(f"  Heads: {config['num_heads']}")
        print(f"  Stack: {stack}")
        print()


if __name__ == "__main__":
    example_basic_usage()
    example_different_depths()
    example_sequential_processing()
    example_parameter_sharing()
    example_gpt2_configurations()
    
    print("\n" * 2)
    print("=" * 70)
    print("All examples completed successfully!")
    print("=" * 70)

