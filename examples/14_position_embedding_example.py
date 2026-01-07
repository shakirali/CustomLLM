"""
Example: Using Position Embeddings

This example demonstrates how to:
1. Create and use PositionEmbedding
2. Combine token embeddings with position embeddings
3. Understand the broadcasting behavior
4. See how position embeddings work in practice
"""

import sys
import os

# Add parent directory to path to import src modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
from src.model import TokenEmbedding, PositionEmbedding


def example_basic_usage():
    """Basic example: Create and use position embeddings."""
    print("=" * 70)
    print("Example 1: Basic Position Embedding Usage")
    print("=" * 70)
    
    vocab_size = 1000
    embed_dim = 64
    max_seq_len = 128
    
    # Create embedding layers
    token_embedding = TokenEmbedding(vocab_size, embed_dim)
    pos_embedding = PositionEmbedding(max_seq_len, embed_dim)
    
    # Create some token IDs
    token_ids = torch.tensor([[10, 20, 30], [40, 50, 60]])  # (batch=2, seq_len=3)
    
    print(f"Input token_ids shape: {token_ids.shape}")
    print(f"Token IDs:\n{token_ids}")
    print()
    
    # Get token embeddings
    token_embeds = token_embedding(token_ids)
    print(f"Token embeddings shape: {token_embeds.shape}")
    print()
    
    # Get position embeddings
    pos_embeds = pos_embedding(token_ids)
    print(f"Position embeddings shape: {pos_embeds.shape}")
    print("Note: No batch dimension! Shape is (seq_len, embed_dim)")
    print()
    
    # Combine them (broadcasting happens automatically)
    combined = token_embeds + pos_embeds
    print(f"Combined embeddings shape: {combined.shape}")
    print("Broadcasting: (batch, seq_len, embed_dim) + (seq_len, embed_dim)")
    print("            → (batch, seq_len, embed_dim)")
    print()
    
    print("✅ Position embeddings work correctly!")


def example_broadcasting_demonstration():
    """Demonstrate how broadcasting works."""
    print("\n" * 2)
    print("=" * 70)
    print("Example 2: Broadcasting Demonstration")
    print("=" * 70)
    
    batch_size = 3
    seq_len = 5
    embed_dim = 8
    vocab_size = 100
    max_seq_len = 128
    
    token_embedding = TokenEmbedding(vocab_size, embed_dim)
    pos_embedding = PositionEmbedding(max_seq_len, embed_dim)
    
    # Create token IDs
    token_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    print(f"Batch size: {batch_size}")
    print(f"Sequence length: {seq_len}")
    print(f"Embedding dimension: {embed_dim}")
    print()
    
    token_embeds = token_embedding(token_ids)
    pos_embeds = pos_embedding(token_ids)
    
    print(f"Token embeddings:   {token_embeds.shape}")
    print(f"Position embeddings: {pos_embeds.shape}")
    print()
    
    # Show that position embeddings are the same for all batch items
    print("Verifying position embeddings are the same for all batch items:")
    print(f"  pos_embeds[0] == pos_embeds[1]: {torch.allclose(pos_embeds[0], pos_embeds[1])}")
    print(f"  pos_embeds[0] == pos_embeds[2]: {torch.allclose(pos_embeds[0], pos_embeds[2])}")
    print()
    
    combined = token_embeds + pos_embeds
    print(f"Combined shape: {combined.shape}")
    print()
    
    # Verify the addition worked correctly
    print("Verifying addition:")
    for b in range(batch_size):
        for s in range(seq_len):
            expected = token_embeds[b, s] + pos_embeds[s]
            actual = combined[b, s]
            assert torch.allclose(expected, actual), "Addition failed!"
    print("  ✅ All additions verified correctly!")
    print()
    
    print("Key Insight:")
    print("  • Position embeddings are shared across all batch items")
    print("  • Broadcasting makes this efficient (no memory duplication)")
    print("  • Each batch item gets the same position information")


def example_different_sequence_lengths():
    """Show how position embeddings handle different sequence lengths."""
    print("\n" * 2)
    print("=" * 70)
    print("Example 3: Different Sequence Lengths")
    print("=" * 70)
    
    embed_dim = 32
    vocab_size = 500
    max_seq_len = 64
    
    token_embedding = TokenEmbedding(vocab_size, embed_dim)
    pos_embedding = PositionEmbedding(max_seq_len, embed_dim)
    
    # Different sequence lengths
    sequences = [
        torch.tensor([[1, 2, 3]]),           # seq_len=3
        torch.tensor([[4, 5, 6, 7, 8]]),     # seq_len=5
        torch.tensor([[9, 10, 11, 12, 13, 14, 15]]),  # seq_len=7
    ]
    
    print("Processing sequences of different lengths:")
    print()
    
    for i, token_ids in enumerate(sequences):
        seq_len = token_ids.shape[1]
        print(f"Sequence {i+1}: seq_len={seq_len}")
        
        token_embeds = token_embedding(token_ids)
        pos_embeds = pos_embedding(token_ids)
        
        print(f"  Token embeddings:   {token_embeds.shape}")
        print(f"  Position embeddings: {pos_embeds.shape}")
        
        combined = token_embeds + pos_embeds
        print(f"  Combined:            {combined.shape}")
        print()
    
    print("✅ Position embeddings adapt to sequence length automatically!")


def example_max_seq_len_validation():
    """Show what happens when sequence length exceeds max_seq_len."""
    print("\n" * 2)
    print("=" * 70)
    print("Example 4: Max Sequence Length Validation")
    print("=" * 70)
    
    embed_dim = 16
    vocab_size = 100
    max_seq_len = 10  # Small max_seq_len for demonstration
    
    pos_embedding = PositionEmbedding(max_seq_len, embed_dim)
    
    # Valid sequence (within max_seq_len)
    valid_ids = torch.randint(0, vocab_size, (2, 8))  # seq_len=8 < max_seq_len=10
    print(f"Valid sequence: seq_len=8, max_seq_len=10")
    try:
        pos_embeds = pos_embedding(valid_ids)
        print(f"  ✅ Success! Position embeddings shape: {pos_embeds.shape}")
    except Exception as e:
        print(f"  ❌ Error: {e}")
    print()
    
    # Invalid sequence (exceeds max_seq_len)
    invalid_ids = torch.randint(0, vocab_size, (2, 15))  # seq_len=15 > max_seq_len=10
    print(f"Invalid sequence: seq_len=15, max_seq_len=10")
    try:
        pos_embeds = pos_embedding(invalid_ids)
        print(f"  ❌ Should have failed but didn't!")
    except ValueError as e:
        print(f"  ✅ Correctly raised ValueError: {e}")
    print()
    
    print("Key Point:")
    print("  • Position embeddings validate sequence length")
    print("  • Prevents errors from sequences longer than max_seq_len")
    print("  • Increase max_seq_len if you need longer sequences")


if __name__ == "__main__":
    example_basic_usage()
    example_broadcasting_demonstration()
    example_different_sequence_lengths()
    example_max_seq_len_validation()
    
    print("\n" * 2)
    print("=" * 70)
    print("All examples completed successfully!")
    print("=" * 70)

