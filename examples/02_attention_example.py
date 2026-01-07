"""
Example: Using MultiHeadAttention with Batched Data

This example demonstrates how to:
1. Create a MultiHeadAttention layer
2. Use it with batched input data
3. Understand causal masking behavior
4. Integrate with the dataset for a complete workflow
"""

import sys
import os

# Add parent directory to path to import src modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.attention import MultiHeadAttention
from src.dataset import create_dataloader
import torch
import torch.nn as nn


def example_basic_attention():
    """Basic example: Using MultiHeadAttention with dummy batched data."""
    print("=" * 60)
    print("Example 1: Basic MultiHeadAttention with Batched Data")
    print("=" * 60)
    
    # Create attention layer
    embed_dim = 768
    num_heads = 12
    attn = MultiHeadAttention(
        embed_dim=embed_dim,
        num_heads=num_heads,
        dropout=0.1,
        max_seq_len=1024
    )
    
    print(f"Created MultiHeadAttention:")
    print(f"  embed_dim: {embed_dim}")
    print(f"  num_heads: {num_heads}")
    print(f"  max_seq_len: {attn.max_seq_len}")
    print()
    
    # Create dummy batched input
    batch_size = 4
    seq_len = 256
    x = torch.randn(batch_size, seq_len, embed_dim)
    
    print(f"Input tensor shape: {x.shape}")
    print(f"  batch_size: {batch_size}")
    print(f"  seq_len: {seq_len}")
    print(f"  embed_dim: {embed_dim}")
    print()
    
    # Self-attention (query, key, value all from same input)
    output, attn_weights = attn(x, x, x, need_weights=False)
    
    print(f"Output tensor shape: {output.shape}")
    print(f"✓ Attention computation successful!")
    print()


def example_causal_masking():
    """Example: Understanding causal masking behavior."""
    print("=" * 60)
    print("Example 2: Understanding Causal Masking")
    print("=" * 60)
    
    # Create a small attention layer for demonstration
    embed_dim = 64
    num_heads = 4
    attn = MultiHeadAttention(
        embed_dim=embed_dim,
        num_heads=num_heads,
        dropout=0.0,  # No dropout for clarity
        max_seq_len=10
    )
    
    # Small batch with short sequence
    batch_size = 2
    seq_len = 5
    x = torch.randn(batch_size, seq_len, embed_dim)
    
    print(f"Input shape: {x.shape}")
    print(f"Sequence length: {seq_len}")
    print()
    print("Causal mask behavior:")
    print("  - Token 0 can only attend to itself")
    print("  - Token 1 can attend to tokens 0-1")
    print("  - Token 2 can attend to tokens 0-2")
    print("  - Token 3 can attend to tokens 0-3")
    print("  - Token 4 can attend to tokens 0-4")
    print()
    
    # Get attention with weights to see the mask effect
    output, attn_weights = attn(x, x, x, need_weights=True)
    
    print(f"Output shape: {output.shape}")
    if attn_weights is not None:
        print(f"Attention weights shape: {attn_weights.shape}")
        print(f"  (batch, num_heads, seq_len, seq_len)")
    print()


def example_with_dataset():
    """Example: Using MultiHeadAttention with actual dataset - processing all batches."""
    print("=" * 60)
    print("Example 3: MultiHeadAttention with Dataset (Full Text Processing)")
    print("=" * 60)
    
    # Check if the-verdict.txt exists
    file_path = "the-verdict.txt"
    
    if not os.path.exists(file_path):
        print(f"Warning: {file_path} not found. Skipping this example.")
        return
    
    # Create dataloader
    batch_size = 2
    max_length = 128
    stride = 64
    dataloader = create_dataloader(
        file_path=file_path,
        batch_size=batch_size,
        max_length=max_length,
        stride=stride,
        shuffle=False  # Keep order to see progression
    )
    
    total_batches = len(dataloader)
    print(f"DataLoader created:")
    print(f"  Total batches: {total_batches}")
    print(f"  Batch size: {batch_size}")
    print(f"  Max length: {max_length}")
    print(f"  Stride: {stride}")
    print()
    
    # Create attention layer
    embed_dim = 768
    num_heads = 12  # 768 / 12 = 64 per head (standard GPT-2 configuration)
    attn = MultiHeadAttention(
        embed_dim=embed_dim,
        num_heads=num_heads,
        dropout=0.1,
        max_seq_len=max_length
    )
    
    # Create a simple embedding layer for demonstration
    vocab_size = 50257  # GPT-2 vocab size
    embedding = nn.Embedding(vocab_size, embed_dim)
    
    print(f"Created attention layer:")
    print(f"  embed_dim: {embed_dim}")
    print(f"  num_heads: {num_heads}")
    print()
    
    # Process ALL batches to cover the entire text
    print("Processing all batches to cover the entire text:")
    print("-" * 60)
    
    total_sequences_processed = 0
    total_tokens_processed = 0
    all_outputs = []
    
    for batch_idx, (input_batch, target_batch) in enumerate(dataloader):
        # Embed the input tokens
        # Shape: (batch_size, seq_len) -> (batch_size, seq_len, embed_dim)
        embedded = embedding(input_batch)
        
        # Apply attention
        output, _ = attn(embedded, embedded, embedded)
        
        # Track statistics
        batch_sequences = input_batch.shape[0]
        batch_tokens = input_batch.shape[0] * input_batch.shape[1]
        total_sequences_processed += batch_sequences
        total_tokens_processed += batch_tokens
        
        # Store output (optional, for further processing)
        all_outputs.append(output)
        
        # Print progress for first few and last few batches
        if batch_idx < 3 or batch_idx >= total_batches - 3:
            print(f"Batch {batch_idx + 1}/{total_batches}:")
            print(f"  Input shape: {input_batch.shape}")
            print(f"  Output shape: {output.shape}")
            print(f"  Sequences in batch: {batch_sequences}")
            print(f"  Tokens in batch: {batch_tokens}")
            if batch_idx < total_batches - 1:
                print()
    
    print("-" * 60)
    print()
    print("Processing Summary:")
    print(f"  Total batches processed: {total_batches}")
    print(f"  Total sequences processed: {total_sequences_processed}")
    print(f"  Total tokens processed: {total_tokens_processed}")
    print(f"  Average tokens per sequence: {max_length}")
    print()
    print("✓ Successfully processed entire text through attention!")
    print(f"✓ All {total_batches} batches completed!")
    print()


def example_multiple_heads():
    """Example: Understanding multiple attention heads."""
    print("=" * 60)
    print("Example 4: Multiple Attention Heads")
    print("=" * 60)
    
    # Create attention with multiple heads
    embed_dim = 512
    num_heads = 8
    head_dim = embed_dim // num_heads
    
    attn = MultiHeadAttention(
        embed_dim=embed_dim,
        num_heads=num_heads,
        dropout=0.1
    )
    
    print(f"Attention configuration:")
    print(f"  embed_dim: {embed_dim}")
    print(f"  num_heads: {num_heads}")
    print(f"  head_dim: {head_dim} (embed_dim / num_heads)")
    print()
    print("Each head processes the input independently:")
    print(f"  - Each head operates on {head_dim}-dimensional vectors")
    print(f"  - {num_heads} heads run in parallel")
    print(f"  - Outputs are concatenated to form {embed_dim}-dim output")
    print()
    
    # Test with batched input
    batch_size = 4
    seq_len = 100
    x = torch.randn(batch_size, seq_len, embed_dim)
    
    output, attn_weights = attn(x, x, x, need_weights=True)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    if attn_weights is not None:
        print(f"Attention weights shape: {attn_weights.shape}")
        print(f"  - {num_heads} heads, each with attention matrix of shape ({seq_len}, {seq_len})")
    print()


def main():
    """Run all examples."""
    print("\n" + "=" * 60)
    print("MultiHeadAttention Examples with Batched Data")
    print("=" * 60)
    print()
    
    try:
        example_basic_attention()
        example_causal_masking()
        example_with_dataset()
        example_multiple_heads()
        
        print("=" * 60)
        print("All examples completed!")
        print("=" * 60)
        print()
        print("Key Takeaways:")
        print("  - MultiHeadAttention expects input shape: (batch, seq_len, embed_dim)")
        print("  - Causal masking prevents attention to future tokens")
        print("  - Multiple heads process attention in parallel")
        print("  - Works seamlessly with batched data from DataLoader")
        
    except Exception as e:
        print(f"\nError running examples: {e}")
        import traceback
        traceback.print_exc()
        print("\nMake sure you have installed the required dependencies:")
        print("  pip install -r requirements.txt")


if __name__ == "__main__":
    main()

