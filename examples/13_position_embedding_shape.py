"""
Why Position Embeddings are (seq_len, embed_dim) not (batch, seq_len, embed_dim)

This example explains:
1. Why position embeddings don't need batch dimension
2. How broadcasting works when adding them
3. The efficiency benefits
4. How they're used in practice
"""

import torch
import torch.nn as nn


def demonstrate_why_no_batch_dimension():
    """Explain why position embeddings don't need batch dimension."""
    print("=" * 70)
    print("Why Position Embeddings: (seq_len, embed_dim) not (batch, seq_len, embed_dim)")
    print("=" * 70)
    
    print("""
KEY INSIGHT:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Position embeddings are the SAME for all sequences in a batch!

Position 0 → Same embedding for all batch items
Position 1 → Same embedding for all batch items
Position 2 → Same embedding for all batch items
...

We don't need different position embeddings for each batch item!
    """)


def demonstrate_position_embedding_structure():
    """Show the structure of position embeddings."""
    print("\n" * 2)
    print("=" * 70)
    print("Position Embedding Structure")
    print("=" * 70)
    
    max_seq_len = 10
    embed_dim = 4
    
    # Create position embedding layer
    pos_embedding = nn.Embedding(max_seq_len, embed_dim)
    
    print(f"Position Embedding Layer:")
    print(f"  Shape: ({max_seq_len}, {embed_dim})")
    print(f"  Each row: Position embedding for that position")
    print()
    
    # Get position embeddings
    positions = torch.arange(max_seq_len)  # [0, 1, 2, ..., 9]
    pos_embeds = pos_embedding(positions)
    
    print(f"Position Embeddings:")
    print(f"  Input positions: {positions}")
    print(f"  Output shape: {pos_embeds.shape}")
    print(f"  Content:")
    print(pos_embeds)
    print()
    
    print("Key Point:")
    print("  • Shape: (seq_len, embed_dim)")
    print("  • One embedding per position")
    print("  • Same for all sequences in batch")
    print()


def demonstrate_broadcasting():
    """Show how broadcasting works when adding position embeddings."""
    print("\n" * 2)
    print("=" * 70)
    print("Broadcasting: How (seq_len, embed_dim) Works with (batch, seq_len, embed_dim)")
    print("=" * 70)
    
    batch_size = 3
    seq_len = 5
    embed_dim = 4
    
    # Token embeddings (different for each batch item)
    token_embeds = torch.randn(batch_size, seq_len, embed_dim)
    print(f"Token Embeddings:")
    print(f"  Shape: {token_embeds.shape}")
    print(f"  Different for each batch item")
    print()
    
    # Position embeddings (same for all batch items)
    pos_embedding = nn.Embedding(seq_len, embed_dim)
    positions = torch.arange(seq_len)  # [0, 1, 2, 3, 4]
    pos_embeds = pos_embedding(positions)
    
    print(f"Position Embeddings:")
    print(f"  Shape: {pos_embeds.shape}")
    print(f"  Same for all batch items")
    print()
    
    # Broadcasting: PyTorch automatically expands (seq_len, embed_dim) to (batch, seq_len, embed_dim)
    combined = token_embeds + pos_embeds
    
    print(f"Combined (token + position):")
    print(f"  Shape: {combined.shape}")
    print(f"  Broadcasting: (3, 5, 4) + (5, 4) → (3, 5, 4)")
    print()
    
    print("How Broadcasting Works:")
    print("  • PyTorch sees: (batch, seq_len, embed_dim) + (seq_len, embed_dim)")
    print("  • Automatically expands: (seq_len, embed_dim) → (1, seq_len, embed_dim)")
    print("  • Then broadcasts: (1, seq_len, embed_dim) → (batch, seq_len, embed_dim)")
    print("  • Result: Each batch item gets the same position embeddings added")
    print()


def demonstrate_why_not_batch_dimension():
    """Show why we don't need batch dimension."""
    print("\n" * 2)
    print("=" * 70)
    print("Why NOT (batch, seq_len, embed_dim)?")
    print("=" * 70)
    
    print("""
If we used (batch, seq_len, embed_dim):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Position Embeddings: (batch, seq_len, embed_dim)
  Batch 0: [pos0_emb, pos1_emb, pos2_emb, ...]
  Batch 1: [pos0_emb, pos1_emb, pos2_emb, ...]  ← Same as Batch 0!
  Batch 2: [pos0_emb, pos1_emb, pos2_emb, ...]  ← Same as Batch 0!
  ...

Problems:
  ✗ Wastes memory (storing same values multiple times)
  ✗ Wastes computation (redundant storage)
  ✗ No benefit (all batch items need same position embeddings)

With (seq_len, embed_dim):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Position Embeddings: (seq_len, embed_dim)
  [pos0_emb, pos1_emb, pos2_emb, ...]

Benefits:
  ✓ Efficient memory (store once, use for all batches)
  ✓ Efficient computation (broadcasting is fast)
  ✓ Correct behavior (all batch items get same position info)
    """)


def demonstrate_memory_efficiency():
    """Show memory efficiency comparison."""
    print("\n" * 2)
    print("=" * 70)
    print("Memory Efficiency Comparison")
    print("=" * 70)
    
    batch_size = 32
    seq_len = 256
    embed_dim = 768
    
    # With batch dimension (inefficient)
    pos_embeds_with_batch = torch.randn(batch_size, seq_len, embed_dim)
    memory_with_batch = pos_embeds_with_batch.numel() * 4  # 4 bytes per float32
    
    # Without batch dimension (efficient)
    pos_embeds_no_batch = torch.randn(seq_len, embed_dim)
    memory_no_batch = pos_embeds_no_batch.numel() * 4
    
    print(f"Scenario: batch_size={batch_size}, seq_len={seq_len}, embed_dim={embed_dim}")
    print()
    print(f"With batch dimension (seq_len, embed_dim):")
    print(f"  Shape: ({batch_size}, {seq_len}, {embed_dim})")
    print(f"  Memory: {memory_with_batch:,} bytes = {memory_with_batch / 1024 / 1024:.2f} MB")
    print()
    print(f"Without batch dimension (seq_len, embed_dim):")
    print(f"  Shape: ({seq_len}, {embed_dim})")
    print(f"  Memory: {memory_no_batch:,} bytes = {memory_no_batch / 1024 / 1024:.2f} MB")
    print()
    print(f"Memory saved: {memory_with_batch - memory_no_batch:,} bytes")
    print(f"  = {(memory_with_batch - memory_no_batch) / 1024 / 1024:.2f} MB")
    print(f"  = {memory_with_batch / memory_no_batch:.1f}x more efficient!")
    print()


def demonstrate_practical_usage():
    """Show practical usage in TokenEmbedding + PositionEmbedding."""
    print("\n" * 2)
    print("=" * 70)
    print("Practical Usage: Token + Position Embeddings")
    print("=" * 70)
    
    batch_size = 2
    seq_len = 5
    vocab_size = 1000
    embed_dim = 4
    
    # Token embeddings
    token_embedding = nn.Embedding(vocab_size, embed_dim)
    token_ids = torch.tensor([[10, 20, 30, 40, 50], [60, 70, 80, 90, 100]])
    
    token_embeds = token_embedding(token_ids)
    print(f"Token Embeddings:")
    print(f"  Input shape: {token_ids.shape}")
    print(f"  Output shape: {token_embeds.shape}")
    print(f"  Different for each token in each batch")
    print()
    
    # Position embeddings
    pos_embedding = nn.Embedding(seq_len, embed_dim)
    positions = torch.arange(seq_len)  # [0, 1, 2, 3, 4]
    
    pos_embeds = pos_embedding(positions)
    print(f"Position Embeddings:")
    print(f"  Input: {positions}")
    print(f"  Output shape: {pos_embeds.shape}")
    print(f"  Same for all batch items")
    print()
    
    # Combine (broadcasting happens automatically)
    combined = token_embeds + pos_embeds
    print(f"Combined (token + position):")
    print(f"  Shape: {combined.shape}")
    print(f"  Operation: (2, 5, 4) + (5, 4) → (2, 5, 4)")
    print()
    
    print("What happens:")
    print("  • Batch 0, Position 0: token_emb[0,0] + pos_emb[0]")
    print("  • Batch 0, Position 1: token_emb[0,1] + pos_emb[1]")
    print("  • Batch 1, Position 0: token_emb[1,0] + pos_emb[0]  ← Same pos_emb[0]!")
    print("  • Batch 1, Position 1: token_emb[1,1] + pos_emb[1]  ← Same pos_emb[1]!")
    print()
    print("Key Point:")
    print("  • Position embeddings are shared across all batch items")
    print("  • Broadcasting makes this efficient")
    print("  • No need to store duplicate position embeddings")


def demonstrate_implementation_example():
    """Show how PositionEmbedding would be implemented."""
    print("\n" * 2)
    print("=" * 70)
    print("Implementation Example: PositionEmbedding")
    print("=" * 70)
    
    class PositionEmbedding(nn.Module):
        def __init__(self, max_seq_len, embed_dim):
            super().__init__()
            # Shape: (max_seq_len, embed_dim) - NO batch dimension!
            self.embedding = nn.Embedding(max_seq_len, embed_dim)
        
        def forward(self, token_ids):
            """
            Args:
                token_ids: (batch, seq_len) - used to get sequence length
            
            Returns:
                Position embeddings: (seq_len, embed_dim)
            """
            batch_size, seq_len = token_ids.shape
            
            # Create position indices [0, 1, 2, ..., seq_len-1]
            positions = torch.arange(seq_len, device=token_ids.device)
            
            # Get position embeddings: (seq_len, embed_dim)
            pos_embeds = self.embedding(positions)
            
            return pos_embeds  # Shape: (seq_len, embed_dim)
    
    # Usage
    pos_embedding = PositionEmbedding(max_seq_len=10, embed_dim=4)
    token_ids = torch.tensor([[1, 2, 3], [4, 5, 6]])  # (batch=2, seq_len=3)
    
    pos_embeds = pos_embedding(token_ids)
    print(f"Input token_ids shape: {token_ids.shape}")
    print(f"Output pos_embeds shape: {pos_embeds.shape}")
    print()
    print("When adding to token embeddings:")
    print("  token_embeds: (batch, seq_len, embed_dim)")
    print("  pos_embeds:   (seq_len, embed_dim)")
    print("  Combined:     (batch, seq_len, embed_dim)  ← Broadcasting!")
    print()
    print("Key Point:")
    print("  • Position embeddings: (seq_len, embed_dim)")
    print("  • PyTorch broadcasting handles the batch dimension")
    print("  • Efficient and correct!")


if __name__ == "__main__":
    demonstrate_why_no_batch_dimension()
    demonstrate_position_embedding_structure()
    demonstrate_broadcasting()
    demonstrate_why_not_batch_dimension()
    demonstrate_memory_efficiency()
    demonstrate_practical_usage()
    demonstrate_implementation_example()

