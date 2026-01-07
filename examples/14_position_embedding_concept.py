"""
Understanding Position Embeddings: The Concept

This example explains:
1. Why position embeddings are needed (the problem they solve)
2. What they do (encode positional information)
3. How they work (learned vs sinusoidal)
4. Why they're added to token embeddings
5. The intuition and theory behind them
"""

import torch
import torch.nn as nn
import math


def demonstrate_the_problem():
    """Explain why position embeddings are needed."""
    print("=" * 70)
    print("The Problem: Transformers Don't Understand Order")
    print("=" * 70)
    
    print("""
The Challenge:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Transformers process tokens in PARALLEL (not sequentially like RNNs).

This means:
  • "The cat sat" and "sat cat The" look the SAME to the model
  • No inherent understanding of token order
  • Position information is LOST

Example:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Sentence 1: "The cat sat on the mat"
  Tokens: [The, cat, sat, on, the, mat]
  
Sentence 2: "The mat sat on the cat"  ← Different meaning!
  Tokens: [The, mat, sat, on, the, cat]

Without position info:
  • Both have same tokens (just different order)
  • Model can't distinguish them
  • Position matters for meaning!

Solution: Position Embeddings
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  • Add position information to each token
  • Encode "where" each token is in the sequence
  • Model can now understand order
    """)


def demonstrate_what_position_embeddings_do():
    """Explain what position embeddings do."""
    print("\n" * 2)
    print("=" * 70)
    print("What Position Embeddings Do")
    print("=" * 70)
    
    print("""
Position Embeddings Encode Position Information:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Each position gets a unique embedding vector:

Position 0 → [0.1, 0.2, 0.3, ..., 0.768]  (embedding for position 0)
Position 1 → [0.4, 0.5, 0.6, ..., 0.768]  (embedding for position 1)
Position 2 → [0.7, 0.8, 0.9, ..., 0.768]  (embedding for position 2)
...

These embeddings are ADDED to token embeddings:

Token "cat" at position 1:
  token_embedding("cat") + position_embedding(1)
  
Token "cat" at position 5:
  token_embedding("cat") + position_embedding(5)

Result:
  • Same token at different positions → Different final embeddings
  • Model can distinguish position
  • Order information is preserved
    """)


def demonstrate_visual_example():
    """Show a visual example."""
    print("\n" * 2)
    print("=" * 70)
    print("Visual Example: How Position Embeddings Work")
    print("=" * 70)
    
    embed_dim = 4
    
    # Token embeddings (what the token is)
    token_embedding = nn.Embedding(100, embed_dim)
    
    # Position embeddings (where the token is)
    pos_embedding = nn.Embedding(10, embed_dim)
    
    # Example sentence: "The cat sat"
    token_ids = torch.tensor([10, 20, 30])  # [The, cat, sat]
    positions = torch.tensor([0, 1, 2])     # [pos0, pos1, pos2]
    
    token_embeds = token_embedding(token_ids)
    pos_embeds = pos_embedding(positions)
    combined = token_embeds + pos_embeds
    
    print("Sentence: 'The cat sat'")
    print()
    print("Token Embeddings (what):")
    print(f"  'The' (token_id=10): {token_embeds[0]}")
    print(f"  'cat' (token_id=20): {token_embeds[1]}")
    print(f"  'sat' (token_id=30): {token_embeds[2]}")
    print()
    print("Position Embeddings (where):")
    print(f"  Position 0: {pos_embeds[0]}")
    print(f"  Position 1: {pos_embeds[1]}")
    print(f"  Position 2: {pos_embeds[2]}")
    print()
    print("Combined (what + where):")
    print(f"  'The' at pos 0: {combined[0]}")
    print(f"  'cat' at pos 1: {combined[1]}")
    print(f"  'sat' at pos 2: {combined[2]}")
    print()
    print("Key Insight:")
    print("  • Token embedding: What the word is")
    print("  • Position embedding: Where the word is")
    print("  • Combined: Complete representation (what + where)")


def demonstrate_why_add_not_concat():
    """Explain why we add, not concatenate."""
    print("\n" * 2)
    print("=" * 70)
    print("Why Add, Not Concatenate?")
    print("=" * 70)
    
    print("""
Two Options:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Option 1: Concatenate
  token_embed (768) + pos_embed (768) → combined (1536)
  
Option 2: Add (What we use)
  token_embed (768) + pos_embed (768) → combined (768)

Why Add?
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. PRESERVES DIMENSION
   • Keeps embedding dimension constant
   • No need to adjust downstream layers
   • Simpler architecture

2. LEARNABLE INTERACTION
   • Model learns how to combine token + position info
   • More flexible than separate dimensions
   • Better representation learning

3. STANDARD PRACTICE
   • Used in GPT, BERT, and other transformers
   • Proven to work well
   • Efficient and effective

4. INTUITIVE
   • Token embedding: "what"
   • Position embedding: "where"
   • Adding them: "what at where"
   • Natural combination

Example:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Token "cat" at position 1:
  combined = token_embed("cat") + pos_embed(1)
  
Token "cat" at position 5:
  combined = token_embed("cat") + pos_embed(5)

Different positions → Different combined embeddings
Same token → Can still recognize it's "cat"
    """)


def demonstrate_learned_vs_sinusoidal():
    """Compare learned vs sinusoidal position embeddings."""
    print("\n" * 2)
    print("=" * 70)
    print("Learned vs Sinusoidal Position Embeddings")
    print("=" * 70)
    
    print("""
Two Approaches:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. LEARNED Position Embeddings (GPT uses this)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   • Similar to token embeddings
   • Learnable parameters (nn.Embedding)
   • Each position has its own embedding vector
   • Learned during training
   
   Implementation:
     pos_embedding = nn.Embedding(max_seq_len, embed_dim)
     pos_embeds = pos_embedding(positions)
   
   Advantages:
     ✓ Flexible (learns optimal position representations)
     ✓ Can adapt to task-specific position patterns
     ✓ Simple to implement
   
   Disadvantages:
     ✗ Fixed maximum sequence length
     ✗ Can't extrapolate to longer sequences


2. SINUSOIDAL Position Embeddings (Original Transformer)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   • Fixed (not learnable)
   • Based on sine/cosine functions
   • Mathematical formula
   
   Formula:
     PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
     PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
   
   Advantages:
     ✓ Can extrapolate to longer sequences
     ✓ No learnable parameters
     ✓ Deterministic
   
   Disadvantages:
     ✗ Less flexible
     ✗ May not be optimal for all tasks

Which to Use?
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   • GPT models: Learned (more flexible)
   • Original Transformer: Sinusoidal (extrapolation)
   • Modern practice: Usually learned
    """)


def demonstrate_learned_implementation():
    """Show learned position embeddings."""
    print("\n" * 2)
    print("=" * 70)
    print("Learned Position Embeddings (What GPT Uses)")
    print("=" * 70)
    
    max_seq_len = 10
    embed_dim = 4
    
    # Learned position embeddings
    pos_embedding = nn.Embedding(max_seq_len, embed_dim)
    
    print("Position Embedding Layer:")
    print(f"  Shape: ({max_seq_len}, {embed_dim})")
    print(f"  Type: Learnable parameters")
    print()
    
    # Get embeddings for all positions
    positions = torch.arange(max_seq_len)
    pos_embeds = pos_embedding(positions)
    
    print("Position Embeddings (learned):")
    print(pos_embeds)
    print()
    print("Key Points:")
    print("  • Each position has unique embedding")
    print("  • Embeddings are learned during training")
    print("  • Model learns optimal position representations")
    print("  • Similar to how token embeddings work")


def demonstrate_sinusoidal_implementation():
    """Show sinusoidal position embeddings."""
    print("\n" * 2)
    print("=" * 70)
    print("Sinusoidal Position Embeddings (Original Transformer)")
    print("=" * 70)
    
    def sinusoidal_position_encoding(max_seq_len, embed_dim):
        """Create sinusoidal position encodings."""
        pe = torch.zeros(max_seq_len, embed_dim)
        position = torch.arange(0, max_seq_len).unsqueeze(1).float()
        
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() *
                            -(math.log(10000.0) / embed_dim))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        return pe
    
    max_seq_len = 10
    embed_dim = 4
    
    pos_embeds = sinusoidal_position_encoding(max_seq_len, embed_dim)
    
    print("Sinusoidal Position Encodings (fixed):")
    print(pos_embeds)
    print()
    print("Key Points:")
    print("  • Fixed (not learnable)")
    print("  • Based on sine/cosine functions")
    print("  • Can extrapolate to longer sequences")
    print("  • Deterministic pattern")


def demonstrate_why_position_matters():
    """Show why position information matters."""
    print("\n" * 2)
    print("=" * 70)
    print("Why Position Information Matters")
    print("=" * 70)
    
    print("""
Examples Where Position Matters:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. WORD ORDER
   "The cat sat" vs "sat cat The"
   → Different meanings, same tokens
   → Position embeddings distinguish them

2. DEPENDENCIES
   "The cat that sat on the mat"
   → "cat" depends on "sat"
   → Position helps model understand relationships

3. CONTEXT
   "I saw a cat" vs "a cat saw I"
   → Subject/object relationships
   → Position encodes grammatical structure

4. TEMPORAL ORDER
   "First, do this. Then, do that."
   → Temporal sequence matters
   → Position encodes order

5. CAUSALITY
   "If X, then Y"
   → Order of conditions matters
   → Position helps model understand flow

Without Position Embeddings:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  ✗ "The cat sat" = "sat cat The" (same representation)
  ✗ Can't understand word order
  ✗ Can't model dependencies
  ✗ Poor language understanding

With Position Embeddings:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  ✓ "The cat sat" ≠ "sat cat The" (different representations)
  ✓ Understands word order
  ✓ Can model dependencies
  ✓ Better language understanding
    """)


def demonstrate_complete_flow():
    """Show complete flow with position embeddings."""
    print("\n" * 2)
    print("=" * 70)
    print("Complete Flow: Token + Position Embeddings")
    print("=" * 70)
    
    vocab_size = 1000
    embed_dim = 4
    max_seq_len = 10
    
    # Token embeddings
    token_embedding = nn.Embedding(vocab_size, embed_dim)
    
    # Position embeddings
    pos_embedding = nn.Embedding(max_seq_len, embed_dim)
    
    # Example: "The cat sat"
    token_ids = torch.tensor([[10, 20, 30]])  # (batch=1, seq_len=3)
    seq_len = token_ids.shape[1]
    
    # Get token embeddings
    token_embeds = token_embedding(token_ids)  # (1, 3, 4)
    print("Step 1: Token Embeddings")
    print(f"  Input: token_ids {token_ids.shape}")
    print(f"  Output: token_embeds {token_embeds.shape}")
    print(f"  Content: What each token is")
    print()
    
    # Get position embeddings
    positions = torch.arange(seq_len)  # [0, 1, 2]
    pos_embeds = pos_embedding(positions)  # (3, 4)
    print("Step 2: Position Embeddings")
    print(f"  Input: positions {positions.shape}")
    print(f"  Output: pos_embeds {pos_embeds.shape}")
    print(f"  Content: Where each token is")
    print()
    
    # Combine
    combined = token_embeds + pos_embeds  # Broadcasting: (1, 3, 4) + (3, 4) → (1, 3, 4)
    print("Step 3: Combined (Token + Position)")
    print(f"  Output: combined {combined.shape}")
    print(f"  Content: What each token is AND where it is")
    print()
    
    print("Final Representation:")
    print(f"  Token 0 ('The'): {combined[0, 0]}")
    print(f"    = token_embed('The') + pos_embed(0)")
    print(f"  Token 1 ('cat'): {combined[0, 1]}")
    print(f"    = token_embed('cat') + pos_embed(1)")
    print(f"  Token 2 ('sat'): {combined[0, 2]}")
    print(f"    = token_embed('sat') + pos_embed(2)")
    print()
    print("Key Point:")
    print("  • Each token now has position information")
    print("  • Model can distinguish order")
    print("  • Ready for transformer blocks!")


if __name__ == "__main__":
    demonstrate_the_problem()
    demonstrate_what_position_embeddings_do()
    demonstrate_visual_example()
    demonstrate_why_add_not_concat()
    demonstrate_learned_vs_sinusoidal()
    demonstrate_learned_implementation()
    demonstrate_sinusoidal_implementation()
    demonstrate_why_position_matters()
    demonstrate_complete_flow()

