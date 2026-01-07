"""
Example: Using Output Projection

This example demonstrates how to:
1. Create and use OutputProjection
2. Understand weight tying
3. Convert logits to probabilities
4. Integrate with the full model pipeline
"""

import sys
import os

# Add parent directory to path to import src modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn.functional as F
from src.model import (
    TokenEmbedding, PositionEmbedding, 
    TransformerStack, OutputProjection
)


def example_basic_usage():
    """Basic example: Create and use output projection."""
    print("=" * 70)
    print("Example 1: Basic OutputProjection Usage")
    print("=" * 70)
    
    embed_dim = 64
    vocab_size = 1000
    
    # Create output projection
    output_proj = OutputProjection(embed_dim=embed_dim, vocab_size=vocab_size)
    
    # Simulate transformer output
    batch_size = 2
    seq_len = 10
    transformer_output = torch.randn(batch_size, seq_len, embed_dim)
    
    print(f"Transformer output shape: {transformer_output.shape}")
    print()
    
    # Project to vocabulary logits
    logits = output_proj(transformer_output)
    
    print(f"Logits shape: {logits.shape}")
    print(f"Expected: (batch={batch_size}, seq_len={seq_len}, vocab_size={vocab_size})")
    print()
    
    print("Logits are unnormalized scores:")
    print(f"  Min value: {logits.min().item():.2f}")
    print(f"  Max value: {logits.max().item():.2f}")
    print(f"  Mean value: {logits.mean().item():.2f}")
    print()
    
    print("✅ Output projection works correctly!")


def example_weight_tying():
    """Demonstrate weight tying with token embedding."""
    print("\n" * 2)
    print("=" * 70)
    print("Example 2: Weight Tying")
    print("=" * 70)
    
    embed_dim = 32
    vocab_size = 100
    
    # Create token embedding
    token_emb = TokenEmbedding(vocab_size=vocab_size, embed_dim=embed_dim)
    
    # Create output projection with weight tying
    output_proj = OutputProjection(
        embed_dim=embed_dim,
        vocab_size=vocab_size,
        tie_weights=True,
        embedding_layer=token_emb.embedding
    )
    
    print("Weight Tying Configuration:")
    print(f"  Token embedding weight shape: {token_emb.embedding.weight.shape}")
    print(f"  Output projection weight shape: {output_proj.projection.weight.shape}")
    print()
    
    # Verify weights are tied (same weights, no transpose needed)
    embedding_weight = token_emb.embedding.weight  # (vocab_size, embed_dim)
    projection_weight = output_proj.projection.weight  # (vocab_size, embed_dim)
    
    print("Verifying weight tying:")
    print(f"  embedding.weight shape: {embedding_weight.shape}")
    print(f"  projection.weight shape: {projection_weight.shape}")
    print(f"  projection.weight == embedding.weight: "
          f"{torch.allclose(projection_weight, embedding_weight)}")
    print()
    
    # Count parameters
    # With weight tying: the projection.weight IS the embedding.weight (same tensor)
    # So we only count unique parameters
    print("Parameter Details:")
    print(f"  Token embedding parameters: {list(token_emb.named_parameters())}")
    print(f"  Output projection parameters: {list(output_proj.named_parameters())}")
    print(f"  Same tensor? projection.weight is embedding.weight: "
          f"{output_proj.projection.weight is token_emb.embedding.weight}")
    print()
    
    # With weight tying, projection.weight points to the same tensor as embedding.weight
    # So the total unique parameters = embedding params only
    params_with_tie = sum(p.numel() for p in token_emb.parameters())
    
    # Compare with untied version
    token_emb_untied = TokenEmbedding(vocab_size=vocab_size, embed_dim=embed_dim)
    output_proj_untied = OutputProjection(embed_dim=embed_dim, vocab_size=vocab_size)
    params_without_tie = sum(p.numel() for p in token_emb_untied.parameters()) + \
                         sum(p.numel() for p in output_proj_untied.parameters())
    
    print("Parameter Comparison:")
    print(f"  With weight tying: {params_with_tie:,} parameters")
    print(f"    - Token embedding: {sum(p.numel() for p in token_emb.parameters()):,}")
    print("    - Output projection: shares embedding weight (not counted separately)")
    print(f"  Without weight tying: {params_without_tie:,} parameters")
    print(f"    - Token embedding: {sum(p.numel() for p in token_emb_untied.parameters()):,}")
    print(f"    - Output projection: {sum(p.numel() for p in output_proj_untied.parameters()):,}")
    print(f"  Parameters saved: {params_without_tie - params_with_tie:,}")
    if params_without_tie > 0:
        print(f"  Reduction: {(1 - params_with_tie / params_without_tie) * 100:.1f}%")
    print()
    
    print("✅ Weight tying works correctly!")


def example_logits_to_probabilities():
    """Show how to convert logits to probabilities."""
    print("\n" * 2)
    print("=" * 70)
    print("Example 3: Converting Logits to Probabilities")
    print("=" * 70)
    
    embed_dim = 16
    vocab_size = 10
    batch_size = 1
    seq_len = 3
    
    output_proj = OutputProjection(embed_dim=embed_dim, vocab_size=vocab_size)
    transformer_output = torch.randn(batch_size, seq_len, embed_dim)
    
    # Get logits
    logits = output_proj(transformer_output)
    print(f"Logits shape: {logits.shape}")
    print(f"Logits (first position): {logits[0, 0, :5].tolist()}...")  # First 5 tokens
    print()
    
    # Convert to probabilities
    probs = F.softmax(logits, dim=-1)
    print(f"Probabilities shape: {probs.shape}")
    print(f"Probabilities (first position, first 5 tokens): {probs[0, 0, :5].tolist()}")
    print(f"Sum of probabilities (should be 1.0): {probs[0, 0, :].sum().item():.4f}")
    print()
    
    # Get log probabilities (useful for training)
    log_probs = F.log_softmax(logits, dim=-1)
    print(f"Log probabilities shape: {log_probs.shape}")
    print(f"Log probabilities (first position, first 5 tokens): {log_probs[0, 0, :5].tolist()}")
    print()
    
    # Find most likely token at each position
    predicted_tokens = torch.argmax(logits, dim=-1)
    print(f"Predicted tokens (most likely at each position): {predicted_tokens[0].tolist()}")
    print()
    
    print("✅ Logits to probabilities conversion works!")


def example_complete_pipeline():
    """Show complete pipeline from token IDs to logits."""
    print("\n" * 2)
    print("=" * 70)
    print("Example 4: Complete Pipeline")
    print("=" * 70)
    
    # Configuration
    vocab_size = 1000
    embed_dim = 64
    num_heads = 4
    num_layers = 2
    max_seq_len = 128
    batch_size = 2
    seq_len = 10
    
    # Create all components
    token_emb = TokenEmbedding(vocab_size=vocab_size, embed_dim=embed_dim)
    pos_emb = PositionEmbedding(max_seq_len=max_seq_len, embed_dim=embed_dim)
    transformer_stack = TransformerStack(
        num_layers=num_layers,
        embed_dim=embed_dim,
        num_heads=num_heads,
        dropout=0.0  # No dropout for this example
    )
    output_proj = OutputProjection(
        embed_dim=embed_dim,
        vocab_size=vocab_size,
        tie_weights=True,
        embedding_layer=token_emb.embedding
    )
    
    print("Model Components:")
    print(f"  TokenEmbedding: {vocab_size} → {embed_dim}")
    print(f"  PositionEmbedding: max_seq_len={max_seq_len}, embed_dim={embed_dim}")
    print(f"  TransformerStack: {num_layers} layers, {num_heads} heads")
    print(f"  OutputProjection: {embed_dim} → {vocab_size} (weights tied)")
    print()
    
    # Create token IDs
    token_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    print(f"Input token_ids shape: {token_ids.shape}")
    print(f"Token IDs (first sequence): {token_ids[0, :5].tolist()}...")
    print()
    
    # Forward pass
    print("Forward Pass:")
    
    # Step 1: Token embeddings
    token_embeds = token_emb(token_ids)
    print(f"  1. Token embeddings: {token_embeds.shape}")
    
    # Step 2: Position embeddings
    pos_embeds = pos_emb(token_ids)
    print(f"  2. Position embeddings: {pos_embeds.shape}")
    
    # Step 3: Combine
    x = token_embeds + pos_embeds
    print(f"  3. Combined embeddings: {x.shape}")
    
    # Step 4: Transformer stack
    transformer_output = transformer_stack(x)
    print(f"  4. Transformer output: {transformer_output.shape}")
    
    # Step 5: Output projection
    logits = output_proj(transformer_output)
    print(f"  5. Logits: {logits.shape}")
    print()
    
    # Show predictions
    predicted_tokens = torch.argmax(logits, dim=-1)
    print("Predicted tokens (most likely next token at each position):")
    print(f"  Sequence 0: {predicted_tokens[0, :5].tolist()}...")
    print(f"  Sequence 1: {predicted_tokens[1, :5].tolist()}...")
    print()
    
    print("✅ Complete pipeline works!")


def example_next_token_prediction():
    """Show how to predict the next token."""
    print("\n" * 2)
    print("=" * 70)
    print("Example 5: Next Token Prediction")
    print("=" * 70)
    
    embed_dim = 32
    vocab_size = 50
    seq_len = 5
    
    output_proj = OutputProjection(embed_dim=embed_dim, vocab_size=vocab_size)
    transformer_output = torch.randn(1, seq_len, embed_dim)  # Single sequence
    
    # Get logits for all positions
    logits = output_proj(transformer_output)  # (1, seq_len, vocab_size)
    
    # Get logits for the last position (next token prediction)
    next_token_logits = logits[:, -1, :]  # (1, vocab_size)
    
    print(f"All positions logits: {logits.shape}")
    print(f"Next token logits: {next_token_logits.shape}")
    print()
    
    # Convert to probabilities
    next_token_probs = F.softmax(next_token_logits, dim=-1)
    
    # Get top 5 most likely tokens
    top_k = 5
    top_probs, top_indices = torch.topk(next_token_probs, k=top_k, dim=-1)
    
    print(f"Top {top_k} most likely next tokens:")
    for i in range(top_k):
        token_id = top_indices[0, i].item()
        prob = top_probs[0, i].item()
        print(f"  Token {token_id}: {prob:.4f} ({prob*100:.2f}%)")
    print()
    
    # Sample from the distribution (for text generation)
    sampled_token = torch.multinomial(next_token_probs, num_samples=1)
    print(f"Sampled next token: {sampled_token.item()}")
    print()
    
    print("✅ Next token prediction works!")


if __name__ == "__main__":
    example_basic_usage()
    example_weight_tying()
    example_logits_to_probabilities()
    example_complete_pipeline()
    example_next_token_prediction()
    
    print("\n" * 2)
    print("=" * 70)
    print("All examples completed successfully!")
    print("=" * 70)

