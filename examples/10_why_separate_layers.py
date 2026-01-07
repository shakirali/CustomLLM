"""
Why Separate LayerNorm and Dropout Layers in TransformerBlock?

This example explains:
1. Why we need two separate LayerNorm layers (norm1 and norm2)
2. Why we need two separate Dropout layers (dropout1 and dropout2)
3. What happens if we try to reuse a single layer
4. The importance of independent learnable parameters
"""

import torch
import torch.nn as nn
from src.model import LayerNorm

def demonstrate_why_separate_layernorm():
    """Explain why we need two separate LayerNorm layers."""
    print("=" * 70)
    print("Why Two Separate LayerNorm Layers?")
    print("=" * 70)
    
    emb_dim = 768
    
    # Create two separate LayerNorm layers
    norm1 = LayerNorm(emb_dim=emb_dim)
    norm2 = LayerNorm(emb_dim=emb_dim)
    
    # Create a single LayerNorm (for comparison)
    single_norm = LayerNorm(emb_dim=emb_dim)
    
    # Create sample input
    x = torch.randn(4, 256, emb_dim)
    
    print("Input shape:", x.shape)
    print()
    
    print("REASON 1: Different Input Distributions")
    print("-" * 70)
    print("""
    norm1 receives: Raw input (x)
    norm2 receives: Attention output + residual (x + attention_output)
    
    These have DIFFERENT statistical properties:
    • Different mean values
    • Different variance values
    • Different feature distributions
    
    A single LayerNorm can't optimally normalize both!
    """)
    
    # Simulate the flow
    print("Demonstration:")
    print("  Step 1: norm1 normalizes raw input")
    norm1_out = norm1(x)
    print(f"    Input mean: {x.mean().item():.4f}, std: {x.std().item():.4f}")
    print(f"    After norm1 mean: {norm1_out.mean().item():.4f}, std: {norm1_out.std().item():.4f}")
    print()
    
    # Simulate attention output (simplified)
    attn_output = torch.randn(4, 256, emb_dim) * 0.1  # Smaller scale
    x_after_attn = x + attn_output
    
    print("  Step 2: norm2 normalizes attention output + residual")
    norm2_out = norm2(x_after_attn)
    print(f"    Input mean: {x_after_attn.mean().item():.4f}, std: {x_after_attn.std().item():.4f}")
    print(f"    After norm2 mean: {norm2_out.mean().item():.4f}, std: {norm2_out.std().item():.4f}")
    print()
    
    print("REASON 2: Separate Learnable Parameters")
    print("-" * 70)
    print("""
    Each LayerNorm has its own learnable parameters:
    • scale: How much to scale normalized values
    • shift: How much to shift normalized values
    
    norm1 learns: How to normalize pre-attention features
    norm2 learns: How to normalize pre-feedforward features
    
    These are DIFFERENT normalization needs!
    """)
    
    print("Parameter comparison:")
    print(f"  norm1.scale[0:5]: {norm1.scale.data[0:5]}")
    print(f"  norm2.scale[0:5]: {norm2.scale.data[0:5]}")
    print("  ↑ Different parameters learn different patterns")
    print()
    
    print("REASON 3: Different Normalization Needs")
    print("-" * 70)
    print("""
    norm1: Normalizes raw token embeddings
           → Needs to handle initial feature distributions
           → Learns normalization for attention input
    
    norm2: Normalizes attention-processed features
           → Needs to handle context-aware features
           → Learns normalization for feedforward input
    
    These require DIFFERENT normalization strategies!
    """)


def demonstrate_why_separate_dropout():
    """Explain why we need two separate Dropout layers."""
    print("\n" * 2)
    print("=" * 70)
    print("Why Two Separate Dropout Layers?")
    print("=" * 70)
    
    dropout1 = nn.Dropout(p=0.1)
    dropout2 = nn.Dropout(p=0.1)
    single_dropout = nn.Dropout(p=0.1)
    
    x = torch.randn(4, 256, 768)
    
    print("REASON 1: Independent Random States")
    print("-" * 70)
    print("""
    Each Dropout layer maintains its own random state.
    This means they independently decide which neurons to drop.
    """)
    
    print("Demonstration with separate dropouts:")
    dropout1.train()
    dropout2.train()
    
    attn_output = torch.randn(4, 256, 768)
    ff_output = torch.randn(4, 256, 768)
    
    print("  Forward pass 1:")
    dropped1 = dropout1(attn_output)
    dropped2 = dropout2(ff_output)
    zeros1 = (dropped1 == 0).sum().item()
    zeros2 = (dropped2 == 0).sum().item()
    print(f"    dropout1 zeros: {zeros1}")
    print(f"    dropout2 zeros: {zeros2}")
    print(f"    Different patterns: {zeros1 != zeros2}")
    print()
    
    print("  Forward pass 2 (different random patterns):")
    dropped1 = dropout1(attn_output)
    dropped2 = dropout2(ff_output)
    zeros1 = (dropped1 == 0).sum().item()
    zeros2 = (dropped2 == 0).sum().item()
    print(f"    dropout1 zeros: {zeros1}")
    print(f"    dropout2 zeros: {zeros2}")
    print()
    
    print("REASON 2: Different Regularization Needs")
    print("-" * 70)
    print("""
    dropout1: Regularizes attention output
              → Prevents over-reliance on specific attention patterns
              → Forces learning multiple attention strategies
    
    dropout2: Regularizes feedforward output
              → Prevents memorizing specific feature combinations
              → Forces robust feature transformations
    
    These are DIFFERENT regularization needs!
    """)
    
    print("REASON 3: Flexibility for Different Rates")
    print("-" * 70)
    print("""
    While we use the same rate (0.1) for both, having separate layers
    allows us to use different dropout rates if needed:
    
    dropout1 = nn.Dropout(p=0.1)  # Light regularization for attention
    dropout2 = nn.Dropout(p=0.2)  # Heavier regularization for feedforward
    
    This flexibility is important for fine-tuning!
    """)


def demonstrate_what_happens_if_reused():
    """Show what happens if we try to reuse a single layer."""
    print("\n" * 2)
    print("=" * 70)
    print("What Happens If We Reuse a Single Layer?")
    print("=" * 70)
    
    emb_dim = 768
    single_norm = LayerNorm(emb_dim=emb_dim)
    single_dropout = nn.Dropout(p=0.1)
    
    x = torch.randn(4, 256, emb_dim)
    
    print("PROBLEM 1: Shared Learnable Parameters")
    print("-" * 70)
    print("""
    If we reuse a single LayerNorm:
    
    norm1_out = single_norm(x)              # Normalizes raw input
    attn_output = attention(norm1_out)
    norm2_out = single_norm(x + attn_output)  # Tries to normalize different distribution
    
    Problem:
    • Same scale and shift parameters for both
    • Can't learn optimal normalization for both cases
    • One normalization will be suboptimal
    • Training becomes less stable
    """)
    
    print("Demonstration:")
    norm1_out = single_norm(x)
    attn_output = torch.randn(4, 256, emb_dim) * 0.1
    x_after_attn = x + attn_output
    norm2_out = single_norm(x_after_attn)
    
    print(f"  norm1_out mean: {norm1_out.mean().item():.4f}")
    print(f"  norm2_out mean: {norm2_out.mean().item():.4f}")
    print("  ↑ Same parameters trying to normalize different distributions!")
    print()
    
    print("PROBLEM 2: Shared Dropout State")
    print("-" * 70)
    print("""
    If we reuse a single Dropout:
    
    dropped1 = single_dropout(attn_output)
    dropped2 = single_dropout(ff_output)
    
    Problem:
    • Same random pattern for both
    • Less effective regularization
    • Attention and feedforward get same dropout pattern
    • Reduces diversity in regularization
    """)
    
    single_dropout.train()
    attn_output = torch.randn(4, 256, emb_dim)
    ff_output = torch.randn(4, 256, emb_dim)
    
    dropped1 = single_dropout(attn_output)
    dropped2 = single_dropout(ff_output)
    
    # Check if patterns are related (they might be sequential)
    print("  Using same dropout sequentially:")
    print(f"    attn_output zeros: {(dropped1 == 0).sum().item()}")
    print(f"    ff_output zeros: {(dropped2 == 0).sum().item()}")
    print("  ↑ Sequential calls might have correlated patterns")
    print()


def demonstrate_architecture_comparison():
    """Compare architectures with separate vs shared layers."""
    print("\n" * 2)
    print("=" * 70)
    print("Architecture Comparison")
    print("=" * 70)
    
    print("""
CORRECT ARCHITECTURE (Separate Layers):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Input (x)
  ↓
norm1 → Attention → dropout1 → Add (x + output)
  ↓
norm2 → FeedForward → dropout2 → Add (x + output)
  ↓
Output

Benefits:
  ✓ norm1 learns optimal normalization for attention input
  ✓ norm2 learns optimal normalization for feedforward input
  ✓ dropout1 independently regularizes attention
  ✓ dropout2 independently regularizes feedforward
  ✓ Better training stability
  ✓ Better model performance


INCORRECT ARCHITECTURE (Shared Layers):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Input (x)
  ↓
single_norm → Attention → single_dropout → Add (x + output)
  ↓
single_norm → FeedForward → single_dropout → Add (x + output)
  ↓
Output

Problems:
  ✗ Single norm tries to normalize different distributions
  ✗ Can't learn optimal normalization for both
  ✗ Dropout patterns might be correlated
  ✗ Less effective regularization
  ✗ Worse training stability
  ✗ Worse model performance
    """)


def demonstrate_parameter_count():
    """Show that separate layers don't significantly increase parameters."""
    print("\n" * 2)
    print("=" * 70)
    print("Parameter Count: Separate vs Shared")
    print("=" * 70)
    
    emb_dim = 768
    
    # Separate layers
    norm1 = LayerNorm(emb_dim=emb_dim)
    norm2 = LayerNorm(emb_dim=emb_dim)
    
    # Single layer
    single_norm = LayerNorm(emb_dim=emb_dim)
    
    params_separate = sum(p.numel() for p in [norm1, norm2])
    params_single = sum(p.numel() for p in [single_norm])
    
    print(f"Separate LayerNorm layers:")
    print(f"  norm1 parameters: {sum(p.numel() for p in norm1.parameters()):,}")
    print(f"  norm2 parameters: {sum(p.numel() for p in norm2.parameters()):,}")
    print(f"  Total: {params_separate:,}")
    print()
    
    print(f"Single LayerNorm layer:")
    print(f"  Total: {params_single:,}")
    print()
    
    print(f"Difference: {params_separate - params_single:,} parameters")
    print(f"  ↑ Only {emb_dim * 2:,} extra parameters (scale + shift)")
    print(f"  ↑ Negligible compared to model size (millions of parameters)")
    print()
    
    print("Key Point:")
    print("  The small parameter increase is worth it for:")
    print("  • Better normalization")
    print("  • Better regularization")
    print("  • Better model performance")
    print("  • Training stability")


if __name__ == "__main__":
    demonstrate_why_separate_layernorm()
    demonstrate_why_separate_dropout()
    demonstrate_what_happens_if_reused()
    demonstrate_architecture_comparison()
    demonstrate_parameter_count()

