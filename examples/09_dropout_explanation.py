"""
Understanding Dropout: What It Is and Why It's Used

This example explains:
1. What Dropout is (randomly setting some neurons to zero)
2. Why it's used (regularization to prevent overfitting)
3. How it works during training vs inference
4. Why it's important in Transformer blocks
"""

import torch
import torch.nn as nn
import numpy as np

def demonstrate_what_is_dropout():
    """Show what dropout does to a tensor."""
    print("=" * 70)
    print("What is Dropout?")
    print("=" * 70)
    
    # Create a sample tensor
    x = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0],
                      [6.0, 7.0, 8.0, 9.0, 10.0]])
    
    print("Original Tensor:")
    print(x)
    print()
    
    # Create dropout layer with 50% probability
    dropout = nn.Dropout(p=0.5)
    
    print("After Dropout (p=0.5) - Training Mode:")
    print("-" * 70)
    dropout.train()  # Set to training mode
    output1 = dropout(x)
    print("Output 1:")
    print(output1)
    print()
    
    output2 = dropout(x)
    print("Output 2 (different random zeros):")
    print(output2)
    print()
    
    print("Key Observations:")
    print("  ✓ Some values are set to 0 (randomly)")
    print("  ✓ Remaining values are scaled up (multiplied by 1/(1-p))")
    print("  ✓ Different zeros each time (random)")
    print()
    
    print("After Dropout - Evaluation Mode:")
    print("-" * 70)
    dropout.eval()  # Set to evaluation mode
    output_eval = dropout(x)
    print("Output (no dropout applied):")
    print(output_eval)
    print("  ✓ All values preserved (no zeros)")
    print("  ✓ No scaling needed")


def demonstrate_why_dropout():
    """Explain why dropout is used."""
    print("\n" * 2)
    print("=" * 70)
    print("Why Use Dropout? (Preventing Overfitting)")
    print("=" * 70)
    
    print("""
The Problem: Overfitting
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Without Dropout:
  • Model learns to rely on specific neurons/features
  • Memorizes training data patterns too closely
  • Fails to generalize to new, unseen data
  • High training accuracy, low validation accuracy

Example:
  Training: 95% accuracy
  Validation: 60% accuracy  ← Overfitting!


The Solution: Dropout
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

With Dropout:
  • Randomly "turns off" some neurons during training
  • Forces model to not rely on any single neuron
  • Encourages learning redundant, robust features
  • Better generalization to new data

Example:
  Training: 85% accuracy
  Validation: 82% accuracy  ← Better generalization!


How It Works:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. During Training:
   - Randomly set some neurons to 0 (with probability p)
   - Scale remaining neurons by 1/(1-p) to maintain expected value
   - Different neurons dropped each forward pass

2. During Inference:
   - All neurons active (no dropout)
   - No scaling needed
   - Use all learned features
    """)


def demonstrate_dropout_effect():
    """Show the effect of dropout on model training."""
    print("\n" * 2)
    print("=" * 70)
    print("Dropout Effect: Training vs Inference")
    print("=" * 70)
    
    # Create a simple linear layer
    linear = nn.Linear(5, 3)
    dropout = nn.Dropout(p=0.3)
    
    x = torch.randn(2, 5)
    
    print("Input shape:", x.shape)
    print()
    
    # Training mode
    print("TRAINING MODE (dropout.active = True):")
    print("-" * 70)
    linear.train()
    dropout.train()
    
    for i in range(3):
        output = dropout(linear(x))
        print(f"Forward pass {i+1}:")
        print(f"  Output: {output}")
        print(f"  Zeros in output: {(output == 0).sum().item()}")
        print()
    
    print("Key Point:")
    print("  • Different neurons dropped each time")
    print("  • Model can't rely on specific neurons")
    print("  • Forces robust feature learning")
    print()
    
    # Evaluation mode
    print("EVALUATION MODE (dropout.active = False):")
    print("-" * 70)
    linear.eval()
    dropout.eval()
    
    output = dropout(linear(x))
    print(f"Output: {output}")
    print(f"Zeros in output: {(output == 0).sum().item()}")
    print()
    
    print("Key Point:")
    print("  • All neurons active")
    print("  • No randomness")
    print("  • Consistent predictions")


def demonstrate_dropout_in_transformer():
    """Explain why dropout is used in Transformer blocks."""
    print("\n" * 2)
    print("=" * 70)
    print("Why Dropout in Transformer Blocks?")
    print("=" * 70)
    
    print("""
Transformer Block Architecture:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Input
  ↓
LayerNorm1 → Attention → Dropout1 → Add (residual)
  ↓
LayerNorm2 → FeedForward → Dropout2 → Add (residual)
  ↓
Output

Why Dropout After Attention?
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  • Attention learns which tokens to focus on
  • Without dropout: Model might over-rely on specific attention patterns
  • With dropout: Forces model to learn multiple attention strategies
  • Better generalization across different contexts

Why Dropout After FeedForward?
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  • FeedForward processes context-aware features
  • Without dropout: Model might memorize specific feature combinations
  • With dropout: Forces model to learn robust feature transformations
  • Better handling of unseen token combinations

Why NOT Before LayerNorm?
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  • LayerNorm needs consistent input statistics
  • Dropout before LayerNorm would change normalization behavior
  • Standard practice: Dropout after the operation, before residual

Typical Dropout Values:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  • 0.1 (10%): Light regularization (common in GPT models)
  • 0.2 (20%): Moderate regularization
  • 0.5 (50%): Heavy regularization (rare in transformers)
  
  GPT models typically use: dropout = 0.1
    """)


def demonstrate_mathematical_effect():
    """Show the mathematical effect of dropout."""
    print("\n" * 2)
    print("=" * 70)
    print("Mathematical Effect of Dropout")
    print("=" * 70)
    
    print("""
During Training (p = dropout probability):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

For each value x:
  • With probability p: set to 0
  • With probability (1-p): keep and scale to x/(1-p)

Expected value:
  E[output] = p × 0 + (1-p) × (x/(1-p))
            = 0 + x
            = x

Result: Expected value is preserved!


Example with p = 0.5:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Input: x = 2.0

50% chance: output = 0
50% chance: output = 2.0 / (1 - 0.5) = 2.0 / 0.5 = 4.0

Expected output: 0.5 × 0 + 0.5 × 4.0 = 2.0 ✓


During Inference:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

All values preserved: output = x
No scaling needed (all neurons active)
    """)


def demonstrate_visual_analogy():
    """Use an analogy to explain dropout."""
    print("\n" * 2)
    print("=" * 70)
    print("Analogy: Dropout as Team Training")
    print("=" * 70)
    
    print("""
Think of a neural network as a team of specialists:

WITHOUT DROPOUT (Overfitting):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  • Team always works with the same people
  • Each person learns to rely on specific teammates
  • Team performs well on familiar tasks
  • Fails when team composition changes
  • "We always do it this way because John is here"


WITH DROPOUT (Better Generalization):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  • Randomly remove team members during practice
  • Team learns to work with different combinations
  • Each person learns multiple roles
  • Team adapts when members are unavailable
  • "We can handle this even if John isn't here"


In Transformer Blocks:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  • Attention: Learns multiple ways to focus on tokens
  • FeedForward: Learns robust feature transformations
  • Model works even when some features are "missing"
  • Better generalization to new text patterns
    """)


if __name__ == "__main__":
    demonstrate_what_is_dropout()
    demonstrate_why_dropout()
    demonstrate_dropout_effect()
    demonstrate_dropout_in_transformer()
    demonstrate_mathematical_effect()
    demonstrate_visual_analogy()

