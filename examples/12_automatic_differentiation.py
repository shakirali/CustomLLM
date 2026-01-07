"""
Why We Only Implement Forward Pass (Automatic Differentiation)

This example explains:
1. How PyTorch's autograd automatically computes gradients
2. Why we don't need to implement backward pass manually
3. How the computation graph works
4. What happens during loss.backward()
"""

import torch
import torch.nn as nn


def demonstrate_automatic_differentiation():
    """Show how PyTorch automatically computes gradients."""
    print("=" * 70)
    print("Automatic Differentiation: The Magic of PyTorch")
    print("=" * 70)
    
    print("""
KEY INSIGHT:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

We DON'T need to implement backward pass manually!

PyTorch's autograd system automatically:
  1. Tracks all operations in forward pass
  2. Builds a computation graph
  3. Computes gradients automatically when we call loss.backward()
  4. Stores gradients in parameter.grad

This is called "Automatic Differentiation" or "Autograd"
    """)


def demonstrate_how_it_works():
    """Show how autograd works step by step."""
    print("\n" * 2)
    print("=" * 70)
    print("How It Works: Step by Step")
    print("=" * 70)
    
    # Create TokenEmbedding
    class TokenEmbedding(nn.Module):
        def __init__(self, vocab_size, embed_dim):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, embed_dim)
        
        def forward(self, token_ids):
            return self.embedding(token_ids)
    
    token_embedding = TokenEmbedding(vocab_size=10, embed_dim=4)
    
    print("Step 1: Forward Pass")
    print("-" * 70)
    token_ids = torch.tensor([[1, 2, 3]])
    print(f"Input token_ids: {token_ids}")
    
    # Forward pass
    embeddings = token_embedding(token_ids)
    print(f"Output embeddings shape: {embeddings.shape}")
    print(f"Output embeddings:\n{embeddings}")
    print()
    print("What PyTorch does:")
    print("  ✓ Tracks the embedding lookup operation")
    print("  ✓ Records: embeddings = embedding(token_ids)")
    print("  ✓ Builds computation graph")
    print("  ✓ Links output to embedding.weight parameter")
    print()
    
    print("Step 2: Compute Loss")
    print("-" * 70)
    # Create dummy target
    target = torch.randn(1, 3, 4)
    loss = nn.MSELoss()(embeddings, target)
    print(f"Loss: {loss.item():.4f}")
    print()
    print("What PyTorch does:")
    print("  ✓ Tracks the loss computation")
    print("  ✓ Records: loss = MSE(embeddings, target)")
    print("  ✓ Extends computation graph")
    print("  ✓ Links loss back to embeddings")
    print()
    
    print("Step 3: Backward Pass (Automatic!)")
    print("-" * 70)
    # Zero gradients first
    token_embedding.zero_grad()
    
    # Backward pass - THIS IS WHERE MAGIC HAPPENS!
    loss.backward()
    
    print("What PyTorch does AUTOMATICALLY:")
    print("  1. Starts from loss")
    print("  2. Traces back through computation graph")
    print("  3. Computes gradients using chain rule")
    print("  4. Stores gradients in parameter.grad")
    print()
    
    # Check if gradients exist
    has_grad = token_embedding.embedding.weight.grad is not None
    print(f"Gradients computed: {has_grad}")
    if has_grad:
        grad = token_embedding.embedding.weight.grad
        print(f"Gradient shape: {grad.shape}")
        print(f"Gradient (first few values):\n{grad[:3, :]}")
        print()
        print("  ✓ Gradients automatically computed!")
        print("  ✓ Stored in embedding.weight.grad")
        print("  ✓ Ready for optimizer to use")
    print()
    
    print("Step 4: Update Weights")
    print("-" * 70)
    optimizer = torch.optim.Adam(token_embedding.parameters(), lr=0.001)
    
    # Store weight before update
    weight_before = token_embedding.embedding.weight.data[1, 0].item()
    
    optimizer.step()
    
    weight_after = token_embedding.embedding.weight.data[1, 0].item()
    
    print(f"Weight before: {weight_before:.6f}")
    print(f"Weight after:  {weight_after:.6f}")
    print(f"Change:        {weight_after - weight_before:+.6f}")
    print()
    print("  ✓ Weights updated using gradients")
    print("  ✓ All automatic - no manual backward needed!")


def demonstrate_computation_graph():
    """Show the computation graph concept."""
    print("\n" * 2)
    print("=" * 70)
    print("Computation Graph: What PyTorch Builds")
    print("=" * 70)
    
    print("""
When we do forward pass, PyTorch builds a computation graph:

Forward Pass:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

token_ids (input)
    ↓
embedding.weight (parameter) ──┐
    ↓                           │
embedding lookup                │
    ↓                           │
embeddings (output)             │
    ↓                           │
MSE loss                        │
    ↓                           │
loss (scalar)                   │
                                │
Computation Graph:              │
  loss ← embeddings ← embedding.weight
                                │
Backward Pass (automatic):      │
  loss → embeddings → embedding.weight
                                │
Gradients flow backwards:       │
  d(loss)/d(weight) computed! ──┘

Key Point:
  • Forward pass: Builds graph
  • Backward pass: Traverses graph in reverse
  • Gradients: Computed automatically using chain rule
    """)


def demonstrate_why_no_manual_backward():
    """Explain why we don't need manual backward."""
    print("\n" * 2)
    print("=" * 70)
    print("Why We Don't Need Manual Backward Implementation")
    print("=" * 70)
    
    print("""
1. AUTOMATIC DIFFERENTIATION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   PyTorch tracks every operation in forward pass
   → Builds computation graph automatically
   → Computes gradients automatically
   
   We just need to:
     • Implement forward() method
     • Call loss.backward()
     • PyTorch does the rest!


2. CHAIN RULE AUTOMATICALLY APPLIED
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   For: loss = f(g(h(embedding.weight)))
   
   Gradient: d(loss)/d(weight) = d(loss)/d(f) × d(f)/d(g) × d(g)/d(h) × d(h)/d(weight)
   
   PyTorch computes this automatically!
   We don't need to derive formulas manually.


3. WORKS FOR ANY OPERATION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   Works for:
     • Matrix multiplication
     • Element-wise operations
     • Convolutions
     • Embedding lookups
     • Any differentiable operation
   
   PyTorch knows how to compute gradients for all of them!


4. EFFICIENT IMPLEMENTATION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   PyTorch's autograd is:
     • Highly optimized (C++ backend)
     • Memory efficient
     • Handles complex graphs
     • Much better than manual implementation
    """)


def demonstrate_what_if_we_implemented_backward():
    """Show what manual backward would look like (not needed)."""
    print("\n" * 2)
    print("=" * 70)
    print("What If We Implemented Backward Manually? (Not Needed)")
    print("=" * 70)
    
    print("""
If we had to implement backward manually:

class TokenEmbedding(nn.Module):
    def forward(self, token_ids):
        return self.embedding(token_ids)
    
    def backward(self, grad_output):
        # Manual gradient computation
        # Need to know:
        #   - How embedding lookup works
        #   - How to compute d(output)/d(weight)
        #   - Chain rule application
        #   - Handle all edge cases
        #   - Optimize for performance
        #   - Handle different tensor shapes
        #   - ... (very complex!)
        
        # This is what PyTorch does automatically!
        pass

Problems:
  ✗ Very complex to implement correctly
  ✗ Error-prone
  ✗ Need deep math knowledge
  ✗ Need to handle all edge cases
  ✗ Performance optimization difficult
  ✗ Maintenance burden

With PyTorch autograd:
  ✓ Just implement forward()
  ✓ PyTorch handles backward automatically
  ✓ Correct, optimized, and tested
  ✓ Works for any operation
    """)


def demonstrate_practical_example():
    """Show practical example with TokenEmbedding."""
    print("\n" * 2)
    print("=" * 70)
    print("Practical Example: TokenEmbedding Training")
    print("=" * 70)
    
    class TokenEmbedding(nn.Module):
        def __init__(self, vocab_size, embed_dim):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, embed_dim)
        
        def forward(self, token_ids):
            # Only forward pass - that's all we need!
            return self.embedding(token_ids)
    
    # Create model
    token_embedding = TokenEmbedding(vocab_size=10, embed_dim=4)
    optimizer = torch.optim.Adam(token_embedding.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    # Training step
    print("Training Step:")
    print("-" * 70)
    
    # 1. Forward pass (we implement this)
    token_ids = torch.tensor([[1, 2, 3]])
    embeddings = token_embedding(token_ids)  # Our forward() method
    print("1. Forward pass: embeddings = token_embedding(token_ids)")
    print(f"   Output shape: {embeddings.shape}")
    
    # 2. Compute loss
    target = torch.randn(1, 3, 4)
    loss = criterion(embeddings, target)
    print(f"2. Loss: {loss.item():.4f}")
    
    # 3. Backward pass (PyTorch does this automatically!)
    optimizer.zero_grad()
    loss.backward()  # PyTorch computes gradients automatically!
    print("3. Backward pass: loss.backward()")
    print("   ✓ PyTorch automatically computes gradients")
    print("   ✓ Gradients stored in embedding.weight.grad")
    
    # 4. Update weights
    optimizer.step()
    print("4. Update: optimizer.step()")
    print("   ✓ Weights updated using gradients")
    print()
    
    print("Summary:")
    print("  • We implement: forward() method")
    print("  • PyTorch handles: backward() automatically")
    print("  • Result: Training works perfectly!")


def demonstrate_requires_grad():
    """Show how requires_grad enables automatic differentiation."""
    print("\n" * 2)
    print("=" * 70)
    print("How requires_grad Enables Automatic Differentiation")
    print("=" * 70)
    
    # Create embedding
    embedding = nn.Embedding(10, 4)
    
    print("Embedding weight properties:")
    weight = embedding.weight
    print(f"  requires_grad: {weight.requires_grad}")
    print(f"  is_leaf: {weight.is_leaf}")
    print(f"  grad_fn: {weight.grad_fn}")
    print()
    print("Key Point:")
    print("  • nn.Embedding creates parameters with requires_grad=True")
    print("  • This tells PyTorch to track gradients for this tensor")
    print("  • When we do operations with it, PyTorch builds computation graph")
    print()
    
    # Forward pass
    token_ids = torch.tensor([[1, 2, 3]])
    output = embedding(token_ids)
    
    print("After forward pass:")
    print(f"  output.requires_grad: {output.requires_grad}")
    print(f"  output.grad_fn: {output.grad_fn}")
    print()
    print("Key Point:")
    print("  • Output has grad_fn (gradient function)")
    print("  • This links output back to embedding.weight")
    print("  • Enables automatic gradient computation")
    print()
    
    # Compute loss
    target = torch.randn(1, 3, 4)
    loss = nn.MSELoss()(output, target)
    
    print("After loss computation:")
    print(f"  loss.requires_grad: {loss.requires_grad}")
    print(f"  loss.grad_fn: {loss.grad_fn}")
    print()
    print("Key Point:")
    print("  • Loss has grad_fn linking back to output")
    print("  • Chain: loss → output → embedding.weight")
    print("  • loss.backward() traverses this chain automatically")


if __name__ == "__main__":
    demonstrate_automatic_differentiation()
    demonstrate_how_it_works()
    demonstrate_computation_graph()
    demonstrate_why_no_manual_backward()
    demonstrate_what_if_we_implemented_backward()
    demonstrate_practical_example()
    demonstrate_requires_grad()

