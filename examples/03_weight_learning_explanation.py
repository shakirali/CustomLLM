"""
Explanation: When and How Weights Are Learned

This example demonstrates:
1. Weight initialization (happens at model creation)
2. Weight learning (happens during training with backpropagation)
3. The training loop process
"""

import sys
import os

# Add parent directory to path to import src modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn as nn
from src.model import FeedForward

def demonstrate_weight_initialization():
    """Show that weights are initialized when you create the model."""
    print("=" * 70)
    print("STEP 1: Weight Initialization (Happens at Model Creation)")
    print("=" * 70)
    
    # Create FeedForward model
    emb_dim = 768
    ff = FeedForward(emb_dim=emb_dim)
    
    print(f"Created FeedForward model with emb_dim={emb_dim}")
    print()
    
    # Access the first linear layer
    first_linear = ff.layers[0]  # nn.Linear(768, 3072)
    
    print("First Linear Layer (768 → 3072):")
    print(f"  Weight matrix shape: {first_linear.weight.shape}")
    print(f"  Bias vector shape: {first_linear.bias.shape}")
    print()
    
    # Show initial weights (randomly initialized by PyTorch)
    print("Initial Weight Values (first 5 values of first row):")
    print(f"  {first_linear.weight.data[0, :5]}")
    print("  ↑ These are RANDOM values (PyTorch default initialization)")
    print()
    
    # Store initial weights for comparison
    initial_weights = first_linear.weight.data.clone()
    
    return ff, initial_weights


def demonstrate_forward_pass_only():
    """Show that forward pass does NOT change weights."""
    print("=" * 70)
    print("STEP 2: Forward Pass (Does NOT Change Weights)")
    print("=" * 70)
    
    ff, initial_weights = demonstrate_weight_initialization()
    
    # Create dummy input
    batch_size = 4
    seq_len = 256
    x = torch.randn(batch_size, seq_len, 768)
    
    print(f"Input shape: {x.shape}")
    print()
    
    # Forward pass
    output = ff(x)
    print(f"Output shape: {output.shape}")
    print()
    
    # Check if weights changed
    first_linear = ff.layers[0]
    current_weights = first_linear.weight.data
    
    weights_changed = not torch.equal(initial_weights, current_weights)
    print(f"Weights changed after forward pass? {weights_changed}")
    print("  ↑ NO! Forward pass only computes output, doesn't update weights")
    print()


def demonstrate_training_process():
    """Show the complete training process where weights ARE learned."""
    print("=" * 70)
    print("STEP 3: Training Process (Where Weights ARE Learned)")
    print("=" * 70)
    
    # Create model
    emb_dim = 768
    ff = FeedForward(emb_dim=emb_dim)
    
    # Store initial weights
    first_linear = ff.layers[0]
    initial_weights = first_linear.weight.data.clone()
    
    print("Initial weights (first value):", initial_weights[0, 0].item())
    print()
    
    # Create dummy data (simulating a training batch)
    batch_size = 4
    seq_len = 256
    x = torch.randn(batch_size, seq_len, emb_dim)
    
    # Create dummy targets (for demonstration - in real training, these come from dataset)
    # In GPT, targets are the next token predictions
    targets = torch.randn(batch_size, seq_len, emb_dim)
    
    # Set up training components
    optimizer = torch.optim.Adam(ff.parameters(), lr=0.001)
    criterion = nn.MSELoss()  # Mean Squared Error loss (for demonstration)
    
    print("Training Components:")
    print("  Optimizer: Adam (learning rate: 0.001)")
    print("  Loss function: MSE")
    print()
    
    # Training loop (one iteration)
    print("Training Step:")
    print("  1. Forward pass: input → model → output")
    
    # Forward pass
    output = ff(x)
    print(f"     Input: {x.shape} → Output: {output.shape}")
    
    print("  2. Compute loss: compare output with targets")
    loss = criterion(output, targets)
    print(f"     Loss: {loss.item():.4f}")
    
    print("  3. Backward pass: compute gradients")
    # Zero gradients (important!)
    optimizer.zero_grad()
    
    # Backward pass (computes gradients)
    loss.backward()
    
    # Check gradients
    grad = first_linear.weight.grad
    print(f"     Gradient shape: {grad.shape}")
    print(f"     Gradient (first value): {grad[0, 0].item():.6f}")
    print("     ↑ Gradients tell us HOW to update weights")
    
    print("  4. Update weights: optimizer adjusts weights using gradients")
    optimizer.step()  # THIS IS WHERE WEIGHTS ARE UPDATED!
    
    # Check if weights changed
    updated_weights = first_linear.weight.data
    weight_change = (updated_weights[0, 0] - initial_weights[0, 0]).item()
    
    print(f"     Weight (first value): {initial_weights[0, 0].item():.6f} → {updated_weights[0, 0].item():.6f}")
    print(f"     Change: {weight_change:+.6f}")
    print("     ↑ YES! Weights changed after optimizer.step()")
    print()
    
    print("=" * 70)
    print("Summary: When Weights Are Learned")
    print("=" * 70)
    print("""
    1. INITIALIZATION (Model Creation):
       - Weights are randomly initialized
       - Shape: (768, 3072) for first linear layer
       - Happens: When you create FeedForward(emb_dim=768)
    
    2. FORWARD PASS (Inference):
       - Weights are USED but NOT CHANGED
       - Input (4, 256, 768) → Output (4, 256, 768)
       - Happens: Every time you call ff(x)
    
    3. TRAINING (Learning):
       - Weights ARE UPDATED
       - Process:
         a. Forward pass: compute output
         b. Compute loss: compare output with targets
         c. Backward pass: compute gradients (loss.backward())
         d. Update weights: optimizer.step() ← THIS IS WHERE LEARNING HAPPENS!
       - Happens: During training loop (not implemented yet in our codebase)
    
    Currently, we only have:
    ✓ Model components (FeedForward, LayerNorm, GELU)
    ✓ Forward pass code
    ✗ Training loop (to be implemented later)
    """)


def show_weight_matrix_details():
    """Show detailed breakdown of the weight matrix."""
    print("=" * 70)
    print("Weight Matrix Details: (768, 3072)")
    print("=" * 70)
    
    ff = FeedForward(emb_dim=768)
    first_linear = ff.layers[0]
    
    W = first_linear.weight.data  # Shape: (768, 3072)
    b = first_linear.bias.data    # Shape: (3072,)
    
    print("Weight Matrix W:")
    print(f"  Shape: {W.shape}")
    print(f"  Total parameters: {W.numel():,} (768 × 3072 = 2,359,296)")
    print("  Each row: 3072 values (one for each output dimension)")
    print("  Each column: 768 values (one for each input dimension)")
    print()
    
    print("Bias Vector b:")
    print(f"  Shape: {b.shape}")
    print(f"  Total parameters: {b.numel():,} (3072)")
    print()
    
    print("How one output value is computed:")
    print("  output[0] = input[0]*W[0,0] + input[1]*W[1,0] + ... + input[767]*W[767,0] + b[0]")
    print("  output[1] = input[0]*W[0,1] + input[1]*W[1,1] + ... + input[767]*W[767,1] + b[1]")
    print("  ...")
    print("  output[3071] = input[0]*W[0,3071] + ... + input[767]*W[767,3071] + b[3071]")
    print()
    
    print("During Training:")
    print("  - All 2,359,296 weight values are updated")
    print("  - All 3072 bias values are updated")
    print("  - Updates are based on gradients computed from loss")
    print("  - Process repeats for many batches over many epochs")


if __name__ == "__main__":
    # Run demonstrations
    demonstrate_forward_pass_only()
    print("\n" * 2)
    demonstrate_training_process()
    print("\n" * 2)
    show_weight_matrix_details()

