"""
Why TokenEmbedding Inherits from nn.Module?

This example explains:
1. What nn.Module provides
2. Why neural network components need it
3. What happens if we don't inherit from nn.Module
4. The benefits for TokenEmbedding specifically
"""

import torch
import torch.nn as nn


def demonstrate_what_nn_module_provides():
    """Show what nn.Module provides."""
    print("=" * 70)
    print("What nn.Module Provides")
    print("=" * 70)
    
    print("""
nn.Module is the BASE CLASS for all neural network components in PyTorch.

It provides essential functionality:

1. PARAMETER REGISTRATION
   • Automatically tracks learnable parameters (nn.Parameter)
   • Makes parameters accessible to optimizers
   • Enables gradient computation

2. BUFFER REGISTRATION
   • Tracks non-trainable tensors (e.g., running statistics)
   • Included in state_dict for saving/loading

3. DEVICE MANAGEMENT
   • .to(device) moves all parameters and buffers
   • .cuda() / .cpu() for device placement

4. STATE DICT MANAGEMENT
   • .state_dict() for saving model weights
   • .load_state_dict() for loading weights

5. TRAINING MODE
   • .train() / .eval() for training vs inference
   • Affects dropout, batch norm, etc.

6. COMPUTATION GRAPH
   • Integrates with autograd for backpropagation
   • Tracks operations for gradient computation

7. MODULE COMPOSITION
   • Can contain other nn.Module instances
   • Hierarchical model structure
    """)


def demonstrate_with_nn_module():
    """Show TokenEmbedding with nn.Module."""
    print("\n" * 2)
    print("=" * 70)
    print("TokenEmbedding WITH nn.Module (Correct)")
    print("=" * 70)
    
    class TokenEmbedding(nn.Module):
        def __init__(self, vocab_size, embed_dim):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, embed_dim)
        
        def forward(self, token_ids):
            return self.embedding(token_ids)
    
    # Create instance
    token_embedding = TokenEmbedding(vocab_size=50257, embed_dim=768)
    
    print("1. Parameter Tracking:")
    print(f"   Parameters: {sum(p.numel() for p in token_embedding.parameters()):,}")
    print(f"   ✓ Embedding weights are tracked as parameters")
    print()
    
    print("2. Device Management:")
    if torch.cuda.is_available():
        token_embedding = token_embedding.cuda()
        print(f"   Device: {next(token_embedding.parameters()).device}")
        print("   ✓ All parameters moved to GPU")
    else:
        print("   Device: CPU (CUDA not available)")
        print("   ✓ All parameters on CPU")
    print()
    
    print("3. State Dict (Saving/Loading):")
    state_dict = token_embedding.state_dict()
    print(f"   Keys: {list(state_dict.keys())}")
    print(f"   Embedding shape: {state_dict['embedding.weight'].shape}")
    print("   ✓ Can save/load model weights")
    print()
    
    print("4. Optimizer Compatibility:")
    optimizer = torch.optim.Adam(token_embedding.parameters(), lr=0.001)
    print(f"   Optimizer parameters: {len(list(optimizer.param_groups[0]['params']))}")
    print("   ✓ Optimizer can access parameters")
    print()
    
    print("5. Gradient Computation:")
    token_ids = torch.tensor([[1, 2, 3]])
    output = token_embedding(token_ids)
    loss = output.sum()
    loss.backward()
    
    has_grad = token_embedding.embedding.weight.grad is not None
    print(f"   Has gradients: {has_grad}")
    print("   ✓ Gradients computed for backpropagation")
    print()
    
    print("6. Module Composition:")
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.token_embedding = TokenEmbedding(50257, 768)
    
    model = SimpleModel()
    print(f"   Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print("   ✓ Can be part of larger models")


def demonstrate_without_nn_module():
    """Show what happens without nn.Module."""
    print("\n" * 2)
    print("=" * 70)
    print("TokenEmbedding WITHOUT nn.Module (Problems)")
    print("=" * 70)
    
    class TokenEmbeddingNoModule:
        """TokenEmbedding without nn.Module - DON'T DO THIS!"""
        def __init__(self, vocab_size, embed_dim):
            # No super().__init__() - not inheriting from nn.Module
            self.embedding = nn.Embedding(vocab_size, embed_dim)
        
        def forward(self, token_ids):
            return self.embedding(token_ids)
    
    token_embedding = TokenEmbeddingNoModule(vocab_size=50257, embed_dim=768)
    
    print("1. Parameter Tracking:")
    try:
        params = list(token_embedding.parameters())
        print(f"   Parameters: {len(params)}")
    except AttributeError:
        print("   ✗ ERROR: No .parameters() method!")
        print("   ✗ Cannot access parameters")
    print()
    
    print("2. Device Management:")
    try:
        token_embedding = token_embedding.cuda()
        print("   ✓ Works, but only because embedding is nn.Module")
    except AttributeError:
        print("   ✗ ERROR: No .cuda() method!")
    print()
    
    print("3. State Dict:")
    try:
        state_dict = token_embedding.state_dict()
        print(f"   Keys: {list(state_dict.keys())}")
    except AttributeError:
        print("   ✗ ERROR: No .state_dict() method!")
        print("   ✗ Cannot save/load model")
    print()
    
    print("4. Optimizer Compatibility:")
    try:
        optimizer = torch.optim.Adam(token_embedding.parameters(), lr=0.001)
        print("   ✓ Works, but only because embedding is nn.Module")
    except AttributeError:
        print("   ✗ ERROR: No .parameters() method!")
        print("   ✗ Optimizer cannot access parameters")
    print()
    
    print("5. Module Composition:")
    try:
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                # This won't work properly
                self.token_embedding = TokenEmbeddingNoModule(50257, 768)
        
        model = SimpleModel()
        # The embedding parameters won't be tracked by the model!
        print(f"   Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        print("   ✗ Embedding parameters NOT tracked by parent model!")
    except Exception as e:
        print(f"   ✗ ERROR: {e}")


def demonstrate_key_benefits():
    """Show key benefits for TokenEmbedding."""
    print("\n" * 2)
    print("=" * 70)
    print("Key Benefits for TokenEmbedding")
    print("=" * 70)
    
    print("""
1. LEARNABLE PARAMETERS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   TokenEmbedding contains embedding weights (vocab_size × embed_dim)
   These are LEARNED during training.
   
   Without nn.Module:
     ✗ Parameters not tracked
     ✗ Optimizer can't update them
     ✗ No gradient computation
   
   With nn.Module:
     ✓ Parameters automatically tracked
     ✓ Optimizer can access and update them
     ✓ Gradients computed during backprop


2. INTEGRATION WITH OTHER MODULES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   TokenEmbedding is used in larger models (GPTModel).
   
   Without nn.Module:
     ✗ Can't be part of nn.Sequential
     ✗ Parent model can't track its parameters
     ✗ Can't use model.children() or model.modules()
   
   With nn.Module:
     ✓ Can be used in nn.Sequential
     ✓ Parent model tracks all parameters
     ✓ Works with model.children() and model.modules()


3. MODEL SAVING AND LOADING
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   Need to save trained embedding weights.
   
   Without nn.Module:
     ✗ No .state_dict() method
     ✗ Can't save/load weights
     ✗ Manual weight management needed
   
   With nn.Module:
     ✓ .state_dict() provides all weights
     ✓ Easy saving/loading
     ✓ Standard PyTorch workflow


4. DEVICE MANAGEMENT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   Need to move model to GPU for training.
   
   Without nn.Module:
     ✗ No .to(device) or .cuda() methods
     ✗ Manual device management
     ✗ Error-prone
   
   With nn.Module:
     ✓ .to(device) moves all parameters
     ✓ .cuda() / .cpu() work automatically
     ✓ Clean and safe


5. TRAINING VS INFERENCE MODE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   Different behavior during training vs inference.
   
   Without nn.Module:
     ✗ No .train() / .eval() methods
     ✗ Can't switch modes
     ✗ Manual mode management
   
   With nn.Module:
     ✓ .train() / .eval() work
     ✓ Consistent with other modules
     ✓ Standard PyTorch pattern
    """)


def demonstrate_practical_example():
    """Show practical example of why it matters."""
    print("\n" * 2)
    print("=" * 70)
    print("Practical Example: Why It Matters")
    print("=" * 70)
    
    print("""
Scenario: Building a GPT Model
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class GPTModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding = TokenEmbedding(...)  # Needs nn.Module!
        self.transformer_blocks = nn.ModuleList([...])
        self.output_proj = nn.Linear(...)
    
    def forward(self, token_ids):
        x = self.token_embedding(token_ids)  # Works because it's nn.Module
        for block in self.transformer_blocks:
            x = block(x)
        return self.output_proj(x)

WITHOUT nn.Module:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  ✗ model.parameters() won't include embedding weights
  ✗ Optimizer won't update embedding weights
  ✗ model.state_dict() won't include embedding weights
  ✗ Can't save/load embedding weights
  ✗ model.to(device) won't move embedding weights
  ✗ Training will fail!

WITH nn.Module:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  ✓ model.parameters() includes all weights
  ✓ Optimizer updates all weights
  ✓ model.state_dict() includes all weights
  ✓ Can save/load entire model
  ✓ model.to(device) moves all weights
  ✓ Training works perfectly!
    """)


if __name__ == "__main__":
    demonstrate_what_nn_module_provides()
    demonstrate_with_nn_module()
    demonstrate_without_nn_module()
    demonstrate_key_benefits()
    demonstrate_practical_example()

