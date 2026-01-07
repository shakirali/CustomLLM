"""
GPTModel Example: Complete Model Usage

This example demonstrates:
1. Creating a GPTModel instance
2. Forward pass (training/inference)
3. Text generation
4. Parameter counting
5. Different model configurations
"""

import sys
import os

# Add parent directory to path to allow importing src
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import tiktoken
from src.model import GPTModel


def demonstrate_model_creation():
    """Show how to create a GPTModel instance."""
    print("=" * 70)
    print("1. Creating GPTModel")
    print("=" * 70)
    
    # Create model with GPT-2 small configuration
    model = GPTModel(
        vocab_size=50257,    # GPT-2 vocabulary size
        embed_dim=768,       # GPT-2 small embedding dimension
        num_layers=12,       # GPT-2 small number of layers
        num_heads=12,        # GPT-2 small number of heads
        max_seq_len=1024,    # GPT-2 maximum sequence length
        dropout=0.1,
        tie_weights=True     # Standard: tie output projection with token embedding
    )
    
    print("Model created:")
    print(model)
    print()
    
    # Count parameters
    num_params = model.get_num_params()
    print(f"Total trainable parameters: {num_params:,}")
    print(f"  ({num_params / 1e6:.2f}M parameters)")
    print()


def demonstrate_forward_pass():
    """Show forward pass through the model."""
    print("=" * 70)
    print("2. Forward Pass (Training/Inference)")
    print("=" * 70)
    
    # Create a small model for demonstration
    model = GPTModel(
        vocab_size=1000,
        embed_dim=128,
        num_layers=2,
        num_heads=4,
        max_seq_len=256
    )
    
    # Create sample input (batch_size=2, seq_len=10)
    batch_size = 2
    seq_len = 10
    token_ids = torch.randint(0, 1000, (batch_size, seq_len))
    
    print(f"Input shape: {token_ids.shape}")
    print(f"Input tokens (first sequence): {token_ids[0].tolist()}")
    print()
    
    # Forward pass
    model.eval()  # Set to evaluation mode
    with torch.no_grad():
        logits = model(token_ids)
    
    print(f"Output logits shape: {logits.shape}")
    print(f"  (batch_size={batch_size}, seq_len={seq_len}, vocab_size=1000)")
    print()
    
    # Get probabilities
    probs = torch.softmax(logits, dim=-1)
    print(f"Probabilities shape: {probs.shape}")
    print(f"Sum of probabilities (should be 1.0): {probs[0, 0, :].sum().item():.6f}")
    print()
    
    # Get predicted tokens (greedy decoding)
    predicted_tokens = torch.argmax(logits, dim=-1)
    print(f"Predicted tokens shape: {predicted_tokens.shape}")
    print(f"Predicted tokens (first sequence): {predicted_tokens[0].tolist()}")
    print()


def demonstrate_text_generation():
    """Show text generation with the model."""
    print("=" * 70)
    print("3. Text Generation")
    print("=" * 70)
    
    # Create a small model for demonstration
    model = GPTModel(
        vocab_size=1000,
        embed_dim=128,
        num_layers=2,
        num_heads=4,
        max_seq_len=256
    )
    
    # Initialize tokenizer (for demonstration - in real use, you'd use GPT-2 tokenizer)
    # For this example, we'll just use random token IDs
    print("Note: Using random token IDs for demonstration.")
    print("In real usage, you would tokenize text using tiktoken.")
    print()
    
    # Create prompt (batch_size=1, seq_len=5)
    prompt = torch.randint(0, 1000, (1, 5))
    print(f"Prompt shape: {prompt.shape}")
    print(f"Prompt tokens: {prompt[0].tolist()}")
    print()
    
    # Generate text
    print("Generating 10 new tokens...")
    generated = model.generate(
        token_ids=prompt,
        max_new_tokens=10,
        temperature=1.0,  # Standard temperature
        top_k=None,       # No top-k filtering
        top_p=None,      # No nucleus sampling
        stop_token_id=None
    )
    
    print(f"Generated sequence shape: {generated.shape}")
    print(f"Generated tokens: {generated[0].tolist()}")
    print(f"  (Original: {prompt[0].tolist()} + New: {generated[0, 5:].tolist()})")
    print()
    
    # Generate with different temperatures
    print("Generating with different temperatures:")
    for temp in [0.5, 1.0, 1.5]:
        generated = model.generate(
            token_ids=prompt,
            max_new_tokens=5,
            temperature=temp
        )
        print(f"  Temperature {temp}: {generated[0, 5:].tolist()}")
    print()


def demonstrate_different_configurations():
    """Show different model configurations."""
    print("=" * 70)
    print("4. Different Model Configurations")
    print("=" * 70)
    
    configs = [
        {
            "name": "Tiny (for testing)",
            "vocab_size": 1000,
            "embed_dim": 128,
            "num_layers": 2,
            "num_heads": 4,
        },
        {
            "name": "Small (GPT-2 small)",
            "vocab_size": 50257,
            "embed_dim": 768,
            "num_layers": 12,
            "num_heads": 12,
        },
        {
            "name": "Medium (GPT-2 medium)",
            "vocab_size": 50257,
            "embed_dim": 1024,
            "num_layers": 24,
            "num_heads": 16,
        },
    ]
    
    for config in configs:
        model = GPTModel(
            vocab_size=config["vocab_size"],
            embed_dim=config["embed_dim"],
            num_layers=config["num_layers"],
            num_heads=config["num_heads"],
            max_seq_len=1024
        )
        num_params = model.get_num_params()
        print(f"{config['name']}:")
        print(f"  Parameters: {num_params:,} ({num_params / 1e6:.2f}M)")
        print(f"  Config: embed_dim={config['embed_dim']}, "
              f"num_layers={config['num_layers']}, num_heads={config['num_heads']}")
        print()


def demonstrate_weight_tying():
    """Show the effect of weight tying."""
    print("=" * 70)
    print("5. Weight Tying Effect")
    print("=" * 70)
    
    vocab_size = 1000
    embed_dim = 128
    
    # Model with weight tying (default)
    model_tied = GPTModel(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        num_layers=2,
        num_heads=4,
        tie_weights=True
    )
    
    # Model without weight tying
    model_untied = GPTModel(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        num_layers=2,
        num_heads=4,
        tie_weights=False
    )
    
    params_tied = model_tied.get_num_params()
    params_untied = model_untied.get_num_params()
    saved = params_untied - params_tied
    
    print(f"With weight tying:    {params_tied:,} parameters")
    print(f"Without weight tying: {params_untied:,} parameters")
    print(f"Parameters saved:     {saved:,} ({saved / params_untied * 100:.1f}%)")
    print()
    print("Note: Weight tying reduces parameters and improves generalization.")
    print()


def demonstrate_training_mode():
    """Show the difference between training and evaluation mode."""
    print("=" * 70)
    print("6. Training vs Evaluation Mode")
    print("=" * 70)
    
    model = GPTModel(
        vocab_size=1000,
        embed_dim=128,
        num_layers=2,
        num_heads=4,
        max_seq_len=256,
        dropout=0.5  # High dropout to see the effect
    )
    
    token_ids = torch.randint(0, 1000, (1, 10))
    
    # Training mode
    model.train()
    output1 = model(token_ids)
    output2 = model(token_ids)
    
    print("Training mode (dropout active):")
    print(f"  Output 1: {output1[0, 0, :5].tolist()}")
    print(f"  Output 2: {output2[0, 0, :5].tolist()}")
    print(f"  Are outputs equal? {torch.equal(output1, output2)}")
    print()
    
    # Evaluation mode
    model.eval()
    output1 = model(token_ids)
    output2 = model(token_ids)
    
    print("Evaluation mode (dropout disabled):")
    print(f"  Output 1: {output1[0, 0, :5].tolist()}")
    print(f"  Output 2: {output2[0, 0, :5].tolist()}")
    print(f"  Are outputs equal? {torch.equal(output1, output2)}")
    print()
    print("Note: In training mode, dropout creates variation. In eval mode, outputs are deterministic.")
    print()


def main():
    """Run all demonstrations."""
    print("\n" * 2)
    print("=" * 70)
    print("GPTModel Complete Example")
    print("=" * 70)
    print()
    
    demonstrate_model_creation()
    demonstrate_forward_pass()
    demonstrate_text_generation()
    demonstrate_different_configurations()
    demonstrate_weight_tying()
    demonstrate_training_mode()
    
    print("=" * 70)
    print("Example Complete!")
    print("=" * 70)
    print()
    print("Next steps:")
    print("  1. Train the model using a training script (Step 8.1)")
    print("  2. Evaluate the model using an evaluation script (Step 8.2)")
    print("  3. Generate text using the generation script (Step 8.3)")


if __name__ == "__main__":
    main()

