"""
Model Size Analysis Example

This example demonstrates how to determine the size of a GPT model, including:
1. Parameter count (trainable and total)
2. Memory size estimation
3. Actual PyTorch memory usage
4. Model configuration details
5. Comparison with standard GPT-2 model sizes
"""

import sys
import os

# Add parent directory to path to allow importing src
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from src.model import GPTModel


def analyze_model_size():
    """Determine model size - complete example"""
    # Create model
    model = GPTModel(
        vocab_size=50257,
        embed_dim=768,
        num_layers=12,
        num_heads=12,
        max_seq_len=1024
    )

    print("=" * 70)
    print("Model Size Analysis")
    print("=" * 70)

    # 1. Parameter count
    num_params = model.get_num_params(trainable_only=True)
    total_params = model.get_num_params(trainable_only=False)

    print(f"\n1. Parameter Count:")
    print(f"   Trainable parameters: {num_params:,}")
    print(f"   Total parameters: {total_params:,}")
    print(f"   Model size: {num_params / 1e6:.2f}M parameters")

    # 2. Memory size
    memory_mb = num_params * 4 / (1024 * 1024)  # float32 = 4 bytes
    print(f"\n2. Memory Size (float32):")
    print(f"   {memory_mb:.2f} MB")
    print(f"   {memory_mb / 1024:.2f} GB")

    # 3. Actual PyTorch size
    param_size = sum(p.nelement() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.nelement() * b.element_size() for b in model.buffers())
    actual_size_mb = (param_size + buffer_size) / (1024 * 1024)

    print(f"\n3. Actual Memory Usage:")
    print(f"   Parameters: {param_size / (1024 * 1024):.2f} MB")
    print(f"   Buffers: {buffer_size / (1024 * 1024):.2f} MB")
    print(f"   Total: {actual_size_mb:.2f} MB")

    # 4. Model configuration
    print(f"\n4. Model Configuration:")
    print(f"   vocab_size: {model.vocab_size:,}")
    print(f"   embed_dim: {model.embed_dim}")
    print(f"   num_layers: {model.num_layers}")
    print(f"   num_heads: {model.num_heads}")
    print(f"   max_seq_len: {model.max_seq_len}")

    # 5. Comparison with GPT-2
    print(f"\n5. Comparison with GPT-2:")
    if num_params / 1e6 < 150:
        print(f"   Similar to GPT-2 Small (~117M)")
    elif num_params / 1e6 < 400:
        print(f"   Similar to GPT-2 Medium (~345M)")
    elif num_params / 1e6 < 800:
        print(f"   Similar to GPT-2 Large (~762M)")
    else:
        print(f"   Similar to GPT-2 XL (~1.5B)")

    print("=" * 70)


def compare_different_model_sizes():
    """Compare different model configurations."""
    print("\n" + "=" * 70)
    print("Comparing Different Model Configurations")
    print("=" * 70)
    
    configs = [
        {
            "name": "Small Model",
            "vocab_size": 50257,
            "embed_dim": 256,
            "num_layers": 6,
            "num_heads": 8,
            "max_seq_len": 512
        },
        {
            "name": "GPT-2 Small",
            "vocab_size": 50257,
            "embed_dim": 768,
            "num_layers": 12,
            "num_heads": 12,
            "max_seq_len": 1024
        },
        {
            "name": "Medium Model",
            "vocab_size": 50257,
            "embed_dim": 1024,
            "num_layers": 24,
            "num_heads": 16,
            "max_seq_len": 1024
        }
    ]
    
    for config in configs:
        model = GPTModel(
            vocab_size=config["vocab_size"],
            embed_dim=config["embed_dim"],
            num_layers=config["num_layers"],
            num_heads=config["num_heads"],
            max_seq_len=config["max_seq_len"]
        )
        
        num_params = model.get_num_params()
        memory_mb = num_params * 4 / (1024 * 1024)
        
        print(f"\n{config['name']}:")
        print(f"  Parameters: {num_params / 1e6:.2f}M")
        print(f"  Memory: {memory_mb:.2f} MB")
        print(f"  Config: {config['embed_dim']}d, {config['num_layers']}L, {config['num_heads']}H")


def main():
    """Main function."""
    analyze_model_size()
    compare_different_model_sizes()
    
    print("\n" + "=" * 70)
    print("Example Complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()

