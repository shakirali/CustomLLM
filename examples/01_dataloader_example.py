"""
Example: Using the Dataset and DataLoader

This example demonstrates how to:
1. Load text from a file and create a DataLoader
2. Iterate through batches
3. Understand the sliding window approach
4. Work with input and target sequences
"""

import sys
import os

# Add parent directory to path to import src modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.dataset import create_dataloader, load_text_from_file, GPTDataset
import tiktoken


def example_basic_usage():
    """Basic example: Create a DataLoader from a text file."""
    print("=" * 60)
    print("Example 1: Basic DataLoader Usage")
    print("=" * 60)
    
    # Create a DataLoader from a text file
    # Using the-verdict.txt file in the project root
    file_path = "the-verdict.txt"
    
    if not os.path.exists(file_path):
        print(f"Warning: {file_path} not found. Please ensure the file exists.")
        return
    
    dataloader = create_dataloader(
        file_path=file_path,
        batch_size=4,
        max_length=256,
        stride=128,
        shuffle=True,
        drop_last=True
    )
    
    print(f"DataLoader created with {len(dataloader)} batches")
    print(f"Batch size: {dataloader.batch_size}")
    print()
    
    # Iterate through a few batches
    print("First 3 batches:")
    for i, (input_batch, target_batch) in enumerate(dataloader):
        if i >= 3:
            break
        
        print(f"\nBatch {i + 1}:")
        print(f"  Input shape: {input_batch.shape}")  # (batch_size, max_length)
        print(f"  Target shape: {target_batch.shape}")  # (batch_size, max_length)
        print(f"  Input dtype: {input_batch.dtype}")
        print(f"  Target dtype: {target_batch.dtype}")
        
        # Show first sequence in the batch
        print(f"  First sequence input (first 20 tokens): {input_batch[0, :20].tolist()}")
        print(f"  First sequence target (first 20 tokens): {target_batch[0, :20].tolist()}")
        
        # Decode and show first sequence text
        tokenizer = tiktoken.get_encoding("gpt2")
        decoded_input = tokenizer.decode(input_batch[0].tolist())
        decoded_target = tokenizer.decode(target_batch[0].tolist())
        print(f"  Decoded input (first 100 chars): {decoded_input[:100]}...")
        print(f"  Decoded target (first 100 chars): {decoded_target[:100]}...")


def example_sliding_window():
    """Example: Understanding the sliding window approach."""
    print("\n" + "=" * 60)
    print("Example 2: Understanding Sliding Window")
    print("=" * 60)
    
    # Load text from the-verdict.txt
    file_path = "the-verdict.txt"
    
    if not os.path.exists(file_path):
        print(f"Warning: {file_path} not found. Please ensure the file exists.")
        return
    
    # Load the text
    sample_text = load_text_from_file(file_path)
    
    # Initialize tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")
    
    # Create dataset with small max_length and stride to see overlap
    dataset = GPTDataset(
        txt=sample_text,
        tokenizer=tokenizer,
        max_length=20,
        stride=10  # 50% overlap
    )
    
    print(f"File: {file_path}")
    print(f"Text length: {len(sample_text)} characters")
    print(f"Number of sequences created: {len(dataset)}")
    print("Max length: 20, Stride: 10 (50% overlap)")
    print()
    
    # Show first few sequences
    print("First 3 sequences (showing overlap):")
    for i in range(min(3, len(dataset))):
        input_seq, target_seq = dataset[i]
        print(f"\nSequence {i + 1}:")
        print(f"  Input tokens (first 10): {input_seq[:10].tolist()}")
        print(f"  Target tokens (first 10): {target_seq[:10].tolist()}")
        decoded_input = tokenizer.decode(input_seq.tolist())
        decoded_target = tokenizer.decode(target_seq.tolist())
        print(f"  Decoded input: {decoded_input[:80]}...")
        print(f"  Decoded target: {decoded_target[:80]}...")


def example_multiple_files():
    """Example: Loading from multiple files."""
    print("\n" + "=" * 60)
    print("Example 3: Loading from Multiple Files")
    print("=" * 60)
    
    # Check if the-verdict.txt exists
    file_path = "the-verdict.txt"
    
    if not os.path.exists(file_path):
        print(f"Warning: {file_path} not found. Skipping this example.")
        return
    
    # Create DataLoader from multiple files (using same file twice as example)
    file_paths = [file_path]  # In practice, you'd have different files
    
    print(f"Loading from {len(file_paths)} file(s)")
    
    dataloader = create_dataloader(
        file_path=file_paths,
        batch_size=2,
        max_length=128,
        stride=64
    )
    
    print(f"DataLoader created with {len(dataloader)} batches")
    
    # Get one batch
    input_batch, target_batch = next(iter(dataloader))
    print(f"Batch shape: {input_batch.shape}")


def example_target_shift():
    """Example: Understanding target sequence shift."""
    print("\n" + "=" * 60)
    print("Example 4: Understanding Target Sequence Shift")
    print("=" * 60)
    
    # Load text from the-verdict.txt
    file_path = "the-verdict.txt"
    
    if not os.path.exists(file_path):
        print(f"Warning: {file_path} not found. Please ensure the file exists.")
        return
    
    # Load the text
    sample_text = load_text_from_file(file_path)
    
    # Get first sentence for demonstration
    first_sentence = sample_text.split('.')[0] + '.'
    
    tokenizer = tiktoken.get_encoding("gpt2")
    token_ids = tokenizer.encode(first_sentence, allowed_special={"<|endoftext|>"})
    
    print(f"Original text (first sentence): {first_sentence}")
    print(f"Token IDs: {token_ids[:15]}... (showing first 15)")
    print()
    
    # Create dataset
    dataset = GPTDataset(
        txt=first_sentence,
        tokenizer=tokenizer,
        max_length=10,
        stride=10
    )
    
    if len(dataset) > 0:
        input_seq, target_seq = dataset[0]
        print(f"Input sequence:  {input_seq.tolist()}")
        print(f"Target sequence: {target_seq.tolist()}")
        print()
        print("Notice: Target is shifted by 1 position (next token prediction)")
        print(f"  Input[0] = {input_seq[0].item()}, Target[0] = {target_seq[0].item()}")
        print(f"  Input[1] = {input_seq[1].item()}, Target[1] = {target_seq[1].item()}")
        print(f"  Input[2] = {input_seq[2].item()}, Target[2] = {target_seq[2].item()}")
        print()
        print("Decoded sequences:")
        print(f"  Input:  {tokenizer.decode(input_seq.tolist())}")
        print(f"  Target: {tokenizer.decode(target_seq.tolist())}")


def main():
    """Run all examples."""
    print("\n" + "=" * 60)
    print("Dataset and DataLoader Examples")
    print("=" * 60)
    print()
    
    try:
        example_basic_usage()
        example_sliding_window()
        example_multiple_files()
        example_target_shift()
        
        print("\n" + "=" * 60)
        print("All examples completed!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\nError running examples: {e}")
        print("Make sure you have installed the required dependencies:")
        print("  pip install -r requirements.txt")


if __name__ == "__main__":
    main()

