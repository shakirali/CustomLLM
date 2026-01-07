"""
Dataset module for GPT model.

This module provides a PyTorch Dataset class for loading and tokenizing text files,
and a function to create DataLoaders with batching.
"""

import tiktoken
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Union, List
import os


class GPTDataset(Dataset):
    """
    A PyTorch Dataset for GPT-style language modeling.
    
    This dataset loads text from a file, tokenizes it using tiktoken's GPT-2 tokenizer,
    and creates overlapping sequences using a sliding window approach for training.
    
    Attributes:
        input_ids: List of input token sequences
        target_ids: List of target token sequences (shifted by 1 position)
    """
    
    def __init__(self, txt: str, tokenizer, max_length: int, stride: int):
        """
        Initialize the dataset.
        
        Args:
            txt: The input text as a string
            tokenizer: The tiktoken tokenizer (from tiktoken.get_encoding("gpt2"))
            max_length: Maximum sequence length
            stride: Step size for sliding window (overlap = max_length - stride)
        """
        self.input_ids = []
        self.target_ids = []
        
        # Tokenize the entire text
        token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})
        
        # Use a sliding window to chunk the text into overlapping sequences
        # This creates multiple training examples from a single text
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1: i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk, dtype=torch.long))
            self.target_ids.append(torch.tensor(target_chunk, dtype=torch.long))
    
    def __len__(self):
        """Return the number of sequences in the dataset."""
        return len(self.input_ids)
    
    def __getitem__(self, idx):
        """
        Get a single sequence pair.
        
        Args:
            idx: Index of the sequence to retrieve
        
        Returns:
            Tuple of (input_ids, target_ids) tensors
        """
        return self.input_ids[idx], self.target_ids[idx]


def load_text_from_file(file_path: str) -> str:
    """
    Load text from a file.
    
    Args:
        file_path: Path to the text file
    
    Returns:
        The text content as a string
    
    Raises:
        FileNotFoundError: If the file doesn't exist
        IOError: If there's an error reading the file
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    if not text.strip():
        raise ValueError(f"File is empty: {file_path}")
    
    return text


def create_dataloader(
    file_path: Union[str, List[str]],
    batch_size: int = 4,
    max_length: int = 256,
    stride: int = 128,
    shuffle: bool = True,
    drop_last: bool = True,
    num_workers: int = 0
) -> DataLoader:
    """
    Create a DataLoader from text file(s).
    
    This function loads text from file(s), tokenizes it, creates sequences using
    a sliding window approach, and returns a PyTorch DataLoader.
    
    Args:
        file_path: Path to a single text file, or list of file paths
        batch_size: Number of sequences per batch
        max_length: Maximum sequence length
        stride: Step size for sliding window (smaller stride = more overlap)
        shuffle: Whether to shuffle the data
        drop_last: Whether to drop the last incomplete batch
        num_workers: Number of worker processes for data loading
    
    Returns:
        A PyTorch DataLoader instance
    
    Example:
        >>> dataloader = create_dataloader("data/sample.txt", batch_size=4, max_length=256)
        >>> for input_batch, target_batch in dataloader:
        ...     print(input_batch.shape)  # (batch_size, max_length)
        ...     break
    """
    # Initialize the tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")
    
    # Load text from file(s)
    if isinstance(file_path, str):
        # Single file
        text = load_text_from_file(file_path)
    elif isinstance(file_path, list):
        # Multiple files - concatenate them
        texts = []
        for path in file_path:
            texts.append(load_text_from_file(path))
        text = "\n".join(texts)
    else:
        raise TypeError(f"file_path must be str or List[str], got {type(file_path)}")
    
    # Create dataset
    dataset = GPTDataset(txt=text, tokenizer=tokenizer, max_length=max_length, stride=stride)
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers
    )
    
    return dataloader

