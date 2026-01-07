# Custom LLM - GPT Foundation Components

A step-by-step implementation of foundational components for a GPT-based LLM model, focusing on tokenization, data loading, and attention mechanisms.

## Project Structure

```
CustomLLM/
├── src/
│   ├── __init__.py
│   ├── dataset.py             # Dataset and DataLoader for text files
│   └── attention.py           # MultiHeadAttention using PyTorch
├── data/                      # Directory for text data files
├── examples/
│   ├── __init__.py
│   ├── 01_dataloader_example.py
│   └── 02_attention_example.py
├── tests/
│   ├── __init__.py
│   ├── test_dataset.py
│   └── test_attention.py
├── requirements.txt
└── README.md
```

## Installation

1. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Components

### 1. Dataset (`src/dataset.py`)
PyTorch Dataset and DataLoader for loading and tokenizing text files with sliding window approach.

### 2. MultiHeadAttention (`src/attention.py`)
Wrapper around PyTorch's built-in `nn.MultiheadAttention` with causal masking for GPT-style autoregressive models.

## Usage Example

```python
import tiktoken
from src.dataset import create_dataloader
from src.attention import MultiHeadAttention
import torch

# Create dataloader from text file (uses tiktoken internally)
dataloader = create_dataloader("data/sample.txt", batch_size=4, max_length=256)

# Create attention layer
attn = MultiHeadAttention(embed_dim=768, num_heads=12, dropout=0.1)

# Process a batch
for input_batch, target_batch in dataloader:
    # Self-attention with causal masking
    output, _ = attn(input_batch, input_batch, input_batch)
    break
```

## Implementation Progress

- [x] Step 1: Project Structure Setup
- [x] Step 2: Tokenizer Usage (using tiktoken directly)
- [ ] Step 3: Dataset and DataLoader
- [ ] Step 4: MultiHeadAttention Implementation
- [ ] Step 5: Examples and Tests

## Dependencies

- `torch >= 2.2.2`: PyTorch for neural network implementation
- `tiktoken >= 0.5.1`: GPT-2 tokenizer
- `numpy >= 1.26.0`: Numerical operations

## License

This is an educational project for learning purposes.

