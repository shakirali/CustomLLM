# Custom LLM - GPT Implementation from Scratch

A complete, step-by-step implementation of a GPT (Generative Pre-trained Transformer) model from scratch, including all foundational components, the complete model, and training infrastructure.

## Overview

This project implements a GPT-based language model with:
- **Modular architecture**: Each component is implemented separately for educational purposes
- **Complete GPT model**: Full forward pass and text generation capabilities
- **Training ready**: All components needed for training and evaluation
- **Comprehensive tests**: Unit tests for all components
- **Educational examples**: Detailed examples explaining each component

## Project Structure

```
CustomLLM/
├── src/
│   ├── __init__.py
│   ├── dataset.py                    # Dataset and DataLoader for text files
│   ├── attention.py                  # MultiHeadAttention with causal masking
│   ├── model.py                      # Compatibility layer
│   └── model/
│       ├── __init__.py               # Model components exports
│       ├── layer_norm.py             # Layer normalization
│       ├── gelu.py                   # GELU activation function
│       ├── feedforward.py           # Feed-forward network
│       ├── transformer_block.py     # Transformer block (attention + FFN)
│       ├── transformer_stack.py     # Stack of transformer blocks
│       ├── token_embedding.py       # Token ID to embedding mapping
│       ├── position_embedding.py    # Position embedding
│       ├── output_projection.py     # Output to vocabulary logits
│       └── gpt_model.py             # Complete GPT model
├── scripts/                          # Training and evaluation scripts (Step 8)
│   ├── train.py                      # Training script (pending)
│   ├── eval.py                       # Evaluation script (pending)
│   └── generate.py                   # Text generation script (pending)
├── examples/                         # Educational examples
│   ├── 01_dataloader_example.py
│   ├── 02_attention_example.py
│   ├── 03_weight_learning_explanation.py
│   ├── 09_dropout_explanation.py
│   ├── 10_why_separate_layers.py
│   ├── 11_why_nn_module.py
│   ├── 12_automatic_differentiation.py
│   ├── 13_position_embedding_shape.py
│   ├── 14_position_embedding_example.py
│   ├── 15_transformer_stack_example.py
│   ├── 16_output_projection_example.py
│   └── 17_gpt_model_example.py      # Complete GPT model example
├── tests/                            # Unit tests
│   ├── test_dataset.py
│   ├── test_attention.py
│   ├── test_model_components.py
│   ├── test_embeddings.py
│   ├── test_transformer.py
│   └── test_output_projection.py
├── data/                             # Text data files
├── checkpoints/                      # Saved model checkpoints (for Step 8)
├── logs/                             # Training logs (for Step 8)
├── requirements.txt
├── PLAN.md                           # Detailed implementation plan
├── IMPLEMENTATION_STATUS.md          # Implementation status tracking
└── README.md
```

## Installation

1. **Clone the repository** (if applicable) or navigate to the project directory

2. **Create a virtual environment** (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

Dependencies:
- `torch >= 2.2.2`: PyTorch for neural network implementation
- `tiktoken >= 0.5.1`: GPT-2 tokenizer
- `numpy >= 1.26.0`: Numerical operations
- `pytest >= 7.0.0`: Testing framework

## Quick Start

### 1. Create a GPT Model

```python
from src.model import GPTModel
import torch

# Create a GPT-2 small model
model = GPTModel(
    vocab_size=50257,    # GPT-2 vocabulary size
    embed_dim=768,       # Embedding dimension
    num_layers=12,       # Number of transformer blocks
    num_heads=12,        # Number of attention heads
    max_seq_len=1024,    # Maximum sequence length
    dropout=0.1,
    tie_weights=True     # Standard: tie output projection with token embedding
)

print(f"Model parameters: {model.get_num_params():,}")
```

### 2. Forward Pass (Training/Inference)

```python
# Create sample input
token_ids = torch.randint(0, 50257, (2, 10))  # (batch_size=2, seq_len=10)

# Forward pass
model.eval()  # Set to evaluation mode
with torch.no_grad():
    logits = model(token_ids)  # (batch_size, seq_len, vocab_size)

# Get predicted tokens
predicted = torch.argmax(logits, dim=-1)
```

### 3. Text Generation

```python
# Create prompt
prompt = torch.randint(0, 50257, (1, 5))  # (batch_size=1, seq_len=5)

# Generate text
generated = model.generate(
    token_ids=prompt,
    max_new_tokens=50,
    temperature=0.8,    # Lower = more deterministic
    top_k=50,           # Top-k sampling (optional)
    top_p=0.9           # Nucleus sampling (optional)
)

print(f"Generated {generated.shape[1]} tokens")
```

### 4. Load Data

```python
from src.dataset import create_dataloader

# Create dataloader from text file
dataloader = create_dataloader(
    file_path="data/sample.txt",
    batch_size=4,
    max_length=256,
    stride=128,
    shuffle=True
)

# Iterate through batches
for input_batch, target_batch in dataloader:
    print(f"Input shape: {input_batch.shape}")   # (batch_size, seq_len)
    print(f"Target shape: {target_batch.shape}") # (batch_size, seq_len)
    break
```

## Components

### Core Components

1. **GPTModel** (`src/model/gpt_model.py`)
   - Complete GPT model combining all components
   - Forward pass for training/inference
   - Text generation with configurable sampling
   - Parameter counting utilities

2. **TokenEmbedding** (`src/model/token_embedding.py`)
   - Maps token IDs to dense embedding vectors
   - Shape: `(batch, seq_len) → (batch, seq_len, embed_dim)`

3. **PositionEmbedding** (`src/model/position_embedding.py`)
   - Adds positional information to token embeddings
   - Uses learned position embeddings
   - Shape: `(batch, seq_len) → (seq_len, embed_dim)` (broadcasts)

4. **TransformerStack** (`src/model/transformer_stack.py`)
   - Stack of multiple TransformerBlocks
   - Configurable number of layers
   - Sequential processing through all blocks

5. **TransformerBlock** (`src/model/transformer_block.py`)
   - Combines MultiHeadAttention and FeedForward
   - Pre-norm architecture (LayerNorm before each component)
   - Residual connections and dropout

6. **MultiHeadAttention** (`src/attention.py`)
   - Wrapper around PyTorch's `nn.MultiheadAttention`
   - Causal masking for autoregressive models
   - Prevents attention to future tokens

7. **FeedForward** (`src/model/feedforward.py`)
   - 2-layer MLP with 4× expansion
   - GELU activation function
   - Standard architecture for GPT models

8. **OutputProjection** (`src/model/output_projection.py`)
   - Maps transformer output to vocabulary logits
   - Supports weight tying with token embedding
   - Shape: `(batch, seq_len, embed_dim) → (batch, seq_len, vocab_size)`

### Supporting Components

- **LayerNorm** (`src/model/layer_norm.py`): Custom layer normalization
- **GELU** (`src/model/gelu.py`): Gaussian Error Linear Unit activation
- **GPTDataset** (`src/dataset.py`): PyTorch Dataset for text files
- **create_dataloader** (`src/dataset.py`): Utility to create DataLoaders

## Model Configurations

The model supports different configurations matching GPT-2 sizes:

| Configuration | vocab_size | embed_dim | num_layers | num_heads | Parameters |
|--------------|------------|-----------|------------|-----------|------------|
| Tiny (testing) | 1000 | 128 | 2 | 4 | ~0.66M |
| GPT-2 Small | 50257 | 768 | 12 | 12 | ~124M |
| GPT-2 Medium | 50257 | 1024 | 24 | 16 | ~355M |
| GPT-2 Large | 50257 | 1280 | 36 | 20 | ~774M |
| GPT-2 XL | 50257 | 1600 | 48 | 25 | ~1.5B |

## Examples

The `examples/` directory contains educational examples:

- **01_dataloader_example.py**: Data loading and tokenization
- **02_attention_example.py**: Multi-head attention with causal masking
- **03_weight_learning_explanation.py**: How weights are learned during training
- **09_dropout_explanation.py**: Dropout mechanism and its effects
- **10_why_separate_layers.py**: Why components are separated
- **11_why_nn_module.py**: Why components inherit from nn.Module
- **12_automatic_differentiation.py**: PyTorch autograd explanation
- **13_position_embedding_shape.py**: Position embedding broadcasting
- **14_position_embedding_example.py**: Position embedding usage
- **15_transformer_stack_example.py**: Transformer stack demonstration
- **16_output_projection_example.py**: Output projection and weight tying
- **17_gpt_model_example.py**: Complete GPT model usage

Run examples:
```bash
python examples/17_gpt_model_example.py
```

## Testing

All components have comprehensive unit tests. Run tests with:

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_gpt_model.py -v

# Run with coverage
pytest tests/ --cov=src -v
```

Test coverage:
- ✅ Dataset and DataLoader (12 tests)
- ✅ MultiHeadAttention (16 tests)
- ✅ Model components: LayerNorm, GELU, FeedForward (24 tests)
- ✅ Embeddings: TokenEmbedding, PositionEmbedding (20 tests)
- ✅ Transformer: TransformerBlock, TransformerStack (24 tests)
- ✅ OutputProjection (22 tests)

**Total: 118 tests, all passing** ✅

## Implementation Progress

### ✅ Completed Steps

- ✅ **Step 1**: Project Structure Setup
- ✅ **Step 2**: Tokenizer Usage (tiktoken)
- ✅ **Step 3**: Dataset and DataLoader
- ✅ **Step 4**: MultiHeadAttention Implementation
- ✅ **Step 5**: Examples and Tests
- ✅ **Step 6.1**: LayerNorm and GELU
- ✅ **Step 6.2**: FeedForward
- ✅ **Step 6.3**: TransformerBlock
- ✅ **Step 7.1**: Token Embeddings
- ✅ **Step 7.2**: Position Embeddings
- ✅ **Step 7.3**: Stack TransformerBlocks
- ✅ **Step 7.4**: Output Projection
- ✅ **Step 7.5**: Complete GPTModel

### ⏳ Pending Steps

- ⏳ **Step 8.1**: Training Script
- ⏳ **Step 8.2**: Evaluation Script
- ⏳ **Step 8.3**: Text Generation Script
- ⏳ **Step 8.4**: Configuration and Utilities

## Architecture

The GPT model follows this architecture:

```
Input: token_ids (batch, seq_len)
    ↓
TokenEmbedding → (batch, seq_len, embed_dim)
    ↓
PositionEmbedding → (seq_len, embed_dim) [broadcasts]
    ↓
Combined: token_emb + pos_emb → (batch, seq_len, embed_dim)
    ↓
TransformerStack (N layers):
    ├─ TransformerBlock 0
    ├─ TransformerBlock 1
    ├─ ...
    └─ TransformerBlock N-1
    ↓
OutputProjection → (batch, seq_len, vocab_size)
    ↓
Output: logits
```

Each TransformerBlock contains:
- Pre-norm LayerNorm
- MultiHeadAttention (with causal masking)
- Residual connection + Dropout
- Pre-norm LayerNorm
- FeedForward (4× expansion)
- Residual connection + Dropout

## Key Features

- **Modular Design**: Each component is separate and testable
- **Educational Focus**: Comprehensive examples and documentation
- **Production Ready**: Follows PyTorch best practices
- **Complete Implementation**: All components needed for a GPT model
- **Weight Tying**: Standard GPT technique to reduce parameters
- **Causal Masking**: Proper autoregressive attention
- **Pre-norm Architecture**: Modern transformer architecture

## Next Steps

To complete the project:

1. **Implement Training Script** (Step 8.1)
   - Training loop with optimizer
   - Loss computation (CrossEntropyLoss)
   - Checkpointing
   - Logging

2. **Implement Evaluation Script** (Step 8.2)
   - Validation/test evaluation
   - Metrics: loss, perplexity
   - Batch evaluation

3. **Implement Generation Script** (Step 8.3)
   - Load trained model
   - Generate text from prompts
   - Configurable sampling parameters

4. **Add Configuration Management** (Step 8.4)
   - Config files for hyperparameters
   - Checkpoint utilities
   - Logging setup

## Documentation

- **PLAN.md**: Detailed implementation plan with all steps
- **IMPLEMENTATION_STATUS.md**: Current implementation status
- **ATTENTION_FEEDFORWARD_EXPLANATION.md**: Explanation of attention and feedforward

## License

This is an educational project for learning purposes.

## Contributing

This is a learning project. Feel free to use it as a reference for understanding GPT architecture and implementation.

## Acknowledgments

This implementation is based on the GPT-2 architecture and follows PyTorch best practices. It's designed for educational purposes to understand how transformer-based language models work.
