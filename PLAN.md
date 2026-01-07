# GPT Foundation Components Implementation Plan

This plan implements the foundational building blocks for a GPT-based LLM model, focusing on the essential components needed before building the complete model.

## Architecture Overview

The implementation follows this progression:

1. **Project Structure**: Set up organized directory structure
2. **Tokenizer**: Use tiktoken directly (no wrapper class needed)
3. **DataLoader**: Create data loading pipeline that tokenizes text files
4. **MultiHeadAttention**: Use PyTorch's built-in `nn.MultiheadAttention` with causal masking wrapper

## Project Structure

```
CustomLLM/
├── src/
│   ├── __init__.py
│   ├── dataset.py             # Dataset and DataLoader for text files
│   ├── attention.py           # MultiHeadAttention implementation
│   └── model.py               # Model components (LayerNorm, GELU, FeedForward, TransformerBlock)
├── scripts/                   # Training and evaluation scripts
│   ├── train.py               # Training script
│   ├── eval.py                # Evaluation script
│   ├── generate.py            # Text generation script
│   └── config.py             # Configuration utilities (optional)
├── configs/                   # Configuration files (optional)
│   └── gpt2_small.yaml        # Model and training hyperparameters
├── data/                      # Directory for text data files
├── checkpoints/               # Saved model checkpoints
├── logs/                      # Training logs
├── examples/
│   ├── __init__.py
│   ├── 01_dataloader_example.py
│   └── 02_attention_example.py
├── tests/
│   ├── __init__.py
│   ├── test_dataset.py
│   └── test_attention.py
├── requirements.txt
├── README.md
└── .gitignore
```

## Implementation Steps

### Step 1: Project Structure Setup ✅

- [x] Create directory structure (`src/`, `data/`, `examples/`, `tests/`)
- [x] Set up `requirements.txt` with dependencies:
  - `torch >= 2.2.2` - PyTorch for neural networks
  - `tiktoken >= 0.5.1` - GPT-2 tokenizer
  - `numpy >= 1.26.0` - Numerical operations
- [x] Create `README.md` with project overview
- [x] Add `.gitignore` for Python projects

**Status**: ✅ Completed

### Step 2: Tokenizer Usage ✅

- [x] Use tiktoken directly (no wrapper class needed)
- [x] Use `tiktoken.get_encoding("gpt2")` to get the GPT-2 tokenizer
- [x] Tokenizer used directly in the dataset implementation:
  - `tokenizer.encode(text, allowed_special={"<|endoftext|>"})` for encoding
  - `tokenizer.decode(token_ids)` for decoding

**Status**: ✅ Completed

### Step 3: Dataset and DataLoader ✅

**File: `src/dataset.py`**

- [x] Create `GPTDataset` class that:
  - Loads text from file(s)
  - Uses tiktoken directly to tokenize text (`tiktoken.get_encoding("gpt2")`)
  - Implements sliding window approach for creating sequences
  - Handles configurable sequence length and stride
- [x] Implement `create_dataloader` function that:
  - Takes text file path(s) as input
  - Creates dataset instance
  - Returns PyTorch DataLoader with batching
  - Supports configurable batch size, shuffle, etc.
- [x] Handle edge cases (empty files, short sequences)
- [x] Add proper documentation
- [x] Create example script (`examples/01_dataloader_example.py`)

**Status**: ✅ Completed

### Step 4: MultiHeadAttention Implementation ⏳

**File: `src/attention.py`**

#### Step 4.1: Basic Wrapper Class ✅
- [x] Create `MultiHeadAttention` wrapper class using PyTorch's built-in `nn.MultiheadAttention`:
  - [x] Use `torch.nn.MultiheadAttention` as the base attention mechanism
  - [x] Initialize with configurable parameters:
    - `embed_dim`: Embedding dimension (must be divisible by num_heads)
    - `num_heads`: Number of attention heads
    - `dropout`: Dropout rate
    - `bias`: Whether to use bias in projections
    - `batch_first`: Set to True for (batch, seq, embed) format
  - [x] Basic forward pass (without causal masking yet)
  - [x] Add proper documentation

#### Step 4.2: Add Causal Masking ✅
- [x] Add causal masking functionality:
  - [x] Create upper triangular mask for causal attention
  - [x] Register mask as buffer
  - [x] Add `_get_causal_mask` helper method for variable sequence lengths
  - [x] Apply mask using `attn_mask` parameter in forward pass
  - [x] Handle variable sequence lengths dynamically
  - [x] Update documentation

**Status**: Step 4.1 ✅ Completed, Step 4.2 ✅ Completed

### Step 5: Examples and Tests ✅

- [x] Create example script for attention (`examples/02_attention_example.py`)
- [x] Add basic unit tests for dataset (`tests/test_dataset.py`)
- [x] Add basic unit tests for attention (`tests/test_attention.py`)
- [x] Add unit tests for model components (`tests/test_model_components.py`)
- [x] Add unit tests for embeddings (`tests/test_embeddings.py`)
- [x] Add unit tests for transformer (`tests/test_transformer.py`)
- [x] Add unit tests for output projection (`tests/test_output_projection.py`)

**Status**: ✅ Completed

### Step 6: Model Components ⏳

**File: `src/model.py`**

#### Step 6.1: LayerNorm and GELU ✅
- [x] Implement `LayerNorm` class:
  - [x] Normalize across embedding dimension
  - [x] Learnable scale and shift parameters
  - [x] Epsilon for numerical stability
- [x] Implement `GELU` class:
  - [x] Gaussian Error Linear Unit activation
  - [x] Smooth non-linear activation function
  - [x] Device and dtype handling

#### Step 6.2: FeedForward ✅
- [x] Implement `FeedForward` class:
  - [x] 2-layer MLP (expand to 4×embed_dim, then back)
  - [x] Use GELU activation
  - [x] Configurable embedding dimension
  - [x] Uses nn.Sequential for clean implementation

#### Step 6.3: TransformerBlock ✅
- [x] Implement `TransformerBlock` class:
  - [x] Combine MultiHeadAttention + FeedForward
  - [x] LayerNorm before each component (Pre-Norm architecture)
  - [x] Residual connections (skip connections)
  - [x] Dropout for regularization
  - [x] Proper documentation and type hints

**Status**: Step 6.1 ✅ Completed, Step 6.2 ✅ Completed, Step 6.3 ✅ Completed

### Step 7: Complete GPT Model ⏳

**File: `src/model.py` (and `src/gpt.py` for full model)**

#### Step 7.1: Token Embeddings ✅
- [x] Implement `TokenEmbedding` class:
  - [x] Use `nn.Embedding` for token-to-vector mapping
  - [x] Configurable vocabulary size and embedding dimension
  - [x] Forward pass: (batch, seq_len) → (batch, seq_len, embed_dim)
  - [x] Proper documentation and type hints

#### Step 7.2: Position Embeddings ✅
- [x] Implement position embedding functionality
- [x] Add positional information to token embeddings
- [x] Support learned position embeddings (using nn.Embedding)

#### Step 7.3: Stack TransformerBlocks ⏳
- [ ] Create mechanism to stack multiple TransformerBlocks
- [ ] Configurable number of layers
- [ ] Sequential processing through blocks

#### Step 7.4: Output Projection ✅
- [x] Implement output projection to vocabulary size
- [x] Convert transformer output to logits
- [x] Shape: (batch, seq_len, embed_dim) → (batch, seq_len, vocab_size)
- [x] Support weight tying with token embedding

#### Step 7.5: Complete GPTModel ✅
- [x] Combine all components into GPTModel class
- [x] Full forward pass implementation
- [x] Text generation function
- [x] Parameter counting utility
- [x] Support for training and evaluation modes
- [x] Add GPTModel to exports
- [x] Create example script (`examples/17_gpt_model_example.py`)

**Status**: Step 7.1 ✅ Completed, Step 7.2 ✅ Completed, Step 7.3 ✅ Completed, Step 7.4 ✅ Completed, Step 7.5 ✅ Completed

### Step 8: Training and Evaluation Scripts ⏳

**Files: `scripts/train.py`, `scripts/eval.py`, `scripts/generate.py`**

#### Step 8.1: Training Script ⏳
- [ ] Create `scripts/train.py` with training loop:
  - [ ] Load training data using `create_dataloader`
  - [ ] Initialize GPTModel with configurable hyperparameters
  - [ ] Set up optimizer (Adam/AdamW with learning rate)
  - [ ] Set up loss function (CrossEntropyLoss for next-token prediction)
  - [ ] Implement training loop:
    - [ ] Set model to training mode (`model.train()`)
    - [ ] Forward pass through model
    - [ ] Reshape logits and targets for loss computation
    - [ ] Compute loss
    - [ ] Backward pass (`loss.backward()`)
    - [ ] Gradient clipping (optional, for stability)
    - [ ] Optimizer step (`optimizer.step()`)
  - [ ] Add logging (loss, learning rate, step count)
  - [ ] Implement checkpointing (save model periodically)
  - [ ] Add validation evaluation during training
  - [ ] Support for resuming from checkpoint
  - [ ] Command-line arguments for hyperparameters
  - [ ] Proper error handling and documentation

#### Step 8.2: Evaluation Script ⏳
- [ ] Create `scripts/eval.py` for model evaluation:
  - [ ] Load trained model from checkpoint
  - [ ] Load validation/test data using `create_dataloader`
  - [ ] Set model to evaluation mode (`model.eval()`)
  - [ ] Implement evaluation loop:
    - [ ] Use `torch.no_grad()` for efficiency
    - [ ] Forward pass through model
    - [ ] Compute loss (same as training)
    - [ ] Accumulate metrics (loss, token count)
  - [ ] Compute final metrics:
    - [ ] Average loss
    - [ ] Perplexity (exp(loss))
    - [ ] Token-level accuracy (optional)
  - [ ] Print evaluation results
  - [ ] Support for multiple evaluation datasets
  - [ ] Command-line arguments for model path and data
  - [ ] Proper error handling and documentation

#### Step 8.3: Text Generation Script ⏳
- [ ] Create `scripts/generate.py` for text generation:
  - [ ] Load trained model from checkpoint
  - [ ] Initialize tokenizer (`tiktoken.get_encoding("gpt2")`)
  - [ ] Accept prompt text as input (command-line or file)
  - [ ] Tokenize prompt
  - [ ] Call `model.generate()` method:
    - [ ] Set model to eval mode
    - [ ] Use `torch.no_grad()` for efficiency
    - [ ] Support generation parameters:
      - [ ] `max_new_tokens`: Maximum tokens to generate
      - [ ] `temperature`: Sampling temperature (default: 1.0)
      - [ ] `top_k`: Top-k sampling (optional)
      - [ ] `top_p`: Nucleus sampling (optional)
      - [ ] `stop_token_id`: Stop generation token (optional)
  - [ ] Decode generated tokens to text
  - [ ] Print or save generated text
  - [ ] Support for batch generation (multiple prompts)
  - [ ] Command-line arguments for all parameters
  - [ ] Proper error handling and documentation

#### Step 8.4: Configuration and Utilities ⏳
- [ ] Create `scripts/config.py` or `configs/` directory:
  - [ ] Model configuration (vocab_size, embed_dim, num_layers, etc.)
  - [ ] Training hyperparameters (learning rate, batch size, epochs, etc.)
  - [ ] Paths for data, checkpoints, logs
  - [ ] Support for YAML/JSON config files (optional)
- [ ] Create utility functions:
  - [ ] `save_checkpoint()`: Save model, optimizer, epoch, loss
  - [ ] `load_checkpoint()`: Load model and resume training
  - [ ] `setup_logging()`: Configure logging to file and console
  - [ ] `count_parameters()`: Count trainable parameters in model

**Status**: Step 8.1 ⏳ Pending, Step 8.2 ⏳ Pending, Step 8.3 ⏳ Pending, Step 8.4 ⏳ Pending

## Key Components Reference

Based on the example code structure:

1. **Tokenizer**: Use tiktoken directly (`tiktoken.get_encoding("gpt2")`)
2. **GPTDataset**: PyTorch Dataset that tokenizes text files and creates sequences
3. **MultiHeadAttention**: Wrapper around PyTorch's built-in `nn.MultiheadAttention` with causal masking for GPT-style autoregressive models

## Dependencies

- `torch >= 2.2.2`: PyTorch for neural network implementation
- `tiktoken >= 0.5.1`: GPT-2 tokenizer
- `numpy >= 1.26.0`: Numerical operations

## Example Usage Flow

1. Initialize tokenizer: `tokenizer = tiktoken.get_encoding("gpt2")`
2. Load and tokenize data: `dataloader = create_dataloader("data/text.txt")` (tokenizer created internally)
3. Create attention layer: `attn = MultiHeadAttention(embed_dim=768, num_heads=12, dropout=0.1)`
4. Process batches through attention: `output, _ = attn(input_tensor, input_tensor, input_tensor)` (self-attention with causal mask)

## Progress Summary

- ✅ Step 1: Project Structure Setup
- ✅ Step 2: Tokenizer Usage
- ✅ Step 3: Dataset and DataLoader
- ✅ Step 4: MultiHeadAttention Implementation
- ✅ Step 5: Examples and Tests
- ✅ Step 6.1: LayerNorm and GELU
- ✅ Step 6.2: FeedForward
- ✅ Step 6.3: TransformerBlock
- ✅ Step 7.1: Token Embeddings
- ✅ Step 7.2: Position Embeddings
- ✅ Step 7.3: Stack TransformerBlocks
- ✅ Step 7.4: Output Projection
- ✅ Step 7.5: Complete GPTModel
- ⏳ Step 8.1: Training Script
- ⏳ Step 8.2: Evaluation Script
- ⏳ Step 8.3: Text Generation Script
- ⏳ Step 8.4: Configuration and Utilities

## Notes

- The tokenizer wrapper class was removed in favor of using tiktoken directly
- The dataset example uses `the-verdict.txt` file for demonstration
- All components follow PyTorch best practices and include comprehensive documentation

