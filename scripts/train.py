"""
Training script for GPT model.

This script implements the complete training loop for the GPT model, including:
- Data loading
- Model initialization
- Training loop with forward/backward passes
- Checkpointing
- Validation
- Logging
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Add parent directory to path to allow importing src modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model import GPTModel
from src.dataset import create_dataloader


def setup_logging(log_dir: str, log_level: str = "INFO"):
    """
    Set up logging configuration.
    
    Args:
        log_dir: Directory to save log files
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
    """
    # Create log directory if it doesn't exist
    os.makedirs(log_dir, exist_ok=True)
    
    # Configure logging
    log_file = os.path.join(log_dir, "training.log")
    
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return logging.getLogger(__name__)


def parse_args():
    """
    Parse command-line arguments.
    
    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="Train GPT model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Data arguments
    parser.add_argument(
        "--train_data",
        type=str,
        required=True,
        help="Path to training data file(s). Can be a single file or comma-separated list."
    )
    parser.add_argument(
        "--val_data",
        type=str,
        default=None,
        help="Path to validation data file(s). Optional."
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size for training"
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=256,
        help="Maximum sequence length"
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=128,
        help="Stride for sliding window (overlap = max_length - stride)"
    )
    
    # Model arguments
    parser.add_argument(
        "--vocab_size",
        type=int,
        default=50257,
        help="Vocabulary size (GPT-2 default: 50257)"
    )
    parser.add_argument(
        "--embed_dim",
        type=int,
        default=768,
        help="Embedding dimension (GPT-2 small: 768)"
    )
    parser.add_argument(
        "--num_layers",
        type=int,
        default=12,
        help="Number of transformer layers (GPT-2 small: 12)"
    )
    parser.add_argument(
        "--num_heads",
        type=int,
        default=12,
        help="Number of attention heads (GPT-2 small: 12)"
    )
    parser.add_argument(
        "--max_seq_len",
        type=int,
        default=1024,
        help="Maximum sequence length (GPT-2: 1024)"
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.1,
        help="Dropout probability"
    )
    parser.add_argument(
        "--tie_weights",
        action="store_true",
        default=True,
        help="Tie output projection weights with token embedding (default: True)"
    )
    parser.add_argument(
        "--no_tie_weights",
        dest="tie_weights",
        action="store_false",
        help="Don't tie output projection weights"
    )
    
    # Training arguments
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Learning rate"
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.01,
        help="Weight decay for optimizer"
    )
    parser.add_argument(
        "--grad_clip",
        type=float,
        default=1.0,
        help="Gradient clipping value (0.0 to disable)"
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=0,
        help="Number of warmup steps for learning rate scheduler"
    )
    
    # Checkpointing arguments
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="checkpoints",
        help="Directory to save checkpoints"
    )
    parser.add_argument(
        "--save_every",
        type=int,
        default=1000,
        help="Save checkpoint every N steps"
    )
    parser.add_argument(
        "--resume_from",
        type=str,
        default=None,
        help="Path to checkpoint to resume training from"
    )
    
    # Logging arguments
    parser.add_argument(
        "--log_dir",
        type=str,
        default="logs",
        help="Directory to save logs"
    )
    parser.add_argument(
        "--log_every",
        type=int,
        default=100,
        help="Log training metrics every N steps"
    )
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )
    
    # Validation arguments
    parser.add_argument(
        "--val_every",
        type=int,
        default=500,
        help="Run validation every N steps"
    )
    
    # Device arguments
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device to use (auto = cuda if available, else cpu)"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="Number of worker processes for data loading"
    )
    
    return parser.parse_args()


def create_model(args, logger):
    """
    Create and initialize GPT model.
    
    Args:
        args: Parsed command-line arguments
        logger: Logger instance
    
    Returns:
        GPTModel: Initialized model
    """
    logger.info("Creating GPT model...")
    logger.info(f"  vocab_size: {args.vocab_size}")
    logger.info(f"  embed_dim: {args.embed_dim}")
    logger.info(f"  num_layers: {args.num_layers}")
    logger.info(f"  num_heads: {args.num_heads}")
    logger.info(f"  max_seq_len: {args.max_seq_len}")
    logger.info(f"  dropout: {args.dropout}")
    logger.info(f"  tie_weights: {args.tie_weights}")
    
    model = GPTModel(
        vocab_size=args.vocab_size,
        embed_dim=args.embed_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        max_seq_len=args.max_seq_len,
        dropout=args.dropout,
        tie_weights=args.tie_weights
    )
    
    num_params = model.get_num_params(trainable_only=True)
    logger.info(f"Model created with {num_params:,} trainable parameters")
    
    return model


def create_dataloaders(args, logger):
    """
    Create training and validation dataloaders.
    
    Args:
        args: Parsed command-line arguments
        logger: Logger instance
    
    Returns:
        tuple: (train_dataloader, val_dataloader)
    """
    logger.info("Creating dataloaders...")
    
    # Parse training data paths
    train_files = [f.strip() for f in args.train_data.split(",")]
    logger.info(f"Training data files: {train_files}")
    
    # Create training dataloader
    train_dataloader = create_dataloader(
        file_path=train_files,
        batch_size=args.batch_size,
        max_length=args.max_length,
        stride=args.stride,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers
    )
    logger.info(f"Training dataloader created: {len(train_dataloader)} batches")
    
    # Create validation dataloader if provided
    val_dataloader = None
    if args.val_data:
        val_files = [f.strip() for f in args.val_data.split(",")]
        logger.info(f"Validation data files: {val_files}")
        
        val_dataloader = create_dataloader(
            file_path=val_files,
            batch_size=args.batch_size,
            max_length=args.max_length,
            stride=args.stride,
            shuffle=False,
            drop_last=False,
            num_workers=args.num_workers
        )
        logger.info(f"Validation dataloader created: {len(val_dataloader)} batches")
    
    return train_dataloader, val_dataloader


def setup_optimizer(model, args, logger):
    """
    Set up optimizer and learning rate scheduler.
    
    Args:
        model: GPT model
        args: Parsed command-line arguments
        logger: Logger instance
    
    Returns:
        tuple: (optimizer, scheduler)
    """
    logger.info("Setting up optimizer...")
    logger.info(f"  Learning rate: {args.learning_rate}")
    logger.info(f"  Weight decay: {args.weight_decay}")
    
    # Use AdamW optimizer (better for transformers)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    
    # Learning rate scheduler (optional, to be implemented)
    scheduler = None
    if args.warmup_steps > 0:
        logger.info(f"  Warmup steps: {args.warmup_steps}")
        # TODO: Implement learning rate scheduler with warmup
    
    return optimizer, scheduler


def save_checkpoint(model, optimizer, epoch, step, loss, checkpoint_dir, logger):
    """
    Save model checkpoint.
    
    Args:
        model: GPT model
        optimizer: Optimizer
        epoch: Current epoch
        step: Current step
        loss: Current loss
        checkpoint_dir: Directory to save checkpoint
        logger: Logger instance
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    checkpoint = {
        "epoch": epoch,
        "step": step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
        "model_config": {
            "vocab_size": model.vocab_size,
            "embed_dim": model.embed_dim,
            "num_layers": model.num_layers,
            "num_heads": model.num_heads,
            "max_seq_len": model.max_seq_len,
            "dropout": model.dropout,
            "tie_weights": model.tie_weights
        }
    }
    
    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch}_step_{step}.pt")
    torch.save(checkpoint, checkpoint_path)
    logger.info(f"Checkpoint saved: {checkpoint_path}")


def load_checkpoint(checkpoint_path, model, optimizer, logger):
    """
    Load model checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file
        model: GPT model
        optimizer: Optimizer
        logger: Logger instance
    
    Returns:
        tuple: (start_epoch, start_step)
    """
    logger.info(f"Loading checkpoint from: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    
    start_epoch = checkpoint.get("epoch", 0)
    start_step = checkpoint.get("step", 0)
    
    logger.info(f"Checkpoint loaded: epoch {start_epoch}, step {start_step}")
    
    return start_epoch, start_step


def train_step(model, batch, criterion, optimizer, device, grad_clip, logger):
    """
    Perform a single training step.
    
    Args:
        model: GPT model
        batch: Training batch (input_ids, target_ids)
        criterion: Loss function
        optimizer: Optimizer
        device: Device to run on
        grad_clip: Gradient clipping value
        logger: Logger instance
    
    Returns:
        float: Loss value
    """
    # 1. Move batch to device
    input_ids, target_ids = batch
    input_ids = input_ids.to(device)
    target_ids = target_ids.to(device)
    
    # 2. Forward pass
    logits = model(input_ids)  # Shape: (batch_size, seq_len, vocab_size)
    
    # 3. Reshape for loss computation
    # CrossEntropyLoss expects: (N, C) for logits, (N,) for targets
    # where N = batch_size * seq_len, C = vocab_size
    vocab_size = logits.shape[-1]
    logits_flat = logits.view(-1, vocab_size)  # (batch_size * seq_len, vocab_size)
    targets_flat = target_ids.view(-1)  # (batch_size * seq_len,)
    
    # 4. Compute loss
    loss = criterion(logits_flat, targets_flat)
    
    # 5. Backward pass
    optimizer.zero_grad()  # Clear previous gradients
    loss.backward()  # Compute gradients
    
    # 6. Gradient clipping (optional, for stability)
    if grad_clip > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    
    # 7. Optimizer step
    optimizer.step()  # Update model weights
    
    # 8. Return loss value (as scalar)
    return loss.item()


def validate(model, val_dataloader, criterion, device, logger):
    """
    Run validation on validation set.
    
    Args:
        model: GPT model
        val_dataloader: Validation dataloader
        criterion: Loss function
        device: Device to run on
        logger: Logger instance
    
    Returns:
        dict: Validation metrics (loss, perplexity, etc.)
    """
    # TODO: Implement validation
    # 1. Set model to eval mode
    # 2. Iterate through validation batches
    # 3. Compute loss
    # 4. Calculate metrics (loss, perplexity)
    
    pass


def train(args):
    """
    Main training function.
    
    Args:
        args: Parsed command-line arguments
    """
    # Set up logging
    logger = setup_logging(args.log_dir, args.log_level)
    logger.info("=" * 70)
    logger.info("Starting GPT Model Training")
    logger.info("=" * 70)
    
    # Determine device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    logger.info(f"Using device: {device}")
    
    # Create model
    model = create_model(args, logger)
    model = model.to(device)
    
    # Create dataloaders
    train_dataloader, val_dataloader = create_dataloaders(args, logger)
    
    # Set up optimizer
    optimizer, scheduler = setup_optimizer(model, args, logger)
    
    # Set up loss function
    criterion = nn.CrossEntropyLoss()
    logger.info("Loss function: CrossEntropyLoss")
    
    # Load checkpoint if resuming
    start_epoch = 0
    start_step = 0
    if args.resume_from:
        start_epoch, start_step = load_checkpoint(
            args.resume_from, model, optimizer, logger
        )
    
    # Create checkpoint directory
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # Training loop
    logger.info("=" * 70)
    logger.info("Starting training loop")
    logger.info("=" * 70)
    
    global_step = start_step
    
    for epoch in range(start_epoch, args.epochs):
        logger.info(f"\nEpoch {epoch + 1}/{args.epochs}")
        logger.info("-" * 70)
        
        # Set model to training mode
        model.train()
        
        # TODO: Implement training loop
        # for batch_idx, batch in enumerate(train_dataloader):
        #     # Training step
        #     loss = train_step(...)
        #     
        #     # Logging
        #     if global_step % args.log_every == 0:
        #         logger.info(...)
        #     
        #     # Validation
        #     if global_step % args.val_every == 0:
        #         val_metrics = validate(...)
        #         logger.info(...)
        #     
        #     # Checkpointing
        #     if global_step % args.save_every == 0:
        #         save_checkpoint(...)
        #     
        #     global_step += 1
        
        logger.info(f"Epoch {epoch + 1} completed")
    
    logger.info("=" * 70)
    logger.info("Training completed!")
    logger.info("=" * 70)
    
    # Save final checkpoint
    save_checkpoint(
        model, optimizer, args.epochs, global_step, 0.0,
        args.checkpoint_dir, logger
    )


def main():
    """Main entry point."""
    args = parse_args()
    train(args)


if __name__ == "__main__":
    main()

