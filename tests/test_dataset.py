"""
Unit tests for GPTDataset and create_dataloader.
"""

import pytest
import torch
import tiktoken
import tempfile
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.dataset import GPTDataset, load_text_from_file, create_dataloader


class TestGPTDataset:
    """Tests for GPTDataset class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.tokenizer = tiktoken.get_encoding("gpt2")
        self.sample_text = "Hello world! This is a test sentence for the GPT dataset."
    
    def test_dataset_creation(self):
        """Test basic dataset creation."""
        dataset = GPTDataset(
            txt=self.sample_text,
            tokenizer=self.tokenizer,
            max_length=10,
            stride=5
        )
        assert len(dataset) > 0
    
    def test_dataset_item_shapes(self):
        """Test that dataset items have correct shapes."""
        max_length = 10
        dataset = GPTDataset(
            txt=self.sample_text,
            tokenizer=self.tokenizer,
            max_length=max_length,
            stride=5
        )
        
        if len(dataset) > 0:
            input_ids, target_ids = dataset[0]
            assert input_ids.shape == (max_length,)
            assert target_ids.shape == (max_length,)
    
    def test_target_is_shifted_input(self):
        """Test that target is input shifted by 1 position."""
        # Create a longer text to ensure we get sequences
        long_text = "A B C D E F G H I J K L M N O P Q R S T U V W X Y Z " * 10
        dataset = GPTDataset(
            txt=long_text,
            tokenizer=self.tokenizer,
            max_length=10,
            stride=5
        )
        
        if len(dataset) > 0:
            # Get the raw tokens for verification
            tokens = self.tokenizer.encode(long_text, allowed_special={"<|endoftext|>"})
            
            # First input should be tokens[0:10], target should be tokens[1:11]
            input_ids, target_ids = dataset[0]
            
            expected_input = torch.tensor(tokens[0:10], dtype=torch.long)
            expected_target = torch.tensor(tokens[1:11], dtype=torch.long)
            
            assert torch.equal(input_ids, expected_input)
            assert torch.equal(target_ids, expected_target)
    
    def test_dataset_dtype(self):
        """Test that dataset returns correct dtype."""
        dataset = GPTDataset(
            txt=self.sample_text,
            tokenizer=self.tokenizer,
            max_length=10,
            stride=5
        )
        
        if len(dataset) > 0:
            input_ids, target_ids = dataset[0]
            assert input_ids.dtype == torch.long
            assert target_ids.dtype == torch.long
    
    def test_empty_text_short_sequence(self):
        """Test with text shorter than max_length produces empty dataset."""
        short_text = "Hi"  # Very short
        dataset = GPTDataset(
            txt=short_text,
            tokenizer=self.tokenizer,
            max_length=100,  # Longer than text
            stride=50
        )
        # Should produce no sequences because text is too short
        assert len(dataset) == 0


class TestLoadTextFromFile:
    """Tests for load_text_from_file function."""
    
    def test_load_existing_file(self):
        """Test loading text from existing file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("Test content")
            temp_path = f.name
        
        try:
            text = load_text_from_file(temp_path)
            assert text == "Test content"
        finally:
            os.unlink(temp_path)
    
    def test_load_nonexistent_file(self):
        """Test that loading nonexistent file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_text_from_file("/nonexistent/path/file.txt")
    
    def test_load_empty_file(self):
        """Test that loading empty file raises ValueError."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("")  # Empty file
            temp_path = f.name
        
        try:
            with pytest.raises(ValueError):
                load_text_from_file(temp_path)
        finally:
            os.unlink(temp_path)


class TestCreateDataloader:
    """Tests for create_dataloader function."""
    
    def setup_method(self):
        """Create temporary test file."""
        self.temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False)
        self.temp_file.write("This is a test file with enough content to create multiple sequences. " * 20)
        self.temp_file.close()
        self.temp_path = self.temp_file.name
    
    def teardown_method(self):
        """Clean up temporary file."""
        if os.path.exists(self.temp_path):
            os.unlink(self.temp_path)
    
    def test_dataloader_creation(self):
        """Test basic dataloader creation."""
        dataloader = create_dataloader(
            self.temp_path,
            batch_size=2,
            max_length=32,
            stride=16
        )
        assert dataloader is not None
    
    def test_dataloader_batch_shapes(self):
        """Test that dataloader produces correct batch shapes."""
        batch_size = 2
        max_length = 32
        
        dataloader = create_dataloader(
            self.temp_path,
            batch_size=batch_size,
            max_length=max_length,
            stride=16
        )
        
        for input_batch, target_batch in dataloader:
            assert input_batch.shape == (batch_size, max_length)
            assert target_batch.shape == (batch_size, max_length)
            break
    
    def test_dataloader_with_multiple_files(self):
        """Test dataloader with multiple files."""
        # Create second temp file
        temp_file2 = tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False)
        temp_file2.write("Another test file with different content. " * 20)
        temp_file2.close()
        
        try:
            dataloader = create_dataloader(
                [self.temp_path, temp_file2.name],
                batch_size=2,
                max_length=32,
                stride=16
            )
            assert dataloader is not None
            # Should have more data with two files
            assert len(dataloader.dataset) > 0
        finally:
            os.unlink(temp_file2.name)
    
    def test_dataloader_invalid_file_type(self):
        """Test that invalid file type raises TypeError."""
        with pytest.raises(TypeError):
            create_dataloader(123)  # Invalid type


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

