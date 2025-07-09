"""
Unit tests for file sequence loader.
"""

import unittest
import tempfile
import os
from pathlib import Path
import asyncio
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from SCAudit.core.loader import FileSequenceLoader
from SCAudit.models.data_models import Strategy


class TestFileSequenceLoader(unittest.TestCase):
    """Test sequence loading functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.loader = FileSequenceLoader()
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up test fixtures."""
        # Remove temporary files
        for file in Path(self.temp_dir).glob("*"):
            file.unlink()
        os.rmdir(self.temp_dir)
    
    def create_test_file(self, name: str, content: list[str]) -> Path:
        """Create a test file with given content."""
        path = Path(self.temp_dir) / name
        with open(path, 'w') as f:
            f.write('\n'.join(content))
        return path
    
    def test_strategy_extraction(self):
        """Test extracting strategy from filename."""
        # Test default pattern
        self.assertEqual(
            self.loader._extract_strategy(Path("sequences_INSET1.txt")),
            Strategy.INSET1
        )
        self.assertEqual(
            self.loader._extract_strategy(Path("CROSS3_data.txt")),
            Strategy.CROSS3
        )
        
        # Test no match
        self.assertIsNone(
            self.loader._extract_strategy(Path("random_file.txt"))
        )
        
        # Test custom pattern
        loader = FileSequenceLoader(strategy_pattern=r"strategy_(\w+)")
        self.assertIsNone(
            loader._extract_strategy(Path("strategy_CUSTOM.txt"))
        )
    
    def test_iter_sequences_single_file(self):
        """Test iterating sequences from a single file."""
        # Create test file
        sequences = [
            "{{{{{",
            "[][][]",
            "<<<>>>",
            "",  # Empty line should be skipped
            "final sequence"
        ]
        file_path = self.create_test_file("test_INSET1.txt", sequences)
        
        # Load sequences
        loaded = list(self.loader.iter_sequences(str(file_path)))
        
        # Check results
        self.assertEqual(len(loaded), 4)  # Empty line skipped
        self.assertEqual(loaded[0].content, "{{{{{")
        self.assertEqual(loaded[1].content, "[][][]")
        self.assertEqual(loaded[2].content, "<<<>>>")
        self.assertEqual(loaded[3].content, "final sequence")
        
        # Check metadata
        for i, seq in enumerate(loaded):
            self.assertEqual(seq.strategy, Strategy.INSET1)
            self.assertEqual(seq.metadata["source_file"], str(file_path))
            self.assertEqual(seq.metadata["line_number"], i + 1 if i < 3 else i + 2)
            self.assertEqual(seq.metadata["file_strategy"], "INSET1")
            self.assertEqual(seq.length, len(seq.content))
            self.assertIsNotNone(seq.sha256)
    
    def test_iter_sequences_glob(self):
        """Test iterating sequences from multiple files."""
        # Create multiple test files
        self.create_test_file("data_CROSS1.txt", ["seq1", "seq2"])
        self.create_test_file("data_CROSS2.txt", ["seq3", "seq4"])
        self.create_test_file("other.txt", ["seq5"])  # Won't match pattern
        
        # Load with glob pattern
        pattern = os.path.join(self.temp_dir, "data_*.txt")
        loaded = list(self.loader.iter_sequences(pattern))
        
        # Check results
        self.assertEqual(len(loaded), 4)  # Only from data_*.txt files
        
        # Check strategies
        strategies = {seq.strategy for seq in loaded}
        self.assertIn(Strategy.CROSS1, strategies)
        self.assertIn(Strategy.CROSS2, strategies)
    
    def test_iter_sequences_directory(self):
        """Test iterating sequences from a directory."""
        # Create test files
        self.create_test_file("file1.txt", ["a", "b"])
        self.create_test_file("file2.txt", ["c", "d"])
        
        # Load from directory
        loaded = list(self.loader.iter_sequences(self.temp_dir))
        
        # Check results
        self.assertEqual(len(loaded), 4)
        contents = {seq.content for seq in loaded}
        self.assertEqual(contents, {"a", "b", "c", "d"})
    
    def test_load_batch(self):
        """Test batch loading."""
        # Create test file with many sequences
        sequences = [f"seq_{i}" for i in range(25)]
        file_path = self.create_test_file("batch_test.txt", sequences)
        
        # Load in batches
        batches = list(self.loader.load_batch(str(file_path), batch_size=10))
        
        # Check batches
        self.assertEqual(len(batches), 3)  # 10, 10, 5
        self.assertEqual(len(batches[0]), 10)
        self.assertEqual(len(batches[1]), 10)
        self.assertEqual(len(batches[2]), 5)
        
        # Check content preservation
        all_contents = []
        for batch in batches:
            all_contents.extend(seq.content for seq in batch)
        self.assertEqual(all_contents, sequences)
    
    def test_count_sequences(self):
        """Test sequence counting."""
        # Create test files
        self.create_test_file("count1.txt", ["a", "b", "c"])
        self.create_test_file("count2.txt", ["d", "e"])
        
        # Count single file
        count = self.loader.count_sequences(
            os.path.join(self.temp_dir, "count1.txt")
        )
        self.assertEqual(count, 3)
        
        # Count with glob
        count = self.loader.count_sequences(
            os.path.join(self.temp_dir, "count*.txt")
        )
        self.assertEqual(count, 5)
    
    def test_get_statistics(self):
        """Test statistics generation."""
        # Create test files with different strategies
        self.create_test_file("stats_INSET1.txt", ["a" * 10, "b" * 20, "c" * 10])
        self.create_test_file("stats_CROSS1.txt", ["x" * 150, "y" * 150])
        
        # Get statistics
        stats = self.loader.get_statistics(
            os.path.join(self.temp_dir, "stats_*.txt")
        )
        
        # Check results
        self.assertEqual(stats["total_sequences"], 5)
        self.assertEqual(stats["unique_sequences"], 5)  # All different
        
        # Check strategy counts
        self.assertEqual(stats["strategy_counts"]["INSET1"], 3)
        self.assertEqual(stats["strategy_counts"]["CROSS1"], 2)
        
        # Check length distribution
        self.assertIn(0, stats["length_distribution"])  # 0-99 bucket
        self.assertIn(100, stats["length_distribution"])  # 100-199 bucket
    
    def test_error_handling(self):
        """Test error handling during file reading."""
        # Non-existent file
        loaded = list(self.loader.iter_sequences("/non/existent/file.txt"))
        self.assertEqual(len(loaded), 0)
        
        # Create file with invalid encoding
        bad_file = Path(self.temp_dir) / "bad_encoding.txt"
        with open(bad_file, 'wb') as f:
            f.write(b'\xff\xfe Invalid UTF-8')
        
        # Should handle error gracefully
        loaded = list(self.loader.iter_sequences(str(bad_file)))
        self.assertEqual(len(loaded), 0)


class TestAsyncFileSequenceLoader(unittest.TestCase):
    """Test async sequence loading functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.loader = FileSequenceLoader()
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up test fixtures."""
        for file in Path(self.temp_dir).glob("*"):
            file.unlink()
        os.rmdir(self.temp_dir)
    
    def create_test_file(self, name: str, content: list[str]) -> Path:
        """Create a test file with given content."""
        path = Path(self.temp_dir) / name
        with open(path, 'w') as f:
            f.write('\n'.join(content))
        return path
    
    async def test_aiter_sequences(self):
        """Test async iteration of sequences."""
        # Create test file
        sequences = ["async1", "async2", "async3"]
        file_path = self.create_test_file("async_test.txt", sequences)
        
        # Load sequences asynchronously
        loaded = []
        async for seq in self.loader.aiter_sequences(str(file_path)):
            loaded.append(seq)
        
        # Check results
        self.assertEqual(len(loaded), 3)
        self.assertEqual([seq.content for seq in loaded], sequences)
    
    def test_async_wrapper(self):
        """Test that async functions work properly."""
        # Create test file
        self.create_test_file("wrapper_test.txt", ["test"])
        
        # Run async test
        asyncio.run(self.test_aiter_sequences())


if __name__ == '__main__':
    unittest.main()