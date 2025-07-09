"""
File sequence loader for SCAudit.

This module provides functionality to load attack sequences from StringGen
output files, with streaming support for large datasets.
"""

import hashlib
import re
from pathlib import Path
from typing import AsyncIterator, Iterator, Optional, Dict, Any
import asyncio

from ..models.data_models import Sequence, Strategy


class FileSequenceLoader:
    """
    Loads attack sequences from StringGen output files.
    
    This loader streams sequences line-by-line to handle large files
    efficiently, and pre-computes metadata for each sequence.
    """
    
    def __init__(self, strategy_pattern: Optional[str] = None):
        """
        Initialize the loader.
        
        Args:
            strategy_pattern: Regex pattern to extract strategy from filename
        """
        self.strategy_pattern = strategy_pattern or r"(INSET1|INSET2|CROSS1|CROSS2|CROSS3)"
    
    def _extract_strategy(self, filepath: Path) -> Optional[Strategy]:
        """
        Extract strategy from filename.
        
        Args:
            filepath: Path to the file
            
        Returns:
            Strategy enum or None if not found
        """
        match = re.search(self.strategy_pattern, filepath.name)
        if match:
            try:
                return Strategy(match.group(1))
            except ValueError:
                pass
        return None
    
    def iter_sequences(self, path: str) -> Iterator[Sequence]:
        """
        Iterate over sequences in a file or glob pattern.
        
        Args:
            path: File path or glob pattern
            
        Yields:
            Sequence objects
        """
        paths = []
        path_obj = Path(path)
        
        if path_obj.is_file():
            paths = [path_obj]
        elif path_obj.is_dir():
            paths = list(path_obj.glob("*.txt"))
        else:
            # Treat as glob pattern
            from glob import glob
            paths = [Path(p) for p in glob(path)]
        
        for filepath in paths:
            strategy = self._extract_strategy(filepath)
            
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    for line_num, line in enumerate(f, 1):
                        line = line.strip()
                        if not line:  # Skip empty lines
                            continue
                        
                        yield Sequence(
                            content=line,
                            strategy=strategy,
                            metadata={
                                "source_file": str(filepath),
                                "line_number": line_num,
                                "file_strategy": strategy.value if strategy else None
                            }
                        )
            except Exception as e:
                # Log error but continue processing other files
                print(f"Error reading {filepath}: {e}")
    
    async def aiter_sequences(self, path: str) -> AsyncIterator[Sequence]:
        """
        Asynchronously iterate over sequences.
        
        Args:
            path: File path or glob pattern
            
        Yields:
            Sequence objects
        """
        # Run the synchronous iterator in a thread pool
        loop = asyncio.get_event_loop()
        
        def _generator():
            return list(self.iter_sequences(path))
        
        sequences = await loop.run_in_executor(None, _generator)
        
        for seq in sequences:
            yield seq
    
    def load_batch(self, path: str, batch_size: int = 1000) -> Iterator[list[Sequence]]:
        """
        Load sequences in batches.
        
        Args:
            path: File path or glob pattern
            batch_size: Number of sequences per batch
            
        Yields:
            Lists of Sequence objects
        """
        batch = []
        for seq in self.iter_sequences(path):
            batch.append(seq)
            if len(batch) >= batch_size:
                yield batch
                batch = []
        
        # Yield remaining sequences
        if batch:
            yield batch
    
    def count_sequences(self, path: str) -> int:
        """
        Count total sequences without loading them all into memory.
        
        Args:
            path: File path or glob pattern
            
        Returns:
            Total number of sequences
        """
        count = 0
        for _ in self.iter_sequences(path):
            count += 1
        return count
    
    def get_statistics(self, path: str) -> Dict[str, Any]:
        """
        Compute statistics about sequences in files.
        
        Args:
            path: File path or glob pattern
            
        Returns:
            Dictionary with statistics
        """
        stats = {
            "total_sequences": 0,
            "unique_sequences": set(),
            "strategy_counts": {},
            "length_distribution": {},
            "files_processed": 0
        }
        
        for seq in self.iter_sequences(path):
            stats["total_sequences"] += 1
            stats["unique_sequences"].add(seq.sha256)
            
            # Count by strategy
            strategy_name = seq.strategy.value if seq.strategy else "unknown"
            stats["strategy_counts"][strategy_name] = stats["strategy_counts"].get(strategy_name, 0) + 1
            
            # Track length distribution (bucketed)
            length_bucket = (seq.length // 100) * 100
            stats["length_distribution"][length_bucket] = stats["length_distribution"].get(length_bucket, 0) + 1
        
        # Convert set to count
        stats["unique_sequences"] = len(stats["unique_sequences"])
        
        return stats