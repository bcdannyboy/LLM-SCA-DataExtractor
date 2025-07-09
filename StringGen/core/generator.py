#!/usr/bin/env python3
"""
High-performance string generator with multi-threading support.

This module provides the main StringGenerator class that orchestrates
the generation of SCA probe sequences using multiple strategies and
parallel processing for maximum throughput.

Key features:
- Multi-threaded batch generation
- Memory-efficient streaming output
- Progress tracking and performance metrics
- Deterministic output with seed support
"""

import os
import time
import random
import threading
import queue
import multiprocessing as mp
from pathlib import Path
from typing import List, Tuple, Optional, Iterator, Dict, Any, Union
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from dataclasses import dataclass
import json
import logging

from .strategies import Strategy, create_strategy, STRATEGY_REGISTRY
from .character_sets import CharacterSets


@dataclass
class GenerationMetrics:
    """Metrics collected during generation process."""
    total_sequences: int = 0
    total_characters: int = 0
    generation_time: float = 0.0
    write_time: float = 0.0
    sequences_per_second: float = 0.0
    characters_per_second: float = 0.0
    strategy_counts: Dict[str, int] = None
    
    def __post_init__(self):
        if self.strategy_counts is None:
            self.strategy_counts = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary for JSON serialization."""
        return {
            'total_sequences': self.total_sequences,
            'total_characters': self.total_characters,
            'generation_time_seconds': self.generation_time,
            'write_time_seconds': self.write_time,
            'sequences_per_second': self.sequences_per_second,
            'characters_per_second': self.characters_per_second,
            'strategy_counts': self.strategy_counts
        }


class StringGenerator:
    """
    High-performance string generator for SCA probe sequences.
    
    This class coordinates the generation of probe sequences using multiple
    strategies and parallel processing. It supports both exhaustive generation
    (all possible combinations) and sampling-based generation.
    """
    
    def __init__(self, 
                 seed: Optional[int] = None,
                 num_workers: Optional[int] = None,
                 use_multiprocessing: bool = True,
                 buffer_size: int = 10000,
                 log_level: str = 'INFO'):
        """
        Initialize string generator with configuration.
        
        Args:
            seed: Random seed for reproducibility. If None, uses system time.
            num_workers: Number of parallel workers. If None, uses CPU count.
            use_multiprocessing: Whether to use processes (True) or threads (False).
            buffer_size: Size of internal buffer for batch processing.
            log_level: Logging level for diagnostics.
        """
        self.seed = seed if seed is not None else int(time.time())
        self.rng = random.Random(self.seed)
        
        # Determine optimal worker count
        if num_workers is None:
            num_workers = mp.cpu_count()
        self.num_workers = max(1, num_workers)
        
        self.use_multiprocessing = use_multiprocessing
        self.buffer_size = buffer_size
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(getattr(logging, log_level.upper()))
        
        # Initialize strategies with different random states
        self._init_strategies()
        
        # Metrics tracking
        self.metrics = GenerationMetrics()
        
    def _init_strategies(self):
        """Initialize strategy instances with independent random states."""
        self.strategies = {}
        for name in STRATEGY_REGISTRY:
            # Create independent random state for each strategy
            strategy_seed = self.rng.randint(0, 2**32 - 1)
            strategy_rng = random.Random(strategy_seed)
            self.strategies[name] = create_strategy(name, strategy_rng)
    
    def generate_single(self, strategy: str, length: int) -> str:
        """
        Generate a single sequence using specified strategy.
        
        Args:
            strategy: Strategy name ('INSET1', 'INSET2', etc.)
            length: Length of sequence to generate
            
        Returns:
            Generated sequence
            
        Raises:
            ValueError: If strategy name is invalid
        """
        if strategy not in self.strategies:
            raise ValueError(f"Unknown strategy: {strategy}")
        
        return self.strategies[strategy].generate(length)
    
    def generate_batch(self, 
                      strategy: str, 
                      length: int, 
                      count: int) -> List[str]:
        """
        Generate multiple sequences in parallel.
        
        Args:
            strategy: Strategy name
            length: Length of each sequence
            count: Number of sequences to generate
            
        Returns:
            List of generated sequences
        """
        if count <= 0:
            return []
        
        if count == 1:
            return [self.generate_single(strategy, length)]
        
        # Use parallel generation for better performance
        strategy_obj = self.strategies[strategy]
        
        if self.use_multiprocessing and count > 100:
            # Use process pool for large batches
            with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
                # Create tasks
                futures = []
                batch_size = max(1, count // self.num_workers)
                
                for i in range(0, count, batch_size):
                    actual_batch_size = min(batch_size, count - i)
                    future = executor.submit(
                        self._generate_batch_worker,
                        strategy, length, actual_batch_size, 
                        self.seed + i  # Unique seed per batch
                    )
                    futures.append(future)
                
                # Collect results
                results = []
                for future in as_completed(futures):
                    results.extend(future.result())
                
                return results[:count]  # Ensure exact count
        else:
            # Use thread pool for smaller batches
            with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                futures = [
                    executor.submit(strategy_obj.generate, length)
                    for _ in range(count)
                ]
                return [f.result() for f in futures]
    
    @staticmethod
    def _generate_batch_worker(strategy_name: str, 
                              length: int, 
                              count: int, 
                              seed: int) -> List[str]:
        """Worker function for process-based parallel generation."""
        rng = random.Random(seed)
        strategy = create_strategy(strategy_name, rng)
        return [strategy.generate(length) for _ in range(count)]
    
    def generate_exhaustive(self, 
                           min_length: int, 
                           max_length: int,
                           strategies: Optional[List[str]] = None) -> Iterator[Tuple[str, int, str]]:
        """
        Generate all possible sequences for given length range.
        
        This is primarily useful for INSET1 strategy which has finite variants.
        
        Args:
            min_length: Minimum sequence length (inclusive)
            max_length: Maximum sequence length (inclusive)
            strategies: List of strategies to use. If None, uses all.
            
        Yields:
            Tuples of (strategy_name, length, sequence)
        """
        if strategies is None:
            strategies = list(self.strategies.keys())
        
        for length in range(min_length, max_length + 1):
            for strategy_name in strategies:
                if strategy_name == 'INSET1':
                    # Generate all variants for INSET1
                    strategy = self.strategies[strategy_name]
                    for set_name, sequence in strategy.generate_all_variants(length):
                        yield (strategy_name, length, sequence)
                else:
                    # Generate one sample for other strategies
                    sequence = self.generate_single(strategy_name, length)
                    yield (strategy_name, length, sequence)
    
    def generate_samples(self,
                        length_range: Union[int, Tuple[int, int]],
                        count: int,
                        strategy: Optional[str] = None) -> Iterator[Tuple[str, int, str]]:
        """
        Generate sample sequences with specified parameters.
        
        Args:
            length_range: Single length or (min, max) tuple
            count: Number of sequences to generate
            strategy: Specific strategy to use. If None, uses all randomly.
            
        Yields:
            Tuples of (strategy_name, length, sequence)
        """
        # Parse length range
        if isinstance(length_range, int):
            min_len = max_len = length_range
        else:
            min_len, max_len = length_range
        
        # Determine strategies to use
        if strategy:
            strategies = [strategy]
        else:
            strategies = list(self.strategies.keys())
        
        for _ in range(count):
            # Random length in range
            length = self.rng.randint(min_len, max_len)
            
            # Random strategy
            strategy_name = self.rng.choice(strategies)
            
            # Generate sequence
            sequence = self.generate_single(strategy_name, length)
            
            yield (strategy_name, length, sequence)
    
    def write_to_file(self,
                     output_path: Union[str, Path],
                     sequences: Iterator[Tuple[str, int, str]],
                     format: str = 'pipe',
                     overwrite: bool = False,
                     chunk_size: int = 1000) -> GenerationMetrics:
        """
        Write sequences to file with progress tracking.
        
        Args:
            output_path: Path to output file
            sequences: Iterator of (strategy, length, sequence) tuples
            format: Output format ('pipe' for STRAT|LEN|SEQ, 'json' for JSON)
            overwrite: Whether to overwrite existing file
            chunk_size: Number of sequences to buffer before writing
            
        Returns:
            Generation metrics
            
        Raises:
            FileExistsError: If file exists and overwrite is False
        """
        output_path = Path(output_path)
        
        # Check if file exists
        if output_path.exists() and not overwrite:
            raise FileExistsError(
                f"Output file {output_path} already exists. "
                "Use overwrite=True to replace."
            )
        
        # Create parent directory if needed
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Start timing
        start_time = time.time()
        write_start = None
        
        # Initialize metrics
        metrics = GenerationMetrics()
        
        # Write sequences in chunks for better performance
        buffer = []
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                for strategy, length, sequence in sequences:
                    # Update metrics
                    metrics.total_sequences += 1
                    metrics.total_characters += length
                    metrics.strategy_counts[strategy] = \
                        metrics.strategy_counts.get(strategy, 0) + 1
                    
                    # Format output
                    if format == 'pipe':
                        line = f"{strategy}|{length}|{sequence}\n"
                    elif format == 'json':
                        line = json.dumps({
                            'strategy': strategy,
                            'length': length,
                            'sequence': sequence
                        }) + '\n'
                    else:
                        raise ValueError(f"Unknown format: {format}")
                    
                    buffer.append(line)
                    
                    # Write chunk if buffer is full
                    if len(buffer) >= chunk_size:
                        if write_start is None:
                            write_start = time.time()
                        f.writelines(buffer)
                        buffer.clear()
                        
                        # Log progress
                        if metrics.total_sequences % 10000 == 0:
                            self.logger.info(
                                f"Generated {metrics.total_sequences:,} sequences "
                                f"({metrics.total_characters:,} characters)"
                            )
                
                # Write remaining buffer
                if buffer:
                    if write_start is None:
                        write_start = time.time()
                    f.writelines(buffer)
        
        finally:
            # Calculate final metrics
            end_time = time.time()
            metrics.generation_time = end_time - start_time
            if write_start:
                metrics.write_time = end_time - write_start
            
            if metrics.generation_time > 0:
                metrics.sequences_per_second = \
                    metrics.total_sequences / metrics.generation_time
                metrics.characters_per_second = \
                    metrics.total_characters / metrics.generation_time
        
        return metrics
    
    def benchmark(self, 
                 length: int = 1000, 
                 count: int = 1000) -> Dict[str, float]:
        """
        Benchmark generation performance for all strategies.
        
        Args:
            length: Length of sequences to generate
            count: Number of sequences per strategy
            
        Returns:
            Dictionary mapping strategy names to sequences/second
        """
        results = {}
        
        for strategy_name in self.strategies:
            start = time.time()
            sequences = self.generate_batch(strategy_name, length, count)
            elapsed = time.time() - start
            
            rate = count / elapsed if elapsed > 0 else float('inf')
            results[strategy_name] = rate
            
            self.logger.info(
                f"{strategy_name}: {rate:.0f} sequences/second "
                f"({len(sequences[0]) * count / elapsed / 1e6:.2f} MB/s)"
            )
        
        return results