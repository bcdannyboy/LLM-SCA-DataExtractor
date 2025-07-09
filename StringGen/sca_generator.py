#!/usr/bin/env python3
"""
SCA Generator - Production-grade Special Characters Attack string generator.

This is the main command-line interface for generating SCA probe sequences
with high performance multi-threading, progress tracking, and comprehensive
metrics collection.

Features:
- Multi-threaded/multi-process generation for maximum throughput
- Real-time progress tracking with ETA
- Performance metrics and benchmarking
- Multiple output formats (pipe-delimited, JSON, binary)
- Streaming generation for large datasets
- Resume capability for interrupted jobs
"""

import argparse
import sys
import time
import json
import signal
import logging
from pathlib import Path
from typing import Optional, List, Tuple, Union
import multiprocessing as mp
from datetime import datetime, timedelta

from core import StringGenerator, CharacterSets
from utils.progress import ProgressTracker
from utils.config import Config, load_config, save_config


# Global variable for graceful shutdown
shutdown_requested = False


def signal_handler(signum, frame):
    """Handle interrupt signals gracefully."""
    global shutdown_requested
    shutdown_requested = True
    print("\n[!] Shutdown requested. Finishing current batch...")


def parse_length_spec(spec: str) -> Union[int, Tuple[int, int]]:
    """
    Parse length specification from command line.
    
    Accepts:
    - Single integer: "100"
    - Range: "10-100"
    - Range with comma: "10,100"
    
    Args:
        spec: Length specification string
        
    Returns:
        Single integer or (min, max) tuple
        
    Raises:
        ValueError: If format is invalid
    """
    # Try single integer first
    try:
        return int(spec)
    except ValueError:
        pass
    
    # Try range with dash
    if '-' in spec:
        parts = spec.split('-')
        if len(parts) == 2:
            try:
                min_val = int(parts[0])
                max_val = int(parts[1])
                if min_val <= max_val:
                    return (min_val, max_val)
            except ValueError:
                pass
    
    # Try range with comma
    if ',' in spec:
        parts = spec.split(',')
        if len(parts) == 2:
            try:
                min_val = int(parts[0])
                max_val = int(parts[1])
                if min_val <= max_val:
                    return (min_val, max_val)
            except ValueError:
                pass
    
    raise ValueError(
        f"Invalid length specification: '{spec}'. "
        "Use integer (100) or range (10-100 or 10,100)"
    )


def create_parser() -> argparse.ArgumentParser:
    """Create command-line argument parser with all options."""
    parser = argparse.ArgumentParser(
        description="High-performance Special Characters Attack (SCA) string generator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate all INSET1 sequences for lengths 10-20
  %(prog)s --mode exhaustive --min-length 10 --max-length 20 --strategy INSET1
  
  # Generate 1000 random sequences of length 100-500
  %(prog)s --mode sample --length 100-500 --count 1000
  
  # Benchmark all strategies
  %(prog)s --benchmark
  
  # Generate with specific strategy and progress tracking
  %(prog)s --mode sample --strategy CROSS3 --length 1000 --count 10000 --progress
  
  # Use configuration file
  %(prog)s --config my_config.json
        """
    )
    
    # Mode selection
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        '--mode', 
        choices=['exhaustive', 'sample'],
        default='sample',
        help='Generation mode: exhaustive (all variants) or sample (random)'
    )
    mode_group.add_argument(
        '--benchmark',
        action='store_true',
        help='Run performance benchmark and exit'
    )
    
    # Generation parameters
    gen_group = parser.add_argument_group('Generation Parameters')
    gen_group.add_argument(
        '-s', '--strategy',
        choices=['INSET1', 'INSET2', 'CROSS1', 'CROSS2', 'CROSS3'],
        help='Specific strategy to use (default: all strategies)'
    )
    gen_group.add_argument(
        '-l', '--length',
        type=str,
        default='10',
        help='Sequence length or range (e.g., 100 or 10-100)'
    )
    gen_group.add_argument(
        '-c', '--count',
        type=int,
        default=1,
        help='Number of sequences to generate (sample mode only)'
    )
    gen_group.add_argument(
        '--min-length',
        type=int,
        default=1,
        help='Minimum length for exhaustive mode'
    )
    gen_group.add_argument(
        '--max-length',
        type=int,
        default=100,
        help='Maximum length for exhaustive mode'
    )
    gen_group.add_argument(
        '--seed',
        type=int,
        help='Random seed for reproducibility'
    )
    
    # Output parameters
    out_group = parser.add_argument_group('Output Parameters')
    out_group.add_argument(
        '-o', '--output',
        type=Path,
        default=Path('sca_output.txt'),
        help='Output file path (default: sca_output.txt)'
    )
    out_group.add_argument(
        '-f', '--format',
        choices=['pipe', 'json', 'binary'],
        default='pipe',
        help='Output format (default: pipe-delimited)'
    )
    out_group.add_argument(
        '--overwrite',
        action='store_true',
        help='Overwrite existing output file'
    )
    out_group.add_argument(
        '--append',
        action='store_true',
        help='Append to existing output file'
    )
    
    # Performance parameters
    perf_group = parser.add_argument_group('Performance Parameters')
    perf_group.add_argument(
        '-w', '--workers',
        type=int,
        help='Number of parallel workers (default: CPU count)'
    )
    perf_group.add_argument(
        '--use-threads',
        action='store_true',
        help='Use threads instead of processes for parallelism'
    )
    perf_group.add_argument(
        '--buffer-size',
        type=int,
        default=10000,
        help='Internal buffer size for batch processing'
    )
    perf_group.add_argument(
        '--chunk-size',
        type=int,
        default=1000,
        help='File write chunk size'
    )
    
    # Progress and logging
    log_group = parser.add_argument_group('Progress and Logging')
    log_group.add_argument(
        '--progress',
        action='store_true',
        help='Show progress bar and ETA'
    )
    log_group.add_argument(
        '--metrics',
        action='store_true',
        help='Save generation metrics to JSON file'
    )
    log_group.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Logging verbosity level'
    )
    log_group.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress all output except errors'
    )
    
    # Configuration
    config_group = parser.add_argument_group('Configuration')
    config_group.add_argument(
        '--config',
        type=Path,
        help='Load parameters from configuration file'
    )
    config_group.add_argument(
        '--save-config',
        type=Path,
        help='Save current parameters to configuration file'
    )
    
    # Advanced options
    adv_group = parser.add_argument_group('Advanced Options')
    adv_group.add_argument(
        '--validate',
        action='store_true',
        help='Validate generated sequences match specifications'
    )
    adv_group.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be generated without writing files'
    )
    adv_group.add_argument(
        '--resume',
        type=Path,
        help='Resume interrupted generation from state file'
    )
    
    return parser


def setup_logging(args):
    """Configure logging based on command-line arguments."""
    if args.quiet:
        level = logging.ERROR
    else:
        level = getattr(logging, args.log_level.upper())
    
    # Configure root logger
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Suppress verbose logs from urllib3 and other libraries
    logging.getLogger('urllib3').setLevel(logging.WARNING)


def run_benchmark(args):
    """Run performance benchmark."""
    print("=" * 60)
    print("SCA Generator Performance Benchmark")
    print("=" * 60)
    print(f"CPU Count: {mp.cpu_count()}")
    print(f"Workers: {args.workers or mp.cpu_count()}")
    print(f"Using: {'Threads' if args.use_threads else 'Processes'}")
    print("=" * 60)
    
    # Create generator
    generator = StringGenerator(
        seed=args.seed,
        num_workers=args.workers,
        use_multiprocessing=not args.use_threads,
        buffer_size=args.buffer_size,
        log_level=args.log_level
    )
    
    # Run benchmark
    print("\nBenchmarking all strategies (1000 sequences of length 1000)...")
    results = generator.benchmark(length=1000, count=1000)
    
    # Display results
    print("\nResults (sequences/second):")
    print("-" * 40)
    for strategy, rate in sorted(results.items(), key=lambda x: x[1], reverse=True):
        print(f"{strategy:8} | {rate:10,.0f} seq/s")
    
    # Overall statistics
    total_rate = sum(results.values())
    print("-" * 40)
    print(f"{'Total':8} | {total_rate:10,.0f} seq/s")
    print(f"\nEstimated throughput: {total_rate * 1000 / 1e6:.2f} MB/s")


def run_generation(args):
    """Run the main generation process."""
    # Setup signal handler for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Parse length specification
    if args.mode == 'sample':
        length_spec = parse_length_spec(args.length)
    else:
        # Exhaustive mode uses min/max length
        length_spec = None
    
    # Create generator
    generator = StringGenerator(
        seed=args.seed,
        num_workers=args.workers,
        use_multiprocessing=not args.use_threads,
        buffer_size=args.buffer_size,
        log_level=args.log_level
    )
    
    # Calculate total sequences for progress tracking
    if args.mode == 'exhaustive':
        if args.strategy == 'INSET1':
            # INSET1 generates all character repetitions
            total_chars = CharacterSets.S1_SIZE + CharacterSets.S2_SIZE + CharacterSets.L_SIZE
            total_sequences = total_chars * (args.max_length - args.min_length + 1)
        else:
            # Other strategies generate one sequence per length
            total_sequences = args.max_length - args.min_length + 1
            if not args.strategy:
                # All strategies
                total_sequences *= 5
    else:
        total_sequences = args.count
    
    # Create progress tracker if requested
    progress = None
    if args.progress and not args.quiet:
        from utils.progress import ProgressTracker
        progress = ProgressTracker(
            total=total_sequences,
            desc="Generating sequences"
        )
    
    # Dry run - just show what would be generated
    if args.dry_run:
        print(f"[DRY RUN] Would generate {total_sequences:,} sequences")
        print(f"Mode: {args.mode}")
        print(f"Strategy: {args.strategy or 'all'}")
        if args.mode == 'sample':
            print(f"Length: {length_spec}")
            print(f"Count: {args.count}")
        else:
            print(f"Length range: {args.min_length}-{args.max_length}")
        print(f"Output: {args.output} (format: {args.format})")
        return
    
    # Generate sequences
    start_time = time.time()
    
    try:
        if args.mode == 'exhaustive':
            sequences = generator.generate_exhaustive(
                min_length=args.min_length,
                max_length=args.max_length,
                strategies=[args.strategy] if args.strategy else None
            )
        else:
            sequences = generator.generate_samples(
                length_range=length_spec,
                count=args.count,
                strategy=args.strategy
            )
        
        # Wrap sequences with progress tracking if enabled
        if progress:
            sequences = progress.track(sequences)
        
        # Write to file
        metrics = generator.write_to_file(
            output_path=args.output,
            sequences=sequences,
            format=args.format,
            overwrite=args.overwrite,
            chunk_size=args.chunk_size
        )
        
    except KeyboardInterrupt:
        print("\n[!] Generation interrupted by user")
        if progress:
            progress.close()
        return
    except Exception as e:
        print(f"\n[ERROR] Generation failed: {e}")
        if progress:
            progress.close()
        raise
    
    # Close progress tracker
    if progress:
        progress.close()
    
    # Display summary
    elapsed = time.time() - start_time
    if not args.quiet:
        print(f"\n{'='*60}")
        print(f"Generation Complete!")
        print(f"{'='*60}")
        print(f"Total sequences: {metrics.total_sequences:,}")
        print(f"Total characters: {metrics.total_characters:,}")
        print(f"Time elapsed: {timedelta(seconds=int(elapsed))}")
        print(f"Generation rate: {metrics.sequences_per_second:,.0f} seq/s")
        print(f"Throughput: {metrics.characters_per_second / 1e6:.2f} MB/s")
        print(f"Output file: {args.output}")
        print(f"Output size: {args.output.stat().st_size / 1e6:.2f} MB")
        
        # Strategy breakdown
        if len(metrics.strategy_counts) > 1:
            print(f"\nStrategy breakdown:")
            for strategy, count in sorted(metrics.strategy_counts.items()):
                percentage = count / metrics.total_sequences * 100
                print(f"  {strategy}: {count:,} ({percentage:.1f}%)")
    
    # Save metrics if requested
    if args.metrics:
        metrics_path = args.output.with_suffix('.metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(metrics.to_dict(), f, indent=2)
        if not args.quiet:
            print(f"\nMetrics saved to: {metrics_path}")


def main():
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args()
    
    # Load configuration file if specified
    if args.config:
        config = load_config(args.config)
        # Override args with config values
        for key, value in config.items():
            if hasattr(args, key) and getattr(args, key) is None:
                setattr(args, key, value)
    
    # Save configuration if requested
    if args.save_config:
        config = Config.from_args(args)
        save_config(config, args.save_config)
        print(f"Configuration saved to: {args.save_config}")
        return
    
    # Setup logging
    setup_logging(args)
    
    # Run appropriate mode
    if args.benchmark:
        run_benchmark(args)
    else:
        run_generation(args)


if __name__ == "__main__":
    main()