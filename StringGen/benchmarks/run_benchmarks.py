#!/usr/bin/env python3
"""
Comprehensive benchmark suite for SCA String Generator.

This script runs various benchmarks to measure performance across:
- Different strategies
- Different sequence lengths
- Different batch sizes
- Different worker configurations
"""

import json
import time
import subprocess
import platform
import psutil
import datetime
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def get_system_info():
    """Collect system information for benchmark context."""
    return {
        'timestamp': datetime.datetime.now().isoformat(),
        'platform': platform.platform(),
        'processor': platform.processor(),
        'python_version': platform.python_version(),
        'cpu_count': psutil.cpu_count(logical=False),
        'cpu_count_logical': psutil.cpu_count(logical=True),
        'memory_total': psutil.virtual_memory().total,
        'memory_available': psutil.virtual_memory().available,
    }


def run_benchmark_command(cmd, description):
    """Run a benchmark command and capture output."""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print('='*60)
    
    start_time = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.time() - start_time
    
    return {
        'description': description,
        'command': ' '.join(cmd),
        'elapsed_time': elapsed,
        'stdout': result.stdout,
        'stderr': result.stderr,
        'returncode': result.returncode
    }


def parse_benchmark_output(output):
    """Parse benchmark output to extract metrics."""
    metrics = {}
    lines = output.split('\n')
    
    for line in lines:
        if 'seq/s' in line and '|' in line:
            parts = line.strip().split('|')
            if len(parts) == 2:
                strategy = parts[0].strip()
                rate_str = parts[1].strip()
                # Extract number from format like "10,725 seq/s"
                rate = float(rate_str.split()[0].replace(',', ''))
                metrics[strategy] = rate
    
    return metrics


def run_all_benchmarks():
    """Run comprehensive benchmark suite."""
    results = {
        'system_info': get_system_info(),
        'benchmarks': []
    }
    
    # 1. Basic benchmark with default settings
    result = run_benchmark_command(
        ['python3', 'sca_generator.py', '--benchmark'],
        'Basic benchmark (default settings)'
    )
    result['metrics'] = parse_benchmark_output(result['stdout'])
    results['benchmarks'].append(result)
    
    # 2. Benchmark with different worker counts
    for workers in [1, 2, 4, 8, 16, 32]:
        if workers <= psutil.cpu_count():
            result = run_benchmark_command(
                ['python3', 'sca_generator.py', '--benchmark', '--workers', str(workers), '--quiet'],
                f'Benchmark with {workers} workers'
            )
            result['metrics'] = parse_benchmark_output(result['stdout'])
            results['benchmarks'].append(result)
    
    # 3. Thread vs Process comparison
    result = run_benchmark_command(
        ['python3', 'sca_generator.py', '--benchmark', '--use-threads', '--quiet'],
        'Benchmark with threads instead of processes'
    )
    result['metrics'] = parse_benchmark_output(result['stdout'])
    results['benchmarks'].append(result)
    
    # 4. Different sequence lengths
    for length in [10, 100, 1000, 10000]:
        # Custom benchmark for specific length
        print(f"\n{'='*60}")
        print(f"Running: Custom benchmark for length {length}")
        print('='*60)
        
        from core import StringGenerator
        generator = StringGenerator(seed=42, num_workers=16)
        
        strategy_results = {}
        for strategy in ['INSET1', 'INSET2', 'CROSS1', 'CROSS2', 'CROSS3']:
            start = time.time()
            sequences = generator.generate_batch(strategy, length, 1000)
            elapsed = time.time() - start
            rate = 1000 / elapsed if elapsed > 0 else 0
            strategy_results[strategy] = rate
            print(f"{strategy}: {rate:.0f} seq/s")
        
        results['benchmarks'].append({
            'description': f'Benchmark with sequence length {length}',
            'length': length,
            'metrics': strategy_results
        })
    
    # 5. Large-scale generation test
    print(f"\n{'='*60}")
    print("Running: Large-scale generation test (1M sequences)")
    print('='*60)
    
    start = time.time()
    cmd = [
        'python3', 'sca_generator.py', 
        '--mode', 'sample',
        '--count', '1000000',
        '--length', '100',
        '--output', 'benchmark_1m.txt',
        '--overwrite',
        '--quiet'
    ]
    subprocess.run(cmd)
    elapsed = time.time() - start
    
    results['benchmarks'].append({
        'description': 'Large-scale generation (1M sequences)',
        'count': 1000000,
        'elapsed_time': elapsed,
        'sequences_per_second': 1000000 / elapsed if elapsed > 0 else 0
    })
    
    # Clean up
    Path('benchmark_1m.txt').unlink(missing_ok=True)
    
    return results


def main():
    """Run benchmarks and save results."""
    print("SCA String Generator - Comprehensive Benchmark Suite")
    print("=" * 60)
    
    # Run benchmarks
    results = run_all_benchmarks()
    
    # Save results
    output_file = Path('benchmarks/benchmark_results.json')
    output_file.parent.mkdir(exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"Benchmark complete! Results saved to: {output_file}")
    
    # Print summary
    print("\nSummary of Results:")
    print("-" * 40)
    
    for benchmark in results['benchmarks']:
        if 'metrics' in benchmark and benchmark['metrics']:
            print(f"\n{benchmark['description']}:")
            for strategy, rate in benchmark['metrics'].items():
                print(f"  {strategy}: {rate:,.0f} seq/s")
        elif 'sequences_per_second' in benchmark:
            print(f"\n{benchmark['description']}:")
            print(f"  Rate: {benchmark['sequences_per_second']:,.0f} seq/s")
            print(f"  Time: {benchmark['elapsed_time']:.2f} seconds")


if __name__ == '__main__':
    main()