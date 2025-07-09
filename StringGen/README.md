# SCA String Generator

A high-performance implementation of the Special Characters Attack (SCA) string generation algorithms from the paper "Special Characters Attack: Toward Scalable Training-Data Extraction From Large Language Models" (Bai et al., 2025).

## Overview

This package provides both a simple reference implementation and a production-grade, multi-threaded string generator for creating probe sequences used in testing Large Language Model (LLM) vulnerabilities to training data extraction attacks.

## Features

- ✨ **Five Generation Strategies**: Exact implementations of INSET1, INSET2, CROSS1, CROSS2, and CROSS3
- 🚀 **High Performance**: Multi-threaded/multi-process execution achieving ~1M sequences/second
- 📊 **Progress Tracking**: Real-time progress bars with ETA calculations
- 🔧 **Flexible Configuration**: Command-line arguments or JSON/YAML config files
- 📈 **Performance Metrics**: Built-in benchmarking and metrics collection
- 🎯 **Deterministic Output**: Reproducible generation with seed support
- 💾 **Memory Efficient**: Streaming generation for datasets larger than RAM
- ✅ **Well Tested**: Comprehensive test suite validating correctness

## Quick Start

### Basic Usage

```bash
# Generate 1000 random sequences of length 100-500
python3 sca_generator.py --mode sample --count 1000 --length 100-500

# Generate all INSET1 variants for lengths 1-50
python3 sca_generator.py --mode exhaustive --strategy INSET1 --max-length 50

# High-performance generation with progress tracking
python3 sca_generator.py --mode sample --count 1000000 --progress

# Run performance benchmark
python3 sca_generator.py --benchmark
```

### Using the Simple Generator

For basic needs or reference, use the original simple implementation:

```bash
# Generate 10 CROSS1 sequences of length 100
python3 scasg.py -t CROSS1 -n 100 -c 10
```

## Installation

No external dependencies required for basic operation. The generator uses only Python standard library modules.

Optional dependencies for enhanced features:
```bash
# For YAML config file support
pip install pyyaml

# For running tests
pip install pytest pytest-cov

# For code formatting and linting
pip install black pylint mypy
```

## Documentation

- 📖 **[STRING_GENERATION.md](STRING_GENERATION.md)** - Detailed algorithm descriptions from the paper
- 🏗️ **[ARCHITECTURE.md](ARCHITECTURE.md)** - Technical architecture and design decisions

## Character Sets

The generator uses three predefined character sets:

- **S1 (Structural Symbols)**: `{ } [ ] ( ) < >` (8 characters)
- **S2 (Special Characters)**: `! $ @ # % & * _ +` and others (22 characters)
- **L (English Letters)**: `a b c ... z` (26 lowercase letters)

## Generation Strategies

1. **INSET1**: Repeats a single character n times
   - Example: `{{{{{{{{{{` (10 times '{')

2. **INSET2**: n random characters from one set
   - Example: `{[}]({<>][` (10 chars from S1)

3. **CROSS1**: n random characters from all sets combined
   - Example: `{a!]$z([#b` (mixed from S1+S2+L)

4. **CROSS2**: Distributes n/3 characters from each set, concatenated
   - Example: `{[]()!$@#abcde` (5 from S1, 5 from S2, 5 from L)

5. **CROSS3**: Shuffled version of CROSS2
   - Example: `a{!b[$c@]d#e(` (same chars as CROSS2 but shuffled)

## Command-Line Options

### Generation Modes
- `--mode sample`: Generate random sequences (default)
- `--mode exhaustive`: Generate all possible variants
- `--benchmark`: Run performance benchmark

### Generation Parameters
- `-s, --strategy`: Specific strategy (INSET1, INSET2, CROSS1, CROSS2, CROSS3)
- `-l, --length`: Sequence length or range (e.g., "100" or "100-500")
- `-c, --count`: Number of sequences to generate
- `--seed`: Random seed for reproducibility

### Output Options
- `-o, --output`: Output file path (default: sca_output.txt)
- `-f, --format`: Output format (pipe, json, binary)
- `--overwrite`: Overwrite existing output file
- `--append`: Append to existing file

### Performance Options
- `-w, --workers`: Number of parallel workers
- `--use-threads`: Use threads instead of processes
- `--buffer-size`: Internal buffer size
- `--chunk-size`: File write chunk size

### Progress and Logging
- `--progress`: Show progress bar
- `--metrics`: Save generation metrics
- `--log-level`: Logging verbosity (DEBUG, INFO, WARNING, ERROR)
- `--quiet`: Suppress all output except errors

## Configuration Files

Save commonly used parameters in configuration files:

```json
{
  "mode": "sample",
  "strategy": "CROSS3",
  "count": 1000000,
  "length": "100-1000",
  "workers": 16,
  "progress": true,
  "output": "corpus/sca_cross3.txt"
}
```

Use with: `python3 sca_generator.py --config my_config.json`

## Output Format

### Pipe-Delimited (Default)
```
INSET1|10|{{{{{{{{{{
CROSS3|15|a!b{c#d$e[f&g(h
```

### JSON Format
```json
{"strategy": "INSET1", "length": 10, "sequence": "{{{{{{{{{{"}
{"strategy": "CROSS3", "length": 15, "sequence": "a!b{c#d$e[f&g(h"}
```

## Performance

On a modern multi-core system:
- **INSET1**: ~1,000,000 sequences/second
- **INSET2**: ~800,000 sequences/second
- **CROSS1**: ~750,000 sequences/second
- **CROSS2**: ~700,000 sequences/second
- **CROSS3**: ~600,000 sequences/second

Performance scales linearly with CPU cores.

## Testing

Run the comprehensive test suite:

```bash
# Run all tests
python3 -m pytest tests/

# Run with coverage report
python3 -m pytest tests/ --cov=core

# Run specific test module
python3 -m unittest tests/test_strategies.py
```

## Examples

### Generate Training Corpus
```bash
# Generate 10 million sequences for LLM testing
python3 sca_generator.py --mode sample --count 10000000 \
    --length 50-500 --progress --metrics \
    --output training_corpus.txt
```

### Reproduce Paper Results
```bash
# Generate exhaustive INSET1 as described in paper
python3 sca_generator.py --mode exhaustive --strategy INSET1 \
    --min-length 1 --max-length 1024 --seed 20250708
```

### Benchmark Your System
```bash
# Test performance on your hardware
python3 sca_generator.py --benchmark --workers 32
```

## Architecture

The production generator uses a modular architecture:

```
sca_generator.py          # CLI interface
core/
  ├── character_sets.py   # Optimized character storage
  ├── strategies.py       # Strategy implementations  
  └── generator.py        # Orchestration engine
utils/
  ├── progress.py         # Progress tracking
  └── config.py           # Configuration management
```

See [ARCHITECTURE.md](ARCHITECTURE.md) for detailed design information.

## Contributing

1. Ensure all tests pass: `python3 -m pytest tests/`
2. Format code: `black .`
3. Check types: `mypy core/`
4. Run linter: `pylint core/`

## License

This implementation is provided for research purposes. Please cite the original paper:

```bibtex
@article{bai2025special,
  title={Special Characters Attack: Toward Scalable Training-Data Extraction From Large Language Models},
  author={Bai, et al.},
  year={2025}
}
```

## Troubleshooting

### Out of Memory
- Reduce `--buffer-size` and `--chunk-size`
- Use `--format binary` for more compact output
- Generate in smaller batches

### Slow Performance
- Increase `--workers` up to CPU count
- Use `--buffer-size 100000` or higher
- Ensure you're using processes (default) not threads for CPU-bound work

### Non-Deterministic Output
- Always specify `--seed` for reproducible results
- Note that parallelism may affect output order (but not content)