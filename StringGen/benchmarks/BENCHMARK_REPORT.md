# SCA String Generator - Benchmark Report

## System Information
- **Date**: 2025-07-08
- **Platform**: macOS 15.3.1 (ARM64)
- **Processor**: ARM (Apple Silicon)
- **Python Version**: 3.11.13
- **CPU Cores**: 16 (physical and logical)
- **Total Memory**: 64 GB
- **Available Memory**: 35 GB

## Performance Summary

### 1. Strategy Performance Comparison (1000 char sequences)

| Strategy | Sequences/Second | Throughput |
|----------|-----------------|------------|
| INSET1   | 10,421          | 10.42 MB/s |
| INSET2   | 10,138          | 10.14 MB/s |
| CROSS1   | 10,844          | 10.84 MB/s |
| CROSS2   | 11,081          | 11.08 MB/s |
| CROSS3   | 8,974           | 8.97 MB/s  |
| **Total**| **51,457**      | **51.46 MB/s** |

### 2. Scalability Analysis (Worker Count Impact)

| Workers | Total Seq/s | Efficiency | Best Strategy |
|---------|-------------|------------|---------------|
| 1       | 38,694      | 100%       | INSET1 (21,506/s) |
| 2       | 49,700      | 64.2%      | INSET1 (21,145/s) |
| 4       | 59,783      | 38.6%      | INSET1 (20,468/s) |
| 8       | 66,460      | 21.5%      | INSET1 (17,818/s) |
| 16      | 53,324      | 8.6%       | INSET1 (12,054/s) |

**Note**: Performance peaks at 8 workers, suggesting optimal parallelization at half the CPU count.

### 3. Thread vs Process Comparison

| Mode      | Total Seq/s | INSET1    | Other Strategies |
|-----------|-------------|-----------|------------------|
| Processes | 51,457      | 10,421/s  | ~10,000/s avg    |
| Threads   | 235,456     | 213,995/s | ~5,400/s avg     |

**Key Insight**: Threads show 20x better performance for INSET1 (simple string multiplication) but worse performance for complex strategies due to Python's GIL.

### 4. Sequence Length Impact

| Length | INSET1 | INSET2 | CROSS1 | CROSS2 | CROSS3 |
|--------|--------|--------|--------|--------|--------|
| 10     | 11,990 | 12,843 | 13,107 | 12,771 | 13,159 |
| 100    | 13,005 | 12,558 | 13,055 | 13,060 | 12,822 |
| 1,000  | 12,924 | 10,483 | 11,441 | 11,150 | 8,794  |
| 10,000 | 12,817 | 5,102  | 5,106  | 4,255  | 2,710  |

**Observations**:
- INSET1 maintains consistent performance across all lengths
- Complex strategies (INSET2, CROSS*) degrade significantly with longer sequences
- Optimal performance for most strategies at 100-character sequences

### 5. Large-Scale Generation Test

- **Task**: Generate 1 million sequences of 100 characters
- **Time**: 18.35 seconds
- **Rate**: 54,505 sequences/second
- **Throughput**: 5.45 MB/s

## Key Findings

1. **INSET1 Optimization**: String multiplication in Python is highly optimized, making INSET1 the fastest strategy, especially with threads.

2. **Parallel Efficiency**: Best performance achieved with 8 workers on a 16-core system, indicating overhead from process management beyond this point.

3. **Memory vs CPU Bound**: 
   - Short sequences (≤100 chars): CPU-bound, benefits from parallelization
   - Long sequences (≥1000 chars): Memory-bound, limited parallelization benefit

4. **Strategy Complexity Impact**:
   - Simple strategies (INSET1): Consistent performance
   - Sampling strategies (INSET2, CROSS1): Moderate degradation
   - Complex strategies (CROSS2, CROSS3): Significant degradation with length

## Recommendations

1. **For Maximum Throughput**: Use 8 workers with process-based parallelization
2. **For INSET1 Only**: Use thread-based execution for 20x performance boost
3. **For Large Datasets**: Keep sequence lengths ≤1000 characters for optimal performance
4. **For Production**: Use `--workers 8` for balanced performance across all strategies

## Benchmark Commands Used

```bash
# Basic benchmark
python3 sca_generator.py --benchmark

# Worker scaling test
python3 sca_generator.py --benchmark --workers [1,2,4,8,16]

# Thread comparison
python3 sca_generator.py --benchmark --use-threads

# Large-scale test
python3 sca_generator.py --mode sample --count 1000000 --length 100
```