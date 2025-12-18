# Iteration 6: FINAL - Ultra-Aggressive CPU Overhead Reduction

## Current State
- **Wall-clock time**: 565 μs
- **CUDA time**: 200 μs ✓ (at target)
- **CPU-GPU overhead**: 365 μs (64% of total time!)
- **Target**: 200 μs wall-clock time

## Problem Analysis

The 365 μs overhead comes from:
1. **Multiple kernel launches**: ~10-20 μs per launch × 30+ launches
2. **PyTorch operation dispatch**: ~5-10 μs per operation
3. **Python interpreter**: ~50-100 μs
4. **Memory allocations**: ~5-10 μs per allocation
5. **CPU-GPU synchronization**: implicit syncs between operations

## Ultra-Aggressive Strategy

### Eliminate ALL unnecessary operations:
1. **Remove double sort**: Don't sort topk indices separately
2. **Pre-compute index ranges**: Use views instead of arange
3. **In-place operations**: Minimize allocations
4. **Batch kernel launches**: Combine where possible
5. **Skip unnecessary checks**: Optimize hot path

### Key Insight:
The heavy indices from top-K don't need to be perfectly sorted - we can generate the final sparse list more efficiently.

## Implementation

### Phase 1: Remove topk sort
- Use unsorted topk results directly
- Let the final sort handle everything

### Phase 2: Optimize index generation  
- Use tensor slicing instead of arange where possible
- Reuse buffers

### Phase 3: Minimize allocations
- Pre-allocate all output tensors
- Use in-place operations

### Phase 4: Reduce PyTorch ops
- Combine multiple small operations
- Use more efficient alternatives

## Expected Outcome

Target breakdown:
- CUDA operations: 200 μs (already achieved)
- Reduced overhead: 150-180 μs (vs 365 μs current)
- **Total target**: 350-380 μs wall-clock time

While we may not reach exactly 200 μs wall-clock (due to framework overhead), we should get significantly closer.

