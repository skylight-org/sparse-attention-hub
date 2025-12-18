# Iteration 4: Aggressive Multi-Front Optimization

## Goal
Implement multiple optimizations simultaneously to maximize performance gains:
1. Eliminate unnecessary operations
2. Optimize memory allocations
3. Reduce top-K overhead
4. Minimize PyTorch operation overhead

## Current Bottlenecks (from Iter-3)
- Top-K: 65.9 μs (33.5%)
- Triton kernel: 70.8 μs (36.0%)
- Sort: 28.9 μs (14.7%)
- Index creation & scatter: ~30 μs (15.8%)

## Optimization Strategies

### 1. Eliminate Clone Operation
Current code does `mask_scores = scores.clone()` before masking. This is expensive.
**Fix**: Mask in-place or pass mask to topk directly.

### 2. Optimize Index Generation
Current code creates sink_indices, window_indices with expand operations.
**Fix**: Create indices more efficiently, potentially in a single kernel.

### 3. Reduce Memory Allocations
Multiple tensor allocations throughout the code.
**Fix**: Pre-allocate or reuse tensors where possible.

### 4. Skip Unnecessary Operations
- The sort operation might not be strictly necessary
- Weight scatter could be optimized

### 5. Use Better Top-K
Instead of masking then top-K, implement a custom selection that avoids the window region.

### 6. Further Kernel Optimization
- Use shared memory for better cache utilization
- Optimize for specific common cases

## Implementation Plan

1. **Remove clone**: Mask directly in top-K input
2. **Fused index generation**: Create all indices in one operation
3. **Skip sort if possible**: Check if consumers require sorted indices
4. **Optimize weight scatter**: Use more efficient indexing
5. **Kernel improvements**: Add shared memory caching

## Target
Achieve < 0.6 ms average time (25% improvement over Iter-3)

