# Iteration 2: Optimized Memory Access with Shared Memory

## Goal
Improve the Triton kernel from Iteration 1 by optimizing memory access patterns using shared memory.

## Analysis of Iteration 1
From the profile results:
- `pq_score_kernel` takes 568 μs (~60% of total time)
- The kernel loads centroids repeatedly for each query
- Memory bandwidth is likely a bottleneck

## Current Kernel Issues
1. **Redundant Centroid Loads**: Each program loads the same centroids multiple times
2. **No Shared Memory Usage**: Centroids could be cached in shared memory
3. **Suboptimal Block Size**: Using BLOCK_CLUSTERED=128 without tuning
4. **Sequential Dimension Access**: Loading query/centroid values one dimension at a time

## Optimization Strategy

### 1. Shared Memory for Centroids
- Load centroids into shared memory once per block
- All threads in a block can reuse these centroids
- Reduces global memory traffic significantly

### 2. Improved Blocking Strategy
- Process multiple queries per block (2D blocking)
- Share centroid loads across queries
- Better amortization of load costs

### 3. Vectorized Loads
- Load multiple dimensions at once where possible
- Use tl.load with appropriate vector widths
- Better memory coalescing

### 4. Block Size Tuning
- Test BLOCK_CLUSTERED values: 64, 128, 256
- Test BLOCK_QUERY values: 1, 2, 4
- Find optimal configuration for typical workloads

## Implementation Plan
1. Rewrite kernel with 2D blocking (queries × keys)
2. Add shared memory for centroid caching
3. Implement vectorized loads for query/centroid values
4. Add auto-tuning decorator for block sizes

## Expected Benefits
- 2-3x reduction in global memory traffic
- Better GPU occupancy with optimized block sizes
- Target: Reduce kernel time from 568 μs to ~250-300 μs

