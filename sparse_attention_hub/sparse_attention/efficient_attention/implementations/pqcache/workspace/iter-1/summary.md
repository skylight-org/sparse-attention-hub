# Iteration 1: Triton Kernel for Score Computation

## Idea Implemented
Implemented a Triton kernel (`pq_score_kernel`) to fuse the PQ-based score computation, which includes:
1. Computing Q @ Centroids.T for all subvectors
2. Gathering scores using codebook indices
3. Summing across subvectors

This replaces the PyTorch operations that involved:
- Matrix multiplication (`torch.matmul`)
- Gather operation (`torch.gather`)
- Reduction (`sum`)

## Timing Results
- **Average time**: 0.955 ms
- **Median time**: 0.948 ms
- **Min time**: 0.941 ms
- **Max time**: 1.114 ms
- **Std dev**: 0.025 ms

## Profile Analysis

### Top Operations by CUDA Time:
1. **indexer_next (overall)**: 1.351 ms (194.29% - includes sync overhead)
2. **pq_score_kernel (Triton)**: 568.293 μs (81.73%) - Our custom kernel
3. **topk**: 66.657 μs (9.59%) - PyTorch's top-K selection
4. **sort**: 29.152 μs (4.19%) - Sorting sparse indices
5. **radixSortKVInPlace**: 29.152 μs (4.19%) - Part of sort operation

### Key Observations:
1. **Triton kernel dominates**: The `pq_score_kernel` takes ~568 μs, which is ~60% of the total execution time
2. **Top-K is second bottleneck**: Takes ~67 μs (7% of total time)
3. **Sort operation**: Takes ~29 μs (3% of total time)
4. **Other operations**: Index creation, scatter for weights are relatively cheap

### Trace File Insights:
The trace file (`profile_indexer_next_trace.json`) shows the detailed timeline of operations. Key points:
- The Triton kernel launches successfully and executes
- There's good GPU utilization during the score computation phase
- CPU-GPU synchronization overhead is present (seen in the 194% metric)

## Opportunities for Further Optimization

### 1. Optimize Triton Kernel Memory Access Pattern
**Current Issue**: The kernel loads centroids multiple times for different queries
**Potential Fix**: 
- Use shared memory to cache centroids for a block of queries
- Adjust block sizes to maximize occupancy
- Consider tiling strategies for better cache utilization

### 2. Fuse Top-K Selection into Triton
**Current Issue**: Top-K is done in PyTorch after score computation
**Potential Fix**:
- Implement a fused kernel that computes scores and immediately selects top-K
- This would eliminate the need to materialize the full score tensor
- Could use block-level reduction followed by merge

### 3. Optimize Key Quantization Path
**Current Issue**: When new keys arrive, quantization is done in PyTorch
**Potential Fix**:
- Write a Triton kernel for distance computation and argmin
- This is less critical as it only runs when new keys arrive (not every decode step)

### 4. Fuse Sparse List Construction
**Current Issue**: Multiple small operations (arange, cat, sort, scatter)
**Potential Fix**:
- Combine index generation and weight scatter into a single kernel
- The sort operation might be hard to fuse but could be optimized

### 5. Tune Triton Kernel Block Sizes
**Current Issue**: Using BLOCK_CLUSTERED=128 without tuning
**Potential Fix**:
- Profile with different block sizes (64, 128, 256)
- Consider 2D blocking (batch/head dimension + key dimension)
- Auto-tune based on input dimensions

## Next Iteration Plan
Focus on **Optimizing Triton Kernel Memory Access** (Opportunity #1):
- Add shared memory caching for centroids
- Experiment with different block sizes
- Profile memory bandwidth utilization

This should give the most immediate performance improvement since the kernel is already the dominant operation.

