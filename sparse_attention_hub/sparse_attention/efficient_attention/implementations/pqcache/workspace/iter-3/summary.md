# Iteration 3: Vectorized Dimension Processing

## Idea Implemented
Improved the Triton kernel with vectorized dimension processing:
1. **Block-level dimension processing**: Process BLOCK_D dimensions at a time instead of one
2. **Vectorized loads**: Load multiple dimensions in a single operation
3. **2D tensor operations**: Load centroids as [BLOCK_KEYS, BLOCK_D] and compute dot products efficiently
4. **Better memory coalescing**: Improved memory access patterns for centroid loading

The key insight was to load centroid values for all keys in a 2D pattern and use broadcasting to compute dot products, rather than processing keys sequentially.

## Timing Results
- **Average time**: 0.796 ms (vs 0.918 ms in Iter-2, 0.955 ms in Iter-1)
- **Median time**: 0.781 ms (vs 0.913 ms in Iter-2, 0.948 ms in Iter-1)
- **Min time**: 0.757 ms (vs 0.902 ms in Iter-2, 0.941 ms in Iter-1)
- **Max time**: 1.266 ms (vs 1.097 ms in Iter-2, 1.114 ms in Iter-1)
- **Std dev**: 0.073 ms (vs 0.028 ms in Iter-2, 0.025 ms in Iter-1)

**Improvement over Iter-2**: ~13.3% faster (0.918 → 0.796 ms, saving ~122 μs)
**Improvement over Iter-1**: ~16.6% faster (0.955 → 0.796 ms, saving ~159 μs)

## Profile Analysis

### Top Operations by CUDA Time:
1. **indexer_next (overall)**: 1.441 ms (733.63% - includes sync overhead)
2. **pq_score_kernel_v3 (Triton)**: 70.784 μs (36.03%) - **Massive improvement!**
3. **topk**: 65.890 μs (33.54%)
4. **sort**: 28.896 μs (14.71%)

### Comparison Across Iterations:
| Operation | Iter-1 | Iter-2 | Iter-3 | Change (1→3) |
|-----------|--------|--------|--------|--------------|
| Triton Kernel | 568.293 μs | 568.447 μs | **70.784 μs** | **-497.5 μs (-87.5%)** |
| Top-K | 66.657 μs | 66.332 μs | 65.890 μs | -0.8 μs (-1.2%) |
| Sort | 29.152 μs | 29.120 μs | 28.896 μs | -0.3 μs (-0.9%) |
| **Total** | **0.955 ms** | **0.918 ms** | **0.796 ms** | **-159 μs (-16.6%)** |

### Key Observations:
1. **Dramatic kernel speedup**: The Triton kernel went from ~568 μs to ~71 μs - an **8x speedup**!
2. **Bottleneck shift**: Top-K is now the dominant operation at 65.9 μs (33.5% of CUDA time)
3. **Memory bandwidth**: The vectorized approach dramatically reduced memory traffic
4. **Auto-tuning effective**: The auto-tuner likely found a good configuration with BLOCK_D parameter

## Analysis: Why Such a Large Improvement?

The 8x kernel speedup can be attributed to:

1. **Vectorized Memory Access**: 
   - Old: Load one dimension at a time, iterate over keys sequentially
   - New: Load BLOCK_D dimensions for all BLOCK_KEYS at once
   - Benefit: Better memory coalescing and bandwidth utilization

2. **Reduced Memory Transactions**:
   - Old: n_subvec × subvec_d × BLOCK_KEYS separate loads
   - New: n_subvec × (subvec_d / BLOCK_D) loads of size [BLOCK_KEYS × BLOCK_D]
   - Benefit: Fewer transactions, better cache utilization

3. **Better Compute Utilization**:
   - Broadcasting operations (q_vals[None, :] * c_vals) are highly optimized
   - GPU can parallelize the dot product computation across keys

4. **Optimal Block Sizes**:
   - Auto-tuning found good BLOCK_KEYS and BLOCK_D values
   - Balanced occupancy and memory access patterns

## Trace File Insights:
The trace shows:
- Kernel execution time is now minimal compared to other operations
- Good GPU utilization during the kernel
- Auto-tuning overhead is amortized across runs
- The kernel is no longer the bottleneck

## Current Performance Breakdown:
- **Top-K**: 65.9 μs (33.5%) - Now the largest bottleneck
- **Triton Kernel**: 70.8 μs (36.0%) - Dramatically reduced
- **Sort**: 28.9 μs (14.7%)
- **Other ops**: ~30 μs (15.8%)

## Opportunities for Further Optimization

### 1. **Fuse Top-K Selection** (Highest Priority)
**Current Issue**: Top-K takes 65.9 μs and is now the largest single operation
**Potential Fix**:
- Implement fused score computation + top-K selection
- Avoid materializing full score tensor
- Use streaming top-K or block-level selection
- **Expected gain**: 40-50 μs reduction

### 2. **Optimize Sort Operation**
**Current Issue**: Sort takes 28.9 μs (14.7%)
**Potential Fix**:
- Generate indices in sorted order if possible
- Use a more efficient sorting algorithm for small arrays
- Consider if sorting is necessary (might be able to use unsorted indices)
- **Expected gain**: 10-15 μs reduction

### 3. **Reduce CPU-GPU Synchronization**
**Current Issue**: 733% CUDA time suggests significant sync overhead
**Potential Fix**:
- Use CUDA streams to overlap operations
- Batch operations where possible
- Reduce number of kernel launches
- **Expected gain**: 20-30 μs reduction

### 4. **Optimize Sparse List Construction**
**Current Issue**: Multiple small operations (arange, cat, scatter)
**Potential Fix**:
- Single fused kernel for index generation and weight scatter
- **Expected gain**: 10-15 μs reduction

### 5. **Further Kernel Optimizations**
**Current Issue**: Kernel is now very fast but could still be improved
**Potential Fix**:
- Use shared memory for centroid caching
- Experiment with different BLOCK_D values
- Consider fp16 computation if precision allows
- **Expected gain**: 10-20 μs reduction

## Next Iteration Plan

Two possible directions:

**Option A (Recommended)**: Focus on **Fusing Top-K Selection**
- This is now the largest bottleneck
- Would eliminate intermediate score tensor
- Complex but high-impact optimization

**Option B**: Focus on **Multiple Small Optimizations**
- Optimize sort (generate sorted indices)
- Reduce synchronization overhead
- Fuse sparse list construction
- Easier to implement, cumulative 40-60 μs gain

Given the complexity of fused top-K and the diminishing returns, **Option B** might be more practical for the next iteration. We can target getting below 0.75 ms total time.

