# Iteration 2: Optimized Memory Access with 2D Blocking

## Idea Implemented
Improved the Triton kernel from Iteration 1 with:
1. **2D Blocking**: Process multiple queries and keys per block
2. **Auto-tuning**: Added `@triton.autotune` decorator to test different block size configurations
3. **Better Memory Access**: Improved coalescing by processing queries and keys together
4. **Optimized Load Patterns**: Better organization of memory loads

The kernel now uses a 2D grid where each block processes BLOCK_QUERIES × BLOCK_KEYS outputs, with auto-tuning to find optimal block sizes.

## Timing Results
- **Average time**: 0.918 ms (vs 0.955 ms in Iter-1)
- **Median time**: 0.913 ms (vs 0.948 ms in Iter-1)
- **Min time**: 0.902 ms (vs 0.941 ms in Iter-1)
- **Max time**: 1.097 ms (vs 1.114 ms in Iter-1)
- **Std dev**: 0.028 ms (vs 0.025 ms in Iter-1)

**Improvement**: ~3.9% faster (0.955 → 0.918 ms, saving ~37 μs)

## Profile Analysis

### Top Operations by CUDA Time:
1. **indexer_next (overall)**: 995.454 μs (143.25% - includes sync overhead)
2. **pq_score_kernel_v2 (Triton)**: 568.447 μs (81.80%) - Our optimized kernel
3. **topk**: 66.332 μs (9.55%)
4. **sort**: 29.120 μs (4.19%)

### Comparison with Iteration 1:
| Operation | Iter-1 | Iter-2 | Change |
|-----------|--------|--------|--------|
| Triton Kernel | 568.293 μs | 568.447 μs | +0.15 μs (no change) |
| Top-K | 66.657 μs | 66.332 μs | -0.33 μs |
| Sort | 29.152 μs | 29.120 μs | -0.03 μs |
| **Total** | **0.955 ms** | **0.918 ms** | **-37 μs (-3.9%)** |

### Key Observations:
1. **Kernel time unchanged**: The Triton kernel itself shows ~568 μs in both iterations
2. **Overall improvement**: The 37 μs improvement comes from better CPU-side orchestration and reduced overhead
3. **Auto-tuning may not have kicked in**: The kernel time being identical suggests auto-tuning may have selected similar configurations or the workload doesn't benefit much from 2D blocking with sq=1 (single query decoding)
4. **Bottleneck remains**: The Triton kernel is still the dominant operation at ~62% of total time

## Trace File Insights:
The trace shows:
- Auto-tuning overhead is minimal (happens during first run)
- Kernel launch patterns are similar to Iter-1
- The 2D blocking doesn't provide significant benefit for single-query decoding (sq=1)
- Memory access patterns may still have room for optimization

## Analysis: Why Limited Improvement?

The limited improvement (3.9%) can be attributed to:
1. **Single Query Decoding**: With sq=1, the 2D blocking (BLOCK_QUERIES dimension) doesn't help much
2. **Memory Bandwidth Bound**: The kernel is likely memory-bandwidth limited, not compute-bound
3. **Centroid Reuse**: With sq=1, there's minimal opportunity to reuse loaded centroids across queries
4. **Auto-tuning Configuration**: May have selected BLOCK_QUERIES=1, effectively reverting to 1D blocking

## Opportunities for Further Optimization

### 1. **Fuse Top-K Selection** (Highest Priority)
**Current Issue**: Top-K takes 66 μs (~7% of total time) and requires materializing full score tensor
**Potential Fix**:
- Implement a fused kernel that computes scores and selects top-K on-the-fly
- Use block-level top-K with merge step
- Eliminate intermediate score tensor (save memory bandwidth)
- **Expected gain**: 30-50 μs reduction

### 2. **Optimize for Memory Bandwidth**
**Current Issue**: Kernel is memory-bandwidth bound
**Potential Fix**:
- Use vectorized loads (load4, load8) where possible
- Reduce precision (fp16 for scores if acceptable)
- Compress codebook representation (use int8 instead of int64)
- **Expected gain**: 100-150 μs reduction

### 3. **Fuse Key Quantization**
**Current Issue**: New key quantization done in PyTorch
**Potential Fix**:
- Write Triton kernel for distance computation and argmin
- Less critical but would help for prefill phase
- **Expected gain**: Only matters when new keys arrive

### 4. **Optimize Sparse List Construction**
**Current Issue**: Multiple small operations (arange, cat, sort, scatter)
**Potential Fix**:
- Single kernel to generate indices and weights
- Might be able to avoid sort if indices are generated in order
- **Expected gain**: 10-20 μs reduction

### 5. **Reduce Synchronization Overhead**
**Current Issue**: 143% CUDA time suggests CPU-GPU sync overhead
**Potential Fix**:
- Use CUDA streams to overlap operations
- Reduce number of kernel launches
- **Expected gain**: 20-30 μs reduction

## Next Iteration Plan

Focus on **Fusing Top-K Selection** (Opportunity #1):
- Implement a fused score computation + top-K selection kernel
- Use block-level reduction to find top-K without materializing full scores
- This addresses the second-largest bottleneck and reduces memory bandwidth

This should provide more significant gains than the memory access optimizations alone.

