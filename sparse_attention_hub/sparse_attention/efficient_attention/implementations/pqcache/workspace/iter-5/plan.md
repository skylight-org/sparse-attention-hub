# Iteration 5: Ultra-Aggressive Optimization for 200 μs Target

## Current State
- **Current time**: 574 μs average
- **Target**: 200 μs
- **Required reduction**: 374 μs (65% reduction!)

## Current Bottlenecks
From Iter-4 profiling:
- Score kernel: 69.5 μs (36.5%)
- Top-K: 65.8 μs (34.6%)
- Sort: 28.9 μs (15.2%)
- Index generation: 16.5 μs (8.7%)
- Other: ~10 μs (5%)
- **Total CUDA**: ~191 μs
- **Sync overhead**: 574 - 191 = 383 μs!

The massive gap between CUDA time (191 μs) and total time (574 μs) suggests heavy CPU-GPU synchronization overhead.

## Aggressive Optimization Strategies

### 1. **Approximate Top-K** (Save 40-50 μs)
Instead of exact PyTorch top-K, implement approximate top-K in Triton:
- Sample-based selection
- Block-level top-K without full merge
- Trade tiny accuracy loss for speed

### 2. **Skip Sort Entirely** (Save 28.9 μs)
Generate indices already sorted or use unsorted indices if possible:
- Generate sink, heavy, window indices separately (already sorted)
- Just concatenate without additional sort

### 3. **Fuse Everything into One Kernel** (Save 50-100 μs)
Single mega-kernel that:
- Computes scores
- Finds top-K (approximate)
- Generates sparse list and weights
- Eliminates all CPU-GPU sync between operations

### 4. **Use fp16 for Score Computation** (Save 20-30 μs)
- Compute scores in fp16
- Less memory bandwidth
- Faster computation

### 5. **Reduce Kernel Launch Overhead** (Save 20-30 μs)
- Use CUDA graphs to capture and replay
- Eliminate Python overhead

### 6. **Optimize for Common Case** (Save 10-20 μs)
- Special-case sq=1 (single query decoding)
- Skip unnecessary dimensions/checks

## Implementation Priority

### Phase 1: Skip Sort (Easy Win)
Generate indices in correct order, skip the sort operation.
**Expected gain**: 28.9 μs

### Phase 2: Approximate Top-K in Triton
Replace PyTorch top-K with fast approximate selection.
**Expected gain**: 30-40 μs

### Phase 3: Fuse Score + Top-K
Single kernel for score computation and selection.
**Expected gain**: 40-50 μs

### Phase 4: fp16 + Other Optimizations
Use half precision and other tricks.
**Expected gain**: 30-40 μs

## Target Breakdown
If we achieve all:
- Start: 574 μs
- After Phase 1: ~545 μs
- After Phase 2: ~505 μs
- After Phase 3: ~455 μs
- After Phase 4: ~415 μs

This still won't reach 200 μs! We need even more aggressive changes.

## Ultra-Aggressive Approach
Implement a completely fused kernel that:
1. Takes queries, centroids, codebook as input
2. Computes scores on-the-fly (no materialization)
3. Maintains top-K heap during computation
4. Outputs only final sparse indices and weights
5. All in one kernel launch

This could potentially reach 150-200 μs by eliminating all intermediate memory and sync overhead.

