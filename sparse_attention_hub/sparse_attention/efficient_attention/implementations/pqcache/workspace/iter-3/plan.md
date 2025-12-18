# Iteration 3: Fused Score Computation + Top-K Selection

## Goal
Fuse the score computation and top-K selection into a single kernel to eliminate the intermediate score tensor and reduce memory bandwidth.

## Analysis of Iteration 2
From the profile results:
- `pq_score_kernel_v2` takes 568 μs (~62% of total time)
- `topk` takes 66 μs (~7% of total time)
- Combined: 634 μs (~69% of total time)
- Current approach materializes full score tensor [b, h, sq, n_clustered] before top-K

## Current Bottlenecks
1. **Memory Bandwidth**: Writing and reading full score tensor
2. **Two-Pass Algorithm**: Compute all scores, then find top-K
3. **Wasted Computation**: Compute scores for all keys but only need top-K

## Optimization Strategy

### Fused Kernel Design
Instead of:
1. Kernel 1: Compute all scores → [b, h, sq, n_clustered]
2. Kernel 2: Find top-K → [b, h, sq, k]

Do:
1. Single Kernel: Compute scores and maintain top-K heap on-the-fly → [b, h, sq, k]

### Implementation Approach

#### Option A: Block-Level Top-K with Merge (Chosen)
1. Each block computes scores for a subset of keys
2. Maintains local top-K heap in shared memory
3. Final merge step combines block-level results
4. More complex but better for large k

#### Option B: Streaming Top-K
1. Process keys in chunks
2. Update global top-K after each chunk
3. Simpler but requires atomic operations

### Detailed Plan
1. **Phase 1**: Each block processes BLOCK_KEYS keys
   - Compute scores for BLOCK_KEYS keys
   - Store in shared memory
   - Find local top-K using parallel reduction

2. **Phase 2**: Merge block-level top-K results
   - Each query has multiple block-level top-K lists
   - Merge using heap or sorting network
   - Output final top-K indices

3. **Optimizations**:
   - Use shared memory for local top-K heaps
   - Vectorized score computation
   - Efficient merge using bitonic sort or heap

## Expected Benefits
- **Memory Bandwidth**: Save writing/reading full score tensor
  - Current: Write n_clustered scores, read n_clustered scores
  - New: Only write k indices
  - Savings: ~2 × n_clustered × 4 bytes (for fp32) = ~2 × 31872 × 4 = ~255 KB per query
  
- **Reduced Kernel Launches**: One kernel instead of two

- **Target Performance**:
  - Current: 568 μs (score) + 66 μs (top-K) = 634 μs
  - Target: 350-400 μs (fused)
  - Expected gain: 200-250 μs (~30-40% improvement)

## Implementation Complexity
This is significantly more complex than previous iterations:
- Need to implement efficient top-K selection in Triton
- Handle merge logic for block-level results
- Careful shared memory management
- May need multiple kernel variants for different k values

## Fallback Plan
If fused top-K proves too complex or doesn't yield expected gains:
- Fall back to optimizing memory bandwidth (vectorized loads, fp16)
- Or implement approximate top-K (faster but slightly less accurate)

