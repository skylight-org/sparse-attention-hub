# Iteration 1: Baseline Tritonization

## Goal
Convert the core computation bottlenecks in `__indexer_next` to Triton kernels to leverage GPU parallelism.

## Analysis of Current Implementation
The imperative code has several key computational sections:

1. **Key Quantization (lines 102-181)**: Quantizes new keys using PQ centroids
   - Reshapes new keys into subvectors
   - Computes distances to centroids
   - Finds nearest centroid (argmin)

2. **Score Computation (lines 183-255)**: Computes attention scores using PQ lookup
   - Reshapes queries into subvectors
   - Computes Q @ Centroids.T 
   - Gathers scores using codebook indices
   - Sums across subvectors

3. **Top-K Selection (lines 257-290)**: Selects heavy hitter tokens
   - Masks window positions
   - Performs top-K operation

4. **Sparse List Construction (lines 292-337)**: Builds final attention pattern
   - Concatenates indices
   - Scatter operation for weights

## Optimization Strategy for Iter-1

### Target: Score Computation (highest impact)
The score computation involves:
- Query-centroid matrix multiplication
- Codebook-based gathering
- Reduction across subvectors

This is the most compute-intensive part and runs for every decoding step.

### Implementation Plan
1. **Fuse Q@Centroids.T + Gather + Reduce** into a single Triton kernel
   - Input: queries [b, h, sq, n_subvec, subvec_d], centroids [b, h, n_subvec, cent_cnt, subvec_d], codebook [b, h, n_subvec, n_clustered]
   - Output: scores [b, h, sq, n_clustered]
   - Each block computes scores for a subset of (batch, head, query, key) combinations
   - Reduce across subvectors within the kernel to avoid intermediate memory

2. **Keep other operations in PyTorch** for correctness in first iteration
   - Key quantization (less frequent, only for new keys)
   - Top-K selection (already well-optimized in PyTorch)
   - Sparse list construction (simple operations)

### Expected Benefits
- Reduced memory bandwidth (no intermediate tensors)
- Better cache locality
- Fusion of gather + reduce operations

