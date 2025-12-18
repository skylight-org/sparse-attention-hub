# Iteration 7: MEGA-KERNEL FUSION - Go Big!

## Goal
Fuse EVERYTHING into minimal kernel launches to approach 200 μs wall-clock time.

## Radical Changes

### 1. Fused Score + Top-K Kernel
**Current**: Compute all scores (208 μs) → Top-K (62 μs) = 270 μs + overhead
**New**: Single kernel that:
- Streams through keys computing scores
- Maintains top-K heap in shared memory
- Never materializes full score tensor
- Outputs only top-K indices
**Expected**: ~120-150 μs (eliminate 120 μs of memory bandwidth)

### 2. Fused Index Generation Kernel
**Current**: Multiple torch operations (arange, cat, sort, scatter) = ~80 μs
**New**: Single kernel that:
- Generates sink indices
- Inserts sorted heavy indices
- Generates window indices
- Writes weights
- All in one pass
**Expected**: ~30-40 μs (save 40-50 μs)

### 3. Eliminate All PyTorch Operations
Replace ALL torch operations in hot path with Triton kernels:
- No torch.arange
- No torch.cat
- No torch.sort (generate sorted output)
- No torch.topk (fused in kernel)

### 4. Pre-allocated Buffers
Reuse the same buffers across calls to eliminate allocation overhead.

## Implementation Strategy

### Phase 1: Streaming Top-K Kernel
```
For each (batch, head):
  Initialize top-K heap in shared memory
  For each key block:
    Compute scores for block
    Update top-K heap
  Output top-K indices directly
```

### Phase 2: Single Index Generation Kernel
```
For each (batch, head):
  Write sink indices [0..sink_size)
  Merge-insert heavy indices (already sorted from top-K)
  Write window indices [window_start..sk)
  Mark weights for all attended indices
```

### Phase 3: Minimize Launches
- Only 2 kernel launches total:
  1. Fused score + top-K
  2. Fused index generation + weights

## Expected Performance

| Component | Current (Iter-6) | Target (Iter-7) | Savings |
|-----------|------------------|-----------------|---------|
| Score kernel | 87 μs | - | (fused) |
| Top-K | 62 μs | - | (fused) |
| **Fused Score+TopK** | - | 120 μs | 29 μs |
| Sort | 29 μs | 0 μs | 29 μs |
| Index gen | 40 μs | 30 μs | 10 μs |
| Other ops | 30 μs | 10 μs | 20 μs |
| **Total CUDA** | 208 μs | **160 μs** | **48 μs** |
| CPU overhead | 310 μs | 200 μs | 110 μs |
| **TOTAL** | 518 μs | **360 μs** | **158 μs** |

## Big Bet
By eliminating intermediate memory and fusing operations, we can:
- Save ~50 μs in CUDA time
- Save ~110 μs in CPU overhead (fewer launches, less dispatch)
- **Target: 360 μs wall-clock time** (much closer to 200 μs!)

This is the "go big" approach - complex but potentially transformative.

