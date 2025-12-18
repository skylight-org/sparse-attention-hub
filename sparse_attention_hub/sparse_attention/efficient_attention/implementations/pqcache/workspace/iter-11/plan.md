# Iteration 11: Eliminate torch.arange with Custom Triton Kernel

## 🎯 Goal

Replace all `torch.arange` calls (which cause scattered kernel launches) with a single custom Triton kernel that generates all indices and weights in one pass.

## 🔍 Problem Analysis

### Current torch.arange Usage (Iter-9)

```python
# 4-5 torch.arange calls causing separate kernel launches:

# 1. Sink indices (line 247)
sparse_list[:, :, :actual_sink_size] = torch.arange(actual_sink_size, ...).view(1, 1, -1)

# 2. Window indices (line 250)  
sparse_list[:, :, end:] = torch.arange(window_start, sk, ...).view(1, 1, -1)

# 3. Batch indices (line 259)
batch_indices = torch.arange(b, ...).view(b, 1, 1)

# 4. Head indices (line 260)
head_indices = torch.arange(h, ...).view(1, h, 1)

# 5. Advanced indexing (line 261) - causes many internal launches
weight_list[batch_indices, head_indices, sparse_list] = 1.0
```

**Each torch.arange**:
- Launches a separate CUDA kernel (~18 μs each)
- Has Python dispatch overhead (~10 μs)
- Creates temporary tensors (allocation overhead)

**Total cost**: ~145 μs from profile data!

## 🚀 Solution: Single Triton Kernel

Write one kernel that does EVERYTHING:
1. Generate sink indices: [0, 1, ..., sink_size-1]
2. Copy heavy hitter indices from topk
3. Generate window indices: [sk-window_size, ..., sk-1]
4. Set weights to 1.0 for all attended positions

### Kernel Design

```python
@triton.jit
def generate_sparse_indices_and_weights_kernel(
    # Outputs
    sparse_list_ptr,     # [b, h, total_attended]
    weight_list_ptr,     # [b, h, sk]
    # Inputs
    topk_indices_ptr,    # [b, h, heavy_size]
    # Dimensions
    b: tl.constexpr,
    h: tl.constexpr,
    sk,
    sink_size,
    heavy_size,
    window_size,
    init_offset,
    # Strides
    ...
):
    """Generate ALL indices and weights in a single kernel pass."""
    
    # Each program handles one (batch, head) pair
    pid = tl.program_id(0)
    batch_idx = pid // h
    head_idx = pid % h
    
    # Calculate base pointers
    sparse_base = sparse_list_ptr + batch_idx * sparse_stride_b + head_idx * sparse_stride_h
    weight_base = weight_list_ptr + batch_idx * weight_stride_b + head_idx * weight_stride_h
    topk_base = topk_indices_ptr + batch_idx * topk_stride_b + head_idx * topk_stride_h
    
    total_attended = sink_size + heavy_size + window_size
    
    # Process in blocks
    BLOCK_SIZE: tl.constexpr = 256
    
    for block_start in range(0, total_attended, BLOCK_SIZE):
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < total_attended
        
        # Determine which region each offset belongs to
        in_sink = offsets < sink_size
        in_heavy = (offsets >= sink_size) & (offsets < sink_size + heavy_size)
        in_window = offsets >= sink_size + heavy_size
        
        # Generate index values
        # Sink: just the offset itself
        sink_vals = offsets
        
        # Heavy: load from topk_indices and add init_offset
        heavy_offset = offsets - sink_size
        heavy_vals = tl.load(topk_base + heavy_offset, mask=in_heavy, other=0) + init_offset
        
        # Window: window_start + (offset - sink_size - heavy_size)
        window_start = sk - window_size
        window_vals = window_start + (offsets - sink_size - heavy_size)
        
        # Select the right value based on region
        sparse_vals = tl.where(in_sink, sink_vals,
                      tl.where(in_heavy, heavy_vals, window_vals))
        
        # Store to sparse_list
        tl.store(sparse_base + offsets, sparse_vals, mask=mask)
        
        # Set weights to 1.0 at these indices
        # This is a scatter operation: weight_list[sparse_vals] = 1.0
        for i in range(BLOCK_SIZE):
            if block_start + i < total_attended:
                idx = tl.load(sparse_base + block_start + i)
                tl.store(weight_base + idx, 1.0)
```

### Launch Parameters

```python
# One thread block per (batch, head) pair
grid = (b * h,)

generate_sparse_indices_and_weights_kernel[grid](
    sparse_list, weight_list, topk_indices,
    b, h, sk, sink_size, heavy_size, window_size, init_offset,
    ...strides...
)
```

## 📊 Expected Performance

### Kernel Count Reduction

```
Current (iter-9): ~29 kernel launches
├─ pq_score_kernel_v6:  1 launch
├─ topk internals:     15 launches
├─ torch.arange:        4 launches ← ELIMINATE!
├─ advanced indexing:   8 launches ← ELIMINATE!
└─ misc:                1 launch

After (iter-11): ~17 kernel launches (-41%)
├─ pq_score_kernel_v6:         1 launch
├─ topk internals:            15 launches
├─ generate_indices_weights:   1 launch ← NEW!
└─ (torch.arange eliminated)

Saved: 12 kernel launches!
```

### Time Breakdown

```
Current overhead from torch.arange + indexing:
├─ torch.arange:        ~72 μs (4 calls × 18 μs)
├─ Advanced indexing:   ~159 μs (weight_list[...] = 1.0)
├─ Launch overhead:     ~30 μs (12 launches × 2.5 μs)
└─ Total:               ~261 μs

After custom kernel:
├─ generate_indices:    ~30-40 μs (single optimized kernel)
├─ Launch overhead:     ~3 μs (1 launch)
└─ Total:               ~33-43 μs

Expected gain: ~218-228 μs (82% reduction in this component!)
```

### Overall Performance

```
Iter-9:    465 μs
├─ CUDA:   171 μs
└─ Overhead: 294 μs
    ├─ arange + indexing: 261 μs ← TARGET
    └─ other: 33 μs

Iter-11:   ~250-270 μs (estimated)
├─ CUDA:   171 + 40 = 211 μs (PQ + indices kernel)
└─ Overhead: ~40-60 μs (minimal!)

Expected improvement: ~195-215 μs (42-46% faster!)
```

## 🎓 Advantages

### 1. Eliminate All torch.arange Calls
- No Python dispatch overhead
- No temporary tensor allocations
- No separate kernel launches

### 2. Fused Weight Assignment
- weight_list[...] = 1.0 done in same kernel
- No advanced indexing overhead
- Direct scatter operation

### 3. Single Kernel Launch
- Minimal overhead (~3 μs vs ~30 μs)
- Better GPU utilization
- Predictable performance

### 4. Optimized Memory Access
- Coalesced writes to sparse_list
- Efficient scatter to weight_list
- No intermediate buffers

## ⚠️ Challenges

### 1. Scatter Operation

```python
# This is tricky in Triton:
weight_list[sparse_vals[i]] = 1.0

# Need to do it carefully to avoid race conditions
# Multiple threads might write to same location (OK since all write 1.0)
```

**Solution**: Use atomic operations or accept races (writing 1.0 is idempotent)

### 2. Variable-length Regions

```python
# sink_size, heavy_size, window_size can vary
# Need to handle different sizes efficiently
```

**Solution**: Use masks and conditional logic in kernel

### 3. Memory Ordering

```python
# Must initialize weight_list to zeros first
# Then scatter 1.0s

weight_list = torch.zeros((b, h, sk), ...)  # Still need this
```

**Alternative**: Have kernel do both initialization AND scatter (slower but cleaner)

## 📝 Implementation Steps

1. Write the Triton kernel
2. Test with simple cases (fixed sizes)
3. Add dynamic size handling
4. Optimize memory access patterns
5. Profile and compare

## ✅ Success Criteria

1. ✅ Eliminate all torch.arange calls
2. ✅ Eliminate advanced indexing overhead
3. ✅ Reduce kernel launches by 12
4. ✅ Improve wall-clock time by 180-220 μs
5. ✅ Pass correctness tests
6. ✅ Lower variance than iter-9

## 🎯 Realistic Expectation

**Conservative estimate**: 280-300 μs (40% faster than iter-9)
**Optimistic estimate**: 250-270 μs (45% faster than iter-9)

This would finally bring us close to the theoretical minimum!

