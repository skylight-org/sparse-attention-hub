# Iteration 10: Reduce Scattered Kernel Launches with torch.compile

## 🎯 Problem

From the profiler trace, we see **29 kernel launches** causing scattered overhead:
- 8× `torch.arange` calls (145 μs total)
- 6× `aten::copy_` operations (95 μs)
- 1× `aten::index_put_` (159 μs)
- 29× `cudaLaunchKernel` overhead (149 μs)

**Total overhead from scattered operations: ~250-300 μs**

## 🔍 Root Cause

Lines 247-261 launch many separate kernels:
```python
# Each line = separate kernel launch!
sparse_list[:, :, :actual_sink_size] = torch.arange(...)           # Launch #1
sparse_list[:, :, start:end] = torch.arange(...)                   # Launch #2
batch_indices = torch.arange(b, ...)                               # Launch #3
head_indices = torch.arange(h, ...)                                # Launch #4
weight_list[batch_indices, head_indices, sparse_list] = 1.0       # Launch #5
```

Each launch has overhead:
- Python call: ~5-8 μs
- PyTorch dispatch: ~10-15 μs
- CUDA API: ~5-8 μs
- **Total: ~20-30 μs per launch**

With 29 launches: **29 × 25 μs = 725 μs overhead!**

(Though actual is lower due to async execution and batching)

## 🚀 Solution: torch.compile

**torch.compile** will:
1. Trace these operations during first run
2. Fuse them into optimized kernels
3. Reduce kernel launches from 29 → ~5-10
4. Eliminate Python/PyTorch dispatch overhead

### Expected Impact

```
Current (iter-9):
├─ CUDA operations:  171 μs
├─ Launch overhead:  149 μs (29 launches)
├─ PyTorch ops:      145 μs (scattered arange, copy, etc.)
└─ Total:            465 μs

With torch.compile:
├─ CUDA operations:  171 μs (same)
├─ Launch overhead:   50 μs (5-10 launches after fusion)
├─ PyTorch ops:       80 μs (fused operations)
└─ Total:            ~300-320 μs

Expected gain: 145-165 μs (31-35% improvement!)
```

## 📝 Implementation

Just wrap the function!

```python
from iter_9.optimized_indexer import __indexer_next
import torch

# That's it!
__indexer_next_compiled = torch.compile(
    __indexer_next,
    mode="reduce-overhead",
    fullgraph=False  # Allow fallback for Triton kernel
)
```

## 🎓 Why This Works

torch.compile uses **TorchInductor** which:

1. **Traces execution**: Records all PyTorch operations
2. **Builds computation graph**: Identifies data dependencies
3. **Fuses operations**: Combines multiple ops into single kernels
4. **Generates code**: Creates optimized Triton/CUDA kernels
5. **Caches compiled code**: Reuses for same input shapes

### What Gets Fused

**Before (29 launches):**
```
torch.arange(sink) → GPU launch #1
  → view → reshape
torch.arange(window) → GPU launch #2
  → view → reshape
sparse_list[:,:,:] = ... → GPU launch #3
sparse_list[:,:,start:] = ... → GPU launch #4
torch.zeros(weight_list) → GPU launch #5
torch.arange(b) → GPU launch #6
torch.arange(h) → GPU launch #7
weight_list[...] = 1.0 → GPU launch #8
... (21 more internal launches)
```

**After compilation (5-10 launches):**
```
[Fused kernel 1: Generate all indices]
  - sink_indices
  - window_indices  
  - batch_indices
  - head_indices
  
[Fused kernel 2: Assign to sparse_list]
  - Combines all slicing operations
  
[Fused kernel 3: Generate weight_list]
  - Zeros + index assignment in one go

[pq_score_kernel_v6: unchanged]
[topk: unchanged]
```

## ⚠️ Considerations

### 1. First-run Overhead
- Compilation takes 10-30 seconds on first call
- Subsequent calls are fast
- Not a problem for production (compile once, use many times)

### 2. Shape Changes
- Recompiles if input shapes change
- Cache compiled versions for common shapes
- Our use case: shapes are usually constant (good!)

### 3. Debugging
- Compiled code harder to debug
- Keep original version for development
- Use `torch.compiler.disable()` when debugging

## 📊 Success Criteria

1. ✅ Reduce kernel launches from 29 → 5-10
2. ✅ Reduce wall-clock time by 100-150 μs
3. ✅ Maintain correctness (10/10 tests pass)
4. ✅ No code changes needed (just wrapper)

## 🔄 Alternative: Manual Kernel Fusion

If torch.compile doesn't work well, we can write a Triton kernel:

```python
@triton.jit
def generate_indices_kernel(
    sparse_list_ptr, weight_list_ptr,
    topk_indices_ptr, b, h, sk,
    sink_size, heavy_size, window_size, init_offset, ...
):
    # Generate ALL indices in single kernel
    # - Sink: [0, 1, 2, ..., sink_size-1]
    # - Heavy: topk_indices + init_offset
    # - Window: [sk-window_size, ..., sk-1]
    # - Set weights to 1.0 for all attended positions
    ...
```

But torch.compile is much easier and likely sufficient!

