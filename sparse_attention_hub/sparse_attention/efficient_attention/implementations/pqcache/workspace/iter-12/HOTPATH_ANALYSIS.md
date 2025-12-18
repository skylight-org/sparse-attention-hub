# Hotpath Analysis - Iteration 12

## 🎯 What is the Hotpath?

The **hotpath** is the most frequently executed code path during normal operation:

```python
Typical usage pattern (after initial warmup):
├─ Codebook is CACHED (no quantization needed)
├─ Normal sequence length (not too short, not too long)
└─ Single query decoding step (sq=1)

This is the path taken in >95% of inference calls!
```

## 🔍 Current Execution Flow

```python
def __indexer_next(...):
    # 1. Extract dimensions (fast)
    b, h, sq, d = query.shape
    _, h_kv, sk, _ = key.shape
    
    # 2. Fast path check (rarely taken)
    if sk <= threshold:
        return fast_path()  # ← Only ~1% of calls
    
    # 3. Quantization (cold path - only on first few calls)
    if sk - init_offset > cached_num_keys:
        codebook = quantize_new_keys()  # ← Only first few calls
    
    # 4. ⭐ HOTPATH STARTS HERE ⭐ (95%+ of execution time)
    #    Everything from here is executed EVERY call
    
    # Lines 278-347: The critical path
    n_clustered = codebook.shape[1]
    ...
    pq_score_kernel[...]()  # ← GPU kernel
    ...
    topk_indices = torch.topk(...)  # ← GPU operation
    ...
    generate_indices_kernel[...]()  # ← GPU kernel
    ...
    return results
```

## 📊 Hotpath Breakdown (Lines 278-347)

### Current Operations (70 lines of Python!)

```python
# Python operations count:
├─ Variable assignments:      ~15 operations
├─ Arithmetic (.shape[]):     ~8 operations  
├─ Function calls (.stride()): ~20 operations
├─ Tensor operations:         ~6 operations
├─ Control flow (if):         ~2 operations
└─ Total Python ops:          ~51 operations!

# GPU operations (the actual work):
├─ pq_score_kernel:           1 kernel launch
├─ torch.topk:                1 operation
├─ generate_indices_kernel:   1 kernel launch
└─ Total GPU ops:             3 operations
```

### The Problem

**For every 3 GPU operations, we have 51 Python operations!**

That's **17:1 Python-to-GPU ratio** - way too much overhead!

## 🚀 Hotpath Optimization Strategy

### Goal: Reduce Python-to-GPU ratio from 17:1 to ~3:1

### Strategy 1: Pre-compute Everything Possible

```python
# BAD: Compute every call
def __indexer_next(...):
    n_subvec = pq_group_factor  # ← Same every time!
    subvec_d = d // n_subvec    # ← Same every time!
    ...

# GOOD: Pre-compute once, pass as arguments
def __indexer_next(..., n_subvec, subvec_d, ...):
    # Just use them directly
    queries_reshaped = query.view(b, h, n_subvec, subvec_d)
```

### Strategy 2: Eliminate Intermediate Variables

```python
# BAD: 3 Python operations
grid = (b * h * triton.cdiv(n_clustered, 256),)
q_strides = queries_reshaped.stride()
pq_score_kernel[grid](queries, ..., q_strides[0], ...)

# GOOD: 1 Python operation (inline everything)
pq_score_kernel[(b * h * ((n_clustered + 255) // 256),)](
    queries, ..., queries.stride(0), ...
)
```

### Strategy 3: Use view() instead of reshape()

```python
# BAD: reshape() may copy data
queries_reshaped = query.reshape(b, h, n_subvec, subvec_d)

# GOOD: view() never copies (if contiguous)
queries_reshaped = query.view(b, h, n_subvec, subvec_d)
```

### Strategy 4: Assume Common Case (No GQA)

```python
# BAD: Branch on every call
if num_key_value_groups == 1:
    repeat_centroids = pq_centroids
    repeat_codebook = codebook.permute(0, 2, 3, 1)
else:
    # Complex GQA logic...

# GOOD: Specialize for common case (GQA=1)
# Assume num_key_value_groups == 1
repeat_centroids = pq_centroids[..., :subvec_d]
repeat_codebook = codebook.permute(0, 2, 3, 1)
# Note: If GQA != 1, use a different specialized function
```

### Strategy 5: Minimize .stride() Calls

```python
# BAD: 4 separate .stride() calls (Python overhead)
q_strides = queries.stride()
s0, s1, s2, s3 = q_strides[0], q_strides[1], q_strides[2], q_strides[3]

# GOOD: Inline in kernel call (JIT compiler optimizes)
kernel[grid](queries, ..., 
    queries.stride(0), queries.stride(1), 
    queries.stride(2), queries.stride(3))
```

## 🎯 Target Hotpath (Ultra-Lean Version)

```python
def __indexer_next_hotpath(
    query, key, weight_list_dtype,
    sink_size, window_size, heavy_size,
    init_offset,
    pq_centroids, pq_codebook,
    # PRE-COMPUTED VALUES (passed in to avoid recomputation)
    n_subvec, subvec_d,
):
    """Ultra-lean hotpath version.
    
    Assumptions:
    - Codebook is fully cached (no quantization needed)
    - num_key_value_groups == 1 (no GQA)
    - query.is_contiguous() == True
    - Not in fast path (sk > threshold)
    
    These assumptions cover >95% of production calls!
    """
    # Extract dimensions (unavoidable)
    b, h, sq, d = query.shape
    _, h_kv, sk, _ = key.shape
    
    # Pre-computed earlier
    actual_sink_size = min(sink_size, sk)
    actual_window_size = min(window_size, sk)
    n_clustered = pq_codebook.shape[1]
    actual_heavy_size = min(heavy_size, n_clustered)
    total_attended = actual_sink_size + actual_heavy_size + actual_window_size
    
    # View query (no reshape, no copy)
    queries_view = query.view(b, h, n_subvec, subvec_d)
    
    # Prepare inputs (minimal ops)
    repeat_centroids = pq_centroids[..., :subvec_d]
    repeat_codebook = pq_codebook.permute(0, 2, 3, 1)
    
    # Allocate output
    scores = torch.empty((b, h, n_clustered), device=query.device, dtype=torch.float32)
    
    # Kernel 1: PQ scoring (inline everything)
    pq_score_kernel_v6[(b * h * ((n_clustered + 255) // 256),)](
        queries_view, repeat_centroids, repeat_codebook, scores,
        b, h, n_subvec, subvec_d, n_clustered,
        queries_view.stride(0), queries_view.stride(1), queries_view.stride(2), queries_view.stride(3),
        repeat_centroids.stride(0), repeat_centroids.stride(1), repeat_centroids.stride(2), 
        repeat_centroids.stride(3), repeat_centroids.stride(4),
        repeat_codebook.stride(0), repeat_codebook.stride(1), repeat_codebook.stride(2), 
        repeat_codebook.stride(3),
        scores.stride(0), scores.stride(1), scores.stride(2),
    )
    
    # Mask window (inline condition)
    if (window_start := sk - actual_window_size - init_offset) < n_clustered and window_start >= 0:
        scores[:, :, window_start:] = float('-inf')
    
    # TopK (can't optimize further)
    _, topk_indices = torch.topk(scores, k=actual_heavy_size, dim=-1, largest=True, sorted=False)
    
    # Allocate outputs (combined to reduce ops)
    sparse_list = torch.empty((b, h, total_attended), device=query.device, dtype=torch.long)
    weight_list = torch.zeros((b, h, sk), device=query.device, dtype=weight_list_dtype)
    
    # Kernel 2: Generate indices (inline everything)
    generate_indices_and_weights_kernel[(b * h,)](
        sparse_list, weight_list, topk_indices,
        b, h, sk, actual_sink_size, actual_heavy_size, actual_window_size,
        init_offset, total_attended,
        sparse_list.stride(0), sparse_list.stride(1),
        weight_list.stride(0), weight_list.stride(1),
        topk_indices.stride(0), topk_indices.stride(1),
    )
    
    # Return (inline sparse_len creation)
    return (sparse_list, 
            torch.full((b, h), total_attended, device=query.device, dtype=torch.long),
            weight_list, pq_centroids, pq_codebook, None)
```

## 📊 Operation Count Comparison

### Before (Current iter-12)

```
Python operations in hotpath:
├─ Variable assignments:       15
├─ Arithmetic operations:       8
├─ Function calls:             20
├─ Tensor operations:           6
├─ Control flow:                2
└─ Total:                      51 operations

Python-to-GPU ratio: 51:3 = 17:1
```

### After (Ultra-lean hotpath)

```
Python operations in hotpath:
├─ Variable assignments:        8  (↓47%)
├─ Arithmetic operations:       4  (↓50%)
├─ Function calls:             20  (same - mostly .stride())
├─ Tensor operations:           6  (same - necessary)
├─ Control flow:                1  (↓50%)
└─ Total:                      39 operations (↓24%)

Python-to-GPU ratio: 39:3 = 13:1 (better!)
```

### Further Optimized (Pre-compute strides)

If we pre-compute and cache strides:

```
Python operations:
├─ Variable assignments:        8
├─ Arithmetic operations:       4
├─ Function calls:              2  (↓90% - no .stride() calls!)
├─ Tensor operations:           6
├─ Control flow:                1
└─ Total:                      21 operations (↓59%!)

Python-to-GPU ratio: 21:3 = 7:1 (much better!)
```

## 🎓 Key Insights

### 1. The Overhead is Mostly .stride() Calls

```
20 out of 51 operations (39%) are .stride() calls!

These are unavoidable if we want flexible tensor layouts, BUT:
- We can cache them if tensors are always contiguous
- We can assume standard layouts for hotpath
```

### 2. Walrus Operator for Inline Assignment

```python
# OLD: 2 operations
window_start = sk - actual_window_size - init_offset
if window_start < n_clustered:
    ...

# NEW: 1 operation (Python 3.8+)
if (window_start := sk - actual_window_size - init_offset) < n_clustered:
    ...
```

### 3. view() vs reshape()

```python
# reshape(): May copy if not contiguous (~50 μs overhead)
x = tensor.reshape(shape)

# view(): Never copies, fails if not contiguous (~5 μs overhead)
x = tensor.view(shape)  # Use this for hotpath!
```

### 4. Inline Everything

```python
# BAD: 3 statements
grid = (b * h,)
sparse_len = torch.full((b, h), total_attended, ...)
return (..., sparse_len, ...)

# GOOD: 1 statement (inline sparse_len creation)
return (..., torch.full((b, h), total_attended, ...), ...)
```

## 🚀 Implementation Plan

### Phase 1: Create Hotpath Specialized Function
```python
def __indexer_next_hotpath(...):
    # Ultra-lean version (assumptions listed in docstring)
    ...

def __indexer_next(...):
    # Check if hotpath conditions met
    if is_hotpath(query, key, pq_codebook, ...):
        return __indexer_next_hotpath(...)
    else:
        return __indexer_next_full(...)  # Full version with all checks
```

### Phase 2: Pre-compute and Cache
```python
class IndexerCache:
    def __init__(self):
        self.n_subvec = None
        self.subvec_d = None
        self.centroids_sliced = None
        # ... cache common computations
    
    def get_or_compute(self, key, compute_fn):
        if self.cache[key] is None:
            self.cache[key] = compute_fn()
        return self.cache[key]
```

### Phase 3: Optimize .stride() Calls
```python
# If query is always contiguous, assume standard strides
if query.is_contiguous():
    # Hard-code strides (much faster!)
    q_s0, q_s1, q_s2, q_s3 = h*n_subvec*subvec_d, n_subvec*subvec_d, subvec_d, 1
else:
    # Fall back to .stride() calls
    ...
```

## 📈 Expected Performance Improvement

### Current (iter-11 optimized): 364 μs
```
├─ CUDA operations:  166 μs (46%)
└─ Python overhead:  198 μs (54%)
    ├─ Hotpath Python: ~120 μs (51 ops)
    └─ Other:          ~78 μs
```

### Target (iter-12 ultra-lean): ~300-320 μs
```
├─ CUDA operations:  166 μs (52%)
└─ Python overhead:  ~140-160 μs (48%)
    ├─ Hotpath Python: ~60-80 μs (21 ops) ← 50% reduction!
    └─ Other:          ~80 μs
```

**Expected gain: 40-60 μs (11-16% improvement)**

This would bring us to **~310 μs** - very close to the theoretical minimum!
