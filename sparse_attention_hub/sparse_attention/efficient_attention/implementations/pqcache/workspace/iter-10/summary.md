# Iteration 10 Summary: torch.compile for Operation Fusion

## 🎯 Optimization Idea

Use `torch.compile` to automatically fuse scattered PyTorch operations and reduce kernel launches from 29 → 15.

## 📊 Performance Results

### Timing Statistics (50 runs)
```
Average time: 0.475 ms (475 μs)
Median time:  0.446 ms (446 μs)
Min time:     0.425 ms (425 μs)
Max time:     1.112 ms (1112 μs)
Std dev:      0.101 ms (101 μs)
```

### Comparison with Iteration 9

| Metric | Iteration 9 (no compile) | Iteration 10 (torch.compile) | Change |
|--------|-------------------------|------------------------------|---------|
| **Average** | 465 μs | **475 μs** | **+10 μs (2.2% WORSE)** ❌ |
| **Median** | 458 μs | **446 μs** | **-12 μs (2.6% better)** ✅ |
| **Min** | 447 μs | **425 μs** | **-22 μs (4.9% better)** ✅ |
| **Max** | 714 μs | **1112 μs** | **+398 μs (WORSE)** ❌ |
| **Std dev** | 38 μs | **101 μs** | **+63 μs (unstable)** ❌ |

### CUDA Operations Breakdown

```
CUDA operations: 186 μs (was 171 μs - 9% WORSE!)
├─ pq_score_kernel_v6: 113 μs (was 85 μs - 33% WORSE!)
├─ topk:                61 μs (was 62 μs - similar)
└─ other:               12 μs (fused operations)

Total CUDA: 186 μs
Kernel launches: 15 (was 29 - 48% reduction! ✅)
```

## 🤔 What Happened?

### The Good News ✅
1. **Kernel launches reduced**: 29 → 15 (48% reduction)
2. **Operations were fused**: Multiple `torch.arange`, `copy_`, etc. combined
3. **Minimum time improved**: 425 μs (best case is better)

### The Bad News ❌
1. **Average time WORSE**: 475 μs vs 465 μs (+10 μs)
2. **CUDA time increased**: 186 μs vs 171 μs (+15 μs) 
3. **PQ kernel slower**: 113 μs vs 85 μs (+28 μs!)
4. **High variance**: 101 μs std dev vs 38 μs (unstable)

## 🔍 Root Cause Analysis

### Why is PQ Kernel Slower?

The **pq_score_kernel_v6** got SLOWER (85 → 113 μs), which is surprising!

**Hypothesis**:
1. **torch.compile interference**: The compiler might be wrapping/modifying the Triton kernel call
2. **Memory layout changes**: Compiled code may change tensor layouts/strides
3. **Launch overhead**: Compiled region adds overhead around kernel launch
4. **Profiler artifacts**: First compilation runs might skew results

From the profile:
```
pq_score_kernel_v6_0: 113 μs  ← torch.compile version
pq_score_kernel_v6:   113 μs  ← The actual kernel
```

The kernel itself is the same, but the compiled wrapper adds overhead!

### Why Average is Worse?

```
Overhead breakdown:
Iter-9:  465 μs = 171 μs CUDA + 294 μs overhead
Iter-10: 475 μs = 186 μs CUDA + 289 μs overhead

CUDA got worse (+15 μs) more than overhead improved (-5 μs)
Net result: +10 μs slower
```

**The compilation overhead around the Triton kernel negates the fusion benefits!**

## 📈 Trace Analysis

### Kernel Reduction (Good!)

**Before (iter-9): 29 launches**
```
cudaLaunchKernel:     29 calls
├─ pq_score_kernel_v6:  1 launch
├─ topk internals:     15 launches  
├─ torch.arange:        8 launches
├─ copy operations:     6 launches
└─ misc:                2 launches
```

**After (iter-10): 15 launches**
```
cudaLaunchKernel:     15 calls (48% reduction!)
├─ pq_score_kernel_v6:  1 launch
├─ topk internals:     15 launches
└─ [fused operations]:  0 launches ← Compiled!
```

The `torch.arange` and `copy` operations were successfully fused!

### But Triton Kernel Got Slower

```
Profile shows:
"Call CompiledFxGraph fo3mk72mnuarhsdjvke6z44oho4g...": 621 μs CUDA time

This is the compiled wrapper, which shows 621 μs but includes
overlapping operations. The actual kernel time is 113 μs.
```

The compiler is adding overhead when calling the Triton kernel!

## 🎓 Lessons Learned

### 1. torch.compile Doesn't Always Help

While it **reduced kernel launches by 48%**, the overall performance got **worse** because:
- Triton kernels don't need compilation (already optimized)
- Compiler adds overhead around external kernels
- Memory layout changes can hurt performance

### 2. Fusion ≠ Faster

We successfully fused operations (arange, copy, etc.), but:
- The fusion benefit was small (~5 μs)
- The overhead from compiled Triton calls was large (+28 μs)
- **Net result**: Slower!

### 3. Profiler Shows What Matters

The trace clearly showed:
- ✅ Fewer kernel launches
- ❌ Slower Triton kernel execution
- ❌ Higher variance

Without profiling, we might have assumed compilation helped!

### 4. When torch.compile Works

torch.compile is great for:
- ✅ Pure PyTorch operations
- ✅ Many small operations to fuse
- ✅ CPU-heavy code

torch.compile is BAD for:
- ❌ Already-optimized custom kernels (Triton, CUDA)
- ❌ Code where custom kernels dominate time
- ❌ Complex control flow with caching

## 🚀 What To Try Next

### Option 1: Selective Compilation (Best!)

Only compile the PyTorch parts, not the Triton kernel:

```python
# Split into two functions
@torch.compile
def generate_indices_and_weights(topk_indices, b, h, sk, ...):
    """Pure PyTorch - compile this!"""
    sparse_list = torch.empty(...)
    sparse_list[:, :, :sink_size] = torch.arange(...)
    # ... all the scattered operations
    weight_list[batch_indices, head_indices, sparse_list] = 1.0
    return sparse_list, weight_list

def __indexer_next(...):
    # ... preprocessing ...
    
    # Call Triton kernel directly (no compilation)
    pq_score_kernel_v6[grid](...)
    
    # TopK
    topk_indices = torch.topk(...)
    
    # Compiled index generation
    sparse_list, weight_list = generate_indices_and_weights(...)
    
    return ...
```

Expected: ~420-430 μs (better than both iter-9 and iter-10!)

### Option 2: CUDA Graphs (Medium effort)

Skip torch.compile, use CUDA graphs instead:
- Reduces launch overhead without compilation
- No interference with Triton kernels
- More predictable performance

Expected: ~330-350 μs

### Option 3: Custom Triton Kernel (Hard)

Write a single Triton kernel for index generation + weight assignment:
- No PyTorch operations at all
- Maximum control
- Best performance

Expected: ~280-310 μs

## ✅ Verdict

**Iteration 10 is WORSE than Iteration 9** ❌

- Average: 475 μs vs 465 μs (+2.2%)
- CUDA time: 186 μs vs 171 μs (+8.8%)
- Variance: 101 μs vs 38 μs (less stable)

**Why**: torch.compile adds overhead around the Triton kernel that exceeds the fusion benefits.

**Recommendation**: 
- ❌ Don't use iter-10 in production
- ✅ Stick with iter-9 (465 μs)
- ✅ Try selective compilation (Option 1 above)
- ✅ Or implement CUDA graphs (Option 2)

## 🎯 Key Insight

**torch.compile is not a silver bullet!**

It helps when you have:
- Many small PyTorch operations
- CPU-heavy dispatch overhead
- Pure PyTorch code

It hurts when you have:
- Custom kernels (Triton, CUDA)
- Already-optimized code
- Complex caching/control flow

**For our case**: The PQ kernel dominates (85 μs), and torch.compile interferes with it. Better to optimize around it, not through it!

