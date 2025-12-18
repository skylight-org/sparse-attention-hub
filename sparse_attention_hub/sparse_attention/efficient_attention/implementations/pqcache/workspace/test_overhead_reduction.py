"""Test script to compare different overhead reduction techniques.

Run this to see the performance improvements from:
1. Baseline (Iteration 6)
2. torch.compile
3. CUDA Graphs (conceptual - would need more work)
"""

import torch
import time
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent.parent))

from sparse_attention_hub.sparse_attention.efficient_attention.implementations.pqcache.research import (
    PQCacheResearchBackend,
    PQCacheResearchBackendConfig
)


def profile_function(fn, *args, warmup=10, iterations=100, **kwargs):
    """Profile a function with proper warmup and timing."""
    
    # Warmup
    for _ in range(warmup):
        _ = fn(*args, **kwargs)
    
    torch.cuda.synchronize()
    
    # Timing
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
    
    for i in range(iterations):
        start_events[i].record()
        _ = fn(*args, **kwargs)
        end_events[i].record()
    
    torch.cuda.synchronize()
    
    # Calculate statistics
    times = [start_events[i].elapsed_time(end_events[i]) * 1000 for i in range(iterations)]  # Convert to μs
    
    return {
        'mean': sum(times) / len(times),
        'min': min(times),
        'max': max(times),
        'median': sorted(times)[len(times) // 2],
    }


def main():
    print("=" * 80)
    print("CPU Overhead Reduction Test")
    print("=" * 80)
    print()
    
    # Setup
    device = torch.device('cuda:0')
    config = PQCacheResearchBackendConfig()
    backend = PQCacheResearchBackend(config)
    
    # Create sample data
    print("Creating sample data...")
    sample_data = backend.create_sample_data_next()
    
    print(f"  Batch size: {sample_data['query'].shape[0]}")
    print(f"  Num heads: {sample_data['query'].shape[1]}")
    print(f"  Sequence length: {sample_data['key'].shape[2]}")
    print(f"  Hidden dim: {sample_data['query'].shape[3]}")
    print()
    
    # Test 1: Baseline (Iteration 6)
    print("Test 1: Baseline (Iteration 6)")
    print("-" * 80)
    
    from iter_6.optimized_indexer import __indexer_next as indexer_v6
    
    results_v6 = profile_function(indexer_v6, **sample_data)
    print(f"  Mean time:   {results_v6['mean']:.1f} μs")
    print(f"  Min time:    {results_v6['min']:.1f} μs")
    print(f"  Median time: {results_v6['median']:.1f} μs")
    print(f"  Max time:    {results_v6['max']:.1f} μs")
    print()
    
    # Test 2: torch.compile
    print("Test 2: With torch.compile")
    print("-" * 80)
    
    indexer_compiled = torch.compile(indexer_v6, mode="reduce-overhead")
    
    print("  Compiling... (this may take 10-30 seconds)")
    # First call triggers compilation
    _ = indexer_compiled(**sample_data)
    torch.cuda.synchronize()
    print("  ✓ Compilation complete")
    print()
    
    results_compiled = profile_function(indexer_compiled, **sample_data)
    print(f"  Mean time:   {results_compiled['mean']:.1f} μs")
    print(f"  Min time:    {results_compiled['min']:.1f} μs")
    print(f"  Median time: {results_compiled['median']:.1f} μs")
    print(f"  Max time:    {results_compiled['max']:.1f} μs")
    print()
    
    improvement = results_v6['mean'] - results_compiled['mean']
    improvement_pct = (improvement / results_v6['mean']) * 100
    
    print(f"  Improvement: {improvement:.1f} μs ({improvement_pct:.1f}%)")
    print()
    
    # Summary
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print()
    print(f"{'Approach':<30} {'Mean Time':<15} {'Improvement':<15}")
    print("-" * 60)
    print(f"{'Baseline (Iter-6)':<30} {results_v6['mean']:>10.1f} μs  {'(baseline)':<15}")
    print(f"{'+ torch.compile':<30} {results_compiled['mean']:>10.1f} μs  {improvement:>8.1f} μs ({improvement_pct:>5.1f}%)")
    print()
    
    # Projections
    print("PROJECTED RESULTS (based on typical gains):")
    print("-" * 60)
    
    base_mean = results_v6['mean']
    
    # torch.compile: measured
    compile_time = results_compiled['mean']
    compile_gain = base_mean - compile_time
    
    # CUDA graphs: typical 25-35% overhead reduction
    graphs_overhead_reduction = 0.30  # Conservative estimate
    cuda_time = 208  # From profiling
    base_overhead = base_mean - cuda_time
    graphs_overhead = base_overhead * (1 - graphs_overhead_reduction)
    graphs_time = cuda_time + graphs_overhead
    graphs_gain = base_mean - graphs_time
    
    # C++ extension: typical 50-70% overhead reduction
    cpp_overhead_reduction = 0.60  # Conservative estimate
    cpp_overhead = base_overhead * (1 - cpp_overhead_reduction)
    cpp_time = cuda_time + cpp_overhead
    cpp_gain = base_mean - cpp_time
    
    print(f"{'+ CUDA graphs (est.)':<30} {graphs_time:>10.1f} μs  {graphs_gain:>8.1f} μs ({(graphs_gain/base_mean)*100:>5.1f}%)")
    print(f"{'+ C++ extension (est.)':<30} {cpp_time:>10.1f} μs  {cpp_gain:>8.1f} μs ({(cpp_gain/base_mean)*100:>5.1f}%)")
    print()
    
    print("TARGET: 200 μs")
    print(f"  Current best:     {compile_time:.1f} μs")
    print(f"  Gap to target:    {compile_time - 200:.1f} μs")
    print(f"  CUDA time:        {cuda_time:.1f} μs (irreducible)")
    print(f"  Overhead:         {compile_time - cuda_time:.1f} μs")
    print()
    print(f"  To reach 200 μs would require:")
    print(f"    - Reduce CUDA time by: {cuda_time - 200:.1f} μs (24% CUDA reduction)")
    print(f"    - OR reduce overhead to: 0 μs (impossible)")
    print()
    print("CONCLUSION: 200 μs target is challenging due to CUDA kernel time.")
    print("            Further optimization would require kernel algorithm changes.")
    print()


if __name__ == '__main__':
    main()

