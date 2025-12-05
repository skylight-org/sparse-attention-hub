#!/usr/bin/env python3
"""Profiling script for kmeans_batched_pytorch function.

This script uses PyTorch profiler to analyze the performance of the
kmeans_batched_pytorch function with configurable tensor dimensions.
"""

import time

import torch
from torch.profiler import profile, ProfilerActivity, record_function

from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed.implementations.utils.pq_utils import (
    kmeans_batched_pytorch,
)


def profile_kmeans_batched(
    n_groups: int = 128,
    n_samples: int = 32548,
    d: int = 8,
    num_clusters: int = 64,
    max_iter: int = 20,
    device: str = "cuda",
    warmup_runs: int = 2,
) -> None:
    """Profile kmeans_batched_pytorch with specified tensor dimensions.
    
    Args:
        n_groups: Number of independent clustering problems (groups)
        n_samples: Number of samples per group
        d: Dimensionality of each sample
        num_clusters: Number of clusters for k-means
        max_iter: Maximum number of k-means iterations
        device: Device to run profiling on ('cuda' or 'cpu')
        warmup_runs: Number of warmup runs before profiling
    """
    # Setup device
    device_obj: torch.device = torch.device(device)
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device_obj = torch.device("cpu")
    
    print("=" * 80)
    print("PyTorch Profiler - kmeans_batched_pytorch")
    print("=" * 80)
    print(f"Tensor shape: (n_groups={n_groups}, n_samples={n_samples}, d={d})")
    print(f"Number of clusters: {num_clusters}")
    print(f"Max iterations: {max_iter}")
    print(f"Device: {device_obj}")
    print(f"Warmup runs: {warmup_runs}")
    print("=" * 80)
    print()
    
    # Create random input tensor
    print("Creating input tensor...")
    data: torch.Tensor = torch.randn(n_groups, n_samples, d, dtype=torch.float32, device=device_obj)
    
    # Warmup runs
    print(f"Running {warmup_runs} warmup iterations...")
    for i in range(warmup_runs):
        with torch.no_grad():
            _, _ = kmeans_batched_pytorch(
                data=data,
                k=num_clusters,
                max_iter=max_iter,
            )
        if device == "cuda":
            torch.cuda.synchronize()
        print(f"  Warmup {i+1}/{warmup_runs} complete")
    
    print()
    print("Starting profiling...")
    print("-" * 80)
    
    # Profile with PyTorch profiler
    activities: list = [ProfilerActivity.CPU]
    if device == "cuda":
        activities.append(ProfilerActivity.CUDA)
    
    with profile(
        activities=activities,
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    ) as prof:
        with record_function("kmeans_batched_pytorch"):
            with torch.no_grad():
                centroids, codes = kmeans_batched_pytorch(
                    data=data,
                    k=num_clusters,
                    max_iter=max_iter,
                )
        if device == "cuda":
            torch.cuda.synchronize()
    
    print("-" * 80)
    print("Profiling complete!")
    print()
    
    # Measure wall clock time (average over 5 runs)
    print("=" * 80)
    print("WALL CLOCK TIME MEASUREMENT")
    print("=" * 80)
    print("Running 5 iterations to measure average wall clock time...")
    print()
    
    timing_runs: int = 5
    timings: list[float] = []
    
    for i in range(timing_runs):
        if device == "cuda":
            torch.cuda.synchronize()
        
        start_time: float = time.perf_counter()
        
        with torch.no_grad():
            _, _ = kmeans_batched_pytorch(
                data=data,
                k=num_clusters,
                max_iter=max_iter,
            )
        
        if device == "cuda":
            torch.cuda.synchronize()
        
        end_time: float = time.perf_counter()
        elapsed: float = end_time - start_time
        timings.append(elapsed)
        print(f"  Run {i+1}/{timing_runs}: {elapsed:.4f} seconds")
    
    avg_time: float = sum(timings) / len(timings)
    min_time: float = min(timings)
    max_time: float = max(timings)
    

    # Display results
    print("=" * 80)
    print("PROFILING RESULTS - Sorted by CPU Time")
    print("=" * 80)
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=20))
    print()

    print()
    print(f"Average wall clock time: {avg_time:.4f} seconds")
    print(f"Min time: {min_time:.4f} seconds")
    print(f"Max time: {max_time:.4f} seconds")
    print(f"Std dev: {(sum((t - avg_time) ** 2 for t in timings) / len(timings)) ** 0.5:.4f} seconds")
    print("=" * 80)
    print()
    
    # if device == "cuda":
    #     print("=" * 80)
    #     print("PROFILING RESULTS - Sorted by CUDA Time")
    #     print("=" * 80)
    #     print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))
    #     print()
    
    # print("=" * 80)
    # print("PROFILING RESULTS - Sorted by CPU Memory")
    # print("=" * 80)
    # print(prof.key_averages().table(sort_by="cpu_memory_usage", row_limit=20))
    # print()
    
    # if device == "cuda":
    #     print("=" * 80)
    #     print("PROFILING RESULTS - Sorted by CUDA Memory")
    #     print("=" * 80)
    #     print(prof.key_averages().table(sort_by="cuda_memory_usage", row_limit=20))
    #     print()
    
    # Export trace
    trace_file: str = "kmeans_batched_trace.json"
    prof.export_chrome_trace(trace_file)
    print(f"Trace exported to: {trace_file}")
    print("View in Chrome: chrome://tracing")
    print()
    
    # # Output shape verification
    # print("=" * 80)
    # print("OUTPUT VERIFICATION")
    # print("=" * 80)
    # print(f"Centroids shape: {centroids.shape} (expected: [{n_groups}, {num_clusters}, {d}])")
    # print(f"Codes shape: {codes.shape} (expected: [{n_groups}, {n_samples}])")
    # print(f"Centroids dtype: {centroids.dtype}")
    # print(f"Codes dtype: {codes.dtype}")
    # print("=" * 80)


if __name__ == "__main__":
    # Profile with the requested dimensions
    profile_kmeans_batched(
        n_groups=128,
        n_samples=32548,
        d=8,
        num_clusters=64,
        max_iter=20,
        device="cuda" if torch.cuda.is_available() else "cpu",
        warmup_runs=2,
    )

