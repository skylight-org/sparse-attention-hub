#!/usr/bin/env python3
"""
Profile the custom_attention function of ResearchAttention using PyTorch profiler.

This script creates a sample sparse attention configuration with multiple maskers
and profiles the custom_attention function to analyze performance bottlenecks.
"""

import os
import sys
import time
import statistics
import torch
from torch import nn
from torch.profiler import profile, record_function, ProfilerActivity

# Try to import flash attention
try:
    from flash_attn import flash_attn_func
    FLASH_ATTN_AVAILABLE = True
except ImportError:
    FLASH_ATTN_AVAILABLE = False
    print("⚠️ Flash Attention not available. Install with: pip install flash-attn")

# Add the project root to the path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from sparse_attention_hub.sparse_attention.research_attention import (
    ResearchAttention,
    ResearchAttentionConfig,
)
from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed.implementations import (
    LocalMaskerConfig,
    SinkMaskerConfig,
    OracleTopKConfig,
)
from sparse_attention_hub.sparse_attention.research_attention.maskers.sampling.implementations import (
    RandomSamplingMaskerConfig,
)


def create_sample_sparse_attention_config() -> ResearchAttentionConfig:
    """Create a comprehensive sparse attention configuration for profiling.
    
    Returns:
        ResearchAttentionConfig: Configuration with multiple maskers including
        sink, local, oracle top-k, and random sampling maskers.
    """
    # Create masker configurations
    masker_configs = [
        # Sink masker - keep first 128 tokens (global context)
        SinkMaskerConfig(sink_size=128),
        
        # Local masker - sliding window of 512 tokens
        LocalMaskerConfig(window_size=512),
        
        # Oracle top-k masker - select top 10% most important tokens
        OracleTopKConfig(heavy_size=0.1),  # 10% of sequence length
        
        ## Random sampling masker - sample 5% of remaining positions
        #RandomSamplingMaskerConfig(sampling_rate=0.05),
    ]
    
    return ResearchAttentionConfig(masker_configs=masker_configs)


def create_sample_tensors(
    batch_size: int = 1,
    num_heads: int = 32,
    num_queries: int = 1,
    seq_len: int = 4096,
    head_dim: int = 128,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> tuple:
    """Create sample input tensors for attention computation.
    
    Args:
        batch_size: Batch size for the tensors
        num_heads: Number of attention heads
        seq_len: Sequence length
        head_dim: Dimension per attention head
        device: Device to place tensors on
        
    Returns:
        Tuple of (queries, keys, values, attention_mask, module, sparse_meta_data)
    """
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    
    # Create attention tensors
    queries = torch.randn(
        batch_size, num_heads, num_queries, head_dim, 
        device=device, dtype=dtype, requires_grad=True
    )
    keys = torch.randn(
        batch_size, num_heads, seq_len, head_dim, 
        device=device, dtype=dtype, requires_grad=True
    )
    values = torch.randn(
        batch_size, num_heads, seq_len, head_dim, 
        device=device, dtype=dtype, requires_grad=True
    )
    
    # Create attention mask (optional)
    attention_mask = torch.ones(
        batch_size, num_heads, num_queries, seq_len,
        device=device, dtype=dtype
    )
    
    # Create a mock attention module
    class MockAttentionModule(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer_idx = 0
            
    module = MockAttentionModule().to(device)
    
    # Create sparse metadata (required for research attention)
    sparse_meta_data = {
        "layer_idx": 0,
        "seq_len": seq_len,
        "batch_size": batch_size,
        "num_heads": num_heads,
        "head_dim": head_dim,
    }
    
    return queries, keys, values, attention_mask, module, sparse_meta_data


def profile_custom_attention(
    num_warmup_runs: int = 5,
    num_profile_runs: int = 1,
    trace_file: str = "sample_trace.json"
) -> None:
    """Profile the ResearchAttention custom_attention function.
    
    Args:
        num_warmup_runs: Number of warmup runs before profiling
        num_profile_runs: Number of runs to profile
        trace_file: Output trace file name
    """
    print("🚀 Starting ResearchAttention custom_attention profiling...")
    
    # Setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"📱 Using device: {device}")
    
    # Create sparse attention configuration and instance
    config = create_sample_sparse_attention_config()
    research_attention = ResearchAttention.create_from_config(config)
    
    print(f"🎭 Created ResearchAttention with {len(research_attention.maskers)} maskers:")
    for i, masker in enumerate(research_attention.maskers):
        print(f"   {i+1}. {masker.__class__.__name__}")
    
    # Create sample tensors
    num_queries = 1
    seq_len = 32678  # Long context scenario
    queries, keys, values, attention_mask, module, sparse_meta_data = create_sample_tensors(
        num_queries=num_queries, seq_len=seq_len, device=device
    )
    
    print(f"📊 Tensor shapes:")
    print(f"   - Queries: {queries.shape}")
    print(f"   - Keys: {keys.shape}")
    print(f"   - Values: {values.shape}")
    print(f"   - Attention mask: {attention_mask.shape}")
    
    # Warmup runs
    print(f"🔥 Running {num_warmup_runs} warmup iterations...")
    with torch.no_grad():
        for _ in range(num_warmup_runs):
            _ = research_attention.custom_attention(
                module=module,
                queries=queries,
                keys=keys,
                values=values,
                attention_mask=attention_mask,
                scaling=1.0 / (queries.shape[-1] ** 0.5),
                dropout=0.0,
                sparse_meta_data=sparse_meta_data,
            )
            if device == "cuda":
                torch.cuda.synchronize()
    
    # Profiling runs
    print(f"📊 Profiling {num_profile_runs} iterations...")
    
    activities = [ProfilerActivity.CPU]
    if device == "cuda":
        activities.append(ProfilerActivity.CUDA)
    
    with profile(
        activities=activities,
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
        with_flops=True,
        with_modules=True,
    ) as prof:
        with record_function("research_attention_profiling"):
            with torch.no_grad():
                for i in range(num_profile_runs):
                    with record_function(f"iteration_{i}"):
                        attention_output, attention_weights = research_attention.custom_attention(
                            module=module,
                            queries=queries,
                            keys=keys,
                            values=values,
                            attention_mask=attention_mask,
                            scaling=1.0 / (queries.shape[-1] ** 0.5),
                            dropout=0.0,
                            sparse_meta_data=sparse_meta_data,
                        )
                    if device == "cuda":
                        torch.cuda.synchronize()
    
    # Save trace
    trace_path = os.path.join(os.path.dirname(__file__), trace_file)
    prof.export_chrome_trace(trace_path)
    print(f"✅ Trace saved to: {trace_path}")
    
    # Print key statistics
    print("\n📈 Top 10 operations by CPU time:")
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
    
    
    # Validate output shapes
    print(f"\n✅ Output validation:")
    print(f"   - Attention output shape: {attention_output.shape}")
    print(f"   - Attention weights shape: {attention_weights.shape if attention_weights is not None else 'None'}")
    
    print(f"\n🎉 Profiling completed! View trace in https://ui.perfetto.dev/")
    print(f"   Load file: {trace_path}")


def time_custom_attention(
    num_timing_runs: int = 100,
    num_warmup_runs: int = 10
) -> float:
    """Time the ResearchAttention custom_attention function with simple measurements.
    
    Args:
        num_timing_runs: Number of timing runs to average over
        num_warmup_runs: Number of warmup runs before timing
        
    Returns:
        Average time in milliseconds for sparse attention
    """
    print("\n⏱️ Starting simple timing measurements...")
    
    # Setup (reuse same configuration as profiling)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Create sparse attention configuration and instance
    config = create_sample_sparse_attention_config()
    research_attention = ResearchAttention.create_from_config(config)
    
    # Create sample tensors (same as profiling)
    num_queries = 1
    seq_len = 32678  # Long context scenario
    queries, keys, values, attention_mask, module, sparse_meta_data = create_sample_tensors(
        num_queries=num_queries, seq_len=seq_len, device=device
    )
    
    print(f"⚙️ Timing configuration:")
    print(f"   - Device: {device}")
    print(f"   - Sequence length: {seq_len}")
    print(f"   - Query length: {num_queries}")
    print(f"   - Warmup runs: {num_warmup_runs}")
    print(f"   - Timing runs: {num_timing_runs}")
    
    # Warmup runs
    print(f"🔥 Running warmup...")
    with torch.no_grad():
        for _ in range(num_warmup_runs):
            _ = research_attention.custom_attention(
                module=module,
                queries=queries,
                keys=keys,
                values=values,
                attention_mask=attention_mask,
                scaling=1.0 / (queries.shape[-1] ** 0.5),
                dropout=0.0,
                sparse_meta_data=sparse_meta_data,
            )
            if device == "cuda":
                torch.cuda.synchronize()
    
    # Timing runs
    print(f"⏱️ Running timing measurements...")
    times = []
    
    with torch.no_grad():
        for i in range(num_timing_runs):
            # Start timing
            if device == "cuda":
                torch.cuda.synchronize()
            start_time = time.perf_counter()
            
            # Run custom attention
            attention_output, attention_weights = research_attention.custom_attention(
                module=module,
                queries=queries,
                keys=keys,
                values=values,
                attention_mask=attention_mask,
                scaling=1.0 / (queries.shape[-1] ** 0.5),
                dropout=0.0,
                sparse_meta_data=sparse_meta_data,
            )
            
            # End timing
            if device == "cuda":
                torch.cuda.synchronize()
            end_time = time.perf_counter()
            
            elapsed_time = (end_time - start_time) * 1000  # Convert to milliseconds
            times.append(elapsed_time)
            
            # Progress indicator
            if (i + 1) % (num_timing_runs // 10) == 0 or i == 0:
                print(f"   Progress: {i + 1}/{num_timing_runs} runs completed")
    
    # Calculate statistics
    avg_time = statistics.mean(times)
    min_time = min(times)
    max_time = max(times)
    std_time = statistics.stdev(times) if len(times) > 1 else 0.0
    median_time = statistics.median(times)
    
    # Print results
    print(f"\n📊 Timing Results:")
    print(f"   - Average time: {avg_time:.3f} ms")
    print(f"   - Median time:  {median_time:.3f} ms")
    print(f"   - Min time:     {min_time:.3f} ms")
    print(f"   - Max time:     {max_time:.3f} ms")
    print(f"   - Std dev:      {std_time:.3f} ms")
    print(f"   - Total runs:   {num_timing_runs}")
        
    print(f"\n✅ Timing completed!")
    return avg_time


def time_baseline_empty_maskers(
    num_timing_runs: int = 100,
    num_warmup_runs: int = 10
) -> float:
    """Time ResearchAttention with empty masker list to measure pipeline overhead.
    
    Args:
        num_timing_runs: Number of timing runs to average over
        num_warmup_runs: Number of warmup runs before timing
        
    Returns:
        Average time in milliseconds for baseline (empty maskers)
    """
    print("\n🔄 Starting baseline timing (empty maskers)...")
    
    # Setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Create empty masker configuration
    empty_config = ResearchAttentionConfig(masker_configs=[])
    research_attention = ResearchAttention.create_from_config(empty_config)
    
    # Create sample tensors (same as main profiling)
    num_queries = 1
    seq_len = 32678  # Long context scenario
    queries, keys, values, attention_mask, module, sparse_meta_data = create_sample_tensors(
        num_queries=num_queries, seq_len=seq_len, device=device
    )
    
    print(f"⚙️ Baseline configuration:")
    print(f"   - Device: {device}")
    print(f"   - Maskers: 0 (empty list)")
    print(f"   - Sequence length: {seq_len}")
    
    # Warmup runs
    print(f"🔥 Running warmup...")
    with torch.no_grad():
        for _ in range(num_warmup_runs):
            _ = research_attention.custom_attention(
                module=module,
                queries=queries,
                keys=keys,
                values=values,
                attention_mask=attention_mask,
                scaling=1.0 / (queries.shape[-1] ** 0.5),
                dropout=0.0,
                sparse_meta_data=sparse_meta_data,
            )
            if device == "cuda":
                torch.cuda.synchronize()
    
    # Timing runs
    print(f"⏱️ Running baseline timing measurements...")
    times = []
    
    with torch.no_grad():
        for i in range(num_timing_runs):
            # Start timing
            if device == "cuda":
                torch.cuda.synchronize()
            start_time = time.perf_counter()
            
            # Run custom attention with empty maskers
            attention_output, attention_weights = research_attention.custom_attention(
                module=module,
                queries=queries,
                keys=keys,
                values=values,
                attention_mask=attention_mask,
                scaling=1.0 / (queries.shape[-1] ** 0.5),
                dropout=0.0,
                sparse_meta_data=sparse_meta_data,
            )
            
            # End timing
            if device == "cuda":
                torch.cuda.synchronize()
            end_time = time.perf_counter()
            
            elapsed_time = (end_time - start_time) * 1000  # Convert to milliseconds
            times.append(elapsed_time)
            
            # Progress indicator
            if (i + 1) % (num_timing_runs // 10) == 0 or i == 0:
                print(f"   Progress: {i + 1}/{num_timing_runs} runs completed")
    
    # Calculate statistics
    avg_time = statistics.mean(times)
    min_time = min(times)
    max_time = max(times)
    std_time = statistics.stdev(times) if len(times) > 1 else 0.0
    median_time = statistics.median(times)
    
    # Print results
    print(f"\n📊 Baseline Results (Empty Maskers):")
    print(f"   - Average time: {avg_time:.3f} ms")
    print(f"   - Median time:  {median_time:.3f} ms")
    print(f"   - Min time:     {min_time:.3f} ms")
    print(f"   - Max time:     {max_time:.3f} ms")
    print(f"   - Std dev:      {std_time:.3f} ms")
    
    print(f"\n✅ Baseline timing completed!")
    return avg_time


def time_flash_attention_baseline(
    num_timing_runs: int = 100,
    num_warmup_runs: int = 10
) -> float:
    """Time Flash Attention as a baseline comparison.
    
    Args:
        num_timing_runs: Number of timing runs to average over
        num_warmup_runs: Number of warmup runs before timing
        
    Returns:
        Average time in milliseconds for Flash Attention, or -1 if not available
    """
    if not FLASH_ATTN_AVAILABLE:
        print("\n❌ Flash Attention not available for baseline comparison")
        return -1.0
    
    print("\n⚡ Starting Flash Attention baseline timing...")
    
    # Setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if device == "cpu":
        print("❌ Flash Attention requires CUDA. Skipping baseline.")
        return -1.0
    
    # Create sample tensors (same dimensions as main profiling)
    num_queries = 1
    seq_len = 32678
    head_dim = 128
    num_heads = 32
    batch_size = 1
    
    dtype = torch.bfloat16
    
    # Flash attention expects different tensor layout: (batch, seq_len, num_heads, head_dim)
    queries_flash = torch.randn(
        batch_size, num_queries, num_heads, head_dim,
        device=device, dtype=dtype, requires_grad=True
    )
    keys_flash = torch.randn(
        batch_size, seq_len, num_heads, head_dim,
        device=device, dtype=dtype, requires_grad=True
    )
    values_flash = torch.randn(
        batch_size, seq_len, num_heads, head_dim,
        device=device, dtype=dtype, requires_grad=True
    )
    
    print(f"⚙️ Flash Attention configuration:")
    print(f"   - Device: {device}")
    print(f"   - Batch size: {batch_size}")
    print(f"   - Sequence length: {seq_len}")
    print(f"   - Query length: {num_queries}")
    print(f"   - Num heads: {num_heads}")
    print(f"   - Head dim: {head_dim}")
    
    # Warmup runs
    print(f"🔥 Running warmup...")
    with torch.no_grad():
        for _ in range(num_warmup_runs):
            try:
                _ = flash_attn_func(
                    q=queries_flash,
                    k=keys_flash, 
                    v=values_flash,
                    causal=False,
                    softmax_scale=1.0 / (head_dim ** 0.5)
                )
                if device == "cuda":
                    torch.cuda.synchronize()
            except Exception as e:
                print(f"❌ Flash Attention error during warmup: {e}")
                return -1.0
    
    # Timing runs
    print(f"⏱️ Running Flash Attention timing measurements...")
    times = []
    
    with torch.no_grad():
        for i in range(num_timing_runs):
            try:
                # Start timing
                if device == "cuda":
                    torch.cuda.synchronize()
                start_time = time.perf_counter()
                
                # Run Flash Attention
                flash_output = flash_attn_func(
                    q=queries_flash,
                    k=keys_flash,
                    v=values_flash,
                    causal=False,
                    softmax_scale=1.0 / (head_dim ** 0.5)
                )
                
                # End timing
                if device == "cuda":
                    torch.cuda.synchronize()
                end_time = time.perf_counter()
                
                elapsed_time = (end_time - start_time) * 1000  # Convert to milliseconds
                times.append(elapsed_time)
                
                # Progress indicator
                if (i + 1) % (num_timing_runs // 10) == 0 or i == 0:
                    print(f"   Progress: {i + 1}/{num_timing_runs} runs completed")
                    
            except Exception as e:
                print(f"❌ Flash Attention error during timing: {e}")
                return -1.0
    
    # Calculate statistics
    avg_time = statistics.mean(times)
    min_time = min(times)
    max_time = max(times)
    std_time = statistics.stdev(times) if len(times) > 1 else 0.0
    median_time = statistics.median(times)
    
    # Print results
    print(f"\n📊 Flash Attention Baseline Results:")
    print(f"   - Average time: {avg_time:.3f} ms")
    print(f"   - Median time:  {median_time:.3f} ms")
    print(f"   - Min time:     {min_time:.3f} ms")
    print(f"   - Max time:     {max_time:.3f} ms")
    print(f"   - Std dev:      {std_time:.3f} ms")
    
    print(f"\n✅ Flash Attention baseline completed!")
    return avg_time


def run_comparative_analysis(
    sparse_time: float, 
    baseline_time: float, 
    flash_time: float
) -> None:
    """Run comparative analysis between sparse attention, baseline, and flash attention.
    
    Args:
        sparse_time: Average time for sparse attention in ms
        baseline_time: Average time for empty maskers in ms  
        flash_time: Average time for flash attention in ms (-1 if not available)
    """
    print("\n🔍 Comparative Analysis:")
    print("=" * 60)
    
    # Sparse vs Baseline comparison
    if baseline_time > 0:
        overhead = sparse_time - baseline_time
        overhead_percent = (overhead / baseline_time) * 100
        print(f"📈 Masker Overhead Analysis:")
        print(f"   - Sparse Attention:     {sparse_time:.3f} ms")
        print(f"   - Baseline (no maskers): {baseline_time:.3f} ms")
        print(f"   - Masker overhead:      {overhead:.3f} ms ({overhead_percent:.1f}%)")
        
        if overhead_percent > 100:
            print(f"   ❌ High overhead - significant masker cost")
    
    # Flash Attention comparison
    if flash_time > 0:
        print(f"\n⚡ Flash Attention Comparison:")
        print(f"   - Sparse Attention:     {sparse_time:.3f} ms")
        print(f"   - Flash Attention:      {flash_time:.3f} ms")
        
        if sparse_time < flash_time:
            speedup = flash_time / sparse_time
            print(f"   🚀 Sparse is {speedup:.2f}x FASTER than Flash Attention!")
        else:
            slowdown = sparse_time / flash_time
            print(f"   🐌 Sparse is {slowdown:.2f}x slower than Flash Attention")
        
        print(f"   - Relative performance: {(flash_time/sparse_time)*100:.1f}% of sparse time")
    else:
        print(f"\n⚡ Flash Attention: Not available for comparison")
    print("=" * 60)


if __name__ == "__main__":
    # Ensure profile directory exists
    profile_dir = os.path.dirname(__file__)
    os.makedirs(profile_dir, exist_ok=True)
    
    # Run profiling
    profile_custom_attention(
        num_warmup_runs=5,
        num_profile_runs=1,
        trace_file="sample_trace.json"
    )
    
    # Run timing measurements for all configurations
    timing_runs = 50
    warmup_runs = 10
    
    # 1. Sparse attention timing (main configuration)
    sparse_time = time_custom_attention(
        num_timing_runs=timing_runs,
        num_warmup_runs=warmup_runs
    )
    
    # 2. Empty maskers baseline timing
    baseline_time = time_baseline_empty_maskers(
        num_timing_runs=timing_runs,
        num_warmup_runs=warmup_runs
    )
    
    # 3. Flash attention baseline timing
    flash_time = time_flash_attention_baseline(
        num_timing_runs=timing_runs,
        num_warmup_runs=warmup_runs
    )
    
    # 4. Comparative analysis
    run_comparative_analysis(sparse_time, baseline_time, flash_time)
