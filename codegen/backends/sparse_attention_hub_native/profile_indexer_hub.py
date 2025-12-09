#!/usr/bin/env python3
"""Profile custom_indexer_hub and optional __indexer functions for sparse attention native backend.

This script profiles the custom_indexer_hub function from custom_attention_hub.py,
and optionally profiles a custom __indexer function if an indexer file is provided.
It generates Perfetto-compatible traces and outputs CPU/CUDA timing information.

Usage:
    # Profile only custom_indexer_hub
    python codegen/backends/sparse_attention_hub_native/profile_indexer_hub.py --output my_profile

    # Profile custom_indexer_hub and a custom __indexer
    python codegen/backends/sparse_attention_hub_native/profile_indexer_hub.py --output my_profile --indexer-file codegen/backends/sparse_attention_hub_native/example_indexer.py

    # Customize profiling parameters
    python codegen/backends/sparse_attention_hub_native/profile_indexer_hub.py --output my_profile --warmup-runs 10 --profile-runs 5 --seq-len 8192
"""

import argparse
import importlib.util
import os
import sys
import time
from typing import Any, Callable, Dict, Optional, Tuple

import torch
from torch import nn
from torch.profiler import ProfilerActivity, profile, record_function

# Add project root to Python path (go up 4 levels: file -> sparse_attention_hub_native -> backends -> codegen -> project_root)
SCRIPT_DIR: str = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT: str = os.path.dirname(os.path.dirname(os.path.dirname(SCRIPT_DIR)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from codegen.custom_attention_hub import custom_indexer_hub  # noqa: E402
from codegen.backends.sparse_attention_hub_native.utils import (  # noqa: E402
    convert_dense_to_sparse_attention_inputs,
)


class DummyAttentionModule(nn.Module):
    """Minimal attention module used only to provide a module interface."""

    def __init__(self) -> None:
        """Initialize dummy attention module."""
        super().__init__()
        self.layer_idx: int = 0


def load_indexer_from_file(file_path: str) -> Callable:
    """Dynamically load a custom indexer function from a Python file.

    The file must contain a function named ``__indexer``.

    Args:
        file_path: Path to the Python file containing the ``__indexer`` function.

    Returns:
        The ``__indexer`` function from the loaded module.

    Raises:
        FileNotFoundError: If the file does not exist.
        AttributeError: If the file does not contain a ``__indexer`` function.
        ImportError: If the file cannot be imported.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Indexer file not found: {file_path}")

    # Load the module dynamically
    spec = importlib.util.spec_from_file_location("custom_indexer_module", file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module from: {file_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # Get the __indexer function
    if not hasattr(module, "__indexer"):
        raise AttributeError(
            f"Module {file_path} does not contain a '__indexer' function"
        )

    indexer_fn: Callable = getattr(module, "__indexer")
    return indexer_fn


def create_sample_tensors(
    batch_size: int = 1,
    num_heads: int = 32,
    num_queries: int = 1,
    seq_len: int = 4096,
    head_dim: int = 128,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, DummyAttentionModule, Dict[str, Any]]:
    """Create sample input tensors for attention computation.

    Args:
        batch_size: Batch size for the tensors.
        num_heads: Number of attention heads.
        num_queries: Number of query tokens.
        seq_len: Sequence length for keys/values.
        head_dim: Dimension per attention head.
        device: Device to place tensors on.

    Returns:
        Tuple of (queries, keys, values, attention_mask, module, sparse_meta_data).
    """
    dtype: torch.dtype = torch.bfloat16 if device == "cuda" else torch.float32

    # Create attention tensors
    queries: torch.Tensor = torch.randn(
        batch_size,
        num_heads,
        num_queries,
        head_dim,
        device=device,
        dtype=dtype,
        requires_grad=True,
    )
    keys: torch.Tensor = torch.randn(
        batch_size, num_heads, seq_len, head_dim, device=device, dtype=dtype, requires_grad=True
    )
    values: torch.Tensor = torch.randn(
        batch_size, num_heads, seq_len, head_dim, device=device, dtype=dtype, requires_grad=True
    )

    # Create attention mask (optional)
    attention_mask: torch.Tensor = torch.ones(
        batch_size, num_heads, num_queries, seq_len, device=device, dtype=dtype
    )

    # Create a mock attention module
    module: DummyAttentionModule = DummyAttentionModule().to(device)

    # Create sparse metadata (required for research attention)
    sparse_meta_data: Dict[str, Any] = {
        "layer_idx": 0,
        "seq_len": seq_len,
        "batch_size": batch_size,
        "num_heads": num_heads,
        "head_dim": head_dim,
    }

    return queries, keys, values, attention_mask, module, sparse_meta_data


def create_sparse_native_sample_inputs(
    batch_size: int = 1,
    num_heads: int = 32,
    head_dim: int = 128,
    seq_len: int = 4096,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create sample inputs for sparse attention native backend indexer function.

    This function creates dense tensors using create_sample_tensors and converts
    them to sparse attention native format using convert_dense_to_sparse_attention_inputs.

    Args:
        batch_size: Batch size.
        num_heads: Number of attention heads.
        head_dim: Head dimension.
        seq_len: Sequence length for keys/values.
        device: Device to place tensors on.

    Returns:
        Tuple of sparse-attention-compatible inputs matching the signature expected by
        custom __indexer functions:
        (query, key, value, sparse_list, sparse_len, weight_list)
    """
    # Create dense sample tensors
    queries, keys, values, _, _, _ = create_sample_tensors(
        batch_size=batch_size,
        num_heads=num_heads,
        num_queries=1,  # Sparse attention expects single query position (decode mode)
        seq_len=seq_len,
        head_dim=head_dim,
        device=device,
    )

    # Convert to sparse attention native format using the utility function
    sparse_inputs: Tuple[torch.Tensor, ...] = convert_dense_to_sparse_attention_inputs(
        queries=queries,
        keys=keys,
        values=values,
    )

    return sparse_inputs


def profile_custom_indexer_hub(
    output_filename: str,
    num_warmup_runs: int = 5,
    num_profile_runs: int = 1,
    num_timing_runs: int = 50,
    seq_len: int = 4096,
    num_queries: int = 1,
    batch_size: int = 1,
    num_heads: int = 32,
    head_dim: int = 128,
) -> Dict[str, float]:
    """Profile the custom_indexer_hub function.

    Args:
        output_filename: Base filename for output files (without extension).
        num_warmup_runs: Number of warmup runs before profiling.
        num_profile_runs: Number of runs to profile.
        num_timing_runs: Number of runs for timing measurements.
        seq_len: Sequence length for keys/values.
        num_queries: Number of query tokens.
        batch_size: Batch size.
        num_heads: Number of attention heads.
        head_dim: Head dimension.

    Returns:
        Dictionary containing timing statistics.
    """
    print("🚀 Starting custom_indexer_hub profiling...")

    # Setup
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"📱 Using device: {device}")

    # Create sample tensors
    queries, keys, values, attention_mask, module, sparse_meta_data = create_sample_tensors(
        batch_size=batch_size,
        num_heads=num_heads,
        num_queries=num_queries,
        seq_len=seq_len,
        head_dim=head_dim,
        device=device,
    )

    print(f"📊 Tensor shapes:")
    print(f"   - Queries: {queries.shape}")
    print(f"   - Keys: {keys.shape}")
    print(f"   - Values: {values.shape}")
    print(f"   - Attention mask: {attention_mask.shape}")

    # Scaling and dropout parameters
    scaling: float = 1.0 / (head_dim**0.5)
    dropout: float = 0.0

    # Warmup runs
    print(f"🔥 Running {num_warmup_runs} warmup iterations...")
    with torch.no_grad():
        for _ in range(num_warmup_runs):
            _ = custom_indexer_hub(
                module=module,
                queries=queries,
                keys=keys,
                values=values,
                attention_mask=attention_mask,
                scaling=scaling,
                dropout=dropout,
                sparse_meta_data=sparse_meta_data,
            )
            if device == "cuda":
                torch.cuda.synchronize()

    # Profiling runs
    print(f"📊 Profiling {num_profile_runs} iterations...")

    activities: list = [ProfilerActivity.CPU]
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
        with record_function("custom_indexer_hub_profiling"):
            with torch.no_grad():
                for i in range(num_profile_runs):
                    with record_function(f"iteration_{i}"):
                        mask = custom_indexer_hub(
                            module=module,
                            queries=queries,
                            keys=keys,
                            values=values,
                            attention_mask=attention_mask,
                            scaling=scaling,
                            dropout=dropout,
                            sparse_meta_data=sparse_meta_data,
                        )
                    if device == "cuda":
                        torch.cuda.synchronize()

    # Save trace
    trace_path: str = f"{output_filename}_indexer_hub.json"
    prof.export_chrome_trace(trace_path)
    print(f"✅ Trace saved to: {trace_path}")

    # Print key statistics
    print("\n📈 Top 10 operations by CPU time:")
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))

    if device == "cuda":
        print("\n📈 Top 10 operations by CUDA time:")
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

    # Manual timing for accurate wall-clock measurements
    print(f"\n⏱️ Running manual timing measurements ({num_timing_runs} runs)...")
    times: list = []

    with torch.no_grad():
        for i in range(num_timing_runs):
            # Start timing
            if device == "cuda":
                torch.cuda.synchronize()
            start_time: float = time.perf_counter()

            # Run custom_indexer_hub
            _ = custom_indexer_hub(
                module=module,
                queries=queries,
                keys=keys,
                values=values,
                attention_mask=attention_mask,
                scaling=scaling,
                dropout=dropout,
                sparse_meta_data=sparse_meta_data,
            )

            # End timing
            if device == "cuda":
                torch.cuda.synchronize()
            end_time: float = time.perf_counter()

            elapsed_time: float = (end_time - start_time) * 1000  # Convert to ms
            times.append(elapsed_time)

    # Calculate statistics
    avg_time: float = sum(times) / len(times)
    min_time: float = min(times)
    max_time: float = max(times)

    timing_stats: Dict[str, float] = {
        "avg_time_ms": avg_time,
        "min_time_ms": min_time,
        "max_time_ms": max_time,
    }

    print(f"\n⏱️ Timing Statistics:")
    print(f"   - Average time: {timing_stats['avg_time_ms']:.3f} ms")
    print(f"   - Min time:     {timing_stats['min_time_ms']:.3f} ms")
    print(f"   - Max time:     {timing_stats['max_time_ms']:.3f} ms")

    print(f"\n🎉 Profiling completed! View trace in https://ui.perfetto.dev/")
    print(f"   Load file: {trace_path}")

    return timing_stats


def profile_custom_indexer(
    indexer_fn: Callable,
    output_filename: str,
    num_warmup_runs: int = 5,
    num_profile_runs: int = 1,
    num_timing_runs: int = 50,
    batch_size: int = 1,
    num_heads: int = 32,
    head_dim: int = 128,
    seq_len: int = 4096,
) -> Dict[str, float]:
    """Profile a custom __indexer function for sparse attention native backend.

    Args:
        indexer_fn: The custom indexer function to profile.
        output_filename: Base filename for output files (without extension).
        num_warmup_runs: Number of warmup runs before profiling.
        num_profile_runs: Number of runs to profile.
        num_timing_runs: Number of runs for timing measurements.
        batch_size: Batch size.
        num_heads: Number of attention heads.
        head_dim: Head dimension.
        seq_len: Sequence length for keys/values.

    Returns:
        Dictionary containing timing statistics.
    """
    print("\n🚀 Starting custom __indexer profiling...")

    # Setup
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"📱 Using device: {device}")

    # Create sample inputs using utility function
    (
        query,
        key,
        value,
        sparse_list,
        sparse_len,
        weight_list,
    ) = create_sparse_native_sample_inputs(
        batch_size=batch_size,
        num_heads=num_heads,
        head_dim=head_dim,
        seq_len=seq_len,
        device=device,
    )

    print(f"📊 Input shapes:")
    print(f"   - Query: {query.shape}")
    print(f"   - Key: {key.shape}")
    print(f"   - Value: {value.shape}")
    print(f"   - Sparse list: {sparse_list.shape}")
    print(f"   - Sparse len: {sparse_len.shape}")
    print(f"   - Weight list: {weight_list.shape}")

    # Warmup runs
    print(f"🔥 Running {num_warmup_runs} warmup iterations...")
    with torch.no_grad():
        for _ in range(num_warmup_runs):
            _ = indexer_fn(
                queries=query,
                keys=key,
                values=value,
                sparse_list=sparse_list,
                sparse_len=sparse_len,
                weight_list=weight_list,
            )
            if device == "cuda":
                torch.cuda.synchronize()

    # Profiling runs
    print(f"📊 Profiling {num_profile_runs} iterations...")

    activities: list = [ProfilerActivity.CPU]
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
        with record_function("custom_indexer_profiling"):
            with torch.no_grad():
                for i in range(num_profile_runs):
                    with record_function(f"iteration_{i}"):
                        result = indexer_fn(
                            queries=query,
                            keys=key,
                            values=value,
                            sparse_list=sparse_list,
                            sparse_len=sparse_len,
                            weight_list=weight_list,
                        )
                    if device == "cuda":
                        torch.cuda.synchronize()

    # Save trace
    trace_path: str = f"{output_filename}_custom_indexer.json"
    prof.export_chrome_trace(trace_path)
    print(f"✅ Trace saved to: {trace_path}")

    # Print key statistics
    print("\n📈 Top 10 operations by CPU time:")
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))

    if device == "cuda":
        print("\n📈 Top 10 operations by CUDA time:")
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

    # Manual timing for accurate wall-clock measurements
    print(f"\n⏱️ Running manual timing measurements ({num_timing_runs} runs)...")
    times: list = []

    with torch.no_grad():
        for i in range(num_timing_runs):
            # Start timing
            if device == "cuda":
                torch.cuda.synchronize()
            start_time: float = time.perf_counter()

            # Run custom indexer
            _ = indexer_fn(
                queries=query,
                keys=key,
                values=value,
                sparse_list=sparse_list,
                sparse_len=sparse_len,
                weight_list=weight_list,
            )

            # End timing
            if device == "cuda":
                torch.cuda.synchronize()
            end_time: float = time.perf_counter()

            elapsed_time: float = (end_time - start_time) * 1000  # Convert to ms
            times.append(elapsed_time)

    # Calculate statistics
    avg_time: float = sum(times) / len(times)
    min_time: float = min(times)
    max_time: float = max(times)

    timing_stats: Dict[str, float] = {
        "avg_time_ms": avg_time,
        "min_time_ms": min_time,
        "max_time_ms": max_time,
    }

    print(f"\n⏱️ Timing Statistics:")
    print(f"   - Average time: {timing_stats['avg_time_ms']:.3f} ms")
    print(f"   - Min time:     {timing_stats['min_time_ms']:.3f} ms")
    print(f"   - Max time:     {timing_stats['max_time_ms']:.3f} ms")

    print(f"\n🎉 Profiling completed! View trace in https://ui.perfetto.dev/")
    print(f"   Load file: {trace_path}")

    return timing_stats


def save_timing_summary(
    output_filename: str,
    indexer_hub_stats: Dict[str, float],
    custom_indexer_stats: Optional[Dict[str, float]] = None,
) -> None:
    """Save timing summary to a text file.

    Args:
        output_filename: Base filename for output files (without extension).
        indexer_hub_stats: Timing statistics for custom_indexer_hub.
        custom_indexer_stats: Optional timing statistics for custom __indexer.
    """
    summary_path: str = f"{output_filename}_timing_summary.txt"

    with open(summary_path, "w") as f:
        f.write("=" * 60 + "\n")
        f.write("Sparse Attention Native Backend - Indexer Hub Profiling Summary\n")
        f.write("=" * 60 + "\n\n")

        f.write("custom_indexer_hub:\n")
        f.write(f"  Average Time: {indexer_hub_stats['avg_time_ms']:.3f} ms\n")
        f.write(f"  Min Time:     {indexer_hub_stats['min_time_ms']:.3f} ms\n")
        f.write(f"  Max Time:     {indexer_hub_stats['max_time_ms']:.3f} ms\n")

        if custom_indexer_stats is not None:
            f.write("\ncustom __indexer:\n")
            f.write(f"  Average Time: {custom_indexer_stats['avg_time_ms']:.3f} ms\n")
            f.write(f"  Min Time:     {custom_indexer_stats['min_time_ms']:.3f} ms\n")
            f.write(f"  Max Time:     {custom_indexer_stats['max_time_ms']:.3f} ms\n")

        f.write("\n" + "=" * 60 + "\n")

    print(f"\n📄 Timing summary saved to: {summary_path}")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Profile custom_indexer_hub and optional custom __indexer functions for sparse attention native backend",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Base filename for output files (without extension)",
    )

    parser.add_argument(
        "--indexer-file",
        type=str,
        default=None,
        help="Path to custom indexer file containing __indexer function",
    )

    parser.add_argument(
        "--warmup-runs",
        type=int,
        default=5,
        help="Number of warmup runs before profiling (default: 5)",
    )

    parser.add_argument(
        "--profile-runs",
        type=int,
        default=1,
        help="Number of profiling runs (default: 1)",
    )

    parser.add_argument(
        "--timing-runs",
        type=int,
        default=50,
        help="Number of timing measurement runs (default: 50)",
    )

    parser.add_argument(
        "--seq-len",
        type=int,
        default=32678,
        help="Sequence length for custom_indexer_hub (default: 32678)",
    )

    parser.add_argument(
        "--num-queries",
        type=int,
        default=1,
        help="Number of query tokens (default: 1)",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size (default: 1)",
    )

    parser.add_argument(
        "--num-heads",
        type=int,
        default=32,
        help="Number of attention heads (default: 32)",
    )

    parser.add_argument(
        "--head-dim",
        type=int,
        default=128,
        help="Head dimension (default: 128)",
    )

    return parser.parse_args()


def main() -> None:
    """Main entry point for the profiling script."""
    args: argparse.Namespace = parse_args()

    print("=" * 60)
    print("Sparse Attention Native Backend - Indexer Hub Profiling Script")
    print("=" * 60)
    print(f"\nConfiguration:")
    print(f"  Output filename: {args.output}")
    print(f"  Indexer file: {args.indexer_file if args.indexer_file else 'None'}")
    print(f"  Warmup runs: {args.warmup_runs}")
    print(f"  Profile runs: {args.profile_runs}")
    print(f"  Timing runs: {args.timing_runs}")
    print(f"  Sequence length: {args.seq_len}")
    print(f"  Num queries: {args.num_queries}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Num heads: {args.num_heads}")
    print(f"  Head dim: {args.head_dim}")
    print()

    # Profile custom_indexer_hub
    indexer_hub_stats: Dict[str, float] = profile_custom_indexer_hub(
        output_filename=args.output,
        num_warmup_runs=args.warmup_runs,
        num_profile_runs=args.profile_runs,
        num_timing_runs=args.timing_runs,
        seq_len=args.seq_len,
        num_queries=args.num_queries,
        batch_size=args.batch_size,
        num_heads=args.num_heads,
        head_dim=args.head_dim,
    )

    # Profile custom __indexer if provided
    custom_indexer_stats: Optional[Dict[str, float]] = None
    if args.indexer_file:
        try:
            print(f"\n📂 Loading custom indexer from: {args.indexer_file}")
            indexer_fn: Callable = load_indexer_from_file(args.indexer_file)
            print("✅ Custom indexer loaded successfully")

            custom_indexer_stats = profile_custom_indexer(
                indexer_fn=indexer_fn,
                output_filename=args.output,
                num_warmup_runs=args.warmup_runs,
                num_profile_runs=args.profile_runs,
                num_timing_runs=args.timing_runs,
                batch_size=args.batch_size,
                num_heads=args.num_heads,
                head_dim=args.head_dim,
                seq_len=args.seq_len,
            )
        except Exception as e:
            print(f"❌ Error profiling custom indexer: {e}")
            print("Continuing without custom indexer profiling...")

    # Save timing summary
    save_timing_summary(args.output, indexer_hub_stats, custom_indexer_stats)

    print("\n" + "=" * 60)
    print("✅ All profiling completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()

