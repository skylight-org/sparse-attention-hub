"""Profile vatt idx computation with Perfetto."""

import argparse
import importlib.util
import os
import sys
from typing import Callable, Tuple

import torch
from torch.profiler import ProfilerActivity, profile, record_function

BATCH_SIZE: int = 1
NUM_QUERY_HEADS: int = 32
NUM_KEY_HEADS: int = 32
SEQ_LEN_KEYS: int = 32000
SEQ_LEN_QUERIES: int = 1
HEAD_DIM: int = 128
BASE_SAMPLE_SIZE: int = 1024
MAX_SAMPLE_SIZE: int = 4096
START_OFFSET: int = 128
END_OFFSET: int = 128


def generate_random_inputs(
    seed: int,
    batch_size: int,
    num_query_heads: int,
    num_key_heads: int,
    seq_len_keys: int,
    seq_len_queries: int,
    head_dim: int,
    base_sample_size: int,
    max_sample_size: int,
    start_offset: int,
    end_offset: int,
    device: str = "cpu",
) -> Tuple[torch.Tensor, torch.Tensor, int, int, float, float, float, int, int]:
    """Generate random inputs for vatt idx computation.

    Args:
        seed: Random seed for reproducibility.
        batch_size: Batch size.
        num_query_heads: Number of query heads.
        num_key_heads: Number of key heads.
        seq_len_keys: Sequence length for keys.
        seq_len_queries: Sequence length for queries (should be 1).
        head_dim: Head dimension.
        base_sample_size: Number of base samples.
        max_sample_size: Max number of samples.
        start_offset: Number of keys from the start.
        end_offset: Number of keys from the end.
        device: Device to create tensors on.

    Returns:
        Tuple of (keys, queries, base_sample_size, max_sample_size, epsilon, delta_ppf, scaling, start_offset, end_offset).
    """
    torch.manual_seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed(seed)

    keys: torch.Tensor = torch.randn(
        batch_size, num_key_heads, seq_len_keys, head_dim, device=device
    ).to(torch.float32)
    queries: torch.Tensor = torch.randn(
        batch_size, num_query_heads, seq_len_queries, head_dim, device=device
    ).to(torch.float32)

    epsilon: float = 0.1
    from scipy.stats import norm

    delta: float = 0.05
    delta_ppf: float = float(norm.ppf(1 - delta))
    scaling: float = 1.0 / (head_dim ** 0.5)

    return (
        keys,
        queries,
        base_sample_size,
        max_sample_size,
        epsilon,
        delta_ppf,
        scaling,
        start_offset,
        end_offset,
    )


def profile_with_perfetto(
    vatt_idx_fn: Callable,
    output_file: str = "vatt_idx_trace.json",
    num_warmup: int = 5,
    num_profile_runs: int = 1,
    device: str = "cpu",
) -> None:
    """Profile function and generate Perfetto trace.

    Args:
        vatt_idx_fn: Function to profile.
        output_file: Output file for Perfetto trace.
        num_warmup: Number of warmup runs.
        num_profile_runs: Number of profiling runs.
        device: Device to run on.
    """
    print(f"\n📊 Profiling with Perfetto ({num_warmup} warmup, {num_profile_runs} profile runs)...")

    (
        keys,
        queries,
        base_sample_size,
        max_sample_size,
        epsilon,
        delta_ppf,
        scaling,
        start_offset,
        end_offset,
    ) = generate_random_inputs(
        seed=42,
        batch_size=BATCH_SIZE,
        num_query_heads=NUM_QUERY_HEADS,
        num_key_heads=NUM_KEY_HEADS,
        seq_len_keys=SEQ_LEN_KEYS,
        seq_len_queries=SEQ_LEN_QUERIES,
        head_dim=HEAD_DIM,
        base_sample_size=BASE_SAMPLE_SIZE,
        max_sample_size=MAX_SAMPLE_SIZE,
        start_offset=START_OFFSET,
        end_offset=END_OFFSET,
        device=device,
    )

    with torch.no_grad():
        for _ in range(num_warmup):
            _ = vatt_idx_fn(
                keys=keys,
                queries=queries,
                base_sample_size=base_sample_size,
                max_sample_size=max_sample_size,
                epsilon=epsilon,
                delta_ppf=delta_ppf,
                scaling=scaling,
                start_offset=start_offset,
                end_offset=end_offset,
            )
        if device == "cuda":
            torch.cuda.synchronize()

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
        with record_function("vatt_idx_computation_profiling"):
            with torch.no_grad():
                for i in range(num_profile_runs):
                    with record_function(f"iteration_{i}"):
                        _ = vatt_idx_fn(
                            keys=keys,
                            queries=queries,
                            base_sample_size=base_sample_size,
                            max_sample_size=max_sample_size,
                            epsilon=epsilon,
                            delta_ppf=delta_ppf,
                            scaling=scaling,
                            start_offset=start_offset,
                            end_offset=end_offset,
                        )
            if device == "cuda":
                torch.cuda.synchronize()

    prof.export_chrome_trace(output_file)
    print(f"✅ Perfetto trace saved to: {output_file}")
    print("   View at: https://ui.perfetto.dev/")

    print("\n📈 Top 10 operations by CPU time:")
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))

    if device == "cuda":
        print("\n📈 Top 10 operations by CUDA time:")
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))


def load_function_from_file(
    file_path: str, function_name: str = "ref_vatt_idx_computation"
) -> Callable:
    """Load a function from a Python file.

    Args:
        file_path: Path to the Python file.
        function_name: Name of the function to load.

    Returns:
        The loaded function.

    Raises:
        FileNotFoundError: If file doesn't exist.
        AttributeError: If function doesn't exist in the file.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    spec = importlib.util.spec_from_file_location("vatt_idx_module", file_path)
    if spec is None or spec.loader is None:
        raise ValueError(f"Could not load module from {file_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules["vatt_idx_module"] = module
    spec.loader.exec_module(module)

    if not hasattr(module, function_name):
        raise AttributeError(
            f"Function '{function_name}' not found in {file_path}. "
            f"Available functions: {[x for x in dir(module) if not x.startswith('_')]}"
        )

    return getattr(module, function_name)


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Perfetto profile for vatt idx computation")
    parser.add_argument(
        "file",
        type=str,
        help="Path to Python file containing vatt idx computation function",
    )
    parser.add_argument(
        "--function-name",
        type=str,
        default="ref_vatt_idx_computation",
        help="Name of the function to profile (default: ref_vatt_idx_computation)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cpu", "cuda"],
        help="Device to run on (default: cpu)",
    )
    parser.add_argument(
        "--num-warmup",
        type=int,
        default=5,
        help="Number of warmup runs (default: 5)",
    )
    parser.add_argument(
        "--num-profile-runs",
        type=int,
        default=1,
        help="Number of profiling runs for Perfetto trace (default: 1)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="vatt_idx_trace.json",
        help="Output file for Perfetto trace (default: vatt_idx_trace.json)",
    )

    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("⚠️  CUDA not available, falling back to CPU")
        args.device = "cpu"

    print("=" * 70)
    print("VATT Index Computation Perfetto Profiler")
    print("=" * 70)
    print(f"File: {args.file}")
    print(f"Function: {args.function_name}")
    print(f"Device: {args.device}")
    print()

    try:
        print(f"📦 Loading function '{args.function_name}' from {args.file}...")
        vatt_idx_fn = load_function_from_file(args.file, args.function_name)
        print("✅ Function loaded successfully")
    except Exception as exc:
        print(f"❌ Failed to load function: {exc}")
        sys.exit(1)

    profile_with_perfetto(
        vatt_idx_fn,
        output_file=args.output,
        num_warmup=args.num_warmup,
        num_profile_runs=args.num_profile_runs,
        device=args.device,
    )

    print("\n" + "=" * 70)
    print("✅ Profiling complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
