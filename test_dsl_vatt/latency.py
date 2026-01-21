"""Measure latency for vatt idx computation."""

import argparse
import importlib.util
import os
import sys
import time
from pathlib import Path
from types import ModuleType
from typing import Callable, Tuple

import torch

BATCH_SIZE: int = 1
NUM_QUERY_HEADS: int = 32
NUM_KEY_HEADS: int = 32
SEQ_LEN_KEYS: int = 128000
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
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    int,
    int,
    float,
    float,
    float,
    int,
    int,
]:
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
        Tuple of (keys, queries, values, base_sample_size, max_sample_size, epsilon, delta_ppf,
        scaling, start_offset, end_offset).
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
    values: torch.Tensor = torch.randn(
        batch_size, num_key_heads, seq_len_keys, head_dim, device=device
    ).to(torch.float32)

    epsilon: float = 0.1
    from scipy.stats import norm

    delta: float = 0.05
    delta_ppf: float = float(norm.ppf(1 - delta))
    scaling: float = 1.0 / (head_dim ** 0.5)

    return (
        keys,
        queries,
        values,
        base_sample_size,
        max_sample_size,
        epsilon,
        delta_ppf,
        scaling,
        start_offset,
        end_offset,
    )


def ensure_stdlib_profile() -> None:
    """Ensure the standard-library profile module is used."""
    filtered_paths: list[str] = [
        path for path in sys.path if Path(path).name != "test_dsl_vatt"
    ]
    sys.path = filtered_paths
    profile_module: ModuleType | None = sys.modules.get("profile")
    if profile_module is not None and not hasattr(profile_module, "run"):
        del sys.modules["profile"]
    import importlib

    stdlib_profile: ModuleType = importlib.import_module("profile")
    sys.modules["profile"] = stdlib_profile


def measure_latency(
    vatt_idx_fn: Callable,
    function_name: str,
    num_warmup: int = 5,
    num_runs: int = 50,
    device: str = "cpu",
) -> dict:
    """Measure latency of vatt idx computation function.

    Args:
        vatt_idx_fn: Function to profile.
        function_name: Name of the function to profile.
        num_warmup: Number of warmup runs.
        num_runs: Number of timing runs.
        device: Device to run on.

    Returns:
        Dictionary with timing statistics.
    """
    print(f"\n⏱️  Measuring latency ({num_warmup} warmup, {num_runs} runs)...")

    keys: torch.Tensor
    queries: torch.Tensor
    values: torch.Tensor
    base_sample_size: int
    max_sample_size: int
    epsilon: float
    delta_ppf: float
    scaling: float
    start_offset: int
    end_offset: int
    (
        keys,
        queries,
        values,
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
            if function_name == "compute_attention":
                _ = vatt_idx_fn(
                    keys=keys,
                    queries=queries,
                    values=values,
                    base_sample_size=base_sample_size,
                    max_sample_size=max_sample_size,
                    epsilon=epsilon,
                    delta_ppf=delta_ppf,
                    scaling=scaling,
                    start_offset=start_offset,
                    end_offset=end_offset,
                )
            else:
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

    times: list[float] = []
    with torch.no_grad():
        for _ in range(num_runs):
            if device == "cuda":
                torch.cuda.synchronize()
            start_time: float = time.perf_counter()

            if function_name == "compute_attention":
                _ = vatt_idx_fn(
                    keys=keys,
                    queries=queries,
                    values=values,
                    base_sample_size=base_sample_size,
                    max_sample_size=max_sample_size,
                    epsilon=epsilon,
                    delta_ppf=delta_ppf,
                    scaling=scaling,
                    start_offset=start_offset,
                    end_offset=end_offset,
                )
            else:
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
            end_time: float = time.perf_counter()
            times.append((end_time - start_time) * 1000)

    times_tensor: torch.Tensor = torch.tensor(times)
    stats: dict = {
        "mean": times_tensor.mean().item(),
        "std": times_tensor.std().item(),
        "min": times_tensor.min().item(),
        "max": times_tensor.max().item(),
        "median": times_tensor.median().item(),
        "p95": torch.quantile(times_tensor, 0.95).item(),
        "p99": torch.quantile(times_tensor, 0.99).item(),
    }

    print(f"   Mean:   {stats['mean']:.3f} ms")
    print(f"   Std:    {stats['std']:.3f} ms")
    print(f"   Min:    {stats['min']:.3f} ms")
    print(f"   Max:    {stats['max']:.3f} ms")
    print(f"   Median: {stats['median']:.3f} ms")
    print(f"   P95:    {stats['p95']:.3f} ms")
    print(f"   P99:    {stats['p99']:.3f} ms")

    return stats


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
    ensure_stdlib_profile()
    parser = argparse.ArgumentParser(description="Latency for vatt idx computation")
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
        "--num-runs",
        type=int,
        default=50,
        help="Number of timing runs (default: 50)",
    )

    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("⚠️  CUDA not available, falling back to CPU")
        args.device = "cpu"

    print("=" * 70)
    print("VATT Index Computation Latency")
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

    _ = measure_latency(
        vatt_idx_fn,
        function_name=args.function_name,
        num_warmup=args.num_warmup,
        num_runs=args.num_runs,
        device=args.device,
    )

    print("\n" + "=" * 70)
    print("✅ Latency run complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
