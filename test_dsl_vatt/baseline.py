"""Benchmark latency for sparse attention forward pass."""

import argparse
import importlib.util
import sys
import time
from pathlib import Path
from typing import Callable, Tuple

import torch

BATCH_SIZE: int = 1
NUM_QUERY_HEADS: int = 32
NUM_KEY_HEADS: int = 32
SEQ_LEN_KEYS: int = 128000
SEQ_LEN_QUERIES: int = 1
HEAD_DIM: int = 128


def load_sparse_attention_fwd() -> Callable:
    """Load sparse_attention_fwd without importing the full package tree.

    This avoids importing packages that can shadow stdlib modules (e.g. profile).

    Returns:
        The sparse_attention_fwd function from the native backend module.
    """
    repo_root: Path = Path(__file__).resolve().parents[1]
    backend_path: Path = (
        repo_root
        / "sparse_attention_hub"
        / "sparse_attention"
        / "efficient_attention"
        / "backends"
        / "native_backend"
        / "sparse_attention_backend.py"
    )
    spec = importlib.util.spec_from_file_location(
        "sparse_attention_backend", backend_path
    )
    if spec is None or spec.loader is None:
        raise ValueError(f"Could not load module from {backend_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, "sparse_attention_fwd")


def generate_random_inputs(
    seed: int,
    batch_size: int,
    num_query_heads: int,
    num_key_heads: int,
    seq_len_keys: int,
    seq_len_queries: int,
    head_dim: int,
    device: str = "cuda",
    dtype: torch.dtype = torch.float16,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate random inputs for sparse attention.

    Args:
        seed: Random seed for reproducibility.
        batch_size: Batch size.
        num_query_heads: Number of query heads.
        num_key_heads: Number of key/value heads.
        seq_len_keys: Sequence length for keys.
        seq_len_queries: Sequence length for queries (should be 1).
        head_dim: Head dimension.
        device: Device to create tensors on.
        dtype: Tensor dtype.

    Returns:
        Tuple of (query, key, value, sparse_list, sparse_len).
    """
    torch.manual_seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed(seed)

    query_full: torch.Tensor = torch.randn(
        batch_size, num_query_heads, seq_len_queries, head_dim, device=device, dtype=dtype
    )
    query: torch.Tensor = query_full[:, :, 0, :].contiguous()

    key: torch.Tensor = torch.randn(
        batch_size, num_key_heads, seq_len_keys, head_dim, device=device, dtype=dtype
    )
    value: torch.Tensor = torch.randn(
        batch_size, num_key_heads, seq_len_keys, head_dim, device=device, dtype=dtype
    )

    base_indices: torch.Tensor = torch.arange(
        seq_len_keys, device=device, dtype=torch.int32
    )
    sparse_list: torch.Tensor = (
        base_indices.view(1, 1, seq_len_keys)
        .expand(batch_size, num_query_heads, seq_len_keys)
        .contiguous()
    )
    sparse_len: torch.Tensor = torch.full(
        (batch_size, num_query_heads),
        seq_len_keys,
        device=device,
        dtype=torch.int32,
    )

    return query, key, value, sparse_list, sparse_len


def measure_latency(
    num_warmup: int = 5,
    num_runs: int = 50,
    device: str = "cuda",
    block_seq: int = 256,
) -> dict:
    """Measure latency of sparse attention forward pass.

    Args:
        num_warmup: Number of warmup runs.
        num_runs: Number of timing runs.
        device: Device to run on.
        block_seq: Sequence block size.

    Returns:
        Dictionary with timing statistics.
    """
    print(f"\nMeasuring latency ({num_warmup} warmup, {num_runs} runs)...")

    sparse_attention_fwd: Callable = load_sparse_attention_fwd()
    query, key, value, sparse_list, sparse_len = generate_random_inputs(
        seed=42,
        batch_size=BATCH_SIZE,
        num_query_heads=NUM_QUERY_HEADS,
        num_key_heads=NUM_KEY_HEADS,
        seq_len_keys=SEQ_LEN_KEYS,
        seq_len_queries=SEQ_LEN_QUERIES,
        head_dim=HEAD_DIM,
        device=device,
    )

    with torch.no_grad():
        for _ in range(num_warmup):
            _ = sparse_attention_fwd(
                query=query,
                key=key,
                value=value,
                sparse_list=sparse_list,
                sparse_len=sparse_len,
                block_seq=block_seq,
            )
        if device == "cuda":
            torch.cuda.synchronize()

    times: list[float] = []
    with torch.no_grad():
        for _ in range(num_runs):
            if device == "cuda":
                torch.cuda.synchronize()
            start_time: float = time.perf_counter()

            _ = sparse_attention_fwd(
                query=query,
                key=key,
                value=value,
                sparse_list=sparse_list,
                sparse_len=sparse_len,
                block_seq=block_seq,
            )

            if device == "cuda":
                torch.cuda.synchronize()
            end_time: float = time.perf_counter()
            times.append((end_time - start_time) * 1000)

    times_tensor: torch.Tensor = torch.tensor(times)
    stats: dict[str, float] = {
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


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Baseline latency for sparse_attention_fwd"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to run on (default: cuda)",
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
    parser.add_argument(
        "--block-seq",
        type=int,
        default=256,
        help="Sequence block size (default: 256)",
    )

    args = parser.parse_args()

    if args.device != "cuda":
        print("Sparse attention backend requires CUDA.")
        sys.exit(1)
    if not torch.cuda.is_available():
        print("CUDA not available.")
        sys.exit(1)

    print("=" * 70)
    print("Sparse Attention Forward Latency")
    print("=" * 70)
    print(f"Device: {args.device}")
    print(f"Block seq: {args.block_seq}")
    print()

    _ = measure_latency(
        num_warmup=args.num_warmup,
        num_runs=args.num_runs,
        device=args.device,
        block_seq=args.block_seq,
    )

    print("\n" + "=" * 70)
    print("Latency run complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
