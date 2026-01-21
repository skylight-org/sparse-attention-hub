"""Check correctness for vatt idx computation."""

import argparse
import importlib.util
import os
import sys
from typing import Callable, Tuple

import torch

from pytorch_fn import ref_vatt_idx_computation


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


def _compare_outputs(
    ref_output: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    test_output: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    rtol: float,
    atol: float,
) -> Tuple[bool, str]:
    """Compare reference and test outputs for correctness.

    Args:
        ref_output: Reference outputs (sparse_lens, sparse_idx, weights).
        test_output: Test outputs (sparse_lens, sparse_idx, weights).
        rtol: Relative tolerance for weights comparison.
        atol: Absolute tolerance for weights comparison.

    Returns:
        Tuple of (is_equal, message).
    """
    if len(ref_output) != 3 or len(test_output) != 3:
        return False, "Expected 3 outputs (sparse_lens, sparse_idx, weights)."

    ref_sparse_lens, ref_sparse_idx, ref_weights = ref_output
    test_sparse_lens, test_sparse_idx, test_weights = test_output

    if ref_sparse_lens.shape != test_sparse_lens.shape:
        return (
            False,
            f"sparse_lens shape mismatch - ref: {ref_sparse_lens.shape}, "
            f"test: {test_sparse_lens.shape}",
        )
    if ref_sparse_idx.shape != test_sparse_idx.shape:
        return (
            False,
            f"sparse_idx shape mismatch - ref: {ref_sparse_idx.shape}, "
            f"test: {test_sparse_idx.shape}",
        )
    if ref_weights.shape != test_weights.shape:
        return (
            False,
            f"weights shape mismatch - ref: {ref_weights.shape}, test: {test_weights.shape}",
        )

    if not torch.equal(ref_sparse_lens, test_sparse_lens):
        max_diff: int = int((ref_sparse_lens - test_sparse_lens).abs().max().item())
        return False, f"sparse_lens mismatch - max diff: {max_diff}"

    if not torch.equal(ref_sparse_idx, test_sparse_idx):
        max_diff = int((ref_sparse_idx - test_sparse_idx).abs().max().item())
        return False, f"sparse_idx mismatch - max diff: {max_diff}"

    if not torch.allclose(ref_weights.float(), test_weights.float(), rtol=rtol, atol=atol):
        max_diff_f: float = float((ref_weights - test_weights).abs().max().item())
        return False, f"weights mismatch - max diff: {max_diff_f:.6f}"

    return True, "Outputs match."


def test_correctness(
    vatt_idx_fn: Callable,
    num_tests: int = 15,
    device: str = "cpu",
    rtol: float = 1e-3,
    atol: float = 1e-4,
) -> bool:
    """Test correctness by comparing against reference implementation.

    Args:
        vatt_idx_fn: Function to test.
        num_tests: Number of random test cases.
        device: Device to run tests on.
        rtol: Relative tolerance for weights.
        atol: Absolute tolerance for weights.

    Returns:
        True if all tests pass, False otherwise.
    """
    print(f"🧪 Testing correctness with {num_tests} random inputs...")
    print(f"   Device: {device}")
    print(f"   Tolerance: rtol={rtol}, atol={atol}")

    all_passed: bool = True
    for test_idx in range(num_tests):
        batch_size: int = int(torch.randint(1, 4, (1,)).item())
        num_key_heads: int = int(torch.randint(1, 5, (1,)).item())
        num_query_heads: int = num_key_heads * int(torch.randint(1, 4, (1,)).item())
        seq_len_keys: int = int(torch.randint(4096, 10240, (1,)).item())
        seq_len_queries: int = 1
        head_dim: int = int(torch.randint(16, 128, (1,)).item())
        start_offset: int = int(torch.randint(0, seq_len_keys // 8, (1,)).item())
        end_offset: int = int(torch.randint(1, seq_len_keys // 8 + 1, (1,)).item())

        sampling_range: int = seq_len_keys - start_offset - end_offset
        if sampling_range <= 0:
            end_offset = max(1, seq_len_keys - start_offset - 1)
            sampling_range = seq_len_keys - start_offset - end_offset

        max_base: int = max(2, min(64, sampling_range))
        base_sample_size: int = int(torch.randint(2, max_base + 1, (1,)).item())
        max_sample_size: int = int(
            torch.randint(base_sample_size, sampling_range + 1, (1,)).item()
        )

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
            seed=test_idx,
            batch_size=batch_size,
            num_query_heads=num_query_heads,
            num_key_heads=num_key_heads,
            seq_len_keys=seq_len_keys,
            seq_len_queries=seq_len_queries,
            head_dim=head_dim,
            base_sample_size=base_sample_size,
            max_sample_size=max_sample_size,
            start_offset=start_offset,
            end_offset=end_offset,
            device=device,
        )
        print(f"Keys: {keys.shape}", f"Queries: {queries.shape}", f"Base sample: {base_sample_size}", f"Max sample: {max_sample_size}", f"Start: {start_offset}", f"End: {end_offset}")
        try:
            ref_output = ref_vatt_idx_computation(
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
        except Exception as exc:
            print(f"❌ Test {test_idx + 1}: Reference implementation failed: {exc}")
            all_passed = False
            continue

        try:
            test_output = vatt_idx_fn(
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
        except Exception as exc:
            print(f"❌ Test {test_idx + 1}: Test implementation failed: {exc}")
            all_passed = False
            continue

        is_equal, message = _compare_outputs(
            ref_output=ref_output,
            test_output=test_output,
            rtol=rtol,
            atol=atol,
        )
        if not is_equal:
            print(f"❌ Test {test_idx + 1}: {message}")
            all_passed = False
            continue

        print(f"✅ Test {test_idx + 1}: Passed")

    if all_passed:
        print(f"\n✅ All {num_tests} correctness tests passed!")
    else:
        print("\n❌ Some correctness tests failed!")

    return all_passed


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
    parser = argparse.ArgumentParser(description="Correctness for vatt idx computation")
    parser.add_argument(
        "file",
        type=str,
        help="Path to Python file containing vatt idx computation function",
    )
    parser.add_argument(
        "--function-name",
        type=str,
        default="ref_vatt_idx_computation",
        help="Name of the function to test (default: ref_vatt_idx_computation)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cpu", "cuda"],
        help="Device to run on (default: cpu)",
    )
    parser.add_argument(
        "--num-tests",
        type=int,
        default=15,
        help="Number of correctness tests (default: 15)",
    )
    parser.add_argument(
        "--rtol",
        type=float,
        default=1e-3,
        help="Relative tolerance for weights (default: 1e-3)",
    )
    parser.add_argument(
        "--atol",
        type=float,
        default=1e-4,
        help="Absolute tolerance for weights (default: 1e-4)",
    )

    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("⚠️  CUDA not available, falling back to CPU")
        args.device = "cpu"

    print("=" * 70)
    print("VATT Index Computation Correctness")
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

    test_correctness(
        vatt_idx_fn,
        num_tests=args.num_tests,
        device=args.device,
        rtol=args.rtol,
        atol=args.atol,
    )

    print("\n" + "=" * 70)
    print("✅ Correctness run complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
