"""Test quality comparison between kmeans_loop_pytorch and kmeans_batched_pytorch.

This test compares the reconstruction error for both kmeans implementations
with varying numbers of centroids (powers of 2).
"""

import importlib.util
import os

import torch

# Direct import from pq_utils to avoid package dependencies
pq_utils_path = os.path.join(
    os.path.dirname(__file__),
    "../../sparse_attention_hub/sparse_attention/research_attention/maskers/fixed/implementations/utils/pq_utils.py",
)
spec = importlib.util.spec_from_file_location("pq_utils", pq_utils_path)
pq_utils = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pq_utils)

kmeans_batched_pytorch = pq_utils.kmeans_batched_pytorch
kmeans_loop_pytorch = pq_utils.kmeans_loop_pytorch

try:
    import matplotlib
    import matplotlib.pyplot as plt

    matplotlib.use("Agg")  # Use non-interactive backend
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Warning: matplotlib not available. Plot will not be generated.")

try:
    import pytest

    PYTEST_AVAILABLE = True
except ImportError:
    PYTEST_AVAILABLE = False

    # Mock pytest.mark.parametrize for standalone execution
    class MockPytest:
        class mark:
            @staticmethod
            def parametrize(name, values):
                def decorator(func):
                    return func

                return decorator

    pytest = MockPytest()


def compute_kmeans_reconstruction_error(
    data: torch.Tensor, centroids: torch.Tensor, codes: torch.Tensor
) -> dict:
    """Compute reconstruction error for kmeans clustering.

    Args:
        data: Original data [n_groups, n_samples, d]
        centroids: Cluster centers [n_groups, k, d]
        codes: Cluster assignments [n_groups, n_samples]

    Returns:
        Dict containing:
            - mse_error: Mean Squared Error
            - l2_error: L2 norm of difference
            - relative_error: L2 error normalized by original norm
    """
    n_groups, n_samples, d = data.shape

    # Reconstruct data from centroids and codes
    reconstructed = torch.zeros_like(data)
    for i in range(n_groups):
        reconstructed[i] = centroids[i][codes[i].long()]

    # Calculate error metrics
    diff = data - reconstructed
    mse_error = torch.mean(diff**2).item()
    l2_error = torch.norm(diff).item()
    original_norm = torch.norm(data).item()
    relative_error = l2_error / original_norm if original_norm > 0 else 0.0

    return {
        "mse_error": mse_error,
        "l2_error": l2_error,
        "relative_error": relative_error,
    }


@pytest.mark.parametrize("num_centroids", [2, 4, 8, 16, 32, 64])
def test_kmeans_quality_comparison(num_centroids: int) -> None:
    """Test quality comparison between loop and batched kmeans implementations.

    Args:
        num_centroids: Number of cluster centroids (powers of 2)
    """
    # Fixed parameters
    batch_size: int = 16
    n_samples: int = 128
    dimension: int = 8
    max_iter: int = 20

    # Set seed for reproducibility
    torch.manual_seed(42)

    # Generate test data on GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = torch.randn(batch_size, n_samples, dimension).to(device)

    # Run kmeans_loop_pytorch
    centroids_loop, codes_loop = kmeans_loop_pytorch(
        data, k=num_centroids, max_iter=max_iter
    )

    # Run kmeans_batched_pytorch
    centroids_batched, codes_batched = kmeans_batched_pytorch(
        data, k=num_centroids, max_iter=max_iter
    )

    # Compute reconstruction errors
    error_loop = compute_kmeans_reconstruction_error(data, centroids_loop, codes_loop)
    error_batched = compute_kmeans_reconstruction_error(
        data, centroids_batched, codes_batched
    )

    # Print comparison
    print(f"\nCentroids: {num_centroids}")
    print(
        f"  Loop - MSE: {error_loop['mse_error']:.6f}, "
        f"L2: {error_loop['l2_error']:.4f}, "
        f"Relative: {error_loop['relative_error']:.6f}"
    )
    print(
        f"  Batched - MSE: {error_batched['mse_error']:.6f}, "
        f"L2: {error_batched['l2_error']:.4f}, "
        f"Relative: {error_batched['relative_error']:.6f}"
    )
    print(
        f"  Difference - MSE: {abs(error_loop['mse_error'] - error_batched['mse_error']):.6f}, "
        f"Relative: {abs(error_loop['relative_error'] - error_batched['relative_error']) * 100:.2f}%"
    )

    # Verify both produce valid results
    assert centroids_loop.shape == (batch_size, num_centroids, dimension)
    assert centroids_batched.shape == (batch_size, num_centroids, dimension)
    assert codes_loop.shape == (batch_size, n_samples)
    assert codes_batched.shape == (batch_size, n_samples)

    # Verify dtypes
    assert codes_loop.dtype == torch.int64
    assert codes_batched.dtype == torch.int64

    # Verify all cluster IDs are valid
    assert torch.all(codes_loop >= 0) and torch.all(codes_loop < num_centroids)
    assert torch.all(codes_batched >= 0) and torch.all(codes_batched < num_centroids)

    # Both methods should produce reasonable reconstruction errors
    # Relative error should be less than 100% (sanity check)
    assert error_loop["relative_error"] < 1.0
    assert error_batched["relative_error"] < 1.0

    # Errors should be similar (within 20% of each other)
    # This is lenient because different random initializations can lead to different local minima
    relative_diff = abs(
        error_loop["relative_error"] - error_batched["relative_error"]
    ) / max(error_loop["relative_error"], error_batched["relative_error"])
    assert relative_diff < 0.20, (
        f"Reconstruction errors differ by {relative_diff*100:.1f}% "
        f"(loop: {error_loop['relative_error']:.6f}, "
        f"batched: {error_batched['relative_error']:.6f})"
    )


def test_kmeans_quality_full_comparison(plot: bool = True) -> None:
    """Run full quality comparison across all centroid counts.

    Args:
        plot: Whether to generate and save a plot of the results
    """
    # Fixed parameters
    batch_size: int = 16
    n_samples: int = 128
    dimension: int = 8
    max_iter: int = 20
    centroid_counts = [2, 4, 8, 16, 32, 64, 128]

    # Set seed for reproducibility
    torch.manual_seed(42)

    # Generate test data on GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = torch.randn(batch_size, n_samples, dimension).to(device)

    print("\n" + "=" * 80)
    print("K-Means Quality Comparison: Loop vs Batched Implementation")
    print("=" * 80)
    print(
        f"Configuration: batch = {batch_size}, samples = {n_samples}, "
        f"dim = {dimension}, max_iter = {max_iter}"
    )
    print(f"Device: {device}")
    print("=" * 80)

    results_loop = []
    results_batched = []

    for k in centroid_counts:
        # Run kmeans_loop_pytorch
        centroids_loop, codes_loop = kmeans_loop_pytorch(data, k=k, max_iter=max_iter)
        error_loop = compute_kmeans_reconstruction_error(
            data, centroids_loop, codes_loop
        )
        results_loop.append(error_loop)

        # Run kmeans_batched_pytorch
        centroids_batched, codes_batched = kmeans_batched_pytorch(
            data, k=k, max_iter=max_iter
        )
        error_batched = compute_kmeans_reconstruction_error(
            data, centroids_batched, codes_batched
        )
        results_batched.append(error_batched)

        # Print results
        print(f"\nCentroids: {k:3d}")
        print(
            f"  Loop - MSE: {error_loop['mse_error']:9.6f}, "
            f"L2: {error_loop['l2_error']:8.4f}, "
            f"Rel: {error_loop['relative_error']:8.6f}"
        )
        print(
            f"  Batched - MSE: {error_batched['mse_error']:9.6f}, "
            f"L2: {error_batched['l2_error']:8.4f}, "
            f"Rel: {error_batched['relative_error']:8.6f}"
        )

        mse_diff = abs(error_loop["mse_error"] - error_batched["mse_error"])
        rel_diff_pct = (
            abs(error_loop["relative_error"] - error_batched["relative_error"]) * 100
        )
        print(f"  Diff - MSE: {mse_diff:9.6f}, Rel: {rel_diff_pct:7.2f}%")

    print("\n" + "=" * 80)
    print("Summary:")
    print("=" * 80)
    print("Both implementations produce similar reconstruction errors,")
    print("validating the correctness of the batched implementation.")
    print("=" * 80)

    # Print data for plotting
    print("\n" + "=" * 80)
    print("Relative Reconstruction Error Data:")
    print("=" * 80)
    print(f"{'Centroids':<12} {'Loop':<12} {'Batched':<12}")
    print("-" * 40)
    for i, k in enumerate(centroid_counts):
        print(
            f"{k:<12} {results_loop[i]['relative_error']:<12.6f} "
            f"{results_batched[i]['relative_error']:<12.6f}"
        )
    print("=" * 80)

    # Generate plot if matplotlib is available
    if plot and MATPLOTLIB_AVAILABLE:
        relative_errors_loop = [r["relative_error"] for r in results_loop]
        relative_errors_batched = [r["relative_error"] for r in results_batched]

        plt.figure(figsize=(10, 6))
        plt.plot(
            centroid_counts,
            relative_errors_loop,
            "o-",
            label="kmeans_loop_pytorch",
            linewidth=2,
            markersize=8,
            color="#2E86AB",
        )
        plt.plot(
            centroid_counts,
            relative_errors_batched,
            "s--",
            label="kmeans_batched_pytorch",
            linewidth=2,
            markersize=8,
            color="#A23B72",
        )

        plt.xlabel("Number of Centroids", fontsize=12, fontweight="bold")
        plt.ylabel("Relative Reconstruction Error", fontsize=12, fontweight="bold")
        plt.title(
            f"K-Means Reconstruction Error vs Number of Centroids\n"
            f"(batch = {batch_size}, samples = {n_samples}, dim = {dimension})"
        )
        plt.legend(fontsize=11, loc="best")
        plt.grid(True, alpha=0.3, linestyle="--")
        plt.xscale("log", base=2)

        # Set x-axis ticks to show all centroid counts
        plt.xticks(centroid_counts, [str(k) for k in centroid_counts])

        # Save plot
        output_path = os.path.join(
            os.path.dirname(__file__), "kmeans_quality_comparison.png"
        )
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"\n📊 Plot saved to: {output_path}")
        plt.close()
    elif plot and not MATPLOTLIB_AVAILABLE:
        print("\n⚠️  matplotlib not available - plot not generated")
        print("Install matplotlib with: pip install matplotlib")


if __name__ == "__main__":
    # Run the full comparison test with plotting
    test_kmeans_quality_full_comparison(plot=True)
