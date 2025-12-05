"""
:author: Aditya Desai
:copyright: 2025 Sparse Attention Hub
:license: Apache 2.0
:date: 2025-06-29
:summary: Tests for PQ utility functions.
"""

import pytest
import torch


def _check_sklearn_available() -> bool:
    """Check if sklearn is available for testing."""
    try:
        import sklearn  # noqa: F401

        return True
    except ImportError:
        return False


@pytest.mark.unit
class TestPQUtilityFunctions:
    def test_ip2l2_augment(self):
        """Test IP2L2 augmentation for key vectors."""
        from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed.implementations.utils.pq_utils import (
            ip2l2_augment,
        )

        n_groups = 2
        n_samples = 5
        d = 4

        torch.manual_seed(42)
        xb = torch.randn(n_groups, n_samples, d, dtype=torch.float32)

        xb_aug, phi = ip2l2_augment(xb)

        assert xb_aug.shape == (
            n_groups,
            n_samples,
            d + 1,
        ), f"Expected shape ({n_groups}, {n_samples}, {d + 1}), got {xb_aug.shape}"
        assert phi.shape == (
            n_groups,
            1,
            1,
        ), f"Expected phi shape ({n_groups}, 1, 1), got {phi.shape}"

        assert xb_aug.device == xb.device
        assert phi.device == xb.device

        norms_sq = (xb**2).sum(dim=2)
        max_norms_sq = norms_sq.max(dim=1)[0]

        for g in range(n_groups):
            expected_phi = max_norms_sq[g].item()
            computed_phi = phi[g, 0, 0].item()
            assert torch.isclose(
                torch.tensor(computed_phi),
                torch.tensor(expected_phi),
                rtol=1e-5,
                atol=1e-6,
            ), f"Group {g}: phi mismatch, expected {expected_phi:.6f}, got {computed_phi:.6f}"

        assert torch.equal(
            xb_aug[:, :, :d], xb
        ), "Original dimensions should remain unchanged"

        for g in range(n_groups):
            for i in range(n_samples):
                norm_sq = norms_sq[g, i].item()
                phi_val = phi[g, 0, 0].item()
                expected_aug = (phi_val - norm_sq) ** 0.5
                computed_aug = xb_aug[g, i, d].item()

                assert torch.isclose(
                    torch.tensor(computed_aug),
                    torch.tensor(expected_aug),
                    rtol=1e-5,
                    atol=1e-6,
                ), (
                    f"Group {g}, sample {i}: augmented value mismatch, "
                    f"expected {expected_aug:.6f}, got {computed_aug:.6f}"
                )

    def test_ip2l2_augment_queries(self):
        """Test IP2L2 augmentation for query vectors (should add zero column)."""
        from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed.implementations.utils.pq_utils import (
            ip2l2_augment_queries,
        )

        n_groups = 3
        n_queries = 4
        d = 8

        torch.manual_seed(42)
        xq = torch.randn(n_groups, n_queries, d, dtype=torch.float32)
        phi = torch.ones(n_groups, 1, 1, dtype=torch.float32) * 10.0

        xq_aug = ip2l2_augment_queries(xq, phi)

        assert xq_aug.shape == (
            n_groups,
            n_queries,
            d + 1,
        ), f"Expected shape ({n_groups}, {n_queries}, {d + 1}), got {xq_aug.shape}"

        assert xq_aug.device == xq.device

        assert torch.equal(
            xq_aug[:, :, :d], xq
        ), "Original dimensions should remain unchanged"

        zero_col = xq_aug[:, :, d]
        assert torch.all(
            zero_col == 0
        ), "Augmented column should be all zeros for query vectors"

        phi_different = torch.ones(n_groups, 1, 1, dtype=torch.float32) * 100.0
        xq_aug_different = ip2l2_augment_queries(xq, phi_different)

        assert torch.equal(
            xq_aug, xq_aug_different
        ), "Query augmentation should be independent of phi value"

        x = torch.tensor([[[1.0, 2.0]]], dtype=torch.float32)
        q = torch.tensor([[[3.0, 4.0]]], dtype=torch.float32)

        ip = torch.sum(x * q).item()

        from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed.implementations.utils.pq_utils import (
            ip2l2_augment,
        )

        x_aug, phi = ip2l2_augment(x)
        q_aug = ip2l2_augment_queries(q, phi)

        l2_sq = torch.sum((x_aug - q_aug) ** 2).item()
        q_norm_sq = torch.sum(q**2).item()
        phi_val = phi.item()
        expected_l2_sq = phi_val + q_norm_sq - 2 * ip

        assert torch.isclose(
            torch.tensor(l2_sq), torch.tensor(expected_l2_sq), rtol=1e-5, atol=1e-6
        ), f"L2 distance relationship doesn't hold: {l2_sq:.6f} != {expected_l2_sq:.6f}"

    def test_initialize_kmeans(self):
        """Test K-means initialization."""
        from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed.implementations.utils.pq_utils import (
            _initialize_kmeans,
        )

        n_samples: int = 20
        n_features: int = 4
        num_clusters: int = 3

        torch.manual_seed(42)
        X: torch.Tensor = torch.randn(n_samples, n_features, dtype=torch.float32)

        # Test with seed for reproducibility
        centers1: torch.Tensor = _initialize_kmeans(X, num_clusters, seed=42)
        centers2: torch.Tensor = _initialize_kmeans(X, num_clusters, seed=42)

        assert centers1.shape == (num_clusters, n_features)
        assert torch.equal(centers1, centers2), "Same seed should produce same centers"

        # Test without seed (should still have correct shape)
        centers_no_seed: torch.Tensor = _initialize_kmeans(X, num_clusters, seed=None)
        assert centers_no_seed.shape == (num_clusters, n_features)

        # Verify centers are from the original data
        for i in range(num_clusters):
            center: torch.Tensor = centers1[i]
            found: bool = False
            for j in range(n_samples):
                if torch.allclose(center, X[j], rtol=1e-5, atol=1e-6):
                    found = True
                    break
            assert found, f"Center {i} not in original data"

    def test_pairwise_distance(self):
        """Test pairwise distance computation."""
        from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed.implementations.utils.pq_utils import (
            _pairwise_distance,
        )

        # Create simple test data
        data1: torch.Tensor = torch.tensor(
            [[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32
        )
        data2: torch.Tensor = torch.tensor(
            [[0.0, 0.0], [1.0, 1.0]], dtype=torch.float32
        )

        distances: torch.Tensor = _pairwise_distance(data1, data2)

        assert distances.shape == (2, 2)

        # Verify known distances
        # distance from [1,0] to [0,0] = sqrt(1) = 1
        assert torch.isclose(distances[0, 0], torch.tensor(1.0), rtol=1e-5, atol=1e-6)
        # distance from [1,0] to [1,1] = sqrt(1) = 1
        assert torch.isclose(distances[0, 1], torch.tensor(1.0), rtol=1e-5, atol=1e-6)
        # distance from [0,1] to [0,0] = sqrt(1) = 1
        assert torch.isclose(distances[1, 0], torch.tensor(1.0), rtol=1e-5, atol=1e-6)
        # distance from [0,1] to [1,1] = sqrt(1) = 1
        assert torch.isclose(distances[1, 1], torch.tensor(1.0), rtol=1e-5, atol=1e-6)

    def test_kmeans_basic(self):
        """Test basic K-means clustering."""
        from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed.implementations.utils.pq_utils import (
            kmeans,
        )

        # Create data with clear clusters
        cluster1: torch.Tensor = torch.randn(10, 2) + torch.tensor([5.0, 5.0])
        cluster2: torch.Tensor = torch.randn(10, 2) + torch.tensor([-5.0, -5.0])
        X: torch.Tensor = torch.cat([cluster1, cluster2], dim=0)

        cluster_ids, centers = kmeans(X, num_clusters=2, seed=42, iter_limit=50)

        assert cluster_ids.shape == (20,)
        assert centers.shape == (2, 2)

        # Verify clusters are separated
        center_dist: torch.Tensor = torch.norm(centers[0] - centers[1])
        assert center_dist > 5.0, "Cluster centers should be well separated"

    def test_kmeans_invalid_distance(self):
        """Test that kmeans raises error for invalid distance metric."""
        from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed.implementations.utils.pq_utils import (
            kmeans,
        )

        X: torch.Tensor = torch.randn(10, 2)

        with pytest.raises(NotImplementedError, match="Distance .* not supported"):
            kmeans(X, num_clusters=2, distance="manhattan")

    def test_initialize_batched(self):
        """Test batched K-means initialization."""
        from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed.implementations.utils.pq_utils import (
            initialize_batched,
        )

        b: int = 3
        n: int = 20
        d: int = 4
        num_clusters: int = 5

        torch.manual_seed(42)
        X: torch.Tensor = torch.randn(b, n, d, dtype=torch.float32)

        centers: torch.Tensor = initialize_batched(X, num_clusters, seed=42)

        assert centers.shape == (b, num_clusters, d)

        # Test reproducibility
        centers2: torch.Tensor = initialize_batched(X, num_clusters, seed=42)
        assert torch.equal(centers, centers2), "Same seed should produce same centers"

    def test_pairwise_distance_batched(self):
        """Test batched pairwise distance computation."""
        from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed.implementations.utils.pq_utils import (
            pairwise_distance_batched,
        )

        b: int = 2
        n: int = 3
        k: int = 2

        # Create simple test data
        data1: torch.Tensor = torch.tensor(
            [
                [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
                [[2.0, 0.0], [0.0, 2.0], [2.0, 2.0]],
            ],
            dtype=torch.float32,
        )  # Shape: (2, 3, 2)

        data2: torch.Tensor = torch.tensor(
            [[[0.0, 0.0], [1.0, 1.0]], [[0.0, 0.0], [2.0, 2.0]]], dtype=torch.float32
        )  # Shape: (2, 2, 2)

        distances: torch.Tensor = pairwise_distance_batched(data1, data2)

        assert distances.shape == (b, n, k)

        # Verify known distance for batch 0, sample 0, cluster 0
        # distance from [1,0] to [0,0] = 1
        assert torch.isclose(
            distances[0, 0, 0], torch.tensor(1.0), rtol=1e-5, atol=1e-6
        )

    def test_kmeans_batched_basic(self):
        """Test batched K-means clustering."""
        from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed.implementations.utils.pq_utils import (
            kmeans_batched,
        )

        b: int = 2
        n: int = 20
        d: int = 2
        num_clusters: int = 2

        # Create data with clear clusters for each batch
        torch.manual_seed(42)
        cluster1: torch.Tensor = torch.randn(b, n // 2, d) + torch.tensor([5.0, 5.0])
        cluster2: torch.Tensor = torch.randn(b, n // 2, d) + torch.tensor([-5.0, -5.0])
        X: torch.Tensor = torch.cat([cluster1, cluster2], dim=1)

        cluster_ids, centers = kmeans_batched(X, num_clusters, seed=42, iter_limit=50)

        assert cluster_ids.shape == (b, n)
        assert centers.shape == (b, num_clusters, d)

        # Verify clusters are separated in each batch
        for i in range(b):
            center_dist: torch.Tensor = torch.norm(centers[i, 0] - centers[i, 1])
            assert center_dist > 3.0, f"Batch {i}: clusters should be well separated"

    def test_kmeans_batched_invalid_input(self):
        """Test that batched kmeans raises error for invalid input."""
        from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed.implementations.utils.pq_utils import (
            kmeans_batched,
        )

        # Test 2D input (should be 3D)
        X_2d: torch.Tensor = torch.randn(10, 5)

        with pytest.raises(ValueError, match="Expected 3D input"):
            kmeans_batched(X_2d, num_clusters=2)

        # Test invalid distance
        X_3d: torch.Tensor = torch.randn(2, 10, 5)
        with pytest.raises(
            NotImplementedError, match="Distance .* not yet implemented"
        ):
            kmeans_batched(X_3d, num_clusters=2, distance="cosine")

    @pytest.mark.skipif(not _check_sklearn_available(), reason="sklearn not available")
    def test_kmeans_loop_sklearn(self):
        """Test sklearn-based K-means loop."""
        from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed.implementations.utils.pq_utils import (
            kmeans_loop_sklearn,
        )

        n_groups: int = 2
        n_samples: int = 30
        d: int = 4
        k: int = 3

        torch.manual_seed(42)
        data: torch.Tensor = torch.randn(n_groups, n_samples, d, dtype=torch.float32)

        centroids, codes = kmeans_loop_sklearn(data, k, max_iter=50)

        assert centroids.shape == (n_groups, k, d)
        assert codes.shape == (n_groups, n_samples)
        assert codes.dtype == torch.int64

        # Verify codes are in valid range
        assert torch.all(codes >= 0) and torch.all(codes < k)

    def test_kmeans_loop_pytorch(self):
        """Test PyTorch loop-based K-means."""
        from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed.implementations.utils.pq_utils import (
            kmeans_loop_pytorch,
        )

        n_groups: int = 2
        n_samples: int = 20
        d: int = 3
        k: int = 2

        torch.manual_seed(42)
        data: torch.Tensor = torch.randn(n_groups, n_samples, d, dtype=torch.float32)

        centroids, codes = kmeans_loop_pytorch(data, k, max_iter=50)

        assert centroids.shape == (n_groups, k, d)
        assert codes.shape == (n_groups, n_samples)
        assert codes.dtype == torch.int64

        # Verify codes are in valid range
        assert torch.all(codes >= 0) and torch.all(codes < k)

    def test_kmeans_batched_pytorch(self):
        """Test batched PyTorch K-means."""
        from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed.implementations.utils.pq_utils import (
            kmeans_batched_pytorch,
        )

        n_groups: int = 2
        n_samples: int = 20
        d: int = 3
        k: int = 2

        torch.manual_seed(42)
        data: torch.Tensor = torch.randn(n_groups, n_samples, d, dtype=torch.float32)

        centroids, codes = kmeans_batched_pytorch(data, k, max_iter=50)

        assert centroids.shape == (n_groups, k, d)
        assert codes.shape == (n_groups, n_samples)
        assert codes.dtype == torch.int64

        # Verify codes are in valid range
        assert torch.all(codes >= 0) and torch.all(codes < k)

    def test_compute_reconstruction_errors(self):
        """Test reconstruction error computation."""
        from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed.implementations.utils.pq_utils import (
            compute_reconstruction_errors,
        )

        bsz: int = 1
        num_heads: int = 2
        n_keys: int = 8
        head_dim: int = 8
        pq_sub_dim: int = 4
        n_subvec: int = head_dim // pq_sub_dim
        cent_cnt: int = 4

        torch.manual_seed(42)
        original_keys: torch.Tensor = torch.randn(
            bsz, num_heads, n_keys, head_dim, dtype=torch.float32
        )

        # Create dummy centroids and codebook
        centroids: torch.Tensor = torch.randn(
            bsz, num_heads, n_subvec, cent_cnt, pq_sub_dim, dtype=torch.float32
        )
        codebook: torch.Tensor = torch.randint(
            0, cent_cnt, (bsz, n_keys, num_heads, n_subvec), dtype=torch.int64
        )

        errors = compute_reconstruction_errors(
            original_keys, centroids, codebook, pq_sub_dim, use_ip_metric=False
        )

        assert "mse_error" in errors
        assert "l2_error" in errors
        assert "relative_error" in errors

        assert isinstance(errors["mse_error"], float)
        assert isinstance(errors["l2_error"], float)
        assert isinstance(errors["relative_error"], float)

        assert errors["mse_error"] >= 0
        assert errors["l2_error"] >= 0
        assert 0 <= errors["relative_error"] <= 1.0 or errors["relative_error"] > 1.0

    def test_compute_reconstruction_errors_with_ip_metric(self):
        """Test reconstruction error computation with IP metric (augmented centroids)."""
        from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed.implementations.utils.pq_utils import (
            compute_reconstruction_errors,
        )

        bsz: int = 1
        num_heads: int = 2
        n_keys: int = 8
        head_dim: int = 8
        pq_sub_dim: int = 4
        n_subvec: int = head_dim // pq_sub_dim
        cent_cnt: int = 4

        torch.manual_seed(42)
        original_keys: torch.Tensor = torch.randn(
            bsz, num_heads, n_keys, head_dim, dtype=torch.float32
        )

        # Create dummy centroids with augmented dimension (d+1)
        centroids: torch.Tensor = torch.randn(
            bsz, num_heads, n_subvec, cent_cnt, pq_sub_dim + 1, dtype=torch.float32
        )
        codebook: torch.Tensor = torch.randint(
            0, cent_cnt, (bsz, n_keys, num_heads, n_subvec), dtype=torch.int64
        )

        errors = compute_reconstruction_errors(
            original_keys, centroids, codebook, pq_sub_dim, use_ip_metric=True
        )

        assert "mse_error" in errors
        assert "l2_error" in errors
        assert "relative_error" in errors

        assert errors["mse_error"] >= 0
        assert errors["l2_error"] >= 0
