import pytest
import torch

from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed.implementations.utils.socket_utils import (
    attention_mask_to_allowed_prob,
    get_hyper_planes,
    get_protos_T,
    hard_hash,
    pack_bits,
    soft_hash,
    topk_soft_collision_scores_blockwise,
)


@pytest.mark.unit
class TestSocketUtils:
    def test_get_hyper_planes_basic(self):
        """get_hyper_planes returns correctly-shaped, cached tensors."""
        cache = {}
        D, L, P = 16, 3, 4
        device = torch.device("cpu")
        dtype = torch.float32

        planes1 = get_hyper_planes(
            cache=cache,
            D=D,
            L=L,
            P=P,
            device=device,
            dtype=dtype,
            rng=torch.Generator(device=device).manual_seed(0),
        )
        assert planes1.shape == (L, P, D)
        assert planes1.dtype == dtype
        assert len(cache) == 1

        # Same key -> same object from cache (no reallocation)
        planes2 = get_hyper_planes(
            cache=cache,
            D=D,
            L=L,
            P=P,
            device=device,
            dtype=dtype,
            rng=torch.Generator(device=device).manual_seed(123),
        )
        assert planes2 is planes1

        # Different (D,L,P) -> new entry in cache
        planes3 = get_hyper_planes(
            cache=cache,
            D=8,
            L=L,
            P=P,
            device=device,
            dtype=dtype,
            rng=torch.Generator(device=device).manual_seed(0),
        )
        assert planes3.shape == (L, P, 8)
        assert planes3 is not planes1
        assert len(cache) == 2

    def test_get_protos_T_basic(self):
        """get_protos_T returns hypercube corners with correct shape and caching."""
        cache = {}
        P = 3
        R = 1 << P
        device = torch.device("cpu")
        dtype = torch.float32

        protos1 = get_protos_T(
            cache=cache,
            P=P,
            device=device,
            dtype=dtype,
        )
        assert protos1.shape == (P, R)
        assert protos1.dtype == dtype
        assert len(cache) == 1

        # All entries must be ±1
        assert torch.all(torch.isin(protos1, torch.tensor([-1.0, 1.0], device=device)))

        # Same key -> cached
        protos2 = get_protos_T(
            cache=cache,
            P=P,
            device=device,
            dtype=dtype,
        )
        assert protos2 is protos1

        # Different P -> new entry
        protos3 = get_protos_T(
            cache=cache,
            P=P + 1,
            device=device,
            dtype=dtype,
        )
        assert protos3.shape == (P + 1, 1 << (P + 1))
        assert len(cache) == 2

    def test_pack_bits_known_values(self):
        """pack_bits should pack bit patterns into integers in big-endian order."""
        bits = torch.tensor(
            [
                [0, 0, 0, 0],  # 0
                [0, 0, 0, 1],  # 1
                [0, 0, 1, 0],  # 2
                [1, 0, 0, 0],  # 8
                [1, 1, 1, 1],  # 15
            ],
            dtype=torch.bool,
        )
        codes = pack_bits(bits)

        expected = torch.tensor([0, 1, 2, 8, 15], dtype=codes.dtype)
        assert torch.equal(codes, expected)

    def test_hard_hash_simple_planes(self):
        """hard_hash should assign predictable buckets for simple planes."""
        B, H, N = 1, 1, 2
        L = 1

        planes = torch.tensor(
            [
                [  # table 0
                    [1.0, 0.0],  # hyperplane 0
                    [0.0, 1.0],  # hyperplane 1
                ]
            ],
            dtype=torch.float32,
        )  # [L,P,D]

        keys = torch.tensor(
            [[[[1.0, 1.0], [-1.0, -1.0]]]], dtype=torch.float32
        )  # [B,H,N,D]

        codes = hard_hash(keys, planes)  # [B,H,L,N]
        assert codes.shape == (B, H, L, N)

        # First key: projections [1,1] => bits [1,1] => code b'11' = 3
        # Second key: projections [-1,-1] => bits [0,0] => code b'00' = 0
        assert codes[0, 0, 0, 0].item() == 3
        assert codes[0, 0, 0, 1].item() == 0

        # Identical keys => identical codes
        codes2 = hard_hash(keys.clone(), planes)
        assert torch.equal(codes, codes2)

    def test_soft_hash_shapes_and_probs(self):
        """soft_hash returns valid probability distributions per bucket."""
        B, H, Q, D = 2, 3, 4, 5
        L, P = 2, 3
        R = 1 << P

        torch.manual_seed(0)
        queries = torch.randn(B, H, Q, D)
        planes = torch.randn(L, P, D)
        protos_T = get_protos_T(
            cache={},
            P=P,
            device=queries.device,
            dtype=queries.dtype,
        )  # [P,R]

        q_probs = soft_hash(queries, planes, protos_T, tau=0.7)  # [B,H,Q,L,R]
        assert q_probs.shape == (B, H, Q, L, R)

        assert torch.all(q_probs >= 0)
        probs_sum = q_probs.sum(dim=-1)  # [B,H,Q,L]
        assert torch.allclose(
            probs_sum, torch.ones_like(probs_sum), atol=1e-5, rtol=1e-5
        )

    def test_soft_hash_invalid_tau_raises(self):
        """soft_hash should raise for tau <= 0."""
        B, H, Q, D = 1, 1, 2, 4
        L, P = 1, 2
        queries = torch.randn(B, H, Q, D)
        planes = torch.randn(L, P, D)
        protos_T = get_protos_T(
            cache={}, P=P, device=queries.device, dtype=queries.dtype
        )

        with pytest.raises(ValueError, match=r"tau must be > 0"):
            _ = soft_hash(queries, planes, protos_T, tau=0.0)

    def test_attention_mask_to_allowed_prob_bool(self):
        """attention_mask_to_allowed_prob for boolean masks."""
        B, K = 2, 5
        attention_mask = torch.tensor(
            [
                [[False, False, True, True, False]],
                [[True, False, True, False, False]],
            ],
            dtype=torch.bool,
        )  # [B,1,K] (dim=3 -> will be unsqueezed to [B,1,1,K])

        allowed_prob = attention_mask_to_allowed_prob(attention_mask, K)
        expected = (~attention_mask).to(torch.float32).unsqueeze(1)  # [B,1,1,K]
        assert allowed_prob.shape == (B, 1, 1, K)
        assert torch.equal(allowed_prob, expected)

    def test_attention_mask_to_allowed_prob_additive(self):
        """attention_mask_to_allowed_prob for additive (float) masks."""
        B, K = 1, 4
        attention_mask = torch.tensor([[[0.0, 1.0, -1e9, -0.5]]])  # [B,1,K]

        allowed_prob = attention_mask_to_allowed_prob(attention_mask, K)
        assert allowed_prob.shape == (B, 1, 1, K)

        expected = torch.tensor([[[[1.0, 1.0, 0.0, 0.0]]]])
        assert torch.equal(allowed_prob, expected)

    def test_topk_soft_collision_scores_blockwise_matches_naive(self):
        """
        Compare topk_soft_collision_scores_blockwise against a naive full-score implementation
        on a small example.
        """
        torch.manual_seed(0)
        B, H, Q, L, P = 1, 2, 3, 2, 3
        R = 1 << P
        N = 16
        M = 5

        # Random q_probs, normalize over R
        q_probs = torch.rand(B, H, Q, L, R)
        q_probs = q_probs / q_probs.sum(dim=-1, keepdim=True)

        # Random buckets in [0, R)
        key_buckets = torch.randint(low=0, high=R, size=(B, H, L, N), dtype=torch.int16)

        # Value magnitudes (positive)
        v_mag = torch.rand(B, H, N) + 0.1

        # Allow all
        allowed_bool = torch.ones(B, H, Q, N, dtype=torch.bool)

        # Blockwise result
        idx_blk, scores_blk = topk_soft_collision_scores_blockwise(
            q_probs=q_probs,
            key_buckets=key_buckets,
            v_mag=v_mag,
            allowed_bool=allowed_bool,
            M=M,
            block=7,  # small block to exercise multi-block path
        )

        # Naive full computation (float32 for stability, like the kernel)
        q_probs_f = q_probs.float()
        v_mag_f = v_mag.float()
        scores_full = torch.zeros(B, H, Q, N, dtype=torch.float32)
        for table_idx in range(L):
            probs_l = q_probs_f[:, :, :, table_idx, :]  # [B,H,Q,R]
            buckets_l = key_buckets[:, :, table_idx, :].to(torch.long)  # [B,H,N]
            idx = buckets_l.unsqueeze(2).expand(B, H, Q, N)  # [B,H,Q,N]
            scores_full += torch.gather(probs_l, dim=-1, index=idx)
        scores_full = scores_full * v_mag_f.unsqueeze(2)  # [B,H,Q,N]

        # Top-M naive
        top = torch.topk(scores_full, k=M, dim=-1, largest=True)
        idx_nv = top.indices

        # Compare as sets per row (ordering may match, but set comparison is more robust)
        # 1) scores should match for the chosen indices
        # 2) indices should match (as a set)
        for b in range(B):
            for h in range(H):
                for q in range(Q):
                    s1 = set(idx_blk[b, h, q].tolist())
                    s2 = set(idx_nv[b, h, q].tolist())
                    assert s1 == s2

                    # Scores at those indices should match closely
                    # (kernel uses iterative merge topk but should be exact for deterministic math)
                    gathered_nv = scores_full[b, h, q, idx_blk[b, h, q]].sort().values
                    gathered_blk = scores_blk[b, h, q].sort().values
                    assert torch.allclose(
                        gathered_blk, gathered_nv, atol=1e-6, rtol=1e-6
                    )

    def test_topk_soft_collision_scores_respects_allowed_bool(self):
        """Ensure disallowed positions are never returned in top indices."""
        torch.manual_seed(0)
        B, H, Q, L, P = 1, 1, 1, 2, 2
        R = 1 << P
        N = 8
        M = 4

        q_probs = torch.rand(B, H, Q, L, R)
        q_probs = q_probs / q_probs.sum(dim=-1, keepdim=True)

        key_buckets = torch.randint(low=0, high=R, size=(B, H, L, N), dtype=torch.int16)
        v_mag = torch.ones(B, H, N)

        allowed_bool = torch.ones(B, H, Q, N, dtype=torch.bool)
        allowed_bool[..., -3:] = False  # forbid last 3 keys

        idx, scores = topk_soft_collision_scores_blockwise(
            q_probs=q_probs,
            key_buckets=key_buckets,
            v_mag=v_mag,
            allowed_bool=allowed_bool,
            M=M,
            block=16,
        )

        # No index may be in the forbidden range
        assert torch.all(idx < (N - 3))
        # Scores should be finite (since enough allowed exist)
        assert torch.all(torch.isfinite(scores))
