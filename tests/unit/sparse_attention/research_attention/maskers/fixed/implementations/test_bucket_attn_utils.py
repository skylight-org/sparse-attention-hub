import pytest
import torch

from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed.implementations.utils.bucket_utils import (
    get_hyper_planes,
    get_protos_T,
    hard_hash,
    soft_hash,
    get_collision_counts,
    attention_mask_to_allowed_prob,
    pack_bits,
)


@pytest.mark.unit
class TestBucketUtils:
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
        assert torch.all(torch.isin(protos1, torch.tensor([-1.0, 1.0])))

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
        # bits: [..., P]
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
        codes = pack_bits(bits)  # [5]
        expected = torch.tensor([0, 1, 2, 8, 15], dtype=torch.int64)
        assert torch.equal(codes, expected)

    def test_hard_hash_simple_planes(self):
        """hard_hash should assign the same buckets for identical inputs and respect planes."""
        # Use simple deterministic planes so behavior is predictable
        B, H, N, D = 1, 1, 2, 2
        L, P = 1, 2

        # Planes: identity-like projections
        planes = torch.tensor(
            [
                [  # table 0
                    [1.0, 0.0],  # hyperplane 0
                    [0.0, 1.0],  # hyperplane 1
                ]
            ]
        )  # [L,P,D]

        # Two keys: [1,1] and [-1,-1]
        keys = torch.tensor([[[[1.0, 1.0], [-1.0, -1.0]]]])  # [B,H,N,D]

        codes = hard_hash(keys, planes)  # [B,H,L,N]
        assert codes.shape == (B, H, L, N)

        # First key: projections [1,1] => bits [1,1] => code b'11' = 3
        # Second key: projections [-1,-1] => bits [0,0] => code b'00' = 0
        assert codes[0, 0, 0, 0].item() == 3
        assert codes[0, 0, 0, 1].item() == 0

        # Identical keys => identical codes
        keys2 = keys.clone()
        codes2 = hard_hash(keys2, planes)
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

        q_probs = soft_hash(queries, planes, protos_T)  # [B,H,Q,L,R]
        assert q_probs.shape == (B, H, Q, L, R)

        # Probabilities should be non-negative and sum to ~1 along R
        assert torch.all(q_probs >= 0)
        probs_sum = q_probs.sum(dim=-1)  # [B,H,Q,L]
        assert torch.allclose(
            probs_sum, torch.ones_like(probs_sum), atol=1e-5, rtol=1e-5
        )

    def test_get_collision_counts_tiny_example(self):
        """get_collision_counts should correctly compute candidate_mask and collision_counts."""
        # Small hand-constructed example
        # B=1,H=1,L=2,N=3, Q=2, top_t=1
        # Table 0 buckets: [0, 1, 2]
        # Table 1 buckets: [1, 1, 0]
        key_buckets = torch.tensor(
            [
                [  # B
                    [  # H
                        [0, 1, 2],  # L=0
                        [1, 1, 0],  # L=1
                    ]
                ]
            ]
        )  # [1,1,2,3] => [B,H,L,N]

        # For q0: in table 0 pick bucket 1, in table 1 pick bucket 0
        # For q1: in table 0 pick bucket 2, in table 1 pick bucket 1
        top_buckets = torch.tensor(
            [
                [
                    [
                        [
                            [1],  # q0, L=0
                            [0],  # q0, L=1
                        ],
                        [
                            [2],  # q1, L=0
                            [1],  # q1, L=1
                        ],
                    ]
                ]
            ]
        )  # shape: [1, 1, 2, 2, 1]

        candidate_mask, collision_counts = get_collision_counts(
            key_buckets, top_buckets
        )
        # Shapes
        assert candidate_mask.shape == (1, 1, 2, 3)
        assert collision_counts.shape == (1, 1, 2, 3)

        # Let's reason expected collisions:
        # keys indices: i=0,1,2

        # q0:
        #  table 0 bucket=1 -> matches key1 only
        #  table 1 bucket=0 -> matches key2 only
        # => collisions(q0) = [0,1,1]
        expected_coll_q0 = torch.tensor([0, 1, 1])

        # q1:
        #  table 0 bucket=2 -> matches key2 only
        #  table 1 bucket=1 -> matches key0? no, key0=1 in T1? actually T1: [1,1,0]
        #   -> matches key0 and key1
        # => collisions(q1) = [1,1,1]  (key2 matched in table0 only)
        expected_coll_q1 = torch.tensor([1, 1, 1])

        assert torch.equal(collision_counts[0, 0, 0], expected_coll_q0)
        assert torch.equal(collision_counts[0, 0, 1], expected_coll_q1)

        # candidate_mask is True where collisions > 0
        assert torch.equal(candidate_mask, collision_counts > 0)

    def test_attention_mask_to_allowed_prob_bool(self):
        """attention_mask_to_allowed_prob for boolean masks."""
        B, K = 2, 5
        # True = blocked, False = allowed
        attention_mask = torch.tensor(
            [
                [[False, False, True, True, False]],
                [[True, False, True, False, False]],
            ],
            dtype=torch.bool,
        )  # [B,1,K] or [B,*,K]

        allowed_prob = attention_mask_to_allowed_prob(attention_mask, K)
        # expected: allowed_prob = 1 where False, 0 where True
        expected = (~attention_mask).to(torch.float32).unsqueeze(1)  # [B,1,1,K]
        assert allowed_prob.shape == (B, 1, 1, K)
        assert torch.equal(allowed_prob, expected)

    def test_attention_mask_to_allowed_prob_additive(self):
        """attention_mask_to_allowed_prob for additive (float) masks."""
        B, K = 1, 4
        # >=0 => allowed (1.0), <0 => forbidden (0.0)
        attention_mask = torch.tensor([[[0.0, 1.0, -1e9, -0.5]]])  # [B,1,K]

        allowed_prob = attention_mask_to_allowed_prob(attention_mask, K)
        assert allowed_prob.shape == (B, 1, 1, K)

        # positions 0,1 >=0 => 1.0; positions 2,3 <0 => 0.0
        expected = torch.tensor([[[[1.0, 1.0, 0.0, 0.0]]]])
        assert torch.equal(allowed_prob, expected)
