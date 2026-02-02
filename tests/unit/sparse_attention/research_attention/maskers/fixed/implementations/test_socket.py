import re

import pytest
import torch

from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed.implementations.socket_top_k import (
    SocketMasker,
    SocketMaskerConfig,
)
from sparse_attention_hub.sparse_attention.utils.mask import Mask


@pytest.mark.unit
class TestSocketMaskerImplementation:
    """Tests for SocketMasker (bucket attention)."""

    def test_bucket_masker_config_creation(self):
        """Config can be created and fields are set correctly."""
        config = SocketMaskerConfig(
            heavy_size=0.05,
            K=4,
            L=2,
            tau=0.6,
        )
        assert config is not None
        assert config.heavy_size == 0.05
        assert config.K == 4
        assert config.L == 2
        assert config.tau == 0.6

    def test_bucket_masker_config_validation(self):
        from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed.implementations.socket_top_k import (
            SocketMasker,
            SocketMaskerConfig,
        )

        msg = "K must be positive"

        with pytest.raises(ValueError, match=re.escape(msg)):
            config = SocketMaskerConfig(heavy_size=0.05, K=0, L=1, tau=0.4)
            SocketMasker(config)

        msg = "L must be positive"
        with pytest.raises(ValueError, match=re.escape(msg)):
            config = SocketMaskerConfig(heavy_size=0.05, K=4, L=0, tau=0.4)
            SocketMasker(config)

    def test_bucket_masker_invalid_tau_raises(self):
        """Non-positive tau should raise a ValueError during hashing."""
        config = SocketMaskerConfig(
            heavy_size=0.1,
            K=2,
            L=1,
            tau=0.0,
        )
        masker = SocketMasker.create_from_config(config)
        keys, queries, values, attention_mask, previous_mask = self._make_dummy_inputs()

        with pytest.raises(ValueError, match=r"tau must be > 0"):
            masker.add_mask(
                keys=keys,
                queries=queries,
                values=values,
                attention_mask=attention_mask,
                scaling=1.0,
                dropout=0.0,
                sparse_meta_data={},
                previous_mask=previous_mask,
            )

    def test_bucket_masker_creation(self):
        """SocketMasker can be created from config."""
        config = SocketMaskerConfig(
            heavy_size=0.05,
            K=4,
            L=2,
            tau=0.4,
        )
        masker = SocketMasker.create_from_config(config)
        assert isinstance(masker, SocketMasker)
        # Optional: check that config got attached
        assert masker.heavy_size == config.heavy_size
        assert masker.P == config.K
        assert masker.L == config.L
        assert masker.tau == config.tau

    def _make_dummy_inputs(self, device="cpu"):
        """Helper to create small synthetic Q/K/V + attention_mask."""
        B, H, N, Q, D = 2, 4, 16, 3, 8
        torch.manual_seed(0)

        keys = torch.randn(B, H, N, D, device=device)
        queries = torch.randn(B, H, Q, D, device=device)
        values = torch.randn(B, H, N, D, device=device)

        # Standard [B,1,1,N] additive mask: allow all
        attention_mask = torch.zeros(B, 1, 1, N, device=device)

        # Empty previous mask (all zeros)
        dense_prev = torch.zeros(B, H, Q, N, device=device)
        previous_mask = Mask.create_mask_from_dense_mask(
            (B, H, Q, N), dense_prev, dtype=torch.float32
        )
        return keys, queries, values, attention_mask, previous_mask

    def test_bucket_masker_basic_add_mask_shapes(self):
        """add_mask should produce a Mask with correct dense shape."""
        config = SocketMaskerConfig(
            heavy_size=0.25,  # select about 25% of tokens
            K=4,
            L=2,
        )
        masker = SocketMasker.create_from_config(config)
        keys, queries, values, attention_mask, previous_mask = self._make_dummy_inputs()

        new_mask = masker.add_mask(
            keys=keys,
            queries=queries,
            values=values,
            attention_mask=attention_mask,
            scaling=1.0,
            dropout=0.0,
            sparse_meta_data={},
            previous_mask=previous_mask,
        )
        assert isinstance(new_mask, Mask)

        dense = new_mask.get_dense_mask()
        B, H, Q, N = 2, 4, 3, 16
        assert dense.shape == (B, H, Q, N)

        # Values should be between 0 and 1
        assert dense.min() >= 0.0
        assert dense.max() <= 1.0

    def test_bucket_masker_respects_heavy_size_budget(self):
        """Total selected tokens per (B,H,Q) should not exceed heavy_size-based budget."""
        B, H, Q, N = 2, 4, 3, 32
        config = SocketMaskerConfig(
            heavy_size=0.25,  # about 8 tokens out of 32
            K=4,
            L=2,
        )
        masker = SocketMasker.create_from_config(config)

        keys = torch.randn(B, H, N, 8)
        queries = torch.randn(B, H, Q, 8)
        values = torch.randn(B, H, N, 8)
        attention_mask = torch.zeros(B, 1, 1, N)
        prev = Mask.create_mask_from_dense_mask(
            (B, H, Q, N),
            torch.zeros(B, H, Q, N),
            dtype=torch.float32,
        )

        new_mask = masker.add_mask(
            keys=keys,
            queries=queries,
            values=values,
            attention_mask=attention_mask,
            scaling=1.0,
            dropout=0.0,
            sparse_meta_data={},
            previous_mask=prev,
        )
        dense = new_mask.get_dense_mask()  # [B,H,Q,N]

        # Compute effective heavy tokens as used inside the masker
        effective_M = masker._calculate_effective_size(masker.heavy_size, N)
        # For each (b,h,q) row, number of active tokens should be <= effective_M
        active_per_row = (dense > 0).sum(dim=-1)  # [B,H,Q]
        assert torch.all(active_per_row <= effective_M)

    def test_bucket_masker_attention_mask_boolean(self):
        """Blocked positions in a boolean attention_mask should remain masked out."""
        config = SocketMaskerConfig(
            heavy_size=0.5,
            K=4,
            L=2,
        )
        masker = SocketMasker.create_from_config(config)

        B, H, N, Q, D = 1, 2, 16, 2, 8
        keys = torch.randn(B, H, N, D)
        queries = torch.randn(B, H, Q, D)
        values = torch.randn(B, H, N, D)

        # Boolean mask: allow first half, forbid second half
        attention_mask = torch.zeros(B, 1, 1, N, dtype=torch.bool)
        attention_mask[..., N // 2 :] = True  # blocked

        prev = Mask.create_mask_from_dense_mask(
            (B, H, Q, N),
            torch.zeros(B, H, Q, N),
            dtype=torch.float32,
        )

        new_mask = masker.add_mask(
            keys=keys,
            queries=queries,
            values=values,
            attention_mask=attention_mask,
            scaling=1.0,
            dropout=0.0,
            sparse_meta_data={},
            previous_mask=prev,
        )
        dense = new_mask.get_dense_mask()  # [B,H,Q,N]

        # Second half must be zeroed out by gating
        tail = dense[..., N // 2 :]
        assert torch.all(tail == 0.0)

    def test_bucket_masker_attention_mask_additive(self):
        """Blocked positions in an additive mask (<0) should remain masked out."""
        config = SocketMaskerConfig(
            heavy_size=0.5,
            K=4,
            L=2,
        )
        masker = SocketMasker.create_from_config(config)

        B, H, N, Q, D = 1, 2, 16, 2, 8
        keys = torch.randn(B, H, N, D)
        queries = torch.randn(B, H, Q, D)
        values = torch.randn(B, H, N, D)

        # Additive mask: 0 = allowed, -1e9 = blocked
        attention_mask = torch.zeros(B, 1, 1, N)
        attention_mask[..., N // 2 :] = -1e9  # blocked

        prev = Mask.create_mask_from_dense_mask(
            (B, H, Q, N),
            torch.zeros(B, H, Q, N),
            dtype=torch.float32,
        )

        new_mask = masker.add_mask(
            keys=keys,
            queries=queries,
            values=values,
            attention_mask=attention_mask,
            scaling=1.0,
            dropout=0.0,
            sparse_meta_data={},
            previous_mask=prev,
        )
        dense = new_mask.get_dense_mask()  # [B,H,Q,N]

        # Second half must be zeroed out
        tail = dense[..., N // 2 :]
        assert torch.all(tail == 0.0)

    def test_bucket_masker_deterministic_given_seed(self):
        """With the same config and inputs, SocketMasker should be deterministic."""
        config = SocketMaskerConfig(
            heavy_size=0.25,
            K=4,
            L=2,
        )
        masker1 = SocketMasker.create_from_config(config)
        masker2 = SocketMasker.create_from_config(config)

        keys, queries, values, attention_mask, previous_mask = self._make_dummy_inputs()

        out1 = masker1.add_mask(
            keys, queries, values, attention_mask, 1.0, 0.0, {}, previous_mask
        )
        out2 = masker2.add_mask(
            keys, queries, values, attention_mask, 1.0, 0.0, {}, previous_mask
        )

        dense1 = out1.get_dense_mask()
        dense2 = out2.get_dense_mask()
        assert torch.allclose(dense1, dense2)
