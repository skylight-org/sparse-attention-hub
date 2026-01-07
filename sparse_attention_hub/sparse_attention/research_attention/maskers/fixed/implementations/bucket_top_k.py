from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch

from sparse_attention_hub.sparse_attention.research_attention.maskers.base import (
    AttentionTensorDimensions,
    MaskerConfig,
    MaskerRegistry,
)
from sparse_attention_hub.sparse_attention.utils.kv_utils import (
    _get_num_key_value_groups,
    repeat_kv,
)
from sparse_attention_hub.sparse_attention.utils.mask import Mask

from ..base import TopKMasker, TopKMaskerConfig
from .utils.bucket_utils import (
    attention_mask_to_allowed_prob,
    build_inverted_index_csr,
    get_collision_counts_indexed,
    get_hyper_planes,
    get_protos_T,
    hard_hash,
    soft_hash,
)



@dataclass
class BucketMaskerConfig(TopKMaskerConfig):
    """
    Minimal masker config:
      • K:      # hyperplanes per table (buckets = 2**K)
      • L:      # hash tables
      • top_t:  # buckets selected per table (per (B,H,Q))

    heavy_size (inherited) is used as the sample budget M.
    """
    K: int = 4
    L: int = 1
    top_t: int = 4


@MaskerRegistry.register(BucketMaskerConfig)
class BucketMasker(TopKMasker):
    """
    L-table mask-only RACE-style:
      1) Hard-hash keys with L sets of K planes  -> bucket id per table.
      2) Soft-hash queries per table            -> probs over R=2^K buckets.
      3) Select top_t buckets per table.
      4) Candidate mask based on collision count threshold (adaptive m).
      5) Within candidates, pick top-Km by score = collision_counts * ||v_i||.
    """

    def __init__(self, config: BucketMaskerConfig) -> None:
        super().__init__(config)

        if config.K <= 0:
            raise ValueError("K must be positive")
        if config.L <= 0:
            raise ValueError("L must be positive")
        if config.top_t <= 0:
            raise ValueError("top_t must be positive")

        self.P: int = int(config.K)
        self.L: int = int(config.L)
        self.top_t: int = int(config.top_t)
        self.heavy_size = config.heavy_size

        self._planes_cache: Dict[Tuple[int, torch.device, torch.dtype, int, int], torch.Tensor] = {}
        self._protos_cache: Dict[Tuple[int, torch.device, torch.dtype], torch.Tensor] = {}

        self._seed = 123456789
        self._rng_cache: Dict[torch.device, torch.Generator] = {}

        self.reset_candidate_stats()

    # ----------------------------
    # Candidate stats accumulators
    # ----------------------------
    def reset_candidate_stats(self) -> None:
        self._cand_rows: int = 0
        self._cand_sum: float = 0.0
        self._cand_min: int = 1 << 30
        self._cand_max: int = 0

        self._final_rows: int = 0
        self._final_sum: float = 0.0
        self._final_min: int = 1 << 30
        self._final_max: int = 0

    def get_candidate_stats(self) -> Dict[str, float]:
        out: Dict[str, float] = {}

        if self._cand_rows == 0:
            out.update({"cand_avg": float("nan"), "cand_min": float("nan"), "cand_max": float("nan")})
        else:
            out.update(
                {
                    "cand_avg": self._cand_sum / self._cand_rows,
                    "cand_min": float(self._cand_min),
                    "cand_max": float(self._cand_max),
                }
            )

        if self._final_rows == 0:
            out.update({"final_avg": float("nan"), "final_min": float("nan"), "final_max": float("nan")})
        else:
            out.update(
                {
                    "final_avg": self._final_sum / self._final_rows,
                    "final_min": float(self._final_min),
                    "final_max": float(self._final_max),
                }
            )
        return out

    def _accum_stats_1d(self, counts_1d: torch.Tensor, kind: str) -> None:
        rows = counts_1d.numel()
        if rows == 0:
            return

        s = int(counts_1d.sum().item())
        mn = int(counts_1d.min().item())
        mx = int(counts_1d.max().item())

        if kind == "cand":
            self._cand_rows += rows
            self._cand_sum += float(s)
            self._cand_min = min(self._cand_min, mn)
            self._cand_max = max(self._cand_max, mx)
        elif kind == "final":
            self._final_rows += rows
            self._final_sum += float(s)
            self._final_min = min(self._final_min, mn)
            self._final_max = max(self._final_max, mx)
        else:
            raise ValueError(f"Unknown stats kind: {kind}")

    def _rng(self, device: torch.device) -> Optional[torch.Generator]:
        if self._seed is None:
            return None
        g = self._rng_cache.get(device)
        if g is None:
            g = torch.Generator(device=device)
            g.manual_seed(self._seed + 7777)
            self._rng_cache[device] = g
        return g

    # ----------
    # Main API
    # ----------
    def add_mask(
        self,
        keys: torch.Tensor,                 # [B, H_k or G, N, D]
        queries: torch.Tensor,              # [B, H, Q, D]
        values: torch.Tensor,               # [B, H_k or G, N, Dv]
        attention_mask: Optional[torch.Tensor],
        scaling: float,
        dropout: float,
        sparse_meta_data: Dict,
        previous_mask: Mask,
        **kwargs,
    ) -> Mask:
        if previous_mask.is_full_mask():
            return previous_mask

        dims: AttentionTensorDimensions = self._extract_tensor_dimensions(keys, queries)

        heavy_tokens: int = self._calculate_effective_size(self.heavy_size, dims.seq_len_keys)
        if self._should_use_full_attention(dims, heavy_tokens):
            return self._create_full_mask(dims, previous_mask.dtype, previous_mask.device)

        # 1) Align to MHA heads if KV are grouped (GQA/MQA)
        ngroups = _get_num_key_value_groups(queries, keys)
        keys_rep = repeat_kv(keys, ngroups)  # [B, H, N, D]
        B, H, N, D = keys_rep.shape
        _, _, Q, _ = queries.shape

        # 2) SRP planes & corners
        planes = get_hyper_planes(
            cache=self._planes_cache,
            D=D,
            L=self.L,
            P=self.P,
            device=keys_rep.device,
            dtype=keys_rep.dtype,
            rng=self._rng(keys_rep.device),
        )  # [L, P, D]

        protosT = get_protos_T(
            cache=self._protos_cache,
            P=self.P,
            device=keys_rep.device,
            dtype=keys_rep.dtype,
        )  # [P, R]

        R = 1 << self.P
        top_t = max(1, min(self.top_t, R))

        # 3) Hard-hash keys per table -> [B, H, L, N]
        key_buckets = hard_hash(keys_rep, planes)

        # 3.5) Build inverted index once per forward
        R = 1 << self.P
        perm, offsets = build_inverted_index_csr(key_buckets, R)

        # 4) Soft-hash queries -> probs [B, H, Q, L, R]
        q_probs = soft_hash(queries, planes, protosT)

        # 5) Top buckets per table -> [B, H, Q, L, top_t]
        top_buckets = torch.topk(q_probs, k=top_t, dim=-1, largest=True).indices

        # 6) Collision counts via inverted index (no scan)
        _, collision_counts = get_collision_counts_indexed(
            perm=perm,
            offsets=offsets,
            top_buckets=top_buckets,
            N=N,
        )  # [B,H,Q,N] int16


        # External attention mask -> allowed_bool [B,H,Q,N]
        allowed_prob = None
        if attention_mask is not None:
            allowed_prob = attention_mask_to_allowed_prob(attention_mask, N)  # [B,1,*,N]
            allowed_bool = (allowed_prob > 0)
            allowed_bool = allowed_bool.expand(B, H, Q, N)
        else:
            allowed_bool = torch.ones((B, H, Q, N), device=keys_rep.device, dtype=torch.bool)

        # -------------------------------
        # Adaptive-m candidate selection
        # -------------------------------
        M = max(0, min(int(self._calculate_effective_size(self.heavy_size, N)), N))
        if M == 0:
            return previous_mask
        Km = min(M, N)

        cand_m3 = (collision_counts >= 3)
        cand_m2 = (collision_counts >= 2)
        cand_m1 = (collision_counts >= 1)

        need3 = max(1, int(0.8 * M))
        need2 = max(1, int(0.8 * M))

        cnt3 = cand_m3.sum(dim=-1)  # [B,H,Q]
        cnt2 = cand_m2.sum(dim=-1)  # [B,H,Q]

        use_m2 = (cnt3 < need3).unsqueeze(-1)  # [B,H,Q,1]
        use_m1 = ((cnt2 < need2) & use_m2.squeeze(-1)).unsqueeze(-1)

        candidate_mask = torch.where(
            use_m2,
            torch.where(use_m1, cand_m1, cand_m2),
            cand_m3,
        )
        candidate_mask = candidate_mask & allowed_bool

        # No dense fallback: only a tiny sink to avoid empty rows.
        no_cands = ~candidate_mask.any(dim=-1, keepdim=True)  # [B,H,Q,1]
        sink_fallback = 1  # set 1/4/8/16 if you want more stability

        if sink_fallback > 0:
            sink_idx = torch.arange(min(sink_fallback, N), device=keys_rep.device)
            sink_mask = torch.zeros_like(candidate_mask)
            sink_mask.index_fill_(-1, sink_idx, True)
            sink_mask = sink_mask & allowed_bool
            candidate_mask = torch.where(no_cands, sink_mask, candidate_mask)

        # ---- stats: candidate size ----
        cand_counts = candidate_mask.sum(dim=-1)  # [B,H,Q]
        self._accum_stats_1d(cand_counts, kind="cand")

        # 7) Value-aware score
        v_rep = repeat_kv(values, _get_num_key_value_groups(queries, values))  # [B,H,N,Dv]
        v_mag = torch.linalg.vector_norm(v_rep.float(), ord=2, dim=-1)         # [B,H,N]

        raw_scores = collision_counts.to(torch.float32) * v_mag.unsqueeze(2)   # [B,H,Q,N]
        scores = raw_scores.masked_fill(~candidate_mask, -torch.inf)

        # 8) Top-k inside candidates
        top_idx = torch.topk(scores, k=Km, dim=-1, largest=True).indices       # [B,H,Q,Km]

        k_each = cand_counts.clamp_max(M)  # [B,H,Q]
        keep = (
            torch.arange(Km, device=keys_rep.device).view(1, 1, 1, Km)
            < k_each.unsqueeze(-1)
        )  # [B,H,Q,Km]

        acc = torch.zeros((B, H, Q, N), device=keys_rep.device, dtype=torch.int16)
        acc.scatter_add_(dim=-1, index=top_idx, src=keep.to(acc.dtype))
        final_mask = acc > 0  # [B,H,Q,N]

        # ---- stats: final size ----
        final_counts = final_mask.sum(dim=-1)
        self._accum_stats_1d(final_counts, kind="final")

        # 9) Merge with previous mask + gate by attention mask probs
        dense_prev = previous_mask.get_dense_mask()
        if not dense_prev.dtype.is_floating_point:
            dense_prev = dense_prev.to(scores.dtype)
        dense_prev = dense_prev.clamp_(0.0, 1.0)

        dense_bucket = final_mask.to(dense_prev.dtype)
        dense_mask = torch.maximum(dense_prev, dense_bucket)

        if allowed_prob is not None:
            ap = allowed_prob.to(dense_mask.dtype)  # [B,1,*,N]
            dense_mask = dense_mask * ap.expand_as(dense_mask)

        mask_shape = (B, H, Q, N)
        return Mask.create_mask_from_dense_mask(mask_shape, dense_mask, dtype=previous_mask.dtype)

    def _should_use_full_attention(self, dims: AttentionTensorDimensions, heavy_tokens: int) -> bool:
        return dims.seq_len_keys <= max(1, heavy_tokens)

    @classmethod
    def create_from_config(cls, config: MaskerConfig) -> "BucketMasker":
        if not isinstance(config, BucketMaskerConfig):
            raise ValueError(f"Invalid config type: {type(config)}")
        return cls(config)
