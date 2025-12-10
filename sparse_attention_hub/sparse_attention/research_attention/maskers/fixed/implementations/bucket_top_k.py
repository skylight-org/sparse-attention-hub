from dataclasses import dataclass
from typing import Dict, Tuple, Optional
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
    get_hyper_planes,
    get_protos_T,
    hard_hash,
    soft_hash,
    get_collision_counts,
    attention_mask_to_allowed_prob,
)


@dataclass
class BucketMaskerConfig(TopKMaskerConfig):
    """
    Minimal masker config:

      • K:      # of hyperplanes per table (buckets = 2**K)
      • L:      # of hash tables (independent sketches)
      • top_t:  # of buckets selected per table (per (B,H,Q))

    heavy_size (inherited from TopKMaskerConfig) is used as the *sample size*:
      M = _calculate_effective_size(heavy_size, N_keys)
    We select up to M keys from the union of selected-bucket tokens using a value-aware score.
    """

    K: int = 4
    L: int = 1
    top_t: int = 4


@MaskerRegistry.register(BucketMaskerConfig)
class BucketMasker(TopKMasker):
    """
    L-table sparsity (mask-only):

      1) Hard SRP hash keys with L sets of K planes → bucket ids per table.
      2) Soft SRP hash queries per table (tanh + /√d vs hypercube corners).
      3) Select top_t buckets per table for each (B,H,Q).
      4) Candidate = union of tokens in any selected bucket across tables.
      5) Within candidates, select up to M keys per (B,H,Q) using a *value-aware* score:
         score[b,h,q,i] ∝ (# collisions across tables) * ||v_i||.

    Returns a packed boolean mask [B,H,Q,N].
    """

    def __init__(self, config: BucketMaskerConfig) -> None:
        super().__init__(config)

        if config.K <= 0:
            raise ValueError("K (hyperplanes) must be a positive integer")
        if config.L <= 0:
            raise ValueError("L (hash tables) must be a positive integer")
        if config.top_t <= 0:
            raise ValueError("top_t must be a positive integer")

        self.P: int = int(config.K)
        self.L: int = int(config.L)
        self.top_t: int = int(config.top_t)
        self.heavy_size = config.heavy_size

        # caches
        self._planes_cache: Dict[
            Tuple[int, torch.device, torch.dtype, int, int], torch.Tensor
        ] = {}
        self._protos_cache: Dict[
            Tuple[int, torch.device, torch.dtype], torch.Tensor
        ] = {}
        self._seed = 123456789
        self._rng_cache: Dict[torch.device, torch.Generator] = {}

    def _rng(self, device: torch.device) -> Optional[torch.Generator]:
        if self._seed is None:
            return None
        g = self._rng_cache.get(device)
        if g is None:
            g = torch.Generator(device=device)
            # Option: offset seeds per device to keep sequences distinct
            g.manual_seed(self._seed + 7777)
            self._rng_cache[device] = g
        return g

    # ---------- Public API ----------

    def add_mask(
        self,
        keys: torch.Tensor,  # [B, H_k or G, N, D]
        queries: torch.Tensor,  # [B, H, Q, D]
        values: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        scaling: float,
        dropout: float,
        sparse_meta_data: Dict,
        previous_mask: Mask,
        **kwargs,
    ) -> Mask:
        # Respect a fully-open previous mask
        if previous_mask.is_full_mask():
            return previous_mask

        dims: AttentionTensorDimensions = self._extract_tensor_dimensions(keys, queries)
        heavy_tokens: int = self._calculate_effective_size(
            self.heavy_size, dims.seq_len_keys
        )
        if self._should_use_full_attention(dims, heavy_tokens):
            return self._create_full_mask(
                dims, previous_mask.dtype, previous_mask.device
            )

        # 1) Align to MHA if KV are grouped (GQA/MQA)
        ngroups = _get_num_key_value_groups(queries, keys)
        keys_rep = repeat_kv(keys, ngroups)  # [B,H,N,D]
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
        )  # [L,P,D]
        protosT = get_protos_T(
            cache=self._protos_cache,
            P=self.P,
            device=keys_rep.device,
            dtype=keys_rep.dtype,
        )  # [P,R]
        R = 1 << self.P
        top_t = max(1, min(self.top_t, R))

        # 3) Hard-hash keys per table → [B,H,L,N]
        key_buckets = hard_hash(keys_rep, planes)  # [B,H,L,N]

        # 4) Soft-hash queries per table → probs [B,H,Q,L,R]
        q_probs = soft_hash(queries, planes, protosT)  # [B,H,Q,L,R]

        # 5) Select top_t buckets per table → [B,H,Q,L,top_t]
        top_buckets = torch.topk(q_probs, k=top_t, dim=-1, largest=True).indices

        # 6) Candidate union across tables + collision counts → [B,H,Q,N], [B,H,Q,N]
        candidate_mask, collision_counts = get_collision_counts(
            key_buckets, top_buckets
        )  # candidate_mask: bool

        # Convert external attention mask to allowed probabilities in [0,1],
        allowed_prob = None
        if attention_mask is not None:
            # [B,1,*,N] float in [0,1]
            allowed_prob = attention_mask_to_allowed_prob(attention_mask, N)

            # For fallback when we have no candidates, we derive a boolean "allowed" mask
            # from the probabilities (allowed iff prob > 0).
            allowed_bool = allowed_prob > 0
            if allowed_bool.dim() == 3:
                # [B,*,N] -> [B,1,*,N] to match allowed_prob
                allowed_bool = allowed_bool.unsqueeze(1)
            allowed_bool = allowed_bool.expand_as(candidate_mask)  # [B,H,Q,N]
        else:
            # Everything allowed
            allowed_bool = torch.ones_like(candidate_mask, dtype=torch.bool)

        no_cands = ~candidate_mask.any(dim=-1, keepdim=True)  # [B,H,Q,1]
        candidate_mask = torch.where(
            no_cands, allowed_bool, candidate_mask
        )  # [B,H,Q,N]

        # 8) Budget from heavy_size
        M = max(0, min(int(self._calculate_effective_size(self.heavy_size, N)), N))
        if M == 0:
            return previous_mask
        Km = min(M, N)

        # 9a) Align values to heads and compute ||v_i|| per key
        v_rep = repeat_kv(
            values, _get_num_key_value_groups(queries, values)
        )  # [B,H,N,Dv]
        v_mag = torch.linalg.vector_norm(v_rep.float(), ord=2, dim=-1)  # [B,H,N]

        # 9b) Value-aware score: score[b,h,q,i] = (# collisions) * ||v_i||
        collision_counts_f = collision_counts.to(torch.float32)  # [B,H,Q,N]
        raw_scores = collision_counts_f * v_mag.unsqueeze(2)  # [B,H,Q,N]

        # 9c) Deterministic top-k on value-aware scores within candidates
        scores = raw_scores.masked_fill(~candidate_mask, -torch.inf)  # [B,H,Q,N]
        top_idx = torch.topk(scores, k=Km, dim=-1, largest=True).indices  # [B,H,Q,Km]

        # 9d) Enforce per-row effective K = min(M, #candidates)
        cand_counts = candidate_mask.sum(dim=-1)  # [B,H,Q]
        k_each = cand_counts.clamp_max(M)  # [B,H,Q]
        keep = torch.arange(Km, device=keys_rep.device).view(
            1, 1, 1, Km
        ) < k_each.unsqueeze(
            -1
        )  # [B,H,Q,Km] bool

        # 9e) Scatter to boolean mask (robust to ties / duplicates)
        acc = torch.zeros((B, H, Q, N), device=keys_rep.device, dtype=torch.int16)
        acc.scatter_add_(dim=-1, index=top_idx, src=keep.to(acc.dtype))
        final_mask = acc > 0  # [B,H,Q,N] bool

        # Previous dense mask as probabilities in [0,1]
        dense_prev = previous_mask.get_dense_mask()  # [B,H,Q,N]
        if not dense_prev.dtype.is_floating_point:
            dense_prev = dense_prev.to(scores.dtype)
        dense_prev = dense_prev.clamp_(0.0, 1.0)

        # Our new bucket mask as {0,1} float
        dense_bucket = final_mask.to(dense_prev.dtype)  # [B,H,Q,N]

        # Probabilistic OR: keep anything that either previous_mask or bucket mask allows
        dense_mask = torch.maximum(dense_prev, dense_bucket)

        # Gate by external attention mask probabilities
        if allowed_prob is not None:
            ap = allowed_prob.to(dense_mask.dtype)  # [B,1,*,N]
            dense_mask = dense_mask * ap.expand_as(dense_mask)

        mask_shape = (B, H, Q, N)
        return Mask.create_mask_from_dense_mask(
            mask_shape, dense_mask, dtype=previous_mask.dtype
        )

    def _should_use_full_attention(
        self, dims: AttentionTensorDimensions, heavy_tokens: int
    ) -> bool:
        """Full attention if the key sequence is within budget."""
        return dims.seq_len_keys <= max(1, heavy_tokens)

    @classmethod
    def create_from_config(cls, config: MaskerConfig) -> "BucketMasker":
        if not isinstance(config, BucketMaskerConfig):
            raise ValueError(f"Invalid config type: {type(config)}")
        return cls(config)
