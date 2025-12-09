"""Bucket utility functions."""

from typing import Dict, Tuple, Optional
import torch
import itertools
import math
import torch.nn.functional as F


PlanesCache = Dict[Tuple[int, torch.device, torch.dtype, int, int], torch.Tensor]
ProtosCache = Dict[Tuple[int, torch.device, torch.dtype], torch.Tensor]


def get_hyper_planes(
    cache: PlanesCache,
    D: int,
    L: int,
    P: int,
    device: torch.device,
    dtype: torch.dtype,
    rng: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """
    Independent SRP planes per table:
        planes: [L, P, D]

    Caches in the provided `cache` dict so that multiple calls
    with the same (D, device, dtype, L, P) reuse the planes.
    """
    key = (D, device, dtype, L, P)
    planes = cache.get(key)
    if planes is None:
        base = torch.randn(
            (L, P, D),
            device=device,
            dtype=torch.float32,
            generator=rng,
        )
        planes = base.to(dtype)
        cache[key] = planes
    return planes


def get_protos_T(
    cache: ProtosCache,
    P: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """
    Hypercube corners: protos_T in {-1,+1}^{P}, shape [P, R]

    Uses the given `cache` dict to memoize by:
        (P, device, dtype)
    """
    key = (P, device, dtype)
    protos_T = cache.get(key)
    if protos_T is None:
        corners = torch.tensor(
            list(itertools.product([-1.0, +1.0], repeat=P)),
            device=device,
            dtype=torch.float32,
        )  # [R, P]
        protos_T = corners.t().to(dtype)  # [P, R]
        cache[key] = protos_T
    return protos_T


def pack_bits(bits: torch.Tensor) -> torch.Tensor:
    """
    Pack last-dim bits into integer codes (big-endian).
    bits: [..., P] bool
    returns: [...] int16
    """
    P = bits.shape[-1]
    weights = (
        1
        << torch.arange(
            P - 1, -1, -1, device=bits.device, dtype=torch.int16
        )
    )  # [P] with MSB first
    return torch.sum(
        bits.to(torch.int16)
        * weights.view(*([1] * (bits.ndim - 1)), P),
        dim=-1,
    )

def hard_hash(tensor: torch.Tensor, planes: torch.Tensor) -> torch.Tensor:
    """
    tensor: [B,H,N,D], planes: [L,P,D]
    returns bucket codes per table: [B,H,L,N]
    """
    # [B,H,N,L,P]
    proj = torch.einsum("bhnd,lkd->bhnlk", tensor, planes)
    bits = (proj >= 0)  # bool
    # [B,H,N,L]
    codes = pack_bits(bits)
    # [B,H,L,N]
    return codes.permute(0, 1, 3, 2).contiguous()

def soft_hash(
    queries: torch.Tensor,
    planes: torch.Tensor,
    protos_T: torch.Tensor,
) -> torch.Tensor:
    """
    queries: [B,H,Q,D]
    planes:  [L,P,D]
    protos_T: [P,R]
    returns soft bucket probabilities: [B,H,Q,L,R]
    """
    # [B,H,Q,L,P]
    q_proj = torch.einsum("bhqd,lkd->bhqlk", queries, planes)
    temp = math.sqrt(queries.size(-1))
    logits = torch.einsum(
        "bhqlk,kr->bhqlr",
        torch.tanh(q_proj) / max(temp, 1e-6),
        protos_T,
    )  # [B,H,Q,L,R]
    return F.softmax(logits, dim=-1)


def get_collision_counts(
    key_buckets: torch.Tensor,   # [B,H,L,N]
    top_buckets: torch.Tensor,   # [B,H,Q,L,top_t]
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    For each table ℓ, mark tokens whose bucket matches any selected bucket in that table.
    Then union across tables, and also return per-(B,H,Q,N) collision counts.

    Returns:
        candidate_mask: [B,H,Q,N] (bool)
        collision_counts:    [B,H,Q,N] (int)  # # of tables where (q,i) matched
    """
    B, H, L, N = key_buckets.shape
    _, _, Q, _, top_t = top_buckets.shape

    # match_any[b,h,q,l,i] = True if key_buckets[b,h,l,i] equals
    # any of top_buckets[b,h,q,l,t] over t.
    match_any = torch.zeros((B, H, Q, L, N), dtype=torch.bool, device=key_buckets.device)

    # [B,H,1,L,N], broadcasts across Q and the last dim
    kb = key_buckets.unsqueeze(2)  # [B,H,1,L,N]

    for t in range(top_t):
        # Select the t-th chosen bucket per (B,H,Q,L)
        tb_t = top_buckets[..., t].unsqueeze(-1)     # [B,H,Q,L,1]
        match_any |= (kb == tb_t)                    # [B,H,Q,L,N]

    # Union across L tables → candidate mask [B,H,Q,N]
    candidate_mask = match_any.any(dim=3)

    # Collision counts: number of tables where (q,i) matched
    collision_counts = match_any.sum(dim=3)          # [B,H,Q,N]

    return candidate_mask, collision_counts


def attention_mask_to_allowed_prob(
    attention_mask: torch.Tensor, K: int
) -> torch.Tensor:
    """
    Convert attention_mask to allowed-probabilities in [0,1], shape [B,1,*,K].
    Heuristics:
        - bool masks: 0 => allow (1.0), 1 => forbid (0.0)
        - additive float masks: >=0 => allow (1.0), negative => forbid (0.0)
    """
    am = attention_mask[..., :K]
    if am.dtype == torch.bool:
        allowed = (am == 0).to(torch.float32)
    else:
        allowed = (am >= 0).to(torch.float32)
    if allowed.dim() == 3:
        allowed = allowed.unsqueeze(1)  # [B,1,*,K]
    return allowed