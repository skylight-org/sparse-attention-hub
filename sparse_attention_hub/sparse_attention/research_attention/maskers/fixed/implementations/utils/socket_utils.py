"""Bucket utility functions."""

import itertools
import math
from typing import Dict, Optional, Tuple

import torch
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
    Memoized by (D, device, dtype, L, P).
    """
    key = (D, device, dtype, L, P)
    planes = cache.get(key)
    if planes is None:
        base = torch.randn((L, P, D), device=device, dtype=torch.float32, generator=rng)
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
    Hypercube corners: protos_T in {-1,+1}^P, shape [P, R], where R = 2^P.
    Memoized by (P, device, dtype).
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
    weights = (1 << torch.arange(P - 1, -1, -1, device=bits.device, dtype=torch.int16))  # [P]
    view_shape = (*([1] * (bits.ndim - 1)), P)
    return torch.sum(bits.to(torch.int16) * weights.view(*view_shape), dim=-1)


def hard_hash(tensor: torch.Tensor, planes: torch.Tensor) -> torch.Tensor:
    """
    tensor: [B, H, N, D]
    planes: [L, P, D]
    returns bucket codes per table: [B, H, L, N]
    """
    proj = torch.einsum("bhnd,lkd->bhnlk", tensor, planes)  # [B,H,N,L,P]
    bits = proj >= 0
    codes = pack_bits(bits)  # [B,H,N,L]
    return codes.permute(0, 1, 3, 2).contiguous()  # [B,H,L,N]


def soft_hash(
    queries: torch.Tensor,
    planes: torch.Tensor,
    protos_T: torch.Tensor,
    tau: float = 1.0,
) -> torch.Tensor:
    """
    queries:   [B, H, Q, D]
    planes:    [L, P, D]
    protos_T:  [P, R]
    returns soft bucket probabilities: [B, H, Q, L, R]
    """
    q_proj = torch.einsum("bhqd,lkd->bhqlk", queries, planes)  # [B,H,Q,L,P]
    temp = math.sqrt(queries.size(-1))
    qh = torch.tanh(q_proj) / max(temp, 1e-6)
    logits = torch.einsum("bhqlk,kr->bhqlr", qh, protos_T)  # [B,H,Q,L,R]

    tau = float(tau)
    if tau <= 0:
        raise ValueError(f"tau must be > 0, got {tau}")


    return F.softmax(logits/tau, dim=-1)


def attention_mask_to_allowed_prob(attention_mask: torch.Tensor, K: int) -> torch.Tensor:
    """
    Convert attention_mask to allowed-probabilities in [0,1], shape [B,1,*,K].

    Heuristics:
      - bool masks:          0 => allow (1.0), 1 => forbid (0.0)
      - additive float mask: >=0 => allow (1.0), <0 => forbid (0.0)
    """
    am = attention_mask[..., :K]
    if am.dtype == torch.bool:
        allowed = (am == 0).to(torch.float32)
    else:
        allowed = (am >= 0).to(torch.float32)
    if allowed.dim() == 3:
        allowed = allowed.unsqueeze(1)
    return allowed

def topk_soft_collision_scores_blockwise(
    q_probs: torch.Tensor,      # [B,H,Q,L,R] float probs (may be bf16/fp16)
    key_buckets: torch.Tensor,  # [B,H,L,N] int bucket ids
    v_mag: torch.Tensor,        # [B,H,N] float (any)
    allowed_bool: torch.Tensor, # [B,H,Q,N] bool
    M: int,
    block: int = 4096,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Deterministic score:
        score_j = (sum_l q_probs[..., l, b_j^(l)]) * ||v_j||_2
    Uses float32 accumulation for stability.
    """
    B, H, Q, L, R = q_probs.shape
    _, _, _L, N = key_buckets.shape
    assert _L == L
    assert v_mag.shape == (B, H, N)
    assert allowed_bool.shape == (B, H, Q, N)

    device = q_probs.device

    M = max(1, min(int(M), N))

    # Force float32 for stable accumulation / ranking
    q_probs_f = q_probs.float()
    v_mag_f = v_mag.float()

    best_scores = torch.full((B, H, Q, M), -torch.inf, device=device, dtype=torch.float32)
    best_idx = torch.zeros((B, H, Q, M), device=device, dtype=torch.long)

    for s in range(0, N, block):
        e = min(N, s + block)
        nb = e - s

        collision_block = torch.zeros((B, H, Q, nb), device=device, dtype=torch.float32)

        for l in range(L):
            probs_l = q_probs_f[:, :, :, l, :]  # [B,H,Q,R] float32
            buckets_l = key_buckets[:, :, l, s:e].to(torch.long)  # [B,H,nb]
            idx = buckets_l.unsqueeze(2).expand(B, H, Q, nb)      # [B,H,Q,nb]
            collision_block += torch.gather(probs_l, dim=-1, index=idx)

        score_block = collision_block * v_mag_f[:, :, s:e].unsqueeze(2)  # float32
        score_block = score_block.masked_fill(~allowed_bool[:, :, :, s:e], -torch.inf)

        idx_block = torch.arange(s, e, device=device, dtype=torch.long).view(1, 1, 1, nb).expand(B, H, Q, nb)

        merged_scores = torch.cat([best_scores, score_block], dim=-1)
        merged_idx = torch.cat([best_idx, idx_block], dim=-1)

        top = torch.topk(merged_scores, k=M, dim=-1, largest=True)
        best_scores = top.values
        best_idx = torch.gather(merged_idx, dim=-1, index=top.indices)

    return best_idx, best_scores