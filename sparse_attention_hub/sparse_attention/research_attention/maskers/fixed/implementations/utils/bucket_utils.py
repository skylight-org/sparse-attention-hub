"""Bucket utility functions."""

import itertools
import math
from typing import Dict, Optional, Tuple
import triton
import triton.language as tl
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
    # proj: [B, H, N, L, P]
    proj = torch.einsum("bhnd,lkd->bhnlk", tensor, planes)
    bits = proj >= 0  # [B, H, N, L, P] bool
    codes = pack_bits(bits)  # [B, H, N, L] int16
    return codes.permute(0, 1, 3, 2).contiguous()  # [B, H, L, N]


def soft_hash(
    queries: torch.Tensor,
    planes: torch.Tensor,
    protos_T: torch.Tensor,
) -> torch.Tensor:
    """
    queries:   [B, H, Q, D]
    planes:    [L, P, D]
    protos_T:  [P, R]
    returns soft bucket probabilities: [B, H, Q, L, R]
    """
    # q_proj: [B, H, Q, L, P]
    q_proj = torch.einsum("bhqd,lkd->bhqlk", queries, planes)

    temp = math.sqrt(queries.size(-1))
    qh = torch.tanh(q_proj) / max(temp, 1e-6)

    # logits: [B, H, Q, L, R]
    logits = torch.einsum("bhqlk,kr->bhqlr", qh, protos_T)
    return F.softmax(logits, dim=-1)


def get_collision_counts(
    key_buckets: torch.Tensor,  # [B, H, L, N]
    top_buckets: torch.Tensor,  # [B, H, Q, L, top_t]
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    For each table ℓ, mark tokens whose bucket matches any selected bucket in that table.
    Then union across tables, and also return per-(B,H,Q,N) collision counts.

    Returns:
      candidate_mask:    [B, H, Q, N] bool
      collision_counts:  [B, H, Q, N] int  (#tables where (q,i) matched)
    """
    B, H, L, N = key_buckets.shape
    _, _, Q, _, top_t = top_buckets.shape

    # match_any[b,h,q,l,i] = True if key_buckets[b,h,l,i] equals any top_buckets[b,h,q,l,t]
    match_any = torch.zeros((B, H, Q, L, N), dtype=torch.bool, device=key_buckets.device)

    kb = key_buckets.unsqueeze(2)  # [B, H, 1, L, N]
    for t in range(top_t):
        tb_t = top_buckets[..., t].unsqueeze(-1)  # [B, H, Q, L, 1]
        match_any |= (kb == tb_t)

    candidate_mask = match_any.any(dim=3)         # [B, H, Q, N]
    collision_counts = match_any.sum(dim=3)       # [B, H, Q, N] int
    return candidate_mask, collision_counts


def attention_mask_to_allowed_prob(attention_mask: torch.Tensor, K: int) -> torch.Tensor:
    """
    Convert attention_mask to allowed-probabilities in [0,1], shape [B,1,*,K].

    Heuristics:
      - bool masks:         0 => allow (1.0), 1 => forbid (0.0)
      - additive float mask: >=0 => allow (1.0), <0 => forbid (0.0)
    """
    am = attention_mask[..., :K]
    if am.dtype == torch.bool:
        allowed = (am == 0).to(torch.float32)
    else:
        allowed = (am >= 0).to(torch.float32)

    if allowed.dim() == 3:
        allowed = allowed.unsqueeze(1)  # [B, 1, *, K]
    return allowed


def build_inverted_index_csr(
    key_buckets: torch.Tensor,  # [B,H,L,N] int
    R: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    CSR-style inverted index for bucket -> token ids, per (B,H,L).

    Returns:
      perm:    [B,H,L,N] int32   token indices sorted by bucket id
      offsets: [B,H,L,R+1] int32 prefix sums over counts; bucket r is
               perm[..., offsets[...,r] : offsets[...,r+1]]
    """
    B, H, L, N = key_buckets.shape
    device = key_buckets.device

    # Sort token indices by bucket id (per row)
    perm = torch.argsort(key_buckets, dim=-1)  # [B,H,L,N] int64
    sorted_b = torch.gather(key_buckets, dim=-1, index=perm)  # [B,H,L,N]

    # Count bucket sizes per row using scatter_add into [BHL, R]
    flat = sorted_b.reshape(-1, N).to(torch.int64)  # [BHL, N]
    BHL = flat.shape[0]

    counts = torch.zeros((BHL, R), device=device, dtype=torch.int32)
    src = torch.ones_like(flat, dtype=torch.int32)
    counts.scatter_add_(dim=1, index=flat, src=src)  # [BHL, R]

    # Prefix sums -> offsets
    offsets = torch.zeros((BHL, R + 1), device=device, dtype=torch.int32)
    offsets[:, 1:] = torch.cumsum(counts, dim=1)  # [BHL, R]
    offsets = offsets.view(B, H, L, R + 1)

    return perm.to(torch.int32), offsets

@triton.jit
def collisions_from_ranges_kernel(
    # per-entry metadata (length S)
    b_ptr, h_ptr, q_ptr, row_ptr, start_ptr, len_ptr,   # int32
    # perm_flat: flattened [BHL*N]
    perm_ptr,                                           # int32
    # output flattened collision: [B*H*Q*N]
    out_ptr,                                            # int32
    # runtime sizes
    S: tl.constexpr,                                    # compile-time grid dim 0 upper bound is S, but we still mask
    # constexpr sizes for address math
    H: tl.constexpr, Q: tl.constexpr, N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    s = tl.program_id(0)        # entry id
    tile = tl.program_id(1)     # tile along positions

    # In practice grid[0] == S, but keep a guard anyway
    s_mask = s < S

    # Load metadata for entry s
    b = tl.load(b_ptr + s, mask=s_mask, other=0).to(tl.int64)
    h = tl.load(h_ptr + s, mask=s_mask, other=0).to(tl.int64)
    q = tl.load(q_ptr + s, mask=s_mask, other=0).to(tl.int64)
    row = tl.load(row_ptr + s, mask=s_mask, other=0).to(tl.int64)
    start = tl.load(start_ptr + s, mask=s_mask, other=0).to(tl.int64)
    length = tl.load(len_ptr + s, mask=s_mask, other=0).to(tl.int64)

    offs = tile * BLOCK + tl.arange(0, BLOCK)
    idx_in_row = start + offs

    # Extra safety: idx_in_row < N
    in_bounds = s_mask & (offs < length) & (idx_in_row < N)

    # perm_flat is [BHL, N] row-major => linear = row*N + idx_in_row
    perm_lin = row * N + idx_in_row
    tok = tl.load(perm_ptr + perm_lin, mask=in_bounds, other=0).to(tl.int64)

    out_lin = (((b * H + h) * Q + q) * N + tok)
    tl.atomic_add(out_ptr + out_lin, 1, mask=in_bounds)


def scatter_from_ranges_triton(
    collision_flat_i32: torch.Tensor,   # [B*H*Q*N] int32
    perm_flat: torch.Tensor,            # [BHL, N] int32
    b: torch.Tensor, h: torch.Tensor, q: torch.Tensor,
    row: torch.Tensor, start: torch.Tensor, lens: torch.Tensor,
    H: int, Q: int, N: int,
    block: int = 256,
    num_warps: int = 4,
) -> torch.Tensor:
    # Ensure correct dtypes / contiguity
    perm_flat = perm_flat.contiguous().to(torch.int32)
    collision_flat_i32 = collision_flat_i32.contiguous()

    b = b.contiguous().to(torch.int32)
    h = h.contiguous().to(torch.int32)
    q = q.contiguous().to(torch.int32)
    row = row.contiguous().to(torch.int32)
    start = start.contiguous().to(torch.int32)
    lens = lens.contiguous().to(torch.int32)

    S = lens.numel()
    if S == 0:
        return collision_flat_i32

    max_len = int(lens.max().item())
    if max_len == 0:
        return collision_flat_i32

    tiles = triton.cdiv(max_len, block)
    grid = (S, tiles)

    # Note: perm_ptr expects flattened memory; passing perm_flat works (it’s contiguous)
    collisions_from_ranges_kernel[grid](
        b, h, q, row, start, lens,
        perm_flat,
        collision_flat_i32,
        S=S,          # compile-time constexpr for masking
        H=H, Q=Q, N=N,
        BLOCK=block,
        num_warps=num_warps,
    )
    return collision_flat_i32


def get_collision_counts_indexed(
    perm: torch.Tensor,         # [B,H,L,N] int32
    offsets: torch.Tensor,      # [B,H,L,R+1] int32
    top_buckets: torch.Tensor,  # [B,H,Q,L,top_t] int
    N: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    B, H, L, _N = perm.shape
    assert _N == N
    _, _, Q, _L, _top_t = top_buckets.shape
    assert _L == L
    device = perm.device

    perm_flat = perm.reshape(B * H * L, N).contiguous()
    offsets_flat = offsets.reshape(B * H * L, -1).contiguous()

    tb = top_buckets.to(torch.int64)
    tb_sorted, _ = torch.sort(tb, dim=-1)

    keep = torch.ones_like(tb_sorted, dtype=torch.bool)
    keep[..., 1:] = tb_sorted[..., 1:] != tb_sorted[..., :-1]

    bhqlt = keep.nonzero(as_tuple=False)
    b = bhqlt[:, 0]
    h = bhqlt[:, 1]
    q = bhqlt[:, 2]
    l = bhqlt[:, 3]
    tpos = bhqlt[:, 4]

    row = (b * H + h) * L + l
    buckets = tb_sorted[b, h, q, l, tpos]

    R = offsets_flat.shape[1] - 1
    buckets = buckets.clamp_(0, R - 1)

    start = offsets_flat[row, buckets]
    end   = offsets_flat[row, buckets + 1]
    lens = (end - start).clamp_min(0)

    collision_i32 = torch.zeros((B, H, Q, N), device=device, dtype=torch.int32)
    collision_flat_i32 = collision_i32.view(-1)

    collision_flat_i32 = scatter_from_ranges_triton(
        collision_flat_i32=collision_flat_i32,
        perm_flat=perm_flat,
        b=b, h=h, q=q,
        row=row, start=start, lens=lens,
        H=H, Q=Q, N=N,
        block=128,      # 128/256/512 worth sweeping
        num_warps=4,
    )

    collision_i32 = collision_flat_i32.view(B, H, Q, N)
    candidate_mask = collision_i32 > 0
    collision_i16 = collision_i32.clamp_max(torch.iinfo(torch.int16).max).to(torch.int16)
    return candidate_mask, collision_i16

