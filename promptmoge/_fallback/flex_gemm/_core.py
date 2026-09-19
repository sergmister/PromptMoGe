"""Pure-PyTorch reimplementation of the FlexGEMM ops MoGe-3 needs.

Upstream FlexGEMM (https://github.com/JeffreyXiang/FlexGEMM) is Triton/CUDA-only, so
without it MoGe-3's refiner cannot run on a CPU or on Apple silicon. This module is a
*semantic* re-derivation of the four primitives the MoGe-3 refiner uses, written
against the upstream source so it can serve as a numerical oracle:

  submanifold_conv   out[o] = bias + sum_v  W[:, v, :] @ in[ nmap[o, v] ]
                     with  nmap[o, v] = index of the input voxel at
                     coord_out[o] + kernel_delta[v],  or -1 if absent.
                     kernel_delta is itertools.product over
                     range(-(k//2)*d, (k//2+1)*d, d) per spatial dim, last dim
                     fastest -- i.e. exactly F.conv3d's cross-correlation
                     convention (verified against a
                     dense F.conv3d).

  sparse_pool        out[o] = reduce_v in[ o*stride - padding + v ] over the
                     input voxels that exist. MoGe uses kernel=stride=factor,
                     padding=0 -> a perfect partition, out_coord = in_coord // f.

  sparse_upsample    nearest: out[c] = in[c // scale_factor]; the caller
                     supplies output_coords (the matching skip level).

  NeighborCache      carries the adjacency + the topology used to build it, and
                     `.T` for the transposed (upsample) view.

Two lookup backends, both exact and cross-checked against each other:
  * "column" -- MoGe's L0 has exactly one voxel per (batch, i, j) image column
    (`_voxelize` keeps every pixel), so 3D adjacency reduces to a 2D image
    stencil plus `z[i+di, j+dj] == z[i, j] + dz`. No hashing, no sorting.
  * "sorted" -- generic: sort the linearised coords, look neighbours up with
    torch.searchsorted. Used at levels >= 1 (and to validate "column").
"""
from __future__ import annotations

import itertools
import math
from typing import Any, Literal, Sequence

import torch
from torch import Tensor

__all__ = [
    "NeighborCache",
    "NeighborCacheT",
    "build_submanifold_cache",
    "build_pool_cache",
    "submanifold_conv",
    "sparse_pool",
    "sparse_upsample",
    "make_conv_kernel_delta",
]


def _broadcast(arg, D: int, name: str):
    if arg is None:
        return None
    if isinstance(arg, int):
        return (arg,) * D
    arg = tuple(int(a) for a in arg)
    assert len(arg) == D, f"{name} must have length {D}, got {arg}"
    return arg


def make_conv_kernel_delta(
    kernel_size: Sequence[int],
    dilation: Sequence[int],
    batch_dims: int = 0,
    dtype: torch.dtype = torch.int32,
    device: torch.device | None = None,
) -> Tensor:
    """Byte-for-byte the upstream ordering (itertools.product, last dim fastest)."""
    spatial_ranges = [
        range(-(k // 2) * l, (k // 2 + 1) * l, l)
        for k, l in zip(kernel_size, dilation)
    ]
    offsets = torch.tensor(
        list(itertools.product(*[*itertools.repeat((0,), batch_dims), *spatial_ranges])),
        dtype=dtype,
        device=device,
    )
    return offsets


# --------------------------------------------------------------------------- #
# Coordinate linearisation
# --------------------------------------------------------------------------- #

def _linearise(coords: Tensor, sparse_shape: Sequence[int]) -> Tensor:
    """(M, D) int coords -> (M,) int64 linear keys, row-major over sparse_shape."""
    key = torch.zeros(coords.shape[0], dtype=torch.int64, device=coords.device)
    for d, extent in enumerate(sparse_shape):
        key = key * int(extent) + coords[:, d].to(torch.int64)
    return key


def _in_bounds(coords: Tensor, sparse_shape: Sequence[int]) -> Tensor:
    ok = torch.ones(coords.shape[0], dtype=torch.bool, device=coords.device)
    for d, extent in enumerate(sparse_shape):
        c = coords[:, d]
        ok &= (c >= 0) & (c < int(extent))
    return ok


class _SortedIndex:
    """Sorted-key lookup table over a coord set. Exact, no hashing."""

    def __init__(self, coords: Tensor, sparse_shape: Sequence[int]) -> None:
        self.sparse_shape = tuple(int(s) for s in sparse_shape)
        keys = _linearise(coords, self.sparse_shape)
        self.sorted_keys, self.order = torch.sort(keys)
        self.n = keys.shape[0]

    def lookup(self, query_coords: Tensor) -> Tensor:
        """(Q, D) coords -> (Q,) int32 index into the original coord order, -1 if absent."""
        ok = _in_bounds(query_coords, self.sparse_shape)
        q = _linearise(query_coords, self.sparse_shape)
        # Out-of-bounds keys must not alias a real key.
        q = torch.where(ok, q, torch.full_like(q, -1))
        pos = torch.searchsorted(self.sorted_keys, q)
        pos_c = pos.clamp(max=self.n - 1)
        hit = ok & (self.sorted_keys[pos_c] == q) & (pos < self.n)
        idx = torch.where(hit, self.order[pos_c], torch.full_like(pos_c, -1))
        return idx.to(torch.int32)


# --------------------------------------------------------------------------- #
# NeighborCache
# --------------------------------------------------------------------------- #

class NeighborCache:
    """Adjacency between an input and an output coord set.

    Representations carried (only what MoGe needs):
      * `nmap`   (num_out, V) int32, -1 padded -- submanifold conv.
      * `seg_indices` / `seg_offsets` CSR -- pooling reductions.
      * `parent`  (num_in,) int32 -- for a perfect-partition pool, the single
        output row each input row falls into. This is what the transposed view
        needs for nearest upsampling, and it is exact only because MoGe pools
        with kernel == stride and padding 0.
    """

    is_transposed = False

    def __init__(
        self,
        input_coords: Tensor,
        output_coords: Tensor,
        *,
        nmap: Tensor | None = None,
        seg_indices: Tensor | None = None,
        seg_offsets: Tensor | None = None,
        parent: Tensor | None = None,
        num_kernels: int | None = None,
        kernel_size: tuple[int, ...] | None = None,
        dilation: tuple[int, ...] | None = None,
        stride: tuple[int, ...] | None = None,
        input_sparse_shape: Sequence[int] | None = None,
        output_sparse_shape: Sequence[int] | None = None,
        symmetric: bool = False,
        build_backend: str = "sorted",
    ) -> None:
        self.input_coords = input_coords
        self.output_coords = output_coords
        self.nmap = nmap
        self.seg_indices = seg_indices
        self.seg_offsets = seg_offsets
        self.parent = parent
        self.num_kernels = num_kernels
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.stride = stride
        self.input_sparse_shape = None if input_sparse_shape is None else torch.Size(input_sparse_shape)
        self.output_sparse_shape = None if output_sparse_shape is None else torch.Size(output_sparse_shape)
        self.symmetric = symmetric
        self.build_backend = build_backend
        self._T: NeighborCacheT | None = None

    # -- topology ---------------------------------------------------------- #
    @property
    def num_input_coords(self) -> int:
        return int(self.input_coords.shape[0])

    @property
    def num_output_coords(self) -> int:
        return int(self.output_coords.shape[0])

    @property
    def T(self) -> "NeighborCacheT":
        if self._T is None:
            self._T = NeighborCacheT(self)
        return self._T

    def transpose(self) -> "NeighborCacheT":
        return self.T

    def assert_match(self, *, input_coords=None, output_coords=None, is_transposed=None, **_kw) -> None:
        if input_coords is not None:
            assert input_coords.shape[0] == self.input_coords.shape[0], "stale neighbor cache (input rows)"
            assert input_coords.data_ptr() == self.input_coords.data_ptr() or torch.equal(
                input_coords, self.input_coords
            ), "stale neighbor cache (input coords)"
        if output_coords is not None:
            assert output_coords.shape[0] == self.output_coords.shape[0], "stale neighbor cache (output rows)"
        if is_transposed is not None:
            assert bool(is_transposed) == bool(self.is_transposed)

    # -- diagnostics ------------------------------------------------------- #
    def occupancy(self) -> dict[str, float]:
        """Mean / max occupied taps per output row (the §1.2-5 open question)."""
        assert self.nmap is not None
        occ = (self.nmap >= 0).sum(dim=1).float()
        return {
            "V": float(self.nmap.shape[1]),
            "mean": float(occ.mean()),
            "max": float(occ.max()),
            "min": float(occ.min()),
            "rows": float(self.nmap.shape[0]),
        }


class NeighborCacheT:
    """Transposed view. MoGe only uses it for nearest upsampling."""

    is_transposed = True

    def __init__(self, base: NeighborCache) -> None:
        self.base = base
        self.input_coords = base.output_coords
        self.output_coords = base.input_coords
        self.parent = base.parent
        self.input_sparse_shape = base.output_sparse_shape
        self.output_sparse_shape = base.input_sparse_shape
        self.kernel_size = base.kernel_size
        self.stride = base.stride

    @property
    def T(self) -> NeighborCache:
        return self.base

    def assert_match(self, **_kw) -> None:
        return None


# --------------------------------------------------------------------------- #
# Cache construction
# --------------------------------------------------------------------------- #

def _is_one_voxel_per_column(coords: Tensor, sparse_shape: Sequence[int]) -> bool:
    """True iff coords is exactly {(b, i, j, z(b,i,j))} for every (b, i, j).

    That is MoGe's L0 structure: `_voxelize` emits one voxel per pixel, in
    row-major (b, i, j) order. Checking it is cheap and the payoff is a
    hash-free stencil build.
    """
    B, H, W = int(sparse_shape[0]), int(sparse_shape[1]), int(sparse_shape[2])
    if coords.shape[0] != B * H * W or coords.shape[1] != 4:
        return False
    b = coords[:, 0].view(B, H, W)
    i = coords[:, 1].view(B, H, W)
    j = coords[:, 2].view(B, H, W)
    dev = coords.device
    eb = torch.arange(B, device=dev, dtype=coords.dtype).view(B, 1, 1).expand(B, H, W)
    ei = torch.arange(H, device=dev, dtype=coords.dtype).view(1, H, 1).expand(B, H, W)
    ej = torch.arange(W, device=dev, dtype=coords.dtype).view(1, 1, W).expand(B, H, W)
    return bool(torch.equal(b, eb) and torch.equal(i, ei) and torch.equal(j, ej))


def _build_submanifold_nmap_column(coords: Tensor, sparse_shape, kernel_delta: Tensor) -> Tensor:
    """L0 fast path: image stencil + depth predicate. No sort, no hash.

    z is a (B, H, W) map; the voxel at (b, i+di, j+dj) is the neighbour at
    kernel slot (di, dj, dz) iff z[b, i+di, j+dj] == z[b, i, j] + dz.
    """
    B, H, W = int(sparse_shape[0]), int(sparse_shape[1]), int(sparse_shape[2])
    z = coords[:, 3].view(B, H, W).to(torch.int64)
    flat = torch.arange(B * H * W, device=coords.device, dtype=torch.int32).view(B, H, W)
    V = kernel_delta.shape[0]
    nmap = torch.full((B, H, W, V), -1, dtype=torch.int32, device=coords.device)
    NEG = torch.iinfo(torch.int64).min // 4
    for v in range(V):
        di, dj, dz = (int(kernel_delta[v, 1]), int(kernel_delta[v, 2]), int(kernel_delta[v, 3]))
        # Shift z and flat by (di, dj) with an out-of-frame sentinel.
        zs = torch.full_like(z, NEG)
        fs = torch.full_like(flat, -1)
        si0, si1 = max(0, -di), min(H, H - di)      # source rows
        ti0, ti1 = max(0, di), min(H, H + di)       # target rows
        sj0, sj1 = max(0, -dj), min(W, W - dj)
        tj0, tj1 = max(0, dj), min(W, W + dj)
        if si1 > si0 and sj1 > sj0:
            zs[:, si0:si1, sj0:sj1] = z[:, ti0:ti1, tj0:tj1]
            fs[:, si0:si1, sj0:sj1] = flat[:, ti0:ti1, tj0:tj1]
        hit = zs == (z + dz)
        nmap[..., v] = torch.where(hit, fs, torch.full_like(fs, -1))
    return nmap.view(B * H * W, V)


def _build_submanifold_nmap_sorted(coords: Tensor, sparse_shape, kernel_delta: Tensor) -> Tensor:
    index = _SortedIndex(coords, sparse_shape)
    V = kernel_delta.shape[0]
    nmap = torch.empty((coords.shape[0], V), dtype=torch.int32, device=coords.device)
    for v in range(V):
        nmap[:, v] = index.lookup(coords + kernel_delta[v].to(coords.dtype).view(1, -1))
    return nmap


def build_submanifold_cache(
    coords: Tensor,
    sparse_shape: Sequence[int],
    kernel_size: Sequence[int],
    dilation: Sequence[int] | None = None,
    backend: Literal["auto", "column", "sorted"] = "auto",
) -> NeighborCache:
    D = len(kernel_size)
    batch_dims = coords.shape[1] - D
    dilation = tuple(dilation) if dilation is not None else (1,) * D
    kernel_delta = make_conv_kernel_delta(
        kernel_size, dilation, batch_dims=batch_dims, dtype=coords.dtype, device=coords.device
    )
    use_column = False
    if backend in ("auto", "column") and batch_dims == 1 and D == 3:
        use_column = _is_one_voxel_per_column(coords, sparse_shape)
        if backend == "column" and not use_column:
            raise ValueError("column backend requested but coords are not one-per-column")
    if use_column:
        nmap = _build_submanifold_nmap_column(coords, sparse_shape, kernel_delta)
        used = "column"
    else:
        nmap = _build_submanifold_nmap_sorted(coords, sparse_shape, kernel_delta)
        used = "sorted"
    return NeighborCache(
        coords, coords, nmap=nmap, num_kernels=int(kernel_delta.shape[0]),
        kernel_size=tuple(kernel_size), dilation=dilation,
        input_sparse_shape=sparse_shape, output_sparse_shape=sparse_shape,
        symmetric=True, build_backend=used,
    )


def build_pool_cache(
    coords: Tensor,
    sparse_shape: Sequence[int],
    kernel_size: Sequence[int],
    stride: Sequence[int],
    padding: Sequence[int],
    output_coords: Tensor | None = None,
    output_sparse_shape: Sequence[int] | None = None,
) -> NeighborCache:
    """Perfect-partition pool only (kernel == stride, padding == 0), which is
    all MoGe uses. Output coords are the unique floor(coord / stride)."""
    D = len(kernel_size)
    assert tuple(kernel_size) == tuple(stride), "only kernel == stride supported"
    assert all(p == 0 for p in padding), "only padding == 0 supported"
    batch_dims = coords.shape[1] - D

    div = torch.ones(coords.shape[1], dtype=coords.dtype, device=coords.device)
    for d in range(D):
        div[batch_dims + d] = int(stride[d])
    down = torch.div(coords, div.view(1, -1), rounding_mode="floor")

    if output_sparse_shape is None:
        output_sparse_shape = [
            int(s) if d < batch_dims else -(-int(s) // int(stride[d - batch_dims]))
            for d, s in enumerate(sparse_shape)
        ]
    down_keys = _linearise(down, output_sparse_shape)

    if output_coords is None:
        uniq_keys, inverse = torch.unique(down_keys, sorted=True, return_inverse=True)
        # Recover coords from the first occurrence of each key.
        first = torch.full((uniq_keys.shape[0],), -1, dtype=torch.int64, device=coords.device)
        order = torch.arange(down.shape[0], device=coords.device)
        first.scatter_reduce_(0, inverse, order, reduce="amin", include_self=False)
        out_coords = down[first].contiguous()
    else:
        out_coords = output_coords
        index = _SortedIndex(out_coords, output_sparse_shape)
        inverse = index.lookup(down).to(torch.int64)
        assert int(inverse.min()) >= 0, "input voxel with no output parent"

    num_out = int(out_coords.shape[0])
    order = torch.argsort(inverse, stable=True)
    seg_indices = order.to(torch.int32)
    counts = torch.bincount(inverse, minlength=num_out)
    seg_offsets = torch.zeros(num_out + 1, dtype=torch.int64, device=coords.device)
    seg_offsets[1:] = torch.cumsum(counts, 0)

    return NeighborCache(
        coords, out_coords,
        seg_indices=seg_indices, seg_offsets=seg_offsets, parent=inverse.to(torch.int32),
        kernel_size=tuple(kernel_size), stride=tuple(stride),
        input_sparse_shape=sparse_shape, output_sparse_shape=output_sparse_shape,
    )


# --------------------------------------------------------------------------- #
# Ops
# --------------------------------------------------------------------------- #

def submanifold_conv(
    feats: Tensor,
    coords: Tensor,
    shape: torch.Size,
    weight: Tensor,
    bias: Tensor | None = None,
    *,
    dilation: Sequence[int] | None = None,
    neighbor_cache: NeighborCache | None = None,
    **_ignored,
) -> tuple[Tensor, NeighborCache]:
    """weight is (Co, K1, ..., KD, Ci)."""
    kernel_size = tuple(weight.shape[1:-1])
    D = len(kernel_size)
    sparse_shape = tuple(shape[: coords.shape[1]])
    if neighbor_cache is None:
        neighbor_cache = build_submanifold_cache(coords, sparse_shape, kernel_size, dilation)
    else:
        neighbor_cache.assert_match(input_coords=coords, output_coords=coords, is_transposed=False)

    nmap = neighbor_cache.nmap
    M = feats.shape[0]
    Co = weight.shape[0]
    Ci = weight.shape[-1]
    V = int(nmap.shape[1])
    w = weight.reshape(Co, V, Ci)

    out = feats.new_zeros((M, Co))
    if bias is not None:
        out += bias.to(out.dtype)

    for v in range(V):
        idx = nmap[:, v]
        valid = idx >= 0
        n = int(valid.sum())
        if n == 0:
            continue
        rows = valid.nonzero(as_tuple=True)[0]
        src = idx[rows].to(torch.int64)
        contrib = feats.index_select(0, src) @ w[:, v, :].t()
        out.index_add_(0, rows, contrib.to(out.dtype))

    return out, neighbor_cache


def sparse_pool(
    feats: Tensor,
    coords: Tensor,
    shape: torch.Size,
    kernel_size,
    stride=None,
    padding=None,
    reduce: str = "mean",
    output_coords: Tensor | None = None,
    output_shape: torch.Size | None = None,
    neighbor_cache: NeighborCache | None = None,
) -> tuple[Tensor, Tensor, torch.Size, NeighborCache]:
    D = len(kernel_size)
    stride = tuple(stride) if stride is not None else tuple(kernel_size)
    padding = tuple(padding) if padding is not None else (0,) * D
    sparse_dim = coords.shape[1]
    sparse_in = tuple(shape[:sparse_dim])
    sparse_out = tuple(output_shape[:sparse_dim]) if output_shape is not None else None

    if neighbor_cache is None:
        neighbor_cache = build_pool_cache(
            coords, sparse_in, kernel_size, stride, padding, output_coords, sparse_out
        )
    out_coords = neighbor_cache.output_coords
    num_out = int(out_coords.shape[0])

    parent = neighbor_cache.parent.to(torch.int64)
    C = feats.shape[1]
    acc = feats.new_zeros((num_out, C))
    acc.index_add_(0, parent, feats)
    if reduce == "mean":
        counts = torch.bincount(parent, minlength=num_out).clamp(min=1).to(feats.dtype)
        acc = acc / counts.unsqueeze(1)
    elif reduce != "sum":
        raise NotImplementedError(f"reduce={reduce!r} not needed by MoGe")

    out_shape = torch.Size([*neighbor_cache.output_sparse_shape, C])
    return acc, out_coords, out_shape, neighbor_cache


def sparse_upsample(
    feats: Tensor,
    coords: Tensor,
    shape: torch.Size,
    scale_factor,
    *,
    mode: str = "nearest",
    output_coords: Tensor | None = None,
    output_shape: torch.Size | None = None,
    neighbor_cache: NeighborCacheT | None = None,
    **_ignored,
) -> tuple[Tensor, Tensor, torch.Size, Any]:
    assert mode == "nearest", "MoGe only uses nearest upsampling"
    assert neighbor_cache is not None and neighbor_cache.is_transposed, (
        "reference path requires the transposed pool cache (MoGe always supplies it)"
    )
    assert output_coords is not None
    parent = neighbor_cache.parent.to(torch.int64)  # per high-res row -> low-res row
    assert parent.shape[0] == output_coords.shape[0]
    out = feats.index_select(0, parent)
    out_shape = torch.Size([*tuple(output_shape[: output_coords.shape[1]]), feats.shape[1]])
    return out, output_coords, out_shape, neighbor_cache
