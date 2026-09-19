"""nn.Module wrappers matching the upstream FlexGEMM signatures."""
from __future__ import annotations

import math
from typing import Literal

import torch
from torch import Tensor, nn

from .._core import (
    NeighborCache, NeighborCacheT,
    submanifold_conv, sparse_pool, sparse_upsample,
    _broadcast,
)

__all__ = ["SubmanifoldConv", "SubmanifoldConv3d", "SparsePool", "SparsePool3d",
           "SparseUpsample", "SparseUpsample3d"]


class SubmanifoldConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=None,
                 bias=True, algorithm=None, allow_tf32=None):
        super().__init__()
        kernel_size = tuple(int(k) for k in kernel_size)
        D = len(kernel_size)
        dilation = (1,) * D if dilation is None else tuple(int(d) for d in dilation)
        self.in_channels, self.out_channels = in_channels, out_channels
        self.kernel_size, self.dilation = kernel_size, dilation
        self.algorithm, self.allow_tf32 = algorithm, allow_tf32
        self.weight = nn.Parameter(torch.empty(out_channels, *kernel_size, in_channels))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in = self.in_channels
            for k in self.kernel_size:
                fan_in *= k
            bound = 1.0 / math.sqrt(fan_in) if fan_in > 0 else 0.0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, feats, coords, shape, neighbor_cache=None):
        return submanifold_conv(feats, coords, shape, self.weight, self.bias,
                                dilation=self.dilation, neighbor_cache=neighbor_cache)

    def extra_repr(self):
        return (f"{self.in_channels}, {self.out_channels}, "
                f"kernel_size={self.kernel_size}, dilation={self.dilation}")


class SubmanifoldConv3d(SubmanifoldConv):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1,
                 bias=True, algorithm=None, allow_tf32=None):
        super().__init__(in_channels, out_channels,
                         _broadcast(kernel_size, 3, "kernel_size"),
                         _broadcast(dilation, 3, "dilation"),
                         bias, algorithm, allow_tf32)


class SparsePool(nn.Module):
    def __init__(self, kernel_size, stride=None, padding=None, reduce="mean"):
        super().__init__()
        kernel_size = tuple(int(k) for k in kernel_size)
        D = len(kernel_size)
        self.kernel_size = kernel_size
        self.stride = kernel_size if stride is None else tuple(int(s) for s in stride)
        self.padding = (0,) * D if padding is None else tuple(int(p) for p in padding)
        self.reduce = reduce

    def forward(self, feats, coords, shape, output_coords=None, output_shape=None,
                neighbor_cache=None):
        return sparse_pool(feats, coords, shape, self.kernel_size, self.stride,
                           self.padding, self.reduce, output_coords, output_shape,
                           neighbor_cache)

    def extra_repr(self):
        return (f"kernel_size={self.kernel_size}, stride={self.stride}, "
                f"padding={self.padding}, reduce={self.reduce!r}")


class SparsePool3d(SparsePool):
    def __init__(self, kernel_size, stride=None, padding=0, reduce="mean"):
        super().__init__(_broadcast(kernel_size, 3, "kernel_size"),
                         _broadcast(stride, 3, "stride"),
                         _broadcast(padding, 3, "padding"), reduce)


class SparseUpsample(nn.Module):
    def __init__(self, scale_factor, mode="nearest", padding_mode="normalize"):
        super().__init__()
        self.scale_factor = tuple(int(s) for s in scale_factor)
        self.mode, self.padding_mode = mode, padding_mode

    def forward(self, feats, coords, shape, output_coords=None, output_shape=None,
                neighbor_cache=None):
        return sparse_upsample(feats, coords, shape, self.scale_factor, mode=self.mode,
                               output_coords=output_coords, output_shape=output_shape,
                               neighbor_cache=neighbor_cache)

    def extra_repr(self):
        return f"scale_factor={self.scale_factor}, mode={self.mode!r}"


class SparseUpsample3d(SparseUpsample):
    def __init__(self, scale_factor, mode="nearest", padding_mode="normalize"):
        super().__init__(_broadcast(scale_factor, 3, "scale_factor"), mode, padding_mode)
