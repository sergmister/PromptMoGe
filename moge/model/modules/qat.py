"""Quantisation-aware training for the sparse refiner, matching the on-device Metal int8 kernels exactly.

The deployment's int8 GEMM is fixed by the kernel, not by us (M5_OPTIMIZATION_GUIDANCE section 2.4):

  * weights      symmetric, **per output channel** (`wScale[Co]`), int8;
  * activations  symmetric, **per tensor** -- not per channel and not per row. The K reduction sums
                 neighbours gathered from different rows, so only one activation scale survives the sum;
  * clipping     the activation scale comes from the **99.95th percentile with saturation**, not the max
                 (measured to cut the RMS error ~4x);
  * accumulation int32, dequantised once in the epilogue as `float(acc) * aScale * wScale[n] + bias`;
                 bias and LayerNorm affine stay fp16 and are therefore never quantised here.

Training with any other granularity would produce a model the kernel cannot run, so these are asserts,
not options. Gradients use the straight-through estimator; the activation clip is an EMA of the
per-batch 99.95th percentile, frozen after `freeze_observers()`.
"""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.utils.parametrize as P

QMAX = 127.0


def _fake_quant(x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Symmetric int8 fake quantisation with saturation, straight-through gradient.

    The rounding is done in fp32 on purpose: the refiner trains under bf16 autocast, and bf16's spacing at
    127 is 0.5, so `round(x / s)` computed in bf16 would land on the wrong integer near the top of the
    range and the simulation would not be the int8 the kernel executes.
    """
    s = scale.float().clamp_min(1e-12)
    q = (torch.clamp(torch.round(x.float() / s), -QMAX, QMAX) * s).to(x.dtype)
    return x + (q - x).detach()


class WeightFakeQuant(nn.Module):
    """Parametrisation: symmetric per-output-channel int8 weights. `ch_dim` is the output-channel axis.

    Conv weights in flex_gemm are (Co, kd, kh, kw, Ci) and nn.Linear weights (Co, Ci): the output channel
    is axis 0 in both, which is what `wScale[Co]` indexes.
    """

    def __init__(self, ch_dim: int = 0, enabled: bool = True):
        super().__init__()
        self.ch_dim = ch_dim
        self.enabled = enabled

    def forward(self, w: torch.Tensor) -> torch.Tensor:
        if not self.enabled:
            return w
        dims = [d for d in range(w.ndim) if d != self.ch_dim]
        scale = w.detach().abs().amax(dim=dims, keepdim=True) / QMAX
        return _fake_quant(w, scale)


class ActFakeQuant(nn.Module):
    """Symmetric PER-TENSOR int8 activation quantisation with a 99.95th-percentile clip.

    The clip is an EMA over training batches of the percentile of |x|; values beyond it saturate.
    `max_elems` subsamples the tensor for the percentile estimate (torch.quantile is capped at 2**24
    elements and the estimate does not need every voxel).
    """

    def __init__(self, pct: float = 0.9995, momentum: float = 0.02, max_elems: int = 1 << 18,
                 enabled: bool = True, observe_every: int = 4):
        super().__init__()
        self.pct, self.momentum, self.max_elems, self.enabled = pct, momentum, max_elems, enabled
        self.observe_every = observe_every
        self.frozen = False
        self._calls = 0
        self.register_buffer("clip", torch.zeros(()))
        self.register_buffer("observed", torch.zeros((), dtype=torch.long))

    @torch.no_grad()
    def _observe(self, x: torch.Tensor):
        v = x.detach().abs().float().flatten()
        if v.numel() > self.max_elems:
            v = v[torch.randint(v.numel(), (self.max_elems,), device=v.device)]
        q = torch.quantile(v, self.pct)
        if not torch.isfinite(q) or q <= 0:
            return
        self.clip.copy_(q if int(self.observed) == 0 else (1 - self.momentum) * self.clip + self.momentum * q)
        self.observed += 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.enabled:
            return x
        if self.training and not self.frozen:
            self._calls += 1
            if int(self.observed) == 0 or self._calls % self.observe_every == 0:
                self._observe(x)
        if int(self.observed) == 0:                       # never observed (e.g. eval before any training)
            self._observe(x)
        return _fake_quant(x, self.clip / QMAX)


def _pre_hook(aq: ActFakeQuant):
    def hook(module, args):
        if not args:
            return None
        return (aq(args[0]), *args[1:])
    return hook


def quantize_module(mod: nn.Module, name: str, owner: nn.Module, pct: float = 0.9995) -> ActFakeQuant:
    """Attach per-output-channel weight quantisation and a per-tensor input quantiser to one GEMM."""
    P.register_parametrization(mod, "weight", WeightFakeQuant(ch_dim=0))
    aq = ActFakeQuant(pct=pct)
    owner.add_module(f"aq_{name}", aq)
    mod.register_forward_pre_hook(_pre_hook(aq))
    return aq


def enable_refiner_qat(refiner: nn.Module, pct: float = 0.9995,
                       skip: Sequence[str] = ("input_proj", "prompt_proj", "out_proj")) -> List[str]:
    """Quantise every wide GEMM of a Sparse3DUNet. Returns the list of quantised layer names.

    `skip` keeps the three narrow projections at the graph boundary in fp16: input_proj (3 -> 32),
    prompt_proj (2 -> 32) and out_proj (32 -> 1) together are < 0.1 % of the refiner's FLOPs, and the
    epilogue dequantises to fp16 there anyway.
    """
    from flex_gemm.nn import SubmanifoldConv3d
    done = []
    for name, mod in list(refiner.named_modules()):
        if not isinstance(mod, (SubmanifoldConv3d, nn.Linear)):
            continue
        if any(name == s or name.endswith("." + s) for s in skip):
            continue
        if P.is_parametrized(mod, "weight"):
            continue
        quantize_module(mod, name.replace(".", "_"), refiner, pct=pct)
        done.append(name)
    refiner._qat_layers = done
    return done


def freeze_observers(module: nn.Module):
    for m in module.modules():
        if isinstance(m, ActFakeQuant):
            m.frozen = True


def set_qat_enabled(module: nn.Module, enabled: bool):
    for m in module.modules():
        if isinstance(m, (ActFakeQuant, WeightFakeQuant)):
            m.enabled = enabled


def strip_parametrization_keys(sd: dict) -> dict:
    """`a.b.parametrizations.weight.original` -> `a.b.weight` so QAT checkpoints stay loadable by the
    plain model (the stored tensor is the fp32 master weight; the quantiser is a pure function of it)."""
    out = {}
    for k, v in sd.items():
        out[k.replace(".parametrizations.weight.original", ".weight")] = v
    return out
