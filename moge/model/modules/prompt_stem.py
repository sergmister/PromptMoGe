"""LiDAR prompt modules for MoGe-3 early fusion.

Components
----------
build_prompt          : (depth, conf) -> 4-channel prompt map [norm_log_depth, valid, conf, uncertainty]
                        self-normalized in log space (confident-median centred), resized to the ViT input grid.
perturb_prompt_depth  : uncertainty-aware multiplicative log-normal perturbation for training.
ZeroConv2d / ZeroLinear: zero-initialised projections (exactly zero output at init).
PromptStem            : conv stem mapping the prompt map (at token_grid*14 resolution) to a shared
                        feature map at the token grid, plus one zero-init projection per injection site.
PromptNeck            : optional multi-level prompt pyramid injected additively into the ConvStack neck.
fit_scale_shift       : batched, masked, trimmed least-squares (s, t) : lidar ~= s * z + t (torch).

Zero-init invariant: with all projections at zero, every injection is exactly zero, so the RGB-only
MoGe-3 forward is bit-identical at step 0 (verified by tests/test_prompt_zero_init.py).
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------------------------------------------------------- prompt construction
LOG_CLAMP = 4.0          # |log d - mu| clamp
RANGE_KNEE_M = 3.0       # sensor falloff knee; uncertainty ramps 3m -> 5m
RANGE_FULL_M = 5.0


def masked_median(x: torch.Tensor, mask: torch.Tensor, fallback: float = 0.0) -> torch.Tensor:
    """Per-sample median of x[mask]; x, mask: [B, ...]. Returns [B]."""
    out = []
    for xi, mi in zip(x, mask):
        v = xi[mi]
        out.append(v.median() if v.numel() > 0 else x.new_tensor(fallback))
    return torch.stack(out)


def range_uncertainty(depth: torch.Tensor) -> torch.Tensor:
    return ((depth - RANGE_KNEE_M) / (RANGE_FULL_M - RANGE_KNEE_M)).clamp(0.0, 1.0)


def build_prompt(
    depth: torch.Tensor,
    conf: torch.Tensor,
    out_hw: Tuple[int, int],
    conf_levels: float = 2.0,
    mono_z: Optional[torch.Tensor] = None,
    prefill: Optional[torch.Tensor] = None,
    conf_phase: bool = False,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    depth: [B,1,h,w] metres, <=0 or non-finite == missing. conf: [B,1,h,w] in [0, conf_levels].
    Returns prompt [B,4,H,W] at out_hw (nearest resampled) and stats {'mu': [B] log-depth centre}.
    Channels: 0 norm_log_depth (0 where invalid), 1 valid, 2 conf/levels, 3 uncertainty in [0,1].
    mono_z (optional): [B,1,h',w'] the model's own RGB-only affine depth (exp(logz)); it is LS-aligned to the
    sensor (s, t on confident pixels) and appended as channels 4 (aligned mono log depth − mu, dense) and
    5 (sensor-vs-mono disagreement log d − log mono_aligned, 0 where invalid) so the network is handed the monocular
    structure and an explicit "the sensor disagrees with the image here" signal (LDCM / DMD3C / PriorDA pre-fill lesson).
    prefill (optional): [B,1,h,w] a dense metric pre-fill in the sensor's units (LDCM Poisson completion of the
    confident sensor pixels with the mono log-gradient field); appended as channel 6 (log prefill − mu, dense).
    """
    depth = depth.float()
    conf = conf.float()
    valid = torch.isfinite(depth) & (depth > 0)
    depth = torch.where(valid, depth, torch.ones_like(depth))
    logd = torch.log(depth)
    c = (conf / conf_levels).clamp(0, 1)
    hi = valid & (c >= 0.999)
    use_hi = hi.flatten(1).any(1).view(-1, 1, 1, 1)
    mu = masked_median(logd, torch.where(use_hi, hi, valid))
    nlog = ((logd - mu.view(-1, 1, 1, 1)).clamp(-LOG_CLAMP, LOG_CLAMP)) * valid
    unc = 1.0 - c * (1.0 - range_uncertainty(depth))
    unc = torch.where(valid, unc, torch.ones_like(unc))
    prompt = torch.cat([nlog, valid.float(), c * valid, unc], dim=1)
    if conf_phase:
        # confidence as a phase — the normalised log depth rotated by 2*pi*conf/3 as a unit complex vector, i.e. two
        # channels nlog*cos(theta_c), nlog*sin(theta_c): an explicit multiplicative depth x confidence coupling the first
        # conv can read directly (levels 0/1/2 map to distinct directions; invalid pixels stay 0)
        theta = 2.0 * torch.pi * conf.clamp(0, conf_levels) / 3.0
        prompt = torch.cat([prompt, nlog * torch.cos(theta) * valid, nlog * torch.sin(theta) * valid], dim=1)
    if prompt.shape[-2:] != tuple(out_hw):
        prompt = F.interpolate(prompt, size=out_hw, mode="nearest")
    if mono_z is not None:
        mz = F.interpolate(mono_z.float(), size=depth.shape[-2:], mode="area").clamp_min(1e-4)
        s_, t_ = fit_scale_shift(mz.flatten(1), depth.flatten(1), (valid & (c >= 0.999)).flatten(1))
        mz_al = (s_.view(-1, 1, 1, 1) * mz + t_.view(-1, 1, 1, 1)).clamp_min(1e-3)
        mono_nlog = (torch.log(mz_al) - mu.view(-1, 1, 1, 1)).clamp(-LOG_CLAMP, LOG_CLAMP)
        disagree = ((logd - torch.log(mz_al)).clamp(-1.0, 1.0)) * valid
        extra = F.interpolate(torch.cat([mono_nlog, disagree], dim=1), size=out_hw, mode="nearest")
        prompt = torch.cat([prompt, extra], dim=1)
    if prefill is not None:
        pf = F.interpolate(prefill.float(), size=depth.shape[-2:], mode="area")
        pf_ok = torch.isfinite(pf) & (pf > 0)
        pf_nlog = (torch.log(pf.clamp_min(1e-3)) - mu.view(-1, 1, 1, 1)).clamp(-LOG_CLAMP, LOG_CLAMP) * pf_ok
        prompt = torch.cat([prompt, F.interpolate(pf_nlog, size=out_hw, mode="nearest")], dim=1)
    return prompt, {"mu": mu}


def perturb_prompt_depth(depth: torch.Tensor, conf: torch.Tensor, lam: float = 0.15, conf_levels: float = 2.0,
                         generator: Optional[torch.Generator] = None) -> torch.Tensor:
    """Uncertainty-aware perturbation d' = d * exp(eps), eps ~ N(0, (lam * u)^2), u = uncertainty(conf, range).

    Multiplicative (log-space) noise realises sigma ~ d. Confident, close-range pixels (u=0) are untouched."""
    valid = torch.isfinite(depth) & (depth > 0)
    c = (conf.float() / conf_levels).clamp(0, 1)
    u = 1.0 - c * (1.0 - range_uncertainty(depth))
    eps = torch.randn(depth.shape, device=depth.device, generator=generator) * (lam * u)
    return torch.where(valid, depth * torch.exp(eps), depth)


# ----------------------------------------------------------------------------- zero-init layers
class ZeroConv2d(nn.Conv2d):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 1):
        super().__init__(in_channels, out_channels, kernel_size, padding=kernel_size // 2)
        nn.init.zeros_(self.weight)
        nn.init.zeros_(self.bias)


class ZeroLinear(nn.Linear):
    def __init__(self, in_features: int, out_features: int):
        super().__init__(in_features, out_features)
        nn.init.zeros_(self.weight)
        nn.init.zeros_(self.bias)


# ----------------------------------------------------------------------------- early stem
class PromptStem(nn.Module):
    """Prompt map [B,4,H*14,W*14] -> token-grid features [B,C,H,W] -> per-site zero-init token injections.

    inject_blocks: ViT block indices before which the (projected) prompt tokens are added to the residual
    stream. Index 0 == the patch-embedding level (input of the first block, after pos-embed).
    """

    def __init__(self, in_channels: int = 4, dim_out: int = 1024, dim_hidden: int = 256,
                 inject_blocks: Sequence[int] = (0,), patch_size: int = 14):
        super().__init__()
        assert patch_size == 14, "stem strides assume patch 14 = 2 * 7"
        self.inject_blocks = list(inject_blocks)
        act = nn.GELU
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1), act(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), act(),
            nn.Conv2d(64, 128, 7, stride=7), act(),
            nn.Conv2d(128, dim_hidden, 3, padding=1), act(),
            nn.Conv2d(dim_hidden, dim_hidden, 3, padding=1), act(),
        )
        # Gated injection: standard-init 1x1 projection behind a zero-init per-channel gate. Zero output at init, and the
        # injection magnitude is controlled by dim_out gate values instead of dim_hidden*dim_out weights (Adam moves every
        # weight by ~lr per step, so a zero-init full projection grows as lr*sqrt(N)*steps and destabilises the frozen ViT).
        self.projections = nn.ModuleDict({str(b): nn.Conv2d(dim_hidden, dim_out, 1) for b in self.inject_blocks})
        self.gates = nn.ParameterDict({str(b): nn.Parameter(torch.zeros(dim_out)) for b in self.inject_blocks})

    def forward(self, prompt: torch.Tensor) -> Tuple[torch.Tensor, Dict[int, torch.Tensor]]:
        """Returns (shared features [B,Ch,H,W], {block_idx: tokens [B, H*W, dim_out]})."""
        feat = self.stem(prompt)
        inj = {int(b): (self.gates[b].view(1, -1, 1, 1) * proj(feat)).flatten(2).transpose(1, 2) for b, proj in self.projections.items()}
        return feat, inj

    def gate_norms(self) -> Dict[str, float]:
        return {f"gate{b}": float(g.detach().norm()) for b, g in self.gates.items()}


# ----------------------------------------------------------------------------- neck pyramid (optional)
class PromptNeck(nn.Module):
    """Prompt map at the finest neck resolution -> pyramid of zero-init additive injections for a ConvStack.

    dims: neck residual-block dims per level, coarse -> fine (e.g. [1024, 256, 128, 64, 32]).
    Level l runs at (base_h * 2**l, base_w * 2**l); the prompt is resized to the finest level and
    encoded with stride-2 convs down to the coarsest.

    `hidden` is indexed coarse -> fine like `dims`, so `hidden[-1]` is the width `enc0` emits at the FINEST
    level (480x640 at 1200 tokens) and `hidden[0]` the width at the coarsest. The stock schedule
    (16, 32, 64, 128, 256) is the wrong way round for both cost and capacity: it spends 256 channels at the
    finest level, where `dims` is only 32 and `projections[-1]` discards 224 of them, and 362 of the module's
    431 GFLOP go into that one 256->256 convolution. An inverted schedule such as
    (512, 256, 128, 64, 32) mirrors `dims`, costs ~20x fewer FLOPs and puts the width where the neck is wide.
    """

    def __init__(self, dims: Sequence[int], in_channels: int = 4, hidden: Optional[Sequence[int]] = None):
        super().__init__()
        # Default = the stock schedule, generalised to any ladder depth: (16, 32, 64, 128, 256) for 5 levels,
        # (16, 32, 64, 128) for the 4-level x8 ladder (Model B).
        hidden = tuple(16 * 2 ** i for i in range(len(dims))) if hidden is None else tuple(hidden)
        assert len(dims) == len(hidden), f"hidden {hidden} needs one width per neck level {list(dims)}"
        act = nn.GELU
        self.dims = list(dims)
        self.enc0 = nn.Sequential(nn.Conv2d(in_channels, hidden[-1], 3, padding=1), act(),
                                  nn.Conv2d(hidden[-1], hidden[-1], 3, padding=1), act())
        self.downs = nn.ModuleList([
            nn.Sequential(nn.Conv2d(hidden[i], hidden[i - 1], 3, stride=2, padding=1), act())
            for i in range(len(hidden) - 1, 0, -1)
        ])  # fine -> coarse
        self.projections = nn.ModuleList([nn.Conv2d(hidden[i], dims[i], 1) for i in range(len(dims))])  # coarse -> fine
        self.gates = nn.ParameterList([nn.Parameter(torch.zeros(d)) for d in dims])                      # zero-init gates

    def forward(self, prompt_fine: torch.Tensor) -> List[torch.Tensor]:
        feats = [self.enc0(prompt_fine)]           # finest first
        for d in self.downs:
            feats.append(d(feats[-1]))
        feats = feats[::-1]                        # coarse -> fine
        return [g.view(1, -1, 1, 1) * proj(f) for g, proj, f in zip(self.gates, self.projections, feats)]

    def gate_norms(self) -> Dict[str, float]:
        return {f"gate_neck{i}": float(g.detach().norm()) for i, g in enumerate(self.gates)}


# ----------------------------------------------------------------------------- alignment (torch)
def fit_scale_shift(z: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, iters: int = 4, trim: float = 0.2,
                    min_points: int = 32, differentiable: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
    """Batched trimmed LS: target ~= s * z + t on mask. z, target, mask: [B, N]. Returns s, t: [B].

    Falls back to (1, 0) when fewer than min_points are available. Detached from the graph unless `differentiable`
    (then the trimming weights are chosen with detached values but the final closed-form solve carries gradients, so a
    global affine change of z is exactly loss-neutral -- required for training, otherwise a detached fit creates a
    gauge treadmill)."""
    z_g = z.float().flatten(1) if differentiable else z.detach().float().flatten(1)
    z = z.detach().float().flatten(1); target = target.detach().float().flatten(1); mask = mask.flatten(1)
    w = mask.float()
    s = torch.ones(z.shape[0], device=z.device); t = torch.zeros_like(s)
    ok = w.sum(1) >= min_points
    for it in range(iters + 1):
        zz = z_g if it == iters else z
        n = w.sum(1).clamp_min(1)
        sx = (w * zz).sum(1); sy = (w * target).sum(1)
        sxx = (w * zz * zz).sum(1); sxy = (w * zz * target).sum(1)
        den = n * sxx - sx * sx
        s_new = torch.where(den > 1e-12, (n * sxy - sx * sy) / den.clamp_min(1e-12), torch.ones_like(s))
        t_new = (sy - s_new * sx) / n
        s = torch.where(ok, s_new, s); t = torch.where(ok, t_new, t)
        if it == iters:
            break
        r = (s[:, None] * z + t[:, None] - target).abs() / target.clamp_min(1e-3)
        r = torch.where(mask, r, torch.full_like(r, float("inf")))
        k = ((1 - trim) * mask.sum(1)).long().clamp_min(min_points)
        thr = torch.stack([ri.kthvalue(min(int(ki), ri.numel())).values for ri, ki in zip(r, k)])
        w = (mask & (r <= thr[:, None])).float()
    return s, t


# ----------------------------------------------------------------------------- LoRA (strictly regularised backbone adaptation)
class LoRALinear(nn.Module):
    """y = W x + b + (alpha / r) * B(A x), with B zero-initialised → identical to the frozen base at init.

    Used on DINOv2 attention `qkv` / `proj` so the frozen backbone can learn to route LiDAR-token information while the pretrained weights stay intact (the delta is low-rank and distilled to the teacher on RGB-only)."""

    def __init__(self, base: nn.Linear, r: int = 16, alpha: float = 16.0):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.r = r; self.scale = alpha / r
        self.lora_A = nn.Parameter(torch.empty(r, base.in_features)); nn.init.kaiming_uniform_(self.lora_A, a=5 ** 0.5)
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, r))

    def forward(self, x):
        return self.base(x) + F.linear(F.linear(x, self.lora_A.to(x.dtype)), self.lora_B.to(x.dtype)) * self.scale


def add_lora_to_dinov2(backbone: nn.Module, blocks: Sequence[int], r: int = 16, alpha: float = 16.0, targets=("qkv", "proj")) -> List[nn.Parameter]:
    """Wrap attention linears of the given ViT blocks with LoRALinear in place. Returns the new parameters."""
    params = []
    for i in blocks:
        attn = backbone.blocks[i].attn
        for name in targets:
            lin = getattr(attn, name)
            if isinstance(lin, LoRALinear):
                continue
            wrapped = LoRALinear(lin, r=r, alpha=alpha)
            setattr(attn, name, wrapped)
            params += [wrapped.lora_A, wrapped.lora_B]
    return params
