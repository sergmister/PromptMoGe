"""Learned ARKit sensor simulator: (Faro GT depth, RGB) at 192x256 -> ARKit-like depth + confidence.

SensorUNet predicts per pixel a 2-component Laplace mixture for the log-ratio log(arkit/gt): a tight core (mu1, b1) and a
heavy, shifted outlier component (prob pi, mu2, b2) that carries rim / flattened-object errors, plus 3-class confidence
logits. Sampling: spatially correlated Laplace residual (blurred white noise -> exponential magnitude), correlated outlier
membership and Gaussian-copula confidence, so synthetic frames have coherent error structure, not speckle. One forward pass per frame (~1 ms at 192x256): fast enough for on-the-fly augmentation.
Inputs (10 ch): log GT depth (0 where invalid), GT valid, RGB (3), GT log-depth gradient magnitude (symmetric), GT edge mask
(jump>10%), inverse depth (range cue), signed near-side and far-side rim cues (log depth minus local max / min). GT holes (no Faro) are inpainted by nearest fill before use so the sensor is always dense.
"""
from __future__ import annotations
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F


def _block(cin, cout):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1), nn.GroupNorm(8, cout), nn.SiLU(),
                         nn.Conv2d(cout, cout, 3, padding=1), nn.GroupNorm(8, cout), nn.SiLU())


class SensorUNet(nn.Module):
    def __init__(self, cin: int = 10, base: int = 32, mixed_pixel: int = 0):
        super().__init__()
        # v4 (mixed_pixel=1): outlier mean = learned fraction of the way to the neighbouring surface across a jump
        # v5 (mixed_pixel=2): + a third, physically motivated component: the pixel is a MIX of this surface and the jump
        #    neighbour with a uniform blend fraction (log-ratio uniform on [0, cue]) -- the flattened-thin-object case
        self.mixed_pixel = int(mixed_pixel)
        self.e1 = _block(cin, base); self.e2 = _block(base, base * 2); self.e3 = _block(base * 2, base * 4); self.e4 = _block(base * 4, base * 8)
        self.d3 = _block(base * 8 + base * 4, base * 4); self.d2 = _block(base * 4 + base * 2, base * 2); self.d1 = _block(base * 2 + base, base)
        # outputs: core mu1, core log_b1, outlier logit (pi), outlier mu2, outlier log_b2, conf logits (3)
        self.head = nn.Conv2d(base, {0: 8, 1: 9, 2: 10}[int(mixed_pixel)], 1)
        nn.init.zeros_(self.head.weight); nn.init.zeros_(self.head.bias)
        with torch.no_grad():
            self.head.bias[1] = -4.0          # core b ~ 0.018
            self.head.bias[2] = -3.0          # outlier prob ~ 5 %
            self.head.bias[3] = 0.05          # outlier shift (rims read too far)
            self.head.bias[4] = -1.5          # outlier b ~ 0.22
            self.head.bias[7] = 3.0           # conf 2 prior

    def forward(self, x):
        e1 = self.e1(x); e2 = self.e2(F.avg_pool2d(e1, 2)); e3 = self.e3(F.avg_pool2d(e2, 2)); e4 = self.e4(F.avg_pool2d(e3, 2))
        d3 = self.d3(torch.cat([F.interpolate(e4, scale_factor=2, mode="bilinear", align_corners=False), e3], 1))
        d2 = self.d2(torch.cat([F.interpolate(d3, scale_factor=2, mode="bilinear", align_corners=False), e2], 1))
        d1 = self.d1(torch.cat([F.interpolate(d2, scale_factor=2, mode="bilinear", align_corners=False), e1], 1))
        o = self.head(d1)
        mu2 = o[:, 3].clamp(-1, 1)
        if self.mixed_pixel:
            # signed cue of the dominant jump in the 5x5 neighbourhood: log(max/this) > 0 on the near side (background
            # behind), log(min/this) < 0 on the far side (foreground in front); the outlier mean is a learned fraction
            # (sigmoid) of the way to that neighbour, so "flattened to the background" and "object grows over the
            # background" are both expressible; the free shift is kept only where there is no jump.
            near_cue, far_cue = x[:, 8], x[:, 9]                      # near_cue = lg - max <= 0 ; far_cue = lg - min >= 0
            to_bg, to_fg = -near_cue, -far_cue                          # >= 0 towards background, <= 0 towards foreground
            cue = torch.where(to_bg > -to_fg, to_bg, to_fg)
            frac = torch.sigmoid(o[:, 8])
            mu2 = torch.where(cue.abs() > 0.02, frac * cue, mu2)
        out = {"mu1": o[:, 0], "log_b1": o[:, 1].clamp(-6, 0), "pi_logit": o[:, 2].clamp(-8, 4), "mu2": mu2,
               "log_b2": o[:, 4].clamp(-3, 0.5), "conf_logits": o[:, 5:8]}
        if self.mixed_pixel >= 2:
            out["cue"] = cue
            out["pi_mix_logit"] = torch.where(cue.abs() > 0.05, o[:, 9].clamp(-8, 4), torch.full_like(o[:, 9], -20.0))
        return out


def mixture_nll(out, tgt):
    """Negative log-likelihood of the Laplace mixture: core (mu1, b1), shifted outlier (mu2, b2) with prob pi and, for v5,
    a mixed-pixel component uniform on [0, cue] (blend towards the jump neighbour) with prob pi_mix."""
    lp1 = -((tgt - out["mu1"]).abs() / out["log_b1"].exp() + out["log_b1"]) - 0.6931
    lp2 = -((tgt - out["mu2"]).abs() / out["log_b2"].exp() + out["log_b2"]) - 0.6931
    if "pi_mix_logit" not in out:
        pi = torch.sigmoid(out["pi_logit"])
        return -torch.logaddexp(torch.log1p(-pi + 1e-6) + lp1, torch.log(pi + 1e-6) + lp2)
    # three-way softmax over (core, outlier, mix) from the two logits: weights (1-pi)(1-pm), pi(1-pm), pm
    pi = torch.sigmoid(out["pi_logit"]); pm = torch.sigmoid(out["pi_mix_logit"]); cue = out["cue"]
    lo, hi = torch.minimum(torch.zeros_like(cue), cue), torch.maximum(torch.zeros_like(cue), cue)
    inside = (tgt >= lo - 0.02) & (tgt <= hi + 0.02)
    lp3 = torch.where(inside, -torch.log(cue.abs() + 0.04), torch.full_like(tgt, -20.0))
    return -torch.logsumexp(torch.stack([torch.log((1 - pi) * (1 - pm) + 1e-6) + lp1, torch.log(pi * (1 - pm) + 1e-6) + lp2, torch.log(pm + 1e-6) + lp3]), 0)


def make_inputs(gt: torch.Tensor, rgb: torch.Tensor) -> torch.Tensor:
    """gt [B,1,192,256] metres (0 invalid), rgb [B,3,192,256] in [0,1] -> [B,8,192,256]. GT holes nearest-filled."""
    valid = gt > 0
    g = gt.clone()
    # cheap hole fill: iterative dilation with max-pool of valid values (nearest-ish)
    for _ in range(12):
        filled = F.max_pool2d(torch.where(valid, g, torch.zeros_like(g)), 3, 1, 1)
        g = torch.where(valid, g, filled); valid = valid | (g > 0)
    g = g.clamp_min(0.1)
    lg = torch.log(g)
    gx = lg[:, :, :, 1:] - lg[:, :, :, :-1]; gy = lg[:, :, 1:, :] - lg[:, :, :-1, :]
    gm = torch.zeros_like(lg); gm[:, :, :, 1:] += gx.abs(); gm[:, :, 1:, :] += gy.abs()
    gm[:, :, :, :-1] += gx.abs(); gm[:, :, :-1, :] += gy.abs()          # symmetric: both sides of a jump are marked
    edge = (gm > 0.1).float()
    # signed rim cues: how far this pixel is in front of / behind its 5x5 neighbourhood (near side of a jump: strongly
    # negative first channel; far side: strongly positive second channel)
    near_cue = (lg - F.max_pool2d(lg, 5, 1, 2)).clamp(-1, 0)
    far_cue = (lg + F.max_pool2d(-lg, 5, 1, 2)).clamp(0, 1)
    return torch.cat([lg, (gt > 0).float(), rgb, gm.clamp_max(1.0), edge, 1.0 / g, near_cue, far_cue], dim=1)


def correlated_noise(shape, sigma_px: float, device, generator=None):
    """Unit-variance spatially correlated Gaussian noise (blurred white noise, renormalised)."""
    eps = torch.randn(shape, device=device, generator=generator)
    k = int(2 * round(2 * sigma_px) + 1)
    ax = torch.arange(k, device=device, dtype=torch.float32) - k // 2
    g1 = torch.exp(-0.5 * (ax / sigma_px) ** 2); g1 = g1 / g1.sum()
    ker = (g1[:, None] * g1[None, :]).view(1, 1, k, k)
    out = F.conv2d(eps.view(-1, 1, *shape[-2:]), ker, padding=k // 2).view(shape)
    return out / out.std().clamp_min(1e-6)


@torch.no_grad()
def _gauss_blur(x: torch.Tensor, sigma_px: float) -> torch.Tensor:
    k = int(2 * round(2 * sigma_px) + 1); ax = torch.arange(k, device=x.device, dtype=torch.float32) - k // 2
    g1 = torch.exp(-0.5 * (ax / sigma_px) ** 2); g1 = g1 / g1.sum(); ker = (g1[:, None] * g1[None, :]).view(1, 1, k, k)
    return F.conv2d(F.pad(x, (k // 2,) * 4, mode="replicate"), ker)


def _bilateral_blur(x: torch.Tensor, guide: torch.Tensor, sigma_s: float, sigma_r: float) -> torch.Tensor:
    """Joint bilateral blur of x [B,1,H,W] guided by `guide` [B,1,H,W] (grey image): the real ARKit map is an RGB-guided
    upsampling, so depth edges that coincide with image edges stay sharp and the rest is spread over several pixels."""
    k = int(2 * round(2 * sigma_s) + 1); r = k // 2
    xp = F.pad(x, (r, r, r, r), mode="replicate"); gp = F.pad(guide, (r, r, r, r), mode="replicate")
    num = torch.zeros_like(x); den = torch.zeros_like(x)
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            ws = float(np.exp(-0.5 * (dx * dx + dy * dy) / sigma_s ** 2))
            if ws < 1e-3:
                continue
            xs = xp[:, :, r + dy: r + dy + x.shape[2], r + dx: r + dx + x.shape[3]]; gs = gp[:, :, r + dy: r + dy + x.shape[2], r + dx: r + dx + x.shape[3]]
            w = ws * torch.exp(-0.5 * ((gs - guide) / sigma_r) ** 2)
            num = num + w * xs; den = den + w
    return num / den.clamp_min(1e-9)


def simulate(model: SensorUNet, gt: torch.Tensor, rgb: torch.Tensor, noise_sigma_px: float = 6.0, conf_temp: float = 1.0,
             generator=None, conf_corr_px: float = 4.0, residual_gain: float = 1.0, edge_blur_px: float = 0.0,
             lowconf_noise_gain: float = 1.0, dark_noise_gain: float = 1.0, dense: bool = False, rim_off: bool = False, gain_smooth_px: float = 0.0,
             blur_sigma_r: float = 0.0):
    """v6 physically structured options (all off by default = v4 behaviour):
    edge_blur_px      blur the sensor's log depth across depth edges BEFORE the learned residual (the real ARKit map is an
                      upsampled sparse LiDAR: only ~14 % of a depth jump appears between adjacent pixels, 52 % across 3 px)
    conf_corr_px      correlation length of the confidence copula (real low-conf / hole blobs are ~200 px, 4 per frame)
    lowconf_noise_gain residual amplitude factor on pixels drawn as conf < 2 (real conf-1 pixels are 5x noisier than conf-2)
    dark_noise_gain   residual amplitude factor on dark pixels (grey < 0.15; real dark surfaces are ~2x noisier)
    dense             keep a depth value at every pixel (ARKit never returns zeros; conf carries the reliability)"""
    """Returns arkit-like depth [B,1,192,256] (metres) and confidence [B,1,192,256] in {0,1,2}."""
    x = make_inputs(gt, rgb)
    out = model(x); mu, logits = out["mu1"], out["conf_logits"]
    # spatially correlated Laplace residual: correlated Gaussian field -> half-normal magnitude -> exponential magnitude
    eps = correlated_noise(mu.shape, noise_sigma_px, mu.device, generator)
    lap = torch.sign(eps) * (-torch.log((1 - torch.erf(eps.abs() / 2 ** 0.5)).clamp_min(1e-6)))
    # outlier membership drawn with its own correlated field (contiguous outlier blobs, e.g. a whole flattened object)
    eps_o = correlated_noise(mu.shape, conf_corr_px, mu.device, generator)
    u_o = 0.5 * (1.0 + torch.erf(eps_o / 2 ** 0.5))
    is_out = (u_o < torch.sigmoid(out["pi_logit"])).float()
    if rim_off:
        is_out = torch.zeros_like(is_out)                   # the edge blur models the rims; no learned outlier/rim component
    log_ratio = (1 - is_out) * (out["mu1"] + out["log_b1"].exp() * lap) + is_out * (out["mu2"] + out["log_b2"].exp() * lap)
    if "pi_mix_logit" in out:
        # v5 mixed pixels: a correlated uniform blend fraction towards the jump neighbour (contiguous along the rim)
        eps_m = correlated_noise(mu.shape, 2.0, mu.device, generator); u_m = 0.5 * (1.0 + torch.erf(eps_m / 2 ** 0.5))
        eps_w = correlated_noise(mu.shape, 2.0, mu.device, generator); w = 0.5 * (1.0 + torch.erf(eps_w / 2 ** 0.5))
        is_mix = (u_m < torch.sigmoid(out["pi_mix_logit"])).float() * (0.0 if rim_off else 1.0)
        log_ratio = (1 - is_mix) * log_ratio + is_mix * (w * out["cue"])
    if residual_gain != 1.0:
        # scale the sampled deviation around the core mean (per-pixel noise amplitude) without moving the bias
        log_ratio = out["mu1"] + residual_gain * (log_ratio - out["mu1"])
    g = x[:, 0:1].exp()                                     # hole-filled GT
    # confidence is sampled first (below) when the residual must depend on it; draw it here in that case
    probs = F.softmax(logits / conf_temp, dim=1)
    eps_c = correlated_noise(mu.shape, conf_corr_px, mu.device, generator)
    u = 0.5 * (1.0 + torch.erf(eps_c / 2 ** 0.5)).unsqueeze(1)
    cdf = probs.cumsum(dim=1)
    conf = (u > cdf[:, :1]).float() + (u > cdf[:, 1:2]).float()                   # 0 / 1 / 2
    # v7: the amplitude gains are applied through a smoothed indicator (gain_smooth_px > 0) -- a hard per-pixel gain makes the
    # depth jump at every confidence / brightness boundary (v6: 5x the real sensor's spurious edges on confident pixels)
    def _soft(ind):
        return ind if gain_smooth_px <= 0 else _gauss_blur(ind.unsqueeze(1), gain_smooth_px).squeeze(1)
    if lowconf_noise_gain != 1.0:
        gain = 1.0 + (lowconf_noise_gain - 1.0) * _soft((conf[:, 0] < 2).float())
        log_ratio = out["mu1"] + gain * (log_ratio - out["mu1"])
    if dark_noise_gain != 1.0:
        gain = 1.0 + (dark_noise_gain - 1.0) * _soft((rgb.mean(1) < 0.15).float())
        log_ratio = out["mu1"] + gain * (log_ratio - out["mu1"])
    if edge_blur_px > 0:
        lg_ = torch.log(g.clamp_min(1e-3))
        if blur_sigma_r > 0:      # RGB-guided (joint bilateral): image-aligned depth edges stay sharp, the rest is spread
            g = torch.exp(_bilateral_blur(lg_, rgb.mean(1, keepdim=True), edge_blur_px, blur_sigma_r))
        else:
            g = torch.exp(_gauss_blur(lg_, edge_blur_px))
    depth = g * torch.exp(log_ratio).unsqueeze(1)
    if dense:
        return depth, conf, {"mu": mu, "sigma": out["log_b1"].exp(), "pi": torch.sigmoid(out["pi_logit"]), "probs": probs.permute(0, 2, 3, 1)}
    # confidence: Gaussian-copula sampling (drawn above) -- contiguous low-confidence blobs with exact marginals
    return depth, conf, {"mu": mu, "sigma": out["log_b1"].exp(), "pi": torch.sigmoid(out["pi_logit"]), "probs": probs.permute(0, 2, 3, 1)}
