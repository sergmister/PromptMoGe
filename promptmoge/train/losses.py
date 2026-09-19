"""Losses for LiDAR-prompted MoGe-3 training. All depth losses are void-safe (masks computed before arithmetic).

Notation: pred coord maps are [B,H,W,3] = (x/z, y/z, logz) in the network's affine frame (before exp/shift).

anchor_metric_loss  -- the inference metric objective: fit (s,t) of exp(logz) to confident LiDAR (detached, trimmed LS),
                       then truncated L1 in log space between s*z+t and Faro GT over all GT-valid pixels.
ssi_log_loss        -- scale-and-shift-invariant (in z) loss vs GT with an oracle LS fit: relative-structure supervision.
grad_loss           -- multi-scale log-depth gradient matching (edge sharpness; PromptDA-style).
distill_coord_loss  -- L1 to the frozen teacher's coord map (x/z, y/z, logz): anti-forgetting for RGB-only / replay.
distill_normal_mask -- L1 normals + BCE mask vs teacher.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F

from moge.model.modules.prompt_stem import fit_scale_shift


def _resize_like(x: torch.Tensor, hw, mode="bilinear"):
    if x.shape[-2:] == tuple(hw):
        return x
    return F.interpolate(x, hw, mode=mode, align_corners=False) if mode == "bilinear" else F.interpolate(x, hw, mode=mode)


def lidar_fit(logz: torch.Tensor, lidar: torch.Tensor, conf: torch.Tensor, differentiable: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
    """(s, t) with lidar ~= s*exp(logz)+t on conf==2 pixels; logz [B,H,W], lidar/conf [B,1,h,w].
    Differentiable through the final closed-form solve by default (see fit_scale_shift)."""
    z = torch.exp(logz.float()).unsqueeze(1)
    z_lr = F.interpolate(z, lidar.shape[-2:], mode="area").squeeze(1)
    m = (conf.squeeze(1) >= 2) & (lidar.squeeze(1) > 0) & (z_lr.detach() > 0)
    return fit_scale_shift(z_lr.flatten(1), lidar.squeeze(1).flatten(1), m.flatten(1), differentiable=differentiable)


@torch.no_grad()
def prompt_gauge_field(z_aligned: torch.Tensor, lidar: torch.Tensor, conf_in: torch.Tensor, degree: int = 2, iters: int = 3) -> torch.Tensor:
    """[B,h,w] multiplicative gauge exp(poly(u,v)) fitted per frame (robust LS, detached) to log(prompt / aligned prediction) on
    confident prompt pixels — the same fit `MoGeModel.infer(gauge_poly=2)` applies at inference. z_aligned [B,h,w] at the sensor grid."""
    B, h, w = z_aligned.shape
    u = torch.linspace(-1, 1, w, device=z_aligned.device); v = torch.linspace(-1, 1, h, device=z_aligned.device)
    vv, uu = torch.meshgrid(v, u, indexing="ij")
    cols = [torch.ones_like(uu), uu, vv] + ([uu * uu, uu * vv, vv * vv] if degree >= 2 else [])
    Bm = torch.stack(cols, -1).view(-1, len(cols)).float()
    out = torch.ones_like(z_aligned)
    ld = lidar.view(B, h, w).float(); cf = conf_in.view(B, h, w)
    for b in range(B):
        m = ((cf[b] >= 2) & (ld[b] > 0) & torch.isfinite(z_aligned[b]) & (z_aligned[b] > 0)).view(-1)
        if m.sum() < 200:
            continue
        A = Bm[m]; y = (torch.log(ld[b].clamp_min(1e-3)) - torch.log(z_aligned[b].float().clamp_min(1e-3))).view(-1)[m]
        wgt = torch.ones_like(y)
        for _ in range(iters):
            coef = torch.linalg.lstsq(A * wgt[:, None], (y * wgt)[:, None]).solution[:, 0]
            r = y - A @ coef; sc = 1.4826 * r.abs().median() + 1e-4; wgt = torch.clamp(2 * sc / r.abs().clamp_min(1e-6), max=1.0)
        if torch.isfinite(coef).all():
            out[b] = torch.exp(Bm @ coef).view(h, w)
    return out


def anchor_metric_loss(logz: torch.Tensor, gt: torch.Tensor, lidar: torch.Tensor, conf: torch.Tensor,
                       trunc: float = 0.3, s_t: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                       calib: Optional[torch.Tensor] = None, offsensor_weight: float = 1.0,
                       conf_in: Optional[torch.Tensor] = None, gauge_poly: int = 0) -> Tuple[torch.Tensor, Dict]:
    """logz [B,H,W] (network res), gt [B,1,Hg,Wg] metres (0 invalid). Returns loss and monitors (absrel on valid).
    calib: optional learnable [log_s, t_c] applied after the (detached) LiDAR fit (learned global calibration)."""
    if s_t is None:
        s, t = lidar_fit(logz, lidar, conf)
    else:
        s, t = s_t
    if calib is not None:
        s = s * torch.exp(calib[0]); t = t * torch.exp(calib[0]) + calib[1]
    z = torch.exp(logz.float())
    if gauge_poly > 0:
        # supervise up to the smooth per-frame gauge that inference recovers from the prompt (detached fit at the sensor grid)
        z_lr = F.interpolate(z.unsqueeze(1), lidar.shape[-2:], mode="area").squeeze(1)
        z_al = (s.view(-1, 1, 1) * z_lr + t.view(-1, 1, 1)).detach()
        g = prompt_gauge_field(z_al, lidar, conf_in if conf_in is not None else conf, degree=gauge_poly)
        g = _resize_like(g.unsqueeze(1), gt.shape[-2:]).squeeze(1)
    z = _resize_like(z.unsqueeze(1), gt.shape[-2:]).squeeze(1)
    pred = (s.view(-1, 1, 1) * z + t.view(-1, 1, 1)).clamp_min(1e-3)
    if gauge_poly > 0:
        pred = pred * g
    gtd = gt.squeeze(1)
    valid = (gtd > 0) & torch.isfinite(gtd)
    gts = torch.where(valid, gtd, torch.ones_like(gtd))
    err = (torch.log(pred) - torch.log(gts)).abs()
    w = valid.float()
    if offsensor_weight != 1.0:
        # completion weighting: pixels the network cannot read off the sensor (conf < 2 in the prompt actually fed, i.e.
        # incl. simulated holes / low-conf blobs) get `offsensor_weight`, confident pixels weight 1
        c_in = conf_in if conf_in is not None else conf
        off = F.interpolate((c_in < 2).float(), gt.shape[-2:], mode="nearest").squeeze(1)
        w = w * (1.0 + (offsensor_weight - 1.0) * off)
    loss = (w * err.clamp_max(trunc)).sum() / w.sum().clamp_min(1)
    with torch.no_grad():
        absrel = torch.where(valid, (pred - gts).abs() / gts, torch.zeros_like(err)).sum() / valid.sum().clamp_min(1)
        off_m = valid & (F.interpolate(((conf_in if conf_in is not None else conf) < 2).float(), gt.shape[-2:], mode="nearest").squeeze(1) > 0)
        absrel_off = (torch.where(off_m, (pred - gts).abs() / gts, torch.zeros_like(err)).sum() / off_m.sum().clamp_min(1)) if off_m.any() else torch.zeros(())
    return loss, {"absrel_anchor": absrel.item(), "absrel_off": float(absrel_off), "s": s.mean().item(), "t": t.mean().item()}


def _masked_gauss(x: torch.Tensor, m: torch.Tensor, sigma: float) -> torch.Tensor:
    """masked Gaussian low-pass on the sensor grid: blur(x*m)/blur(m); x, m [B,h,w]."""
    k = int(6 * sigma + 1) | 1; ax = torch.arange(k, device=x.device, dtype=torch.float32) - k // 2
    g = torch.exp(-0.5 * (ax / sigma) ** 2); g = (g / g.sum()).view(1, 1, 1, k)
    def blur(t):
        t = F.conv2d(F.pad(t.unsqueeze(1), (k // 2, k // 2, 0, 0), mode="replicate"), g)
        return F.conv2d(F.pad(t, (0, 0, k // 2, k // 2), mode="replicate"), g.view(1, 1, k, 1)).squeeze(1)
    num = blur(x * m); den = blur(m)
    return num / den.clamp_min(1e-3), den


def prompt_fidelity_loss(logz: torch.Tensor, lidar: torch.Tensor, conf_in: torch.Tensor, margin: float = 0.01, trunc: float = 0.3, sigma_px: float = 0.0) -> torch.Tensor:
    """Prompt fidelity as a soft loss: after the LS fit to the prompt, the prediction (area-averaged to the sensor grid) may deviate from the
    calibrated prompt on confident pixels only within `margin` (log units ≈ the sensor noise); larger deviations are penalised
    (log-L1 beyond the margin, truncated). Keeps the model faithful to a view-consistent prompt without copying its noise."""
    s, t = lidar_fit(logz, lidar, conf_in)
    z_lr = F.interpolate(torch.exp(logz.float()).unsqueeze(1), lidar.shape[-2:], mode="area").squeeze(1)
    pred = (s.view(-1, 1, 1) * z_lr + t.view(-1, 1, 1)).clamp_min(1e-3)
    ld = lidar.squeeze(1); m = (conf_in.squeeze(1) >= 2) & (ld > 0)
    if not m.any():
        return logz.sum() * 0
    d = torch.log(pred) - torch.log(ld.clamp_min(1e-3))
    if sigma_px > 0:                                     # low-frequency version: only the sensor-scale smooth deviation is penalised
        mf = m.float(); d, den = _masked_gauss(d * 0 + d, mf, sigma_px); m = m & (den > 0.2)
    dev = d.abs()
    return ((dev - margin).clamp(min=0, max=trunc) * m.float()).sum() / m.float().sum().clamp_min(1)


def ssi_log_loss(logz: torch.Tensor, gt: torch.Tensor, trunc: float = 0.3) -> torch.Tensor:
    """Oracle LS (s,t) in z-space to GT (detached), then truncated log L1. Pure relative-structure term."""
    z = torch.exp(logz.float())
    z = _resize_like(z.unsqueeze(1), gt.shape[-2:]).squeeze(1)
    gtd = gt.squeeze(1); valid = (gtd > 0) & torch.isfinite(gtd)
    s, t = fit_scale_shift(z.flatten(1), torch.where(valid, gtd, torch.zeros_like(gtd)).flatten(1), valid.flatten(1), differentiable=True)
    pred = (s.view(-1, 1, 1) * z + t.view(-1, 1, 1)).clamp_min(1e-3)
    gts = torch.where(valid, gtd, torch.ones_like(gtd))
    err = (torch.log(pred) - torch.log(gts)).abs().clamp_max(trunc)
    return torch.where(valid, err, torch.zeros_like(err)).sum() / valid.sum().clamp_min(1)


def grad_loss(logz: torch.Tensor, gt: torch.Tensor, scales=(1, 2, 4)) -> torch.Tensor:
    """Multi-scale gradient matching of log depth (scale-invariant). Uses the *metric-free* logz vs log gt:
    gradients are invariant to a global log offset, so no alignment needed (shift in z is ignored: small)."""
    lz = _resize_like(logz.float().unsqueeze(1), gt.shape[-2:])
    gtd = gt; valid = (gtd > 0) & torch.isfinite(gtd)
    lg = torch.log(torch.where(valid, gtd, torch.ones_like(gtd)))
    total = 0.0
    for sc in scales:
        if sc > 1:
            p = F.avg_pool2d(lz, sc); g = F.avg_pool2d(lg, sc); v = F.avg_pool2d(valid.float(), sc) > 0.999
        else:
            p, g, v = lz, lg, valid
        dxp, dyp = p[..., :, 1:] - p[..., :, :-1], p[..., 1:, :] - p[..., :-1, :]
        dxg, dyg = g[..., :, 1:] - g[..., :, :-1], g[..., 1:, :] - g[..., :-1, :]
        vx, vy = v[..., :, 1:] & v[..., :, :-1], v[..., 1:, :] & v[..., :-1, :]
        ex = torch.where(vx, (dxp - dxg).abs().clamp_max(0.5), torch.zeros_like(dxp))
        ey = torch.where(vy, (dyp - dyg).abs().clamp_max(0.5), torch.zeros_like(dyp))
        total = total + (ex.sum() + ey.sum()) / (vx.sum() + vy.sum()).clamp_min(1)
    return total / len(scales)


def distill_coord_loss(coord: torch.Tensor, coord_t: torch.Tensor, w_uv: float = 1.0, w_logz: float = 1.0) -> torch.Tensor:
    """coord, coord_t: [B,H,W,3] student / teacher (same gauge by construction at init)."""
    d = (coord.float() - coord_t.float()).abs()
    return w_uv * d[..., :2].mean() + w_logz * d[..., 2].mean()


def distill_normal_mask(normal, normal_t, mask, mask_t) -> Tuple[torch.Tensor, torch.Tensor]:
    ln = (normal.float() - normal_t.float()).abs().mean() if normal is not None else torch.zeros((), device=mask.device)
    lm = F.binary_cross_entropy(mask.float().clamp(1e-4, 1 - 1e-4), mask_t.float()) if mask is not None else torch.zeros((), device=mask_t.device)
    return ln, lm


def gauge_loss(coord: torch.Tensor, coord_t: torch.Tensor) -> torch.Tensor:
    """Weak tie of the prompted output's global log-depth gauge to the teacher's (per-sample mean log z difference).
    The affine-invariant losses leave this direction free; without a tie it random-walks."""
    d = (coord[..., 2].float() - coord_t[..., 2].float()).flatten(1).mean(1)
    return d.abs().mean()


def _normals_torch(depth: torch.Tensor, fx: float, fy: float, cx: float, cy: float) -> torch.Tensor:
    """depth [B,H,W] (metres) -> unit normals [B,H-2,W-2,3] via central differences of back-projected points."""
    B, H, W = depth.shape
    u = torch.arange(W, device=depth.device, dtype=depth.dtype).view(1, 1, W)
    v = torch.arange(H, device=depth.device, dtype=depth.dtype).view(1, H, 1)
    P = torch.stack([(u - cx) / fx * depth, (v - cy) / fy * depth, depth], dim=-1)
    dx = P[:, 1:-1, 2:] - P[:, 1:-1, :-2]; dy = P[:, 2:, 1:-1] - P[:, :-2, 1:-1]
    n = torch.cross(dx, dy, dim=-1)
    return n / n.norm(dim=-1, keepdim=True).clamp_min(1e-9)


def geometric_normal_loss(logz: torch.Tensor, gt: torch.Tensor, lidar: torch.Tensor, conf: torch.Tensor,
                          fx_norm: float = 1601.96 / 1920, scales=(1, 4)) -> torch.Tensor:
    """Cosine loss between normals of the LiDAR-aligned predicted depth and normals of the Faro depth, at the given
    downsampling scales (1 = pixel scale, 4 = low-frequency / wall angles). Only pixels whose 3x3 GT neighbourhood is
    valid and jump-free count. Uses constant ARKit wide intrinsics (fx/W = fy/W_h are identical for this camera)."""
    s, t = lidar_fit(logz, lidar, conf)
    z = _resize_like(torch.exp(logz.float()).unsqueeze(1), gt.shape[-2:]).squeeze(1)
    pred = (s.view(-1, 1, 1) * z + t.view(-1, 1, 1)).clamp_min(1e-3)
    gtd = gt.squeeze(1); valid = (gtd > 0) & torch.isfinite(gtd)
    total = 0.0
    for sc in scales:
        if sc > 1:
            p = F.avg_pool2d(pred.unsqueeze(1), sc).squeeze(1)
            g = F.avg_pool2d(torch.where(valid, gtd, torch.zeros_like(gtd)).unsqueeze(1), sc).squeeze(1)
            vc = F.avg_pool2d(valid.float().unsqueeze(1), sc).squeeze(1) > 0.999
            g = torch.where(vc, g, torch.ones_like(g))
        else:
            p, g, vc = pred, torch.where(valid, gtd, torch.ones_like(gtd)), valid
        H, W = g.shape[-2:]
        fx = fx_norm * W; cx = W / 2; cy = H / 2
        ng = _normals_torch(g, fx, fx, cx, cy); npred = _normals_torch(p, fx, fx, cx, cy)
        lg = torch.log(g)
        jump = torch.zeros_like(vc)
        jx = (lg[:, :, 1:] - lg[:, :, :-1]).abs() > 0.1; jy = (lg[:, 1:, :] - lg[:, :-1, :]).abs() > 0.1
        jump[:, :, 1:] |= jx; jump[:, :, :-1] |= jx; jump[:, 1:, :] |= jy; jump[:, :-1, :] |= jy
        bad = (~vc) | jump
        m = ~(F.max_pool2d(bad.float().unsqueeze(1), 3, stride=1, padding=1).squeeze(1) > 0)
        m = m[:, 1:-1, 1:-1]
        cos = (ng * npred).sum(-1)
        total = total + torch.where(m, 1.0 - cos, torch.zeros_like(cos)).sum() / m.sum().clamp_min(1)
    return total / len(scales)


def edge_jump_loss(logz: torch.Tensor, gt: torch.Tensor, thresh: float = 0.1, cap: float = 1.0) -> torch.Tensor:
    """Boundary-preservation loss: over horizontal / vertical neighbour pairs whose Faro log-depth jump exceeds `thresh`,
    L1 between the predicted log-depth jump and the GT jump (capped). Scale/shift-free (log differences), so it only
    asks the prediction to *have* the discontinuity the GT has, where it has it."""
    lz = _resize_like(logz.float().unsqueeze(1), gt.shape[-2:]).squeeze(1)
    gtd = gt.squeeze(1); valid = (gtd > 0) & torch.isfinite(gtd)
    lg = torch.log(torch.where(valid, gtd, torch.ones_like(gtd)))
    total = 0.0; n = 0
    for (a, b) in (((slice(None), slice(1, None)), (slice(None), slice(None, -1))), ((slice(1, None), slice(None)), (slice(None, -1), slice(None)))):
        dg = lg[:, a[0], a[1]] - lg[:, b[0], b[1]]; dp = lz[:, a[0], a[1]] - lz[:, b[0], b[1]]
        m = valid[:, a[0], a[1]] & valid[:, b[0], b[1]] & (dg.abs() > thresh)
        total = total + torch.where(m, (dp - dg).abs().clamp_max(cap), torch.zeros_like(dg)).sum(); n = n + m.sum()
    return total / n.clamp_min(1)


def rpnl_loss(logz: torch.Tensor, gt: torch.Tensor, num_crops: int = 16, min_ratio: float = 0.125, max_ratio: float = 0.5,
              generator: Optional[torch.Generator] = None) -> torch.Tensor:
    """DMD3C-style Random Proposal Normalisation Loss in log-depth: random crops, each median/MAD-normalised for
    prediction and GT, L1 between them. Local structure supervision: inside a crop a small object spans a large
    fraction of the range, so flattening it is penalised where a global loss barely notices."""
    lz = _resize_like(logz.float().unsqueeze(1), gt.shape[-2:]).squeeze(1)
    gtd = gt.squeeze(1); valid = (gtd > 0) & torch.isfinite(gtd)
    lg = torch.log(torch.where(valid, gtd, torch.ones_like(gtd)))
    B, H, W = lg.shape
    total = 0.0; n = 0
    for _ in range(num_crops):
        r = float(torch.empty(1).uniform_(min_ratio, max_ratio, generator=generator))
        ch, cw = max(int(H * r), 8), max(int(W * r), 8)
        y0 = int(torch.randint(0, H - ch + 1, (1,), generator=generator)); x0 = int(torch.randint(0, W - cw + 1, (1,), generator=generator))
        p = lz[:, y0:y0 + ch, x0:x0 + cw].flatten(1); g = lg[:, y0:y0 + ch, x0:x0 + cw].flatten(1); m = valid[:, y0:y0 + ch, x0:x0 + cw].flatten(1)
        for b in range(B):
            if m[b].sum() < 64:
                continue
            pb, gb = p[b][m[b]], g[b][m[b]]
            pm, gm = pb.median(), gb.median()
            pmad = (pb - pm).abs().median().clamp_min(1e-3); gmad = (gb - gm).abs().median().clamp_min(1e-3)
            total = total + ((pb - pm) / pmad - (gb - gm) / gmad).abs().mean(); n += 1
    return total / max(n, 1)


def head_normal_loss(normal_pred: torch.Tensor, gt: torch.Tensor, fx_norm: float = 1601.96 / 1920, scales=(4,)) -> torch.Tensor:
    """Cosine loss between the normal HEAD output [B,H,W,3] and Faro-derived normals (from the GT depth with the
    constant wide intrinsics), on interior pixels whose 3x3 GT neighbourhood is valid and jump-free.
    Default scales=(4,): Faro depth is a 1 mm-quantised mesh render, so only its low-frequency (wall-angle) normals are
    trustworthy; pixel-scale normal supervision is reserved for synthetic data with exact normals."""
    gtd = gt.squeeze(1); valid = (gtd > 0) & torch.isfinite(gtd)
    total = 0.0
    for sc in scales:
        if sc > 1:
            g = F.avg_pool2d(torch.where(valid, gtd, torch.zeros_like(gtd)).unsqueeze(1), sc).squeeze(1)
            vc = F.avg_pool2d(valid.float().unsqueeze(1), sc).squeeze(1) > 0.999
            g = torch.where(vc, g, torch.ones_like(g))
            npred = F.avg_pool2d(normal_pred.permute(0, 3, 1, 2).float(), sc).permute(0, 2, 3, 1)
            npred = npred / npred.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        else:
            g, vc, npred = torch.where(valid, gtd, torch.ones_like(gtd)), valid, normal_pred.float()
        H, W = g.shape[-2:]; fx = fx_norm * W
        # MoGe's normal head points TOWARDS the camera (n_z < 0; measured 174.6° mean angle against our
        # finite-difference normals on a real frame), so the finite-difference normals are negated here.
        ng = -_normals_torch(g, fx, fx, W / 2, H / 2)
        lg = torch.log(g); jump = torch.zeros_like(vc)
        jx = (lg[:, :, 1:] - lg[:, :, :-1]).abs() > 0.1; jy = (lg[:, 1:, :] - lg[:, :-1, :]).abs() > 0.1
        jump[:, :, 1:] |= jx; jump[:, :, :-1] |= jx; jump[:, 1:, :] |= jy; jump[:, :-1, :] |= jy
        m = ~(F.max_pool2d(((~vc) | jump).float().unsqueeze(1), 3, stride=1, padding=1).squeeze(1) > 0)
        m = m[:, 1:-1, 1:-1]
        cos = (ng * npred[:, 1:-1, 1:-1]).sum(-1)
        total = total + torch.where(m, 1.0 - cos, torch.zeros_like(cos)).sum() / m.sum().clamp_min(1)
    return total / len(scales)


def exact_normal_loss(normal_pred: torch.Tensor, normal_gt: torch.Tensor, scales=(1, 4)) -> torch.Tensor:
    """Synthetic data only: cosine loss between the normal head [B,H,W,3] and exact renderer normals [B,3,H,W] (already in
    the head's convention, unit length where defined, 0 where undefined), at pixel scale and 4x (block-averaged)."""
    total = 0.0
    for sc in scales:
        npred = normal_pred.permute(0, 3, 1, 2).float(); ngt = normal_gt.float()
        if sc > 1:
            npred = F.avg_pool2d(npred, sc); ngt = F.avg_pool2d(ngt, sc)
        ok = ngt.norm(dim=1) > 0.9                      # undefined / mixed-orientation blocks excluded
        npred = npred / npred.norm(dim=1, keepdim=True).clamp_min(1e-6); ngt = ngt / ngt.norm(dim=1, keepdim=True).clamp_min(1e-6)
        cos = (npred * ngt).sum(1)
        total = total + torch.where(ok, 1.0 - cos, torch.zeros_like(cos)).sum() / ok.sum().clamp_min(1)
    return total / len(scales)


def teacher_edge_grad_loss(logz: torch.Tensor, logz_teacher: torch.Tensor, band_px: int = 3, jump: float = 0.1,
                           cap: float = 0.5) -> torch.Tensor:
    """Edge-aware structure term with an image-aligned pseudo-GT (PromptDA's L_grad idea).

    The frozen RGB-only MoGe-3 teacher has sharp, image-registered depth edges, whereas the Faro GT is misregistered by
    ~5 px and banded out. Its z is aligned to the student's affine gauge by a per-sample LS scale + shift on z (detached).
    The loss is the L1 between neighbouring log-depth differences of student and aligned teacher, inside a band of
    `band_px` pixels around the aligned teacher's own log-depth edges (|jump| > `jump`), with a per-pair cap.
    A transition spread over several pixels (a "curtain") pays on every pair of the band.
    logz, logz_teacher: [B, H, W]."""
    ls = logz.float(); lt0 = logz_teacher.detach().float()
    zs = torch.exp(ls.detach()).flatten(1); zt = torch.exp(lt0).flatten(1)
    ok = torch.isfinite(zs) & torch.isfinite(zt) & (zs > 1e-4) & (zt > 1e-4)
    s, t = fit_scale_shift(zt, zs, ok)
    lt = torch.log((s.view(-1, 1, 1) * torch.exp(lt0) + t.view(-1, 1, 1)).clamp_min(1e-4))
    dxt = lt[:, :, 1:] - lt[:, :, :-1]; dyt = lt[:, 1:] - lt[:, :-1]
    dxs = ls[:, :, 1:] - ls[:, :, :-1]; dys = ls[:, 1:] - ls[:, :-1]
    e = torch.zeros_like(lt, dtype=torch.bool)
    ex = dxt.abs() > jump; ey = dyt.abs() > jump
    e[:, :, 1:] |= ex; e[:, :, :-1] |= ex; e[:, 1:] |= ey; e[:, :-1] |= ey
    band = F.max_pool2d(e.float().unsqueeze(1), 2 * band_px + 1, 1, band_px).squeeze(1) > 0
    bx = band[:, :, 1:] & band[:, :, :-1]; by = band[:, 1:] & band[:, :-1]
    num = ((dxs - dxt).abs().clamp(max=cap) * bx).sum() + ((dys - dyt).abs().clamp(max=cap) * by).sum()
    return num / (bx.sum() + by.sum()).clamp_min(1)
