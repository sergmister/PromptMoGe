"""Sensor simulator v7 — mechanism-based. ARKit's depth is a sparse direct-ToF measurement densified by an RGB-guided network, so
its structural errors (flattened small objects, edges bleeding onto RGB edges, thin structures lost, rims) come from (1) what the
sparse beams see and (2) how the densifier interpolates. v7 reproduces both:
  1. sparse_samples(gt, rgb): a jittered regular grid of beams (~30x23 over 256x192), each returning a depth drawn from its
     footprint (mixed pixels at edges: the nearer or farther surface, footprint-weighted), dropped on dark / far / grazing pixels;
  2. Densifier: a small U-Net trained on the REAL pairs to map (sparse samples of the Faro GT + RGB) -> the real ARKit depth and
     confidence (heteroscedastic log-depth: mu, log_b; 3 conf logits); applied to renders it interpolates the way ARKit does.
simulate_v7(model, gt, rgb) -> (depth [B,1,h,w] dense, conf [B,1,h,w] in {0,1,2}) with a spatially correlated Laplace draw.

train:  python -m promptmoge.train.sensor_sim.v7 --train --steps 6000 --out runs/sensor_sim/densifier_v7.pt
"""
from __future__ import annotations
import argparse, math, random, time
from pathlib import Path
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from promptmoge.train.sensor_sim.model import correlated_noise

GRID = (23, 30)            # default beams (rows, cols); the iPhone LiDAR has 112 beams on a 14x8 triangular grid -> grid (8, 14) is the physical setting
FOOT = 3                   # beam footprint in sensor px (odd)


def sparse_samples(gt: torch.Tensor, rgb: torch.Tensor, grid=GRID, foot=FOOT, jitter=True, p_mixed=0.5, drop_dark=0.6, drop_far_m=4.5, seed=None, lattice="rect", lattice_aug=True):
    """gt [B,1,h,w] metres (0 invalid, hole-filled preferred), rgb [B,3,h,w] 0-1 -> (sparse [B,1,h,w] depth at sample pixels else 0, mask [B,1,h,w]).
    Batched: every frame's lattice (phase / rotation / scale / per-beam jitter) is drawn at once; footprint gathers use advanced indexing."""
    B, _, h, w = gt.shape; dev = gt.device; g = torch.Generator(device=dev)
    if seed is not None: g.manual_seed(seed)
    r = foot // 2; R, C = grid; N = R * C
    ys = torch.linspace(foot, h - 1 - foot, R, device=dev); xs = torch.linspace(foot, w - 1 - foot, C, device=dev)
    jy = (torch.rand(B, R, C, generator=g, device=dev) * 2 - 1) * (h / R / 2 - r) if jitter else torch.zeros(B, R, C, device=dev)
    jx = (torch.rand(B, R, C, generator=g, device=dev) * 2 - 1) * (w / C / 2 - r) if jitter else torch.zeros(B, R, C, device=dev)
    xoff = torch.zeros(R, C, device=dev)
    if lattice == "tri" and C > 1:
        xoff[1::2, :] = (xs[1] - xs[0]) / 2
    Y = ys.view(1, R, 1) + jy; X = xs.view(1, 1, C) + xoff.view(1, R, C) + jx
    if lattice_aug:
        th = (torch.rand(B, 1, 1, generator=g, device=dev) * 2 - 1) * math.radians(3.0); sc = 1.0 + (torch.rand(B, 1, 1, generator=g, device=dev) * 2 - 1) * 0.05
        oy = (torch.rand(B, 1, 1, generator=g, device=dev) * 2 - 1) * (h / R) / 2; ox = (torch.rand(B, 1, 1, generator=g, device=dev) * 2 - 1) * (w / C) / 2
        Yc, Xc = Y - h / 2, X - w / 2
        Y = h / 2 + sc * (torch.cos(th) * Yc - torch.sin(th) * Xc) + oy; X = w / 2 + sc * (torch.sin(th) * Yc + torch.cos(th) * Xc) + ox
    cy = Y.round().long().clamp(r, h - 1 - r).view(B, N); cx = X.round().long().clamp(r, w - 1 - r).view(B, N)
    dy, dx = torch.meshgrid(torch.arange(-r, r + 1, device=dev), torch.arange(-r, r + 1, device=dev), indexing="ij")
    py = cy.unsqueeze(-1) + dy.reshape(1, 1, -1); px = cx.unsqueeze(-1) + dx.reshape(1, 1, -1)                 # [B,N,F]
    bidx = torch.arange(B, device=dev).view(B, 1, 1)
    patch = gt[:, 0][bidx, py, px]                                                                                # [B,N,F]
    valid = patch > 0
    med = torch.where(valid, patch, torch.full_like(patch, float("nan"))).nanmedian(-1).values                   # [B,N]
    idx = torch.randint(0, patch.shape[-1], (B, N), generator=g, device=dev)
    rnd = torch.gather(patch, -1, idx.unsqueeze(-1)).squeeze(-1); rnd = torch.where(rnd > 0, rnd, med)
    d = torch.where(torch.rand(B, N, generator=g, device=dev) < p_mixed, rnd, med)
    gr = rgb.mean(1)[bidx.view(B, 1), cy, cx]                                                                    # [B,N]
    keep = torch.isfinite(d) & (d > 0)
    keep &= ~((gr < 0.15) & (torch.rand(B, N, generator=g, device=dev) < drop_dark))
    keep &= ~((d > drop_far_m) & (torch.rand(B, N, generator=g, device=dev) < 0.7))
    keep &= torch.rand(B, N, generator=g, device=dev) > 0.03
    sparse = torch.zeros_like(gt); mask = torch.zeros_like(gt)
    bsel = bidx.view(B, 1).expand(B, N)[keep]; ysel = cy[keep]; xsel = cx[keep]
    sparse[bsel, 0, ysel, xsel] = d[keep]; mask[bsel, 0, ysel, xsel] = 1.0
    return sparse, mask


def guided_fill(sparse: torch.Tensor, mask: torch.Tensor, rgb: torch.Tensor, sigma_r: float = 0.08, iters: int = 24):
    """RGB-guided fill of the sparse log-depth samples by iterative weighted propagation over the 3x3 neighbourhood, each
    neighbour weighted by its grey similarity exp(-(dg/sigma_r)^2): a cheap geodesic (guided) interpolation — depth spreads
    along similar-colour regions and stops at image edges, so RGB-visible edges and thin structures are preserved (as ARKit's
    guidance does). Returns (filled log-depth, coverage)."""
    x = torch.log(sparse.clamp_min(1e-3)) * mask; cov = mask.clone(); g = rgb.mean(1, keepdim=True)
    B, _, h, w = x.shape
    gp = F.pad(g, (1, 1, 1, 1), mode="replicate")
    shifts = [(dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1) if (dy, dx) != (0, 0)]
    wts = [torch.exp(-((gp[:, :, 1 + dy:1 + dy + h, 1 + dx:1 + dx + w] - g) / sigma_r) ** 2) * (0.5 if dy and dx else 1.0) for dy, dx in shifts]
    for _ in range(iters):
        xp = F.pad(x, (1, 1, 1, 1), mode="replicate"); cp = F.pad(cov, (1, 1, 1, 1), mode="replicate")
        num = torch.zeros_like(x); den = torch.zeros_like(x)
        for (dy, dx), wgt in zip(shifts, wts):
            c = cp[:, :, 1 + dy:1 + dy + h, 1 + dx:1 + dx + w]; v = xp[:, :, 1 + dy:1 + dy + h, 1 + dx:1 + dx + w]
            num = num + wgt * c * v; den = den + wgt * c
        upd = (cov == 0) & (den > 0.05)
        x = torch.where(upd, num / den.clamp_min(1e-6), x); cov = torch.where(upd, torch.ones_like(cov), cov)
    return x, cov


def nearest_fill(sparse: torch.Tensor, mask: torch.Tensor, iters: int = 12):
    """cheap dense cue: iterative masked dilation of the sparse log-depth (nearest-ish fill)."""
    x = torch.log(sparse.clamp_min(1e-3)) * mask; m = mask.clone()
    k = torch.ones(1, 1, 3, 3, device=sparse.device)
    for _ in range(iters):
        num = F.conv2d(x, k, padding=1); den = F.conv2d(m, k, padding=1)
        upd = (m == 0) & (den > 0)
        x = torch.where(upd, num / den.clamp_min(1e-6), x); m = torch.where(upd, torch.ones_like(m), m)
    return x, m


class Densifier(nn.Module):
    """U-Net: [sparse log-depth, sample mask, nearest-filled log-depth, fill mask, rgb(3)] -> [mu, log_b, conf logits x3]."""
    def __init__(self, cin=7, base=32, grid=GRID, lattice="rect", fill="nearest"):
        super().__init__()
        self.grid = tuple(grid); self.lattice = lattice; self.fill = fill
        def blk(a, b): return nn.Sequential(nn.Conv2d(a, b, 3, padding=1), nn.GELU(), nn.Conv2d(b, b, 3, padding=1), nn.GELU())
        self.e1 = blk(cin, base); self.e2 = blk(base, base * 2); self.e3 = blk(base * 2, base * 4); self.e4 = blk(base * 4, base * 8)
        self.d3 = blk(base * 8 + base * 4, base * 4); self.d2 = blk(base * 4 + base * 2, base * 2); self.d1 = blk(base * 2 + base, base)
        self.head = nn.Conv2d(base, 5, 1)
        nn.init.zeros_(self.head.weight); self.head.bias.data = torch.tensor([0.0, -3.0, -3.0, -1.0, 3.0])
    def forward(self, x):
        e1 = self.e1(x); e2 = self.e2(F.avg_pool2d(e1, 2)); e3 = self.e3(F.avg_pool2d(e2, 2)); e4 = self.e4(F.avg_pool2d(e3, 2))
        d3 = self.d3(torch.cat([F.interpolate(e4, scale_factor=2, mode="bilinear", align_corners=False), e3], 1))
        d2 = self.d2(torch.cat([F.interpolate(d3, scale_factor=2, mode="bilinear", align_corners=False), e2], 1))
        d1 = self.d1(torch.cat([F.interpolate(d2, scale_factor=2, mode="bilinear", align_corners=False), e1], 1))
        return self.head(d1)


def load_densifier(path, device="cuda"):
    ck = torch.load(path, map_location=device)
    grid = tuple(ck.get("grid", GRID)) if isinstance(ck, dict) and "state_dict" in ck else GRID
    lattice = ck.get("lattice", "rect") if isinstance(ck, dict) and "state_dict" in ck else "rect"
    base = ck.get("base", 32) if isinstance(ck, dict) and "state_dict" in ck else 32
    fill = ck.get("fill", "nearest") if isinstance(ck, dict) and "state_dict" in ck else "nearest"
    m = Densifier(base=base, grid=grid, lattice=lattice, fill=fill).to(device).eval(); m.load_state_dict(ck["state_dict"] if isinstance(ck, dict) and "state_dict" in ck else ck)
    return m


FILL = "guided"            # "guided" (RGB joint-bilateral, v7b) or "nearest" (v7a)


def densifier_inputs(sparse, mask, rgb, fill_mode=None):
    mode = fill_mode or FILL
    if mode == "guided":
        fill, fm = guided_fill(sparse, mask, rgb)
        nn_fill, _ = nearest_fill(sparse, mask); fill = torch.where(fm > 0, fill, nn_fill)
    else:
        fill, fm = nearest_fill(sparse, mask)
    ls = torch.log(sparse.clamp_min(1e-3)) * mask
    return torch.cat([ls, mask, fill, fm, rgb], 1), fill


def forward_dense(model, sparse, mask, rgb):
    """returns mu (log depth, predicted as fill + residual), log_b, conf logits."""
    x, fill = densifier_inputs(sparse, mask, rgb, getattr(model, "fill", None)); o = model(x)
    return fill + o[:, 0:1], o[:, 1:2], o[:, 2:5]


@torch.no_grad()
def simulate_v7(model, gt, rgb, noise_sigma_px=12.0, conf_corr_px=20.0, noise_gain=1.0, seed=None, dense=True):
    """gt [B,1,h,w] hole-filled metres, rgb [B,3,h,w] -> depth [B,1,h,w] (dense), conf [B,1,h,w] in {0,1,2}."""
    sparse, mask = sparse_samples(gt, rgb, grid=getattr(model, "grid", GRID), seed=seed, lattice=getattr(model, "lattice", "rect"))
    mu, log_b, logits = forward_dense(model, sparse, mask, rgb)
    g = None
    if seed is not None:
        g = torch.Generator(device=gt.device); g.manual_seed(seed + 1)
    eps = correlated_noise(mu.shape, noise_sigma_px, mu.device, g)
    # correlated Laplace: sign * exponential magnitude from the Gaussian field
    u = 0.5 * (1.0 + torch.erf(eps / 2 ** 0.5)).clamp(1e-4, 1 - 1e-4)
    lap = torch.where(u < 0.5, torch.log(2 * u), -torch.log(2 * (1 - u)))
    logd = mu + noise_gain * torch.exp(log_b) * lap
    probs = F.softmax(logits, 1); eps_c = correlated_noise(mu.shape, conf_corr_px, mu.device, g)
    uc = 0.5 * (1.0 + torch.erf(eps_c / 2 ** 0.5)); cdf = probs.cumsum(1)
    conf = (uc > cdf[:, :1]).float() + (uc > cdf[:, 1:2]).float()
    depth = torch.exp(logd)
    if not dense:
        depth = torch.where(conf > 0, depth, torch.zeros_like(depth))
    return depth, conf


@torch.no_grad()
def simulate_hybrid(model, sim_v6, gt, rgb, v6_kw: dict, thresh: float = 0.05, open_px: int = 1, seed=None, generator=None, feather_px: float = 1.5):
    """v7h: the v6 draw (edges, noise, thin structures, confidence) with the densifier's structural deviation applied
    multiplicatively inside failure regions (|log(mu7/GT)| > thresh, mask opened by `open_px` so thin structures stay v6).
    gt hole-free metres [B,1,h,w] (0 = undefined), rgb [B,3,h,w]. Returns depth, conf (dense)."""
    from promptmoge.train.sensor_sim.model import simulate
    d6, c6, _ = simulate(sim_v6, gt.clamp(0, 10), rgb, generator=generator, **v6_kw)
    sparse, mask = sparse_samples(gt, rgb, grid=getattr(model, "grid", GRID), seed=seed, lattice=getattr(model, "lattice", "rect"))
    mu, _, _ = forward_dense(model, sparse, mask, rgb)
    valid = gt > 0
    dev = torch.where(valid, mu - torch.log(gt.clamp_min(1e-3)), torch.zeros_like(mu))
    fail = (dev.abs() > thresh) & valid
    if open_px > 0:                                     # morphological opening: erode then dilate (drops thin structures)
        k = 2 * open_px + 1
        er = 1.0 - F.max_pool2d(1.0 - fail.float(), k, 1, open_px); fail = F.max_pool2d(er, k, 1, open_px) > 0.5
    w = fail.float()
    if feather_px > 0:                                  # soft mask: no step at the border of a failure region (spurious edges)
        from promptmoge.train.sensor_sim.model import _gauss_blur
        w = _gauss_blur(w, feather_px).clamp(0, 1)
    depth = d6 * torch.exp(dev * w)
    return depth, c6


def train(a):
    from torch.utils.data import DataLoader
    from promptmoge.train.sensor_sim.train import SensorPairs, _list_frames, STAGE
    root = Path(a.arkit_root) if a.arkit_root else STAGE
    frames = _list_frames(root); vids = sorted({v for v, _ in frames}); hold = set(vids[int(0.9 * len(vids)):])
    tr = [f for f in frames if f[0] not in hold]; va = [f for f in frames if f[0] in hold]
    dl = DataLoader(SensorPairs(tr, root=root), batch_size=a.batch, shuffle=True, num_workers=3, drop_last=True, persistent_workers=True)
    dv = DataLoader(SensorPairs(va[:: max(1, len(va) // 200)], flip=False, root=root), batch_size=8, shuffle=False, num_workers=2)
    grid = tuple(a.grid); model = Densifier(base=a.base, grid=grid, lattice=a.lattice, fill=a.fill).cuda(); opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    print(f"beam grid {grid} = {grid[0]*grid[1]} beams, {a.lattice} lattice", flush=True)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.steps, pct_start=0.05)
    it = iter(dl); t0 = time.time(); step = 0
    while step < a.steps:
        try: gt, rgb, lr, cf = next(it)
        except StopIteration: it = iter(dl); continue
        gt, rgb, lr, cf = gt.cuda(), rgb.cuda(), lr.cuda(), cf.cuda()
        # hole-fill the Faro GT (nearest) before sampling: the beams see the real scene, not the Faro holes
        gfill, _ = nearest_fill(gt, (gt > 0).float(), iters=30); gsrc = torch.where(gt > 0, gt, torch.exp(gfill))
        sparse, mask = sparse_samples(gsrc, rgb, grid=grid, lattice=a.lattice)
        mu, log_b, logits = forward_dense(model, sparse, mask, rgb)
        tgt = torch.log(lr.clamp_min(1e-3)); valid = (lr > 0) & (gt > 0)
        nll = (torch.abs(mu - tgt) * torch.exp(-log_b) + log_b)[valid].mean()          # Laplace NLL on log depth
        ce = F.cross_entropy(logits, cf, reduction="none")[valid[:, 0]].mean()
        loss = nll + 0.3 * ce
        opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step(); step += 1
        if step % 200 == 0 or step == a.steps:
            with torch.no_grad():
                err = torch.abs(torch.exp(mu) - lr)[valid] / lr[valid]
            print(f"step {step} loss {loss.item():.4f} nll {nll.item():.4f} ce {ce.item():.4f} absrel(mu vs ARKit) {err.mean().item():.4f} {time.time()-t0:.0f}s", flush=True)
    torch.save({"state_dict": model.state_dict(), "grid": grid, "lattice": a.lattice, "base": a.base, "fill": a.fill}, a.out); print("saved", a.out)
    # quick validation: imitation error (mu vs ARKit) and how the draw's error vs Faro GT compares with the real sensor's
    model.eval(); r_real = []; r_sim = []; r_mu = []
    with torch.no_grad():
        for gt, rgb, lr, cf in dv:
            gt, rgb, lr = gt.cuda(), rgb.cuda(), lr.cuda()
            gfill, _ = nearest_fill(gt, (gt > 0).float(), iters=30); gsrc = torch.where(gt > 0, gt, torch.exp(gfill))
            d, c = simulate_v7(model, gsrc, rgb, seed=0); sparse, mask = sparse_samples(gsrc, rgb, grid=grid, seed=0, lattice=a.lattice); mu, _, _ = forward_dense(model, sparse, mask, rgb)
            v = (gt > 0.3) & (gt < 5) & (lr > 0)
            r_real.append((torch.abs(lr - gt) / gt)[v]); r_sim.append((torch.abs(d - gt) / gt)[v]); r_mu.append((torch.abs(torch.exp(mu) - lr) / lr)[v])
    print(f"holdout: real ARKit AbsRel vs Faro {torch.cat(r_real).mean():.4f} | v7 draw AbsRel vs Faro {torch.cat(r_sim).mean():.4f} | densifier mean vs ARKit {torch.cat(r_mu).mean():.4f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--train", action="store_true"); ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--batch", type=int, default=8); ap.add_argument("--base", type=int, default=32); ap.add_argument("--grid", type=int, nargs=2, default=list(GRID), help="beam grid rows cols (iPhone: 9 12 ≈ 110 beams per pulse; ARKit integrates over time -> ~700 effective)"); ap.add_argument("--lattice", default="rect", choices=["rect", "tri"], help="iPhone: tri (triangular lattice)"); ap.add_argument("--fill", default="guided", choices=["guided", "nearest"], help="densifier base: RGB-guided joint-bilateral fill (v7b) or nearest fill (v7a)"); ap.add_argument("--lr", type=float, default=1e-3); ap.add_argument("--out", default="runs/sensor_sim/densifier_v7.pt"); ap.add_argument("--arkit_root", default=None, help="real training stage root (default: data/arkit_stage)")
    a = ap.parse_args()
    if a.train: train(a)
