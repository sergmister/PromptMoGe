"""Score depth predictions against Faro GT on the g768 grid (768x1024), by region and distance stratum.

Frame-mean and pixel-pooled metrics; regions by ARKit confidence, distance strata, edges, normals, coverage,
and a LiDAR-anchored affine alignment.

Alignments
  none      raw metric output.
  lidar_ls  robust (trimmed-L2) scale+shift fitted to conf==2 LiDAR pixels at 192x256 ("late global
            anchoring"). No GT is used.
  gt_ss     oracle LS scale+shift fitted to GT on all_annot pixels -- measures relative geometry.

Usage: python -m promptmoge.train.score --sources raw ours=preds/<tag>_s1 ... --out results/x.json
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
cv2.setNumThreads(1)

from .data import SCENES, Frame, iter_frames, load_depth_png, resize
from .align import fit_scale_shift_l2, fit_scale_shift_robust

GRID = (768, 1024)      # scoring grid; `--grid H W` overrides it (Model B is judged at its own 240x320)
_LUT_PATH = Path(__file__).resolve().parent / "calibration" / "train_lidar_calibration_lut.json"
try:
    import json as _json
    _L = _json.load(open(_LUT_PATH)); _LUT = (np.asarray(_L["log_depth"]), np.asarray(_L["log_ratio"]))
except Exception:
    _LUT = None
REGIONS = ["valid_conf", "lowconf", "noreturn", "all_annot"]
STRATA = [("d0_1", 0.0, 1.0), ("d1_2", 1.0, 2.0), ("d2_3", 2.0, 3.0), ("d3_inf", 3.0, np.inf)]
ALIGNS = ["none", "lidar_ls", "gt_ss"]
KEYS = ["absrel", "rmse", "d125", "d105"]


def _metrics(pred: np.ndarray, gt: np.ndarray) -> Tuple[np.ndarray, int]:
    """Return sums [sum absrel, sum sq err, n d<1.25, n d<1.05] and n."""
    pred = pred.astype(np.float32); gt = gt.astype(np.float32)
    err = pred - gt
    ae = np.abs(err) / gt
    r = pred / gt
    ratio = np.maximum(r, 1.0 / np.maximum(r, 1e-9))
    return np.array([ae.sum(dtype=np.float64), (err * err).sum(dtype=np.float64), (ratio < 1.25).sum(), (ratio < 1.05).sum()], dtype=np.float64), pred.size


def _normals_from_depth(d: np.ndarray, fx: float, fy: float, cx: float, cy: float) -> np.ndarray:
    """Unit normals [H,W,3] from a depth map via central differences of the back-projected points (camera frame)."""
    H, W = d.shape
    d = d.astype(np.float32)
    u = ((np.arange(W, dtype=np.float32) - cx) / fx)[None, :]; v = ((np.arange(H, dtype=np.float32) - cy) / fy)[:, None]
    X = u * d; Y = v * d
    ax, ay, az = X[1:-1, 2:] - X[1:-1, :-2], Y[1:-1, 2:] - Y[1:-1, :-2], d[1:-1, 2:] - d[1:-1, :-2]
    bx, by, bz = X[2:, 1:-1] - X[:-2, 1:-1], Y[2:, 1:-1] - Y[:-2, 1:-1], d[2:, 1:-1] - d[:-2, 1:-1]
    nx = ay * bz - az * by; ny = az * bx - ax * bz; nz = ax * by - ay * bx
    out = np.zeros((H, W, 3), np.float32)
    nrm = np.sqrt(nx * nx + ny * ny + nz * nz); nrm = np.maximum(nrm, 1e-9)
    out[1:-1, 1:-1, 0] = nx / nrm; out[1:-1, 1:-1, 1] = ny / nrm; out[1:-1, 1:-1, 2] = nz / nrm
    return out


def _gt_normal_ctx(gt: np.ndarray, gtv: np.ndarray, K: np.ndarray, scale: float):
    """GT-side quantities for the normal metric (computed once per frame and scale): GT normals and the evaluation
    mask (interior pixels whose 3x3 GT neighbourhood is valid and free of depth jumps > 10 %)."""
    fx, fy, cx, cy = K[0, 0] * scale, K[1, 1] * scale, K[0, 2] * scale, K[1, 2] * scale
    ng = _normals_from_depth(gt.astype(np.float32), fx, fy, cx, cy)
    ok = gtv.copy(); ok[[0, -1], :] = False; ok[:, [0, -1]] = False
    lg = np.log(np.where(gtv, gt, 1.0).astype(np.float32))
    jump = np.zeros_like(gtv)
    jx = np.abs(lg[:, 1:] - lg[:, :-1]) > 0.1; jy = np.abs(lg[1:, :] - lg[:-1, :]) > 0.1
    jump[:, 1:] |= jx; jump[:, :-1] |= jx; jump[1:, :] |= jy; jump[:-1, :] |= jy
    nb = cv2.dilate((jump | ~gtv).astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    return {"ng": ng, "mask": ok & ~nb, "intr": (fx, fy, cx, cy)}


def _normal_error(pred: np.ndarray, ctx: dict):
    """Mean angular error (deg) of depth-derived normals vs the cached GT context. Returns (sum_deg, n, n_within_11.25)."""
    fx, fy, cx, cy = ctx["intr"]
    m = ctx["mask"] & (pred > 0)
    if not m.any():
        return 0.0, 0, 0
    npr = _normals_from_depth(np.where(pred > 0, pred, 1.0).astype(np.float32), fx, fy, cx, cy)
    cosang = np.clip((ctx["ng"][m] * npr[m]).sum(-1), -1, 1)
    ang = np.degrees(np.arccos(cosang))
    return float(ang.sum()), int(m.sum()), int((ang < 11.25).sum())


def score_frame(args) -> Dict:
    fr, sources = args
    h, w = GRID
    K = fr.intrinsics()
    if K is None:
        K = (np.array([[1601.96, 0, 936.545], [0, 1601.96, 709.61], [0, 0, 1]]), (1920, 1440))  # scene 41069046 ships no pincam; constant wide intrinsics
    K, (kw, kh) = K
    kscale = w / kw
    gt = resize(fr.gt(), h, w, cv2.INTER_NEAREST)
    lr = fr.lidar(); cf = fr.conf()
    lr_g = resize(lr, h, w, cv2.INTER_NEAREST)
    cf_g = resize(cf, h, w, cv2.INTER_NEAREST)
    gtv = gt > 0
    masks = {"all_annot": gtv, "valid_conf": gtv & (cf_g == 2), "lowconf": gtv & (cf_g == 1), "noreturn": gtv & (cf_g == 0)}
    for name, lo, hi in STRATA:
        masks[name] = gtv & (gt >= lo) & (gt < hi)
    # depth-discontinuity regions: within 3 px of a GT log-depth jump > 0.1 (~10 %) between valid neighbours
    lg = np.log(np.where(gtv, gt, 1.0))
    jump = np.zeros_like(gtv)
    dx = (np.abs(lg[:, 1:] - lg[:, :-1]) > 0.1) & gtv[:, 1:] & gtv[:, :-1]
    dy = (np.abs(lg[1:, :] - lg[:-1, :]) > 0.1) & gtv[1:, :] & gtv[:-1, :]
    jump[:, 1:] |= dx; jump[:, :-1] |= dx; jump[1:, :] |= dy; jump[:-1, :] |= dy
    edge = cv2.dilate(jump.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
    masks["edge"] = gtv & edge
    masks["nonedge"] = gtv & ~edge
    # sensor confidently wrong: conf==2 but LiDAR off by > 15 % from Faro (thin structures smoothed into the background,
    # small objects flattened, cables against walls). Split by sign: 'far' = sensor reads too far (object flattened away).
    with np.errstate(divide="ignore", invalid="ignore"):
        rel = np.where(gtv, (lr_g - gt) / np.maximum(gt, 1e-3), 0.0)
    masks["conf_wrong"] = gtv & (cf_g == 2) & (np.abs(rel) > 0.15)
    masks["conf_wrong_far"] = gtv & (cf_g == 2) & (rel > 0.15)
    # thin / near-side structure: within 3 px of a GT jump AND on the near side of it (gt < local max by > 10 %)
    gmax = cv2.dilate(np.where(gtv, gt, 0).astype(np.float32), np.ones((7, 7), np.uint8))
    masks["thin_near"] = gtv & edge & (gt < 0.9 * gmax)
    # boundary F-score is evaluated at half resolution (384x512): every source natively supports that grid, so the
    # 2x bilinear upsampling of a 480x640 output does not blur its edges below the per-pixel jump test
    gt_h = cv2.resize(gt, (w // 2, h // 2), interpolation=cv2.INTER_NEAREST); gtv_h = gt_h > 0
    lg_h = np.log(np.where(gtv_h, gt_h, 1.0)); jump_h = np.zeros_like(gtv_h)
    jhx = (np.abs(lg_h[:, 1:] - lg_h[:, :-1]) > 0.1) & gtv_h[:, 1:] & gtv_h[:, :-1]; jhy = (np.abs(lg_h[1:, :] - lg_h[:-1, :]) > 0.1) & gtv_h[1:, :] & gtv_h[:-1, :]
    jump_h[:, 1:] |= jhx; jump_h[:, :-1] |= jhx; jump_h[1:, :] |= jhy; jump_h[:-1, :] |= jhy
    gt_edges = jump_h & gtv_h
    gt_edge_dil = cv2.dilate(gt_edges.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    nctx = _gt_normal_ctx(gt, gtv, K, kscale)
    gt4 = cv2.resize(gt, (w // 4, h // 4), interpolation=cv2.INTER_NEAREST)
    nctx4 = _gt_normal_ctx(gt4, gt4 > 0, K, kscale / 4)
    out = {}
    for sname, sdir in sources.items():
        calib = None; lut = None
        if sdir is not None and "@" in sdir:      # "dir@a,b": global affine calibration; "dir@lut:file.json": depth-dependent
            sdir, cab = sdir.split("@", 1)
            if cab.startswith("lut:"):
                import json as _json
                L = _json.load(open(cab[4:])); lut = (np.asarray(L["log_depth"]), np.asarray(L["log_ratio"]))
            else:
                calib = tuple(float(x) for x in cab.split(","))
        if sname.startswith("raw"):
            pred = lr_g.copy(); pred_lr = lr.copy()
        else:
            p = Path(sdir) / fr.scene / f"{fr.ts}.png"
            if not p.exists():
                continue
            pn = load_depth_png(p)
            pred = resize(pn, h, w, cv2.INTER_AREA if pn.shape[0] > h else cv2.INTER_LINEAR)
            pred_lr = resize(pn, lr.shape[0], lr.shape[1], cv2.INTER_AREA)
        if calib is not None:
            pred = np.where(pred > 0, calib[0] * pred + calib[1], 0.0).astype(np.float32)
            pred_lr = np.where(pred_lr > 0, calib[0] * pred_lr + calib[1], 0.0).astype(np.float32)
        if lut is not None:
            def _apply(x):
                lx = np.log(np.maximum(x, 1e-3)); return np.where(x > 0, x * np.exp(np.interp(lx, lut[0], lut[1])), 0.0).astype(np.float32)
            pred = _apply(pred); pred_lr = _apply(pred_lr)
        pv = pred > 0
        # alignments
        aligned = {"none": pred}
        m_l = (cf == 2) & (pred_lr > 0) & (lr > 0)
        if m_l.sum() >= 64:
            a, b = fit_scale_shift_robust(pred_lr[m_l], lr[m_l])
        else:
            a, b = 1.0, 0.0
        aligned["lidar_ls"] = np.where(pv, a * pred + b, 0.0)
        m_g = gtv & pv
        a2, b2 = fit_scale_shift_l2(pred[m_g], gt[m_g]) if m_g.sum() >= 64 else (1.0, 0.0)
        aligned["gt_ss"] = np.where(pv, a2 * pred + b2, 0.0)
        res = {"fit_lidar": [a, b], "fit_gt": [a2, b2], "coverage": float((pv & gtv).sum() / max(gtv.sum(), 1))}
        for al, ap in aligned.items():
            apv = ap > 0
            if al == "none" or (al == "lidar_ls" and sname != "raw"):
                # boundary F-score: pred depth edges (log-jump > 10 %) vs GT edges, 3 px tolerance, on GT-valid area
                ap_h = cv2.resize(np.where(apv, ap, 0.0).astype(np.float32), (w // 2, h // 2), interpolation=cv2.INTER_AREA); apv_h = ap_h > 0
                lp = np.log(np.where(apv_h, ap_h, 1.0))
                pj = np.zeros_like(apv_h)
                pjx = (np.abs(lp[:, 1:] - lp[:, :-1]) > 0.1) & apv_h[:, 1:] & apv_h[:, :-1]; pjy = (np.abs(lp[1:, :] - lp[:-1, :]) > 0.1) & apv_h[1:, :] & apv_h[:-1, :]
                pj[:, 1:] |= pjx; pj[:, :-1] |= pjx; pj[1:, :] |= pjy; pj[:-1, :] |= pjy
                pj &= gtv_h
                p_dil = cv2.dilate(pj.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
                tp_p = float((pj & gt_edge_dil).sum()); tp_r = float((gt_edges & p_dil).sum())
                prec = tp_p / max(float(pj.sum()), 1.0); rec = tp_r / max(float(gt_edges.sum()), 1.0)
                f1 = 2 * prec * rec / max(prec + rec, 1e-9)
                res[f"{al}|edge_f1"] = ([f1, prec, rec, 0.0], 1)
            if al in ("none", "lidar_ls") and sname != "raw":
                # per-pixel "beats the (calibrated) sensor" rate on confident pixels: |pred-gt| < |lidar_cal-gt|
                mm = masks["valid_conf"] & apv
                if mm.any():
                    lr_c = lr_g * np.exp(np.interp(np.log(np.maximum(lr_g, 1e-3)), _LUT[0], _LUT[1])) if _LUT is not None else lr_g
                    win = (np.abs(ap[mm] - gt[mm]) < np.abs(lr_c[mm] - gt[mm]))
                    res[f"{al}|beats_sensor"] = ([float(win.sum()), 0.0, float(win.sum()), 0.0], int(mm.sum()))
            if al == "none" and sdir is not None:
                # normal HEAD output (if saved by evaluate.py --save_normals): compare with Faro depth-derived normals at half res
                pn_path = Path(str(sdir).rsplit("_s", 1)[0] + "_normal") / fr.scene / f"{fr.ts}.npz"
                if pn_path.exists():
                    nh = np.load(pn_path)["n"].astype(np.float32) / 127.0
                    nh = cv2.resize(nh, (w // 2, h // 2), interpolation=cv2.INTER_LINEAR)
                    nh /= np.maximum(np.linalg.norm(nh, axis=-1, keepdims=True), 1e-6)
                    gt_h2 = cv2.resize(gt, (w // 2, h // 2), interpolation=cv2.INTER_NEAREST)
                    ctx2 = _gt_normal_ctx(gt_h2, gt_h2 > 0, K, kscale / 2)
                    mh = ctx2["mask"] & (np.abs(nh).sum(-1) > 0.5)
                    if mh.any():
                        # MoGe's normal head uses the opposite sign convention to our finite-difference normals (measured 168°
                        # mean angle = 12° after the flip); resolve the convention per frame by the sign of the mean dot product
                        dots = (ctx2["ng"][mh] * nh[mh]).sum(-1); sgn = 1.0 if dots.mean() >= 0 else -1.0
                        cosang = np.clip(sgn * dots, -1, 1); ang = np.degrees(np.arccos(cosang))
                        res["none|head_normals"] = ([float(ang.sum()), 0.0, int((ang < 11.25).sum()), 0.0], int(mh.sum()))
            if al in ("none", "lidar_ls"):
                sd, nn_, n11 = _normal_error(ap, nctx)
                if nn_ > 0:
                    res[f"{al}|normals"] = ([sd, 0.0, n11, 0.0], nn_)
                # low-frequency normals (wall angles): depth area-downsampled 4x to 192x256 first
                ap4 = cv2.resize(np.where(apv, ap, 0.0).astype(np.float32), (w // 4, h // 4), interpolation=cv2.INTER_AREA)
                sd4, nn4, n114 = _normal_error(ap4, nctx4)
                if nn4 > 0:
                    res[f"{al}|normals_lf"] = ([sd4, 0.0, n114, 0.0], nn4)
            for rname, m in masks.items():
                mm = m & apv
                n = int(mm.sum())
                if n == 0:
                    continue
                sums, _ = _metrics(ap[mm], gt[mm])
                res[f"{al}|{rname}"] = (sums.tolist(), n)
        out[sname] = res
    return fr.key, out


class Acc:
    def __init__(self):
        self.frames: List[np.ndarray] = []; self.sums = np.zeros(4); self.n = 0
    def add(self, sums, n):
        s = np.asarray(sums); self.frames.append(s / n); self.sums += s; self.n += n
    def result(self):
        if not self.frames:
            return None
        fm = np.mean(self.frames, axis=0)
        def fmt(v): return {"absrel": float(v[0]), "rmse": float(np.sqrt(v[1])), "d125": float(v[2]), "d105": float(v[3])}
        pooled = self.sums / self.n
        return {"n_frames": len(self.frames), "n_px": int(self.n), "frame_mean": fmt(fm), "pixel_pooled": fmt(pooled)}


def aggregate(per_frame: Dict[str, Dict], scenes: List[str]) -> Dict:
    """per_frame: key -> {source -> {metric_key -> (sums, n)}}. Returns nested source->scene->align|region."""
    out = {}
    for key, srcs in per_frame.items():
        scene = key.split("/")[0]
        for sname, res in srcs.items():
            for k, v in res.items():
                if "|" not in k:
                    continue
                for sc in (scene, "ALL"):
                    out.setdefault(sname, {}).setdefault(sc, {}).setdefault(k, Acc()).add(*v)
    return {s: {sc: {k: a.result() for k, a in d.items()} for sc, d in scs.items()} for s, scs in out.items()}


def table(agg: Dict, align: str = "none", scene: str = "ALL", regions=("valid_conf", "lowconf", "noreturn", "all_annot", "edge", "nonedge", "d0_1", "d1_2", "d2_3", "d3_inf", "normals", "normals_lf", "beats_sensor", "conf_wrong", "conf_wrong_far", "thin_near", "edge_f1", "head_normals")) -> str:
    """For the 'normals' pseudo-region the columns read: mean angle [deg] / - / frac < 11.25 deg / -."""
    lines = [f"### align={align} scene={scene} (frame-mean AbsRel / RMSE[m] / d<1.25 / d<1.05)", "",
             "| source | " + " | ".join(regions) + " |", "|---|" + "---|" * len(regions)]
    for s, scs in agg.items():
        row = [s]
        for r in regions:
            m = scs.get(scene, {}).get(f"{align}|{r}")
            if m is None:
                row.append("-")
            else:
                f = m["frame_mean"]
                row.append(f"{f['absrel']:.4f} / {f['rmse']:.3f} / {f['d125']:.3f} / {f['d105']:.3f}")
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", nargs="+", required=True, help="name=dir or 'raw'")
    ap.add_argument("--scenes", nargs="*", default=SCENES)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--grid", type=int, nargs=2, default=None, help="scoring grid H W (default 768 1024); use 240 320 for a 240x320 model: a larger grid penalises it for its resolution")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--pred_root", default="preds", help="a bare source tag resolves under this directory (where evaluate.py writes)")
    ap.add_argument("--workers", type=int, default=32)
    a = ap.parse_args()
    if a.grid:
        global GRID
        GRID = tuple(a.grid)
    sources = {}
    for s in a.sources:
        if s == "raw":
            sources["raw"] = None
        else:
            n, d = s.split("=", 1)      # raw_cal=@a,b applies a calibration to the raw LiDAR
            if not d.startswith("@") and not d.startswith("raw"):
                # a bare tag resolves under --pred_root (where evaluate.py writes); a missing directory is an error, never a silent drop
                pred_root = Path(a.pred_root)
                if not Path(d).is_dir() and (pred_root / d).is_dir():
                    d = str(pred_root / d)
                if not Path(d).is_dir():
                    raise SystemExit(f"score.py: prediction directory for source {n!r} not found: {d}")
            sources[n] = d
    frames = list(iter_frames(a.scenes, a.stride, a.limit))
    per_frame = {}
    import time
    t0 = time.time()
    with ProcessPoolExecutor(a.workers) as ex:
        for i, (key, out) in enumerate(ex.map(score_frame, [(fr, sources) for fr in frames], chunksize=4)):
            per_frame[key] = out
            if (i + 1) % 100 == 0:
                print(f"scored {i+1}/{len(frames)} ({time.time()-t0:.0f}s)", file=sys.stderr, flush=True)
    agg = aggregate(per_frame, a.scenes)
    fits = {s: {k: v["fit_lidar"] for k, v in ((k, r[s]) for k, r in per_frame.items() if s in r)} for s in sources}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps({"grid": GRID, "scenes": a.scenes, "stride": a.stride, "agg": agg, "fits_lidar": fits}, indent=1))
    for al in ALIGNS:
        print(table(agg, al)); print()
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
