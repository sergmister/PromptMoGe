"""Evaluation of (LiDAR-prompted) MoGe-3 on the ARKitScenes dev scenes (layout: see data.py).

quick_eval(model, teacher, stride)  -- in-memory deployed-metric evaluation on the dev subset: input-side calibration,
                                       3 refine steps, scored unaligned (the same treatment as PromptDA), plus the RGB-only
                                       conformance to the frozen teacher and the null-prompt vs skip-path agreement. The
                                       result carries PromptDA's numbers on the same subset and the gaps.
run_model(...)                      -- full run writing per-step depth PNGs under <pred_root>/<tag>_s<k> for score.py
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

os.environ.setdefault("FLEX_GEMM_AUTOTUNE_MODE", "always")
from promptmoge.train.data import SCENES, AsyncWriter, Frame, iter_frames, resize
from promptmoge.train.align import fit_scale_shift_l2

PRED = Path("preds")                                                                   # `--pred_root` overrides it
CKPT = os.environ.get("MOGE3_WEIGHTS", "checkpoints/moge-3-vitl/model.pt")             # `--base` overrides it
# PromptDA-Large on the same stride-10 subset, deployed metric (its output is already Faro-calibrated): reference row
PROMPTDA_SUBSET = {"valid_conf": 0.0137, "lowconf": 0.0593, "noreturn": 0.0637, "all_annot": 0.0155, "d2_3": 0.0328, "normals_lf": 8.6}
DEFAULT_LUT = str(Path(__file__).resolve().parent / "calibration" / "train_lidar_calibration_lut.json")
WIDE_K = (np.array([[1601.96, 0, 936.545], [0, 1601.96, 709.61], [0, 0, 1]]), (1920, 1440))   # constant ARKit wide intrinsics


def _absrel(pred, gt, m):
    if m.sum() == 0:
        return float("nan")
    return float((np.abs(pred[m] - gt[m]) / gt[m]).mean())


PROMPT_SHIFT = (0, 0)          # (dx, dy) sensor px: shift the LiDAR prompt relative to RGB (robustness test, --prompt_shift)
# Hard-prompt robustness: replace the frame's ARKit prompt by a single-pulse draw of the physical 14x8 = 112-beam
# lattice densified by `sensor_sim/v7`. ARKitScenes' sceneDepth is temporally integrated and
# behaves like several hundred effective samples; a device that hands over one pulse does not, and
# this is the instrument for how much a model leans on the prompt rather than on the image.
SIM_PROMPT = {"model": None, "seed": 0, "gain": 1.0}


def _sim_prompt(ld, cf, rgb_full):
    import torch.nn.functional as _F
    from promptmoge.train.sensor_sim.v7 import simulate_v7
    src = ld.clone()
    if not bool((src > 0).any()):
        return ld, cf
    med = src[src > 0].median()
    src = torch.where(src > 0, src, med)                      # the beams see the scene, not the sensor's holes
    rgb_lr = _F.interpolate(rgb_full, src.shape[-2:], mode="area")
    d, c = simulate_v7(SIM_PROMPT["model"], src, rgb_lr, noise_gain=SIM_PROMPT["gain"], seed=SIM_PROMPT["seed"])
    return d.float(), c.float()


def _shift(a: np.ndarray, dx: int, dy: int) -> np.ndarray:
    M = np.float32([[1, 0, dx], [0, 1, dy]]); h, w = a.shape[:2]
    return cv2.warpAffine(a, M, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REPLICATE)


def frame_tensors(fr: Frame):
    img = torch.from_numpy(fr.rgb().copy()).float().div(255).permute(2, 0, 1)[None].cuda()
    ld_np, cf_np = fr.lidar().copy(), fr.conf().copy()
    if PROMPT_SHIFT != (0, 0):
        ld_np = _shift(ld_np, *PROMPT_SHIFT); cf_np = _shift(cf_np, *PROMPT_SHIFT)
    ld = torch.from_numpy(ld_np)[None, None].cuda()
    cf = torch.from_numpy(cf_np).float()[None, None].cuda()
    if SIM_PROMPT["model"] is not None:
        ld, cf = _sim_prompt(ld, cf, img)
    return img, ld, cf


def _lf_normal_deg(pred, gt, K, kscale):
    """low-frequency (4x) depth-derived normal angle vs GT, as in score.py."""
    from promptmoge.train.score import _gt_normal_ctx, _normal_error
    h, w = gt.shape
    gt4 = cv2.resize(gt, (w // 4, h // 4), interpolation=cv2.INTER_NEAREST)
    ctx = _gt_normal_ctx(gt4, gt4 > 0, K, kscale / 4)
    p4 = cv2.resize(np.where(pred > 0, pred, 0.0).astype(np.float32), (w // 4, h // 4), interpolation=cv2.INTER_AREA)
    sd, n, _ = _normal_error(p4, ctx)
    return sd / n if n else float("nan")


@torch.no_grad()
def quick_eval(model, teacher=None, stride: int = 10, refine_steps: int = 3, num_tokens: int = 1200, use_fp16: bool = True,
               metric_from: str = "lidar_ls", calib_lut: str = DEFAULT_LUT, calibrate_input: bool = True):
    """Deployed-metric evaluation on the dev subset. Returns our numbers plus the PromptDA reference and the gaps."""
    h, w = 768, 1024
    lut = json.load(open(calib_lut)) if calib_lut else None
    acc = {k: [] for k in ["valid_conf", "lowconf", "noreturn", "all_annot", "d2_3", "normals_lf", "rgb_ls_all",
                           "conform_absrel", "conform_normal_deg", "null_vs_skip"]}
    for fr in iter_frames(stride=stride):
        img, ld, cf = frame_tensors(fr)
        gt = resize(fr.gt(), h, w, cv2.INTER_NEAREST); cfg = resize(fr.conf(), h, w, cv2.INTER_NEAREST); gtv = gt > 0
        out = model.infer(img, num_tokens=num_tokens, refine_steps=refine_steps, use_fp16=use_fp16, lidar_depth=ld, lidar_conf=cf,
                          metric_from=metric_from, calibration_lut=lut, calibrate_input=calibrate_input)
        d = out["depth"][0].float().cpu().numpy(); d = np.where(np.isfinite(d), d, 0)
        pred = resize(d, h, w, cv2.INTER_AREA); pv = pred > 0
        acc["valid_conf"].append(_absrel(pred, gt, gtv & pv & (cfg == 2)))
        acc["lowconf"].append(_absrel(pred, gt, gtv & pv & (cfg == 1)))
        acc["noreturn"].append(_absrel(pred, gt, gtv & pv & (cfg == 0)))
        acc["all_annot"].append(_absrel(pred, gt, gtv & pv))
        acc["d2_3"].append(_absrel(pred, gt, gtv & pv & (gt >= 2) & (gt < 3)))
        K = fr.intrinsics() or WIDE_K
        acc["normals_lf"].append(_lf_normal_deg(pred, gt, K[0], w / K[1][0]))
        # RGB-only path (skip) with LS anchoring, and conformance to teacher
        out_rgb = model.infer(img, num_tokens=num_tokens, refine_steps=refine_steps, use_fp16=use_fp16, lidar_depth=ld, lidar_conf=cf,
                              metric_from="lidar_ls", use_prompt=False)
        d2 = out_rgb["depth"][0].float().cpu().numpy(); d2 = np.where(np.isfinite(d2), d2, 0)
        pred2 = resize(d2, h, w, cv2.INTER_AREA)
        acc["rgb_ls_all"].append(_absrel(pred2, gt, gtv & (pred2 > 0)))
        if teacher is not None:
            with torch.autocast("cuda", dtype=torch.float16, enabled=use_fp16):
                ts = teacher.forward(img, num_tokens=num_tokens, refine_steps=0)
                ss = model.forward(img, num_tokens=num_tokens, refine_steps=0)
                # null prompt (all invalid) through the stem vs the skip path
                sn = model.forward(img, num_tokens=num_tokens, refine_steps=0, lidar_depth=torch.zeros_like(ld), lidar_conf=torch.zeros_like(cf))
            zt = ts["points"][0, ..., 2].float().cpu().numpy(); zs = ss["points"][0, ..., 2].float().cpu().numpy(); zn = sn["points"][0, ..., 2].float().cpu().numpy()
            a_, b_ = fit_scale_shift_l2(zs.ravel(), zt.ravel())
            acc["conform_absrel"].append(float(np.mean(np.abs(a_ * zs + b_ - zt) / np.maximum(zt, 1e-3))))
            a2, b2 = fit_scale_shift_l2(zn.ravel(), zs.ravel())
            acc["null_vs_skip"].append(float(np.mean(np.abs(a2 * zn + b2 - zs) / np.maximum(zs, 1e-3))))
            nt, ns = ts["normal"][0].float(), ss["normal"][0].float()
            acc["conform_normal_deg"].append(float(torch.rad2deg(torch.acos((nt * ns).sum(-1).clamp(-1, 1))).mean()))
    res = {k: (float(np.nanmean(v)) if v else None) for k, v in acc.items()}
    res["ref_promptda"] = PROMPTDA_SUBSET
    res["gap_vs_promptda"] = {k: round(res[k] - PROMPTDA_SUBSET[k], 4) for k in PROMPTDA_SUBSET if res.get(k) is not None}
    return res


def set_prompt_trust(model, tau: float):
    """Scale every zero-init injection gate by `tau`, in place.

    The prompt reaches the network only through per-channel gates (`prompt_stem.gates`, `prompt_neck.gates`),
    so scaling them interpolates continuously between the trained prompted model (tau=1) and the RGB-only
    model (tau=0) with no retraining. The metric attachment is untouched: scale still comes from the sensor,
    only the *shape* information does not. This is the deployment knob for a device whose sensor is worse
    than the one the model was trained on.
    """
    import torch as _t
    with _t.no_grad():
        # Absolute, not cumulative. The first call snapshots the trained gates on the model; every call then
        # sets g = g_trained * tau. The earlier in-place `g.mul_(tau)` compounded across calls, so sweeping
        # tau silently multiplied the factors together: tau=1 after tau=0 left the gates at zero, and every
        # later measurement was the RGB-only model wearing the wrong label.
        for mod in ("prompt_stem", "prompt_neck"):
            m = getattr(model, mod, None)
            if m is None:
                continue
            gates = list(m.gates.values() if hasattr(m.gates, "values") else m.gates)
            key = f"_trust_orig_{mod}"
            if not hasattr(model, key):
                setattr(model, key, [g.detach().clone() for g in gates])
            for g, g0 in zip(gates, getattr(model, key)):
                g.copy_(g0 * tau)
    return model


def load_model(ckpt: str | None, base: str | None = None):
    """Base MoGe-3 weights (`base`, default CKPT) + the trainable overlay stored in `ckpt`."""
    from moge.model.v3 import MoGeModel
    base = base or CKPT
    if ckpt is None:
        return MoGeModel.from_pretrained(base).cuda().eval()
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    # A checkpoint may carry architecture overrides (Model B: a 4-level neck/heads and an x8 refiner);
    # they have to be applied at construction, exactly as they were during training.
    mk = dict(ck.get("model_kwargs") or {})
    mk["lidar_prompt"] = ck["lidar_prompt"]
    model = MoGeModel.from_pretrained(base, model_kwargs=mk).cuda()
    model.load_trainable_state(ck["state_dict"])
    return model.eval()


def _warp_depth(prev_depth: np.ndarray, K_prev: np.ndarray, K_cur: np.ndarray, M: np.ndarray, hw) -> np.ndarray:
    """Back-project the previous frame's depth, move it with M (cam_prev -> cam_cur), z-buffer it onto the current grid,
    nearest-fill the holes. prev_depth and the output are at resolution hw (K scaled accordingly)."""
    h, w = hw
    u, v = np.meshgrid(np.arange(w), np.arange(h)); z = prev_depth; ok = z > 0
    P = np.stack([(u - K_prev[0, 2]) / K_prev[0, 0] * z, (v - K_prev[1, 2]) / K_prev[1, 1] * z, z], -1)[ok]
    Q = (M[:3, :3] @ P.T).T + M[:3, 3]; zq = Q[:, 2]; inb = zq > 0.05
    uu = np.round(K_cur[0, 0] * Q[inb, 0] / zq[inb] + K_cur[0, 2]).astype(int); vv = np.round(K_cur[1, 1] * Q[inb, 1] / zq[inb] + K_cur[1, 2]).astype(int)
    m = (uu >= 0) & (uu < w) & (vv >= 0) & (vv < h)
    out = np.full((h, w), np.inf, np.float32)
    idx = vv[m] * w + uu[m]; order = np.argsort(-zq[inb][m])           # far first, near overwrites (z-buffer)
    out.ravel()[idx[order]] = zq[inb][m][order]
    out[~np.isfinite(out)] = 0
    for _ in range(12):                                                  # nearest-ish hole fill
        if (out > 0).all():
            break
        filled = cv2.dilate(out, np.ones((3, 3), np.uint8)); out = np.where(out > 0, out, filled)
    return out


def run_model(a):
    model = load_model(a.ckpt)
    # `run_model` may be called with a caller-built Namespace that does not carry every option:
    # read the newer ones defensively so a missing one is the default, not an AttributeError.
    _trust = getattr(a, "prompt_trust", 1.0)
    if _trust != 1.0:
        set_prompt_trust(model, _trust)
        print(f"prompt trust tau = {_trust}", flush=True)
    if getattr(a, "no_refiner_residual", False) and getattr(model.refiner, "prompt_channels", 0) > 0:
        with torch.no_grad():
            model.refiner.prompt_proj.weight.zero_(); model.refiner.prompt_proj.bias.zero_()
        print("refiner LiDAR residual disabled", flush=True)
    if a.mono_from != "rgb":
        from promptmoge.train.data import load_traj, pose_for, intrinsics_at
        trajs = {}
    lut = json.load(open(a.calib_lut)) if a.calib_lut else None
    if a.output_gate is not None:
        model.refiner_output_gate = bool(a.output_gate)
        if not hasattr(model, "output_gate_floor"):
            model.output_gate_floor = 0.1
    writer = AsyncWriter()
    t0, n = time.time(), 0
    fits = {}
    prev = None                                                          # (scene, depth [192,256] metres, pose w2c, K)
    for fr in iter_frames(a.scenes, a.stride, a.limit):
        img, ld, cf = frame_tensors(fr)
        mono_override = None
        if a.mono_from != "rgb":
            if fr.scene not in trajs:
                trajs[fr.scene] = load_traj(fr.scene)
            T, _ = pose_for(*trajs[fr.scene], float(fr.ts)); K = intrinsics_at(fr, (192, 256))
            if prev is not None and prev[0] == fr.scene and T is not None and prev[2] is not None:
                if a.mono_from == "prev_warp":
                    mono = _warp_depth(prev[1], prev[3], K, T @ np.linalg.inv(prev[2]), (192, 256))
                else:
                    mono = prev[1]
                if (mono > 0).mean() > 0.5:
                    mono_override = torch.from_numpy(np.where(mono > 0, mono, np.median(mono[mono > 0])).astype(np.float32))[None, None].cuda()
            # first frame of a scene (or a missing pose): one RGB-only pass, as a cold start would on device
        fov_x = None
        if a.fov_from_intrinsics:
            K = fr.intrinsics() or WIDE_K
            fov_x = float(np.degrees(2 * np.arctan(K[1][0] / (2 * K[0][0, 0]))))
        with torch.no_grad():
            out = model.infer(img, num_tokens=a.num_tokens, refine_steps=a.refine_steps, return_per_step=True, use_fp16=not a.fp32,
                              lidar_depth=None if a.no_lidar else ld, lidar_conf=None if a.no_lidar else cf,
                              metric_from=a.metric_from, use_prompt=not a.no_prompt, anchor_strength=a.anchor, fov_x=fov_x,
                              use_learned_conf=a.learned_conf, calibration_lut=lut, calibrate_input=a.calib_input, gauge_poly=a.gauge_poly, mono_z_override=mono_override)
        if a.mono_from != "rgb":
            dfin = out["depth"][0].float().cpu().numpy(); dfin = np.where(np.isfinite(dfin), dfin, 0).astype(np.float32)
            prev = (fr.scene, cv2.resize(dfin, (256, 192), interpolation=cv2.INTER_AREA), T, K)
        for k, d in enumerate(out["depth_per_step"]):
            writer.save(PRED / f"{a.tag}_s{k}" / fr.scene / f"{fr.ts}.png", d[0].float().cpu().numpy())
        if a.save_err and out.get("model_err") is not None:
            e = out["model_err"][0].float().cpu().numpy(); p = PRED / f"{a.tag}_err" / fr.scene / f"{fr.ts}.png"; p.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(p), np.clip(e * 10000, 0, 65535).astype(np.uint16))        # relative error x 1e4
        if a.save_normals and "normal" in out:
            nrm = out["normal"][0].float().cpu().numpy()                      # [H,W,3] unit normals (0 where masked)
            p = PRED / f"{a.tag}_normal" / fr.scene / f"{fr.ts}.npz"; p.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(p, n=(nrm[::2, ::2] * 127).astype(np.int8))   # half-res int8, ~1/8 the size of float
        if "metric_fit" in out:
            fits[fr.key] = out["metric_fit"][0].tolist()
        n += 1
        if n % 100 == 0:
            print(f"{a.tag} {n} frames {(time.time()-t0)/n:.3f}s/frame", flush=True)
    writer.close()
    (PRED / f"{a.tag}_meta.json").write_text(json.dumps({"n": n, "s_per_frame": (time.time() - t0) / max(n, 1), "fits": fits, "args": vars(a)}))
    print(json.dumps({"tag": a.tag, "n": n, "s_per_frame": (time.time() - t0) / max(n, 1)}))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None, help="trained checkpoint (default: the unprompted base model)")
    ap.add_argument("--base", default=CKPT, help="base MoGe-3 ViT-L weights (default: $MOGE3_WEIGHTS or checkpoints/moge-3-vitl/model.pt)")
    ap.add_argument("--pred_root", default=str(PRED), help="predictions are written to <pred_root>/<tag>_s<k>")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--metric_from", default="lidar_ls")
    ap.add_argument("--refine_steps", type=int, default=3)
    ap.add_argument("--num_tokens", type=int, default=1200)
    ap.add_argument("--anchor", type=float, default=None)
    ap.add_argument("--output_gate", type=int, default=None, help="override the refiner output gating at inference (1/0)")
    ap.add_argument("--no_lidar", action="store_true")
    ap.add_argument("--no_prompt", action="store_true")
    ap.add_argument("--fp32", action="store_true")
    ap.add_argument("--calib_lut", default=None, help="apply this calibration LUT json inside infer() (deployed end result)")
    ap.add_argument("--calib_input", action="store_true", help="apply the LUT to the LiDAR input instead of the output")
    ap.add_argument("--gauge_poly", type=int, default=0, help="per-frame polynomial prompt gauge at inference (2 = quadratic)")
    ap.add_argument("--prompt_shift", type=int, nargs=2, default=[0, 0], help="dx dy (sensor px): shift the LiDAR prompt relative to RGB (robustness test)")
    ap.add_argument("--sim_prompt", default=None, help="replace the ARKit prompt by a single-pulse draw of the physical 112-beam lattice densified by this v7 checkpoint (robustness: how much the model leans on the prompt)")
    ap.add_argument("--sim_seed", type=int, default=0)
    ap.add_argument("--prompt_trust", type=float, default=1.0, help="scale the prompt injection gates (1 = as trained, 0 = RGB-only geometry with the sensor still setting metric scale)")
    ap.add_argument("--no_refiner_residual", action="store_true", help="zero the refiner's LiDAR residual channel at inference: the second path by which the sensor reaches the geometry")
    ap.add_argument("--sim_gain", type=float, default=1.0, help="multiplier on the simulated sensor's noise amplitude")
    ap.add_argument("--learned_conf", action="store_true", help="use the model's predicted sensor error as the refiner/anchor confidence")
    ap.add_argument("--save_normals", action="store_true", help="also save the normal head output (half-res int8 npz) for the head-normal metric")
    ap.add_argument("--save_err", action="store_true", help="also save the model-error head output (uint16 png, relative error x 1e4) for uncertainty-weighted fusion")
    ap.add_argument("--fov_from_intrinsics", action="store_true", help="fix the horizontal FoV from the ARKit wide intrinsics instead of estimating it")
    ap.add_argument("--mono_from", default="rgb", choices=["rgb", "prev", "prev_warp"], help="mono prompt channel: fresh RGB-only pass | previous frame's output | previous output warped by the pose (streaming, no second backbone pass)")
    ap.add_argument("--scenes", nargs="*", default=SCENES)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    _a = ap.parse_args(); PROMPT_SHIFT = tuple(_a.prompt_shift); CKPT = _a.base; PRED = Path(_a.pred_root)
    if _a.sim_prompt:
        from promptmoge.train.sensor_sim.v7 import load_densifier
        SIM_PROMPT["model"] = load_densifier(_a.sim_prompt)
        SIM_PROMPT["seed"] = _a.sim_seed; SIM_PROMPT["gain"] = _a.sim_gain
        print(f"hard prompt from {_a.sim_prompt}: grid {getattr(SIM_PROMPT['model'], 'grid', None)}, "
              f"lattice {getattr(SIM_PROMPT['model'], 'lattice', None)}", flush=True)
    run_model(_a)
