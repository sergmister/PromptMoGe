"""Fit Model B's head output projections to reproduce the 5-level heads' output at 240x320.

The 5-level ladder computes its output as

    out = Conv1x1_(32->n) [ Conv3x3_(64->32)( Upsample2(x3) ) + Conv1x1(neck_level4) ]

where `neck_level4` is itself an upsample+3x3 of the neck's level-3 stream. The 4-level ladder has to produce
the same thing from `x3` alone, at half the resolution. No closed-form composition exists (the neck's level-4
branch is a different function of a different tensor), so the projection is **fitted**: capture `x3` and the
5-level output on real frames, area-downsample the output to 240x320, and solve the least-squares problem for
a 3x3 convolution 64 -> n. Levels 0-3 are bit-identical between the two ladders, so `x3` is the same tensor in
both and the fit is exact in the only sense that matters.

    python -m promptmoge.compress.fit_model_b_heads --ref teacher.pt --ckpt init_b_neck.pt --out init_b_fit.pt
"""
import argparse, json, os
os.environ.setdefault("FLEX_GEMM_AUTOTUNE_MODE", "always")
import numpy as np, torch, torch.nn.functional as F
from promptmoge.train.data import iter_frames
from promptmoge.train.evaluate import frame_tensors, load_model, DEFAULT_LUT

HEADS = ["points_head", "normal_head", "mask_head"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True, help="the 5-level (480x640) reference checkpoint")
    ap.add_argument("--ckpt", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--frames", type=int, default=40); ap.add_argument("--px", type=int, default=30000)
    ap.add_argument("--ridge", type=float, default=1e-4)
    ap.add_argument("--base", default=None, help="base MoGe-3 ViT-L weights (default: $MOGE3_WEIGHTS or checkpoints/moge-3-vitl/model.pt)")
    ap.add_argument("--calib_lut", default=DEFAULT_LUT, help="sensor calibration LUT json")
    a = ap.parse_args()

    LUT = json.load(open(a.calib_lut))
    ref = load_model(a.ref, a.base)
    cap = {}
    for h in HEADS:
        mod = getattr(ref, h)
        def mk_in(name):
            def f(m, inp):
                cap[name + "_x3"] = inp[0].detach().float()
            return f
        def mk_out(name):
            def f(m, inp, out):
                cap[name + "_y"] = out.detach().float()
            return f
        mod.resamplers[3].register_forward_pre_hook(mk_in(h))
        mod.output_blocks[4].register_forward_hook(mk_out(h))

    K = 3
    G = {h: None for h in HEADS}; B = {h: None for h in HEADS}
    n_used = 0
    with torch.no_grad():
        for k, fr in enumerate(iter_frames(stride=max(1, 1094 // a.frames))):
            if k >= a.frames: break
            img, ld, cf = frame_tensors(fr)
            ref.infer(img, num_tokens=1200, refine_steps=0, use_fp16=False, lidar_depth=ld, lidar_conf=cf,
                      metric_from="lidar_ls", calibration_lut=LUT, calibrate_input=True, gauge_poly=0)
            for h in HEADS:
                x3 = cap[h + "_x3"]                                   # [1, 64, 240, 320]
                y = cap[h + "_y"]                                     # [1, n, 480, 640]
                y = F.interpolate(y, x3.shape[-2:], mode="area")      # [1, n, 240, 320]
                A = F.unfold(F.pad(x3, (1, 1, 1, 1), mode="replicate"), K)   # [1, 64*9, HW]
                A = A[0].T                                            # [HW, 576]
                A = torch.cat([A, torch.ones(A.shape[0], 1, device=A.device)], 1)
                Y = y[0].flatten(1).T                                 # [HW, n]
                idx = torch.randperm(A.shape[0], device=A.device)[: a.px]
                A, Y = A[idx].double(), Y[idx].double()
                G[h] = (G[h] + A.T @ A) if G[h] is not None else A.T @ A
                B[h] = (B[h] + A.T @ Y) if B[h] is not None else A.T @ Y
            n_used += 1
    print(f"accumulated {n_used} frames x {a.px} px")

    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    sd = dict(ck["state_dict"]); mk = dict(ck["model_kwargs"])
    for h in HEADS:
        n_out = mk[h]["dim_out"][-1]
        Gm = G[h] + a.ridge * torch.eye(G[h].shape[0], device=G[h].device, dtype=G[h].dtype) * torch.diagonal(G[h]).mean()
        sol = torch.linalg.solve(Gm, B[h])                            # [577, n]
        W = sol[:-1].T.reshape(n_out, 64, K, K).float().cpu()         # unfold order is (C, kh, kw)
        b = sol[-1].float().cpu()
        # residual of the fit, relative to the target's own RMS
        res = float(((G[h] @ sol - B[h]) ** 2).sum())
        sd[f"{h}.output_blocks.3.weight"] = W.contiguous()
        sd[f"{h}.output_blocks.3.bias"] = b.contiguous()
        mk[h]["output_kernel"] = K
        print(f"{h}: fitted {tuple(W.shape)} + bias, normal-equation residual {res:.3e}")
    ck["state_dict"] = sd; ck["model_kwargs"] = mk
    ck.setdefault("compress", {})["head_fit"] = {"frames": n_used, "px": a.px, "kernel": K}
    torch.save(ck, a.out); print("wrote", a.out)


if __name__ == "__main__":
    main()
