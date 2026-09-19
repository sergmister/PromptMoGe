"""Shrink the dynamic range of the deployed tail (neck + DPT heads) by an EXACT rescaling.

The neck and the three heads are ConvStacks built with `res_block_in_norm='none'` and
`res_block_hidden_norm='none'`: between the module's input projection and its output projection every
operation is either linear (Conv2d / ConvTranspose2d) or a ReLU. ReLU is positively homogeneous, so if
every bias inside such a block is divided by a, and the block's input projection weight is divided by a,
then EVERY internal activation is divided by a and the block's output projection only has to multiply its
weight by a to restore the original output exactly. Nothing is retrained and nothing is approximated.

That buys headroom on the Apple Neural Engine: its accumulator saturates at +/-32768, and the certified
per-conv bound  A * max_c ||W_c||_1 + |b_c|  scales linearly with a.

    python -m promptmoge.compress.rescale_tail --ckpt model.pt --out model_bounded.pt --target_max 8
"""
import argparse, json, os, collections
os.environ.setdefault("FLEX_GEMM_AUTOTUNE_MODE", "always")
import numpy as np, torch, torch.nn as nn
from promptmoge.train.data import iter_frames
from promptmoge.train.evaluate import frame_tensors, load_model, DEFAULT_LUT

BLOCKS = ["neck", "points_head", "normal_head", "mask_head", "conf_head", "err_head"]


def measure_max(model, n_frames, lut, stride=137):
    """max |activation| over every Conv/ConvTranspose/ReLU output inside each tail block."""
    mx = collections.defaultdict(float)
    handles = []
    def mk(root):
        def h(m, inp, out):
            if torch.is_tensor(out) and out.is_floating_point():
                mx[root] = max(mx[root], float(out.abs().max()))
        return h
    for root in BLOCKS:
        mod = getattr(model, root, None)
        if mod is None: continue
        for nm, sub in mod.named_modules():
            if isinstance(sub, (nn.Conv2d, nn.ConvTranspose2d, nn.ReLU)):
                if nm.startswith("output_blocks"):      # the module's output is not an internal activation
                    continue
                handles.append(sub.register_forward_hook(mk(root)))
    with torch.no_grad():
        for k, fr in enumerate(iter_frames(stride=stride)):
            if k >= n_frames: break
            img, ld, cf = frame_tensors(fr)
            model.infer(img, num_tokens=1200, refine_steps=1, use_fp16=True, lidar_depth=ld, lidar_conf=cf,
                        metric_from="lidar_ls", calibration_lut=lut, calibrate_input=True, gauge_poly=2)
    for h in handles:
        h.remove()
    return dict(mx)


@torch.no_grad()
def rescale(model, alphas: dict):
    """alphas: {block: a}. neck's a also divides the heads' input projections (they read the neck stream)."""
    a_neck = alphas.get("neck", 1.0)
    for root in BLOCKS:
        mod = getattr(model, root, None)
        if mod is None: continue
        a = alphas.get(root, 1.0)
        # input projections: scale the incoming stream to 1/a  (heads additionally undo the neck's 1/a_neck)
        w_in = (a_neck if root != "neck" else 1.0) / a
        for m in mod.input_blocks:
            if isinstance(m, nn.Conv2d):
                m.weight.mul_(w_in)
                if m.bias is not None: m.bias.div_(a)
        # every internal conv keeps its weights and divides its bias
        for nm, m in mod.named_modules():
            if not isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)): continue
            if nm.startswith("input_blocks") or nm.startswith("output_blocks"): continue
            if m.bias is not None: m.bias.div_(a)
        # output projections restore the original scale
        for m in mod.output_blocks:
            if isinstance(m, nn.Conv2d):
                m.weight.mul_(a)
    # The neck is not a closed ConvStack: the LiDAR prompt pyramid is ADDED to its stream at every level, so
    # those injections have to be scaled with it. The injection is `gate * projection(f)`, so dividing the
    # (per-channel, zero-init) gates by a_neck scales it exactly.
    if a_neck != 1.0 and hasattr(model, "prompt_neck"):
        for g in model.prompt_neck.gates:
            g.div_(a_neck)
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--target_max", type=float, default=8.0)
    ap.add_argument("--n_measure", type=int, default=8)
    ap.add_argument("--n_verify", type=int, default=6)
    ap.add_argument("--base", default=None, help="base MoGe-3 ViT-L weights (default: $MOGE3_WEIGHTS or checkpoints/moge-3-vitl/model.pt)")
    ap.add_argument("--calib_lut", default=DEFAULT_LUT, help="sensor calibration LUT json")
    a = ap.parse_args()

    lut = json.load(open(a.calib_lut))
    model = load_model(a.ckpt, a.base)
    before = measure_max(model, a.n_measure, lut)
    alphas = {k: max(v / a.target_max, 1.0) for k, v in before.items()}
    print("measured max |activation| per tail block, and the exact rescaling applied:")
    for k in sorted(before):
        print(f"  {k:<14} max {before[k]:9.2f}  ->  alpha {alphas[k]:8.2f}")

    # golden: outputs before vs after on held-out frames
    ref = []
    with torch.no_grad():
        for k, fr in enumerate(iter_frames(stride=97)):
            if k >= a.n_verify: break
            img, ld, cf = frame_tensors(fr)
            o = model.infer(img, num_tokens=1200, refine_steps=1, use_fp16=True, lidar_depth=ld, lidar_conf=cf,
                            metric_from="lidar_ls", calibration_lut=lut, calibrate_input=True, gauge_poly=2)
            ref.append((o["depth"].float().cpu().clone(), o["normal"].float().cpu().clone(), o["mask"].float().cpu().clone()))

    rescale(model, alphas)
    after = measure_max(model, a.n_measure, lut)
    print("after:", {k: round(v, 3) for k, v in sorted(after.items())})

    rels, nmax, mmax = [], 0.0, 0.0
    with torch.no_grad():
        for k, fr in enumerate(iter_frames(stride=97)):
            if k >= a.n_verify: break
            img, ld, cf = frame_tensors(fr)
            o = model.infer(img, num_tokens=1200, refine_steps=1, use_fp16=True, lidar_depth=ld, lidar_conf=cf,
                            metric_from="lidar_ls", calibration_lut=lut, calibrate_input=True, gauge_poly=2)
            d0, n0, m0 = ref[k]
            d1 = o["depth"].float().cpu(); n1 = o["normal"].float().cpu(); m1 = o["mask"].float().cpu()
            ok = torch.isfinite(d0) & torch.isfinite(d1) & (d0 > 0)
            rels.append(((d1[ok] - d0[ok]).abs() / d0[ok]))
            nmax = max(nmax, float((n1 - n0).abs().max())); mmax = max(mmax, float((m1 - m0).abs().max()))
    r = torch.cat(rels)
    dmean, d99, d999, dmax = [float(x) for x in (r.mean(), r.quantile(0.99), r.quantile(0.999), r.max())]
    print(f"golden vs original, {a.n_verify} frames, {r.numel()} px: rel depth mean {dmean:.2e} p99 {d99:.2e} "
          f"p99.9 {d999:.2e} max {dmax:.2e}; max |dnormal| {nmax:.2e}, max |dmask| {mmax:.2e}")

    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    # a frozen distillation reference may still be in a training checkpoint; it is not part of the deliverable
    ck["state_dict"] = {k: v for k, v in ck["state_dict"].items() if not k.startswith("refiner_teacher")}
    sd = {k: v.detach().cpu() for k, v in model.state_dict().items() if k in ck["state_dict"]}
    missing = [k for k in ck["state_dict"] if k not in sd]
    assert not missing, missing
    ck["state_dict"] = sd
    ck.setdefault("compress", {})["tail_rescale"] = {"alphas": alphas, "target_max": a.target_max,
                                              "max_before": before, "max_after": after,
                                              "golden_rel_depth": {"mean": dmean, "p99": d99, "p999": d999, "max": dmax},
                                              "golden_normal_max": nmax, "golden_mask_max": mmax}
    torch.save(ck, a.out)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
