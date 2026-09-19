"""Certified accumulator bound for every conv in the deployed tail (neck + points/normal/mask/conf/err heads).

The Apple Neural Engine's multiply-accumulate saturates at +/-32768. For a conv with weights W and an input whose
observed maximum magnitude is A, no partial sum can exceed

    bound_c = A * ||W_c||_1 + |b_c|      (maximised over output channels c)

so `max_c bound_c < 32768` is a *sufficient* condition for no saturation, independent of the input.
A is measured on real dev frames. Layers are reported worst-first.

    python -m promptmoge.compress.cert_bound model_a_int8.pt 8 cert_bound.json
"""
import argparse, json, os, collections
os.environ.setdefault("FLEX_GEMM_AUTOTUNE_MODE", "always")
import numpy as np, torch, torch.nn as nn
from promptmoge.train.data import iter_frames
from promptmoge.train.evaluate import frame_tensors, load_model, DEFAULT_LUT

CEIL = 32768.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt"); ap.add_argument("n", type=int, nargs="?", default=8, help="number of dev frames")
    ap.add_argument("out", nargs="?", default="cert_bound.json")
    ap.add_argument("--base", default=None, help="base MoGe-3 ViT-L weights (default: $MOGE3_WEIGHTS or checkpoints/moge-3-vitl/model.pt)")
    ap.add_argument("--calib_lut", default=DEFAULT_LUT, help="sensor calibration LUT json")
    a = ap.parse_args()
    CK, N, OUT = a.ckpt, a.n, a.out
    lut = json.load(open(a.calib_lut))
    model = load_model(CK, a.base)

    inmax = collections.defaultdict(float)
    def hook(name):
        def h(mod, inp, out):
            t = inp[0]
            if torch.is_tensor(t) and t.is_floating_point():
                inmax[name] = max(inmax[name], float(t.abs().max()))
        return h

    convs = {}
    # conf_head / err_head are in the exported graph whenever the checkpoints with confidence heads are shipped
    # (deploy_drop_conf_head = false), so they are certified too.
    for root in ["neck", "points_head", "normal_head", "mask_head", "conf_head", "err_head"]:
        mod = getattr(model, root, None)
        if mod is None: continue
        for nm, sub in mod.named_modules():
            if isinstance(sub, (nn.Conv2d, nn.ConvTranspose2d)):
                key = f"{root}.{nm}"; convs[key] = sub
                sub.register_forward_hook(hook(key))

    with torch.no_grad():
        for k, fr in enumerate(iter_frames(stride=137)):
            if k >= N: break
            img, ld, cf = frame_tensors(fr)
            model.infer(img, num_tokens=1200, refine_steps=1, use_fp16=True, lidar_depth=ld, lidar_conf=cf,
                        metric_from="lidar_ls", calibration_lut=lut, calibrate_input=True, gauge_poly=2)

    res = {}
    for key, m in convs.items():
        W = m.weight.detach().float()
        l1 = W.abs().flatten(1).sum(1) if isinstance(m, nn.Conv2d) else W.abs().transpose(0, 1).flatten(1).sum(1)
        b = m.bias.detach().float().abs() if m.bias is not None else torch.zeros_like(l1)
        A = inmax[key]
        bound = float((A * l1 + b).max())
        res[key] = {"in_max": round(A, 2), "w_l1_max": round(float(l1.max()), 3), "bound": round(bound, 1),
                    "over_ceiling": bound > CEIL}
    top = sorted(res.items(), key=lambda kv: -kv[1]["bound"])
    print(f"{'layer':<50}{'in_max':>9}{'|W|_1':>9}{'bound':>12}  over")
    for k, v in top[:20]:
        print(f"{k:<50}{v['in_max']:>9.1f}{v['w_l1_max']:>9.2f}{v['bound']:>12.0f}  {'YES' if v['over_ceiling'] else ''}")
    n_over = sum(v["over_ceiling"] for v in res.values())
    print(f"\n{n_over} / {len(res)} conv layers exceed the +/-32768 accumulator ceiling; worst bound {top[0][1]['bound']:.0f} ({top[0][0]})")
    json.dump({"ceiling": CEIL, "n_over": n_over, "n_layers": len(res), "layers": res}, open(OUT, "w"), indent=1)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
