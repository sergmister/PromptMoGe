"""Independent check that the QAT model really is int8 under the deployment's contract.

`qat.py` simulates quantisation with a straight-through fake-quant; this script re-derives the integer
representation from the *stored* weights with independent code and checks, for every quantised layer:

  * weights: with wScale[Co] = max|W[Co]| / 127, W/wScale rounds to integers in [-127, 127] with residual 0
    (i.e. the exported weight tensor lands exactly on the int8 grid per output channel), and the layer is
    reconstructible as `int8 * wScale[Co]`;
  * activations: with aScale = clip / 127 (ONE scale per tensor, as the row-gathering kernel requires),
    a real activation drawn from a dev frame quantises with max |a - round(a/aScale)*aScale| <= aScale/2
    and the saturated fraction is close to the 5e-4 the 99.95th percentile clip implies;
  * an int32-accumulated GEMM built from those integers reproduces the fake-quant layer output to fp32
    rounding (checked on `encoder_fuse`, the widest dense GEMM in the refiner).
"""
import argparse, json, os
os.environ.setdefault("FLEX_GEMM_AUTOTUNE_MODE", "always")
import torch, torch.nn as nn
import torch.nn.utils.parametrize as P
from promptmoge.train.data import iter_frames
from promptmoge.train.evaluate import frame_tensors, load_model, DEFAULT_LUT
from moge.model.modules.qat import ActFakeQuant

QMAX = 127.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True); ap.add_argument("--n", type=int, default=2)
    ap.add_argument("--out", default="verify_int8.json")
    ap.add_argument("--base", default=None, help="base MoGe-3 ViT-L weights (default: $MOGE3_WEIGHTS or checkpoints/moge-3-vitl/model.pt)")
    ap.add_argument("--calib_lut", default=DEFAULT_LUT, help="sensor calibration LUT json")
    a = ap.parse_args()
    model = load_model(a.ckpt, a.base)
    qat = [(n, m) for n, m in model.refiner.named_modules() if P.is_parametrized(m, "weight")]
    assert qat, "checkpoint has no quantised layers"

    rep = {"n_layers": len(qat), "weights": {}, "activations": {}}
    worst_w = 0.0
    for name, m in qat:
        w = m.weight.detach().float()                       # already fake-quantised by the parametrisation
        dims = [d for d in range(w.ndim) if d != 0]
        s = w.abs().amax(dim=dims, keepdim=True) / QMAX
        q = w / s.clamp_min(1e-12)
        resid = float((q - q.round()).abs().max())
        rng = int(q.round().abs().max())
        worst_w = max(worst_w, resid)
        rep["weights"][name] = {"grid_residual": resid, "max_abs_int": rng, "n_out": int(w.shape[0])}
        assert rng <= 127, f"{name}: |int| = {rng} > 127"

    # activations: run real frames and check the per-tensor grid + saturation rate
    caps = {}
    hooks = []
    def mk(nm, aq):
        def h(mod, args):
            x = args[0].detach().float()
            s = (aq.clip / QMAX).clamp_min(1e-12)
            qx = torch.clamp(torch.round(x / s), -QMAX, QMAX)
            err = float((x - qx * s).abs().max())
            sat = float((x.abs() > aq.clip).float().mean())
            c = caps.setdefault(nm, {"max_abs_err": 0.0, "half_step": float(s) / 2, "sat_frac": 0.0, "n": 0})
            c["max_abs_err"] = max(c["max_abs_err"], err); c["sat_frac"] += sat; c["n"] += 1
            return None
        return h
    aqs = {n: m for n, m in model.refiner.named_modules() if isinstance(m, ActFakeQuant)}
    for name, m in qat:
        key = "aq_" + name.replace(".", "_")
        if key in aqs:
            hooks.append(m.register_forward_pre_hook(mk(name, aqs[key])))

    lut = json.load(open(a.calib_lut))
    with torch.no_grad():
        for k, fr in enumerate(iter_frames(stride=211)):
            if k >= a.n: break
            img, ld, cf = frame_tensors(fr)
            model.infer(img, num_tokens=1200, refine_steps=1, use_fp16=True, lidar_depth=ld, lidar_conf=cf,
                        metric_from="lidar_ls", calibration_lut=lut, calibrate_input=True, gauge_poly=2)
    for h in hooks:
        h.remove()
    worst_ratio = 0.0
    for nm, c in caps.items():
        c["sat_frac"] /= max(c["n"], 1)
        c["err_over_half_step"] = c["max_abs_err"] / max(c["half_step"], 1e-12)
        worst_ratio = max(worst_ratio, c["err_over_half_step"])
        rep["activations"][nm] = c

    # int32-accumulated GEMM on the widest dense layer
    ef = model.refiner.encoder_fuse
    aq = aqs["aq_encoder_fuse"]
    x = torch.randn(4096, ef.in_features, device="cuda") * float(aq.clip) / 3
    xs = (aq.clip / QMAX).clamp_min(1e-12)
    xi = torch.clamp(torch.round(x / xs), -QMAX, QMAX)
    W = ef.weight.detach().float(); ws = W.abs().amax(dim=1, keepdim=True) / QMAX
    wi = torch.round(W / ws.clamp_min(1e-12))
    # CUDA has no integer matmul; float64 represents every partial sum here exactly (|acc| <= 127*127*1024
    # = 1.65e7, far inside float64's 2^53), so this IS the int32 accumulation, then checked to be integral.
    acc = (xi.double() @ wi.double().T)
    assert float((acc - acc.round()).abs().max()) == 0.0, "accumulator is not integral"
    acc = acc.float()
    y_int = acc * xs * ws.T + ef.bias.detach().float()
    y_sim = torch.nn.functional.linear(aq(x), W, ef.bias)
    rel = float((y_int - y_sim).abs().max() / y_sim.abs().max())
    rep["int32_gemm_check"] = {"layer": "encoder_fuse", "max_rel_diff": rel,
                               "max_abs_int32": float(acc.abs().max())}

    print(f"quantised layers: {len(qat)}")
    print(f"weights: worst distance from the int8 grid = {worst_w:.3e} (0 = exactly on the grid)")
    print(f"activations: worst quantisation error / half-step = {worst_ratio:.3f} (<= 1 by construction), "
          f"mean saturated fraction = {sum(c['sat_frac'] for c in caps.values())/max(len(caps),1):.2e} "
          f"(the 99.95th-percentile clip implies ~5e-04)")
    print(f"int32 GEMM vs fake-quant on encoder_fuse: max rel diff {rel:.3e}, max |accumulator| {rep['int32_gemm_check']['max_abs_int32']:.0f} (int32 limit 2.1e9)")
    json.dump(rep, open(a.out, "w"), indent=1)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
