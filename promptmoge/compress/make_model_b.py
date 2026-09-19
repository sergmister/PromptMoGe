"""Build **Model B** (240x320 output) from a 480x640 checkpoint.

`point_map = token_grid x prod(downsample_factors)`, so dropping one factor gives x8 and a 240x320 map from
the SAME 30x40 / 1200-token grid. The token count is untouched; only the output-resolution ladder changes.

What transfers exactly (levels 0-3 are the same tensors at the same resolutions in both ladders):
  neck            input_blocks[0..3], resamplers[0..2], res_blocks[0..3]
  each head       input_blocks[0..3], resamplers[0..2], res_blocks[0..3]
  refiner         input_proj, prompt_proj, out_proj, down_stages[0..3], downsample_blocks[0..2],
                  upsample_blocks[1..3] -> [0..2], up_stages[1..3] -> [0..2]
  refiner         bottleneck_stage <- the old up_stages[0] (a 256-channel residual block at the same width)

What is new, and how it is initialised:
  head output     the 5-level head produced its output as Conv1x1(32->3) . Conv3x3(64->32) . Upsample2 applied
                  to the level-3 stream; the 4-level head applies Conv1x1(64->3) to that same stream. The new
                  1x1 is initialised as the composition with the 3x3 collapsed over its taps (exact on a
                  locally constant input, which a bilinearly upsampled feature map nearly is).
  refiner fusion  encoder_fuse is pruned from 512 to 256 outputs by row norm; fuse_proj is re-initialised
                  (its input is now the level-3 feature space, which the old 512-wide tensor does not index).
  prompt pyramid  4 levels, distilled separately (`distill_prompt_neck.py --levels 4`) against the old
                  pyramid's levels 0-3, which already run at 30x40 ... 240x320.

    python -m promptmoge.compress.make_model_b --ckpt teacher.pt --out init_b.pt --uv_fold
"""
import argparse, json, os, torch

BASE = os.environ.get("MOGE3_WEIGHTS", "checkpoints/moge-3-vitl/model.pt")


def four_level(cfg, dim_out_last):
    c = dict(cfg)
    c["dim_in"] = list(cfg["dim_in"])[:4]
    c["dim_res_blocks"] = list(cfg["dim_res_blocks"])[:4]
    c["num_res_blocks"] = list(cfg["num_res_blocks"])[:4]
    c["resamplers"] = list(cfg["resamplers"])[:3]
    c["dim_out"] = [None, None, None, dim_out_last] if dim_out_last is not None else None
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="trained 480x640 checkpoint (the teacher)")
    ap.add_argument("--base", default=BASE, help="base MoGe-3 ViT-L weights (default: $MOGE3_WEIGHTS or checkpoints/moge-3-vitl/model.pt)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--refiner_channels", type=int, nargs="*", default=[32, 64, 128, 256])
    ap.add_argument("--uv_fold", action="store_true")
    a = ap.parse_args()

    base = torch.load(a.base, map_location="cpu", weights_only=True)
    mc = base["model_config"]
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    sd = dict(ck["state_dict"]); cfg = dict(ck["lidar_prompt"])

    mk = {
        "neck": four_level(mc["neck"], None),
        "points_head": four_level(mc["points_head"], 3),
        "normal_head": four_level(mc["normal_head"], 3),
        "mask_head": four_level(mc["mask_head"], 1),
        "refiner": {**mc["refiner"], "model_channels": list(a.refiner_channels),
                    "downsample_factors": [2, 2, 2], "encoder_downsample": 8},
    }
    mk["neck"]["dim_out"] = None

    out = {}
    # ---- neck and the three heads -------------------------------------------------------------
    for mod, n_out in (("neck", None), ("points_head", 3), ("normal_head", 3), ("mask_head", 1)):
        for k, v in sd.items():
            if not k.startswith(mod + "."):
                continue
            rest = k[len(mod) + 1:]
            keep = True
            if rest.startswith("input_blocks.") or rest.startswith("res_blocks."):
                keep = int(rest.split(".")[1]) <= 3
            elif rest.startswith("resamplers."):
                keep = int(rest.split(".")[1]) <= 2
            elif rest.startswith("output_blocks."):
                keep = False                       # rebuilt below
            if keep:
                out[k] = v
        if n_out is not None:
            # compose Conv1x1(32->n) . [Conv3x3(64->32) collapsed over taps] -> Conv1x1(64->n)
            wr = sd[f"{mod}.resamplers.3.1.weight"].float()        # (32, 64, 3, 3)
            br = sd[f"{mod}.resamplers.3.1.bias"].float()          # (32,)
            w1 = sd[f"{mod}.output_blocks.4.weight"].float()[:, :, 0, 0]   # (n, 32)
            b1 = sd[f"{mod}.output_blocks.4.bias"].float()                 # (n,)
            w_eff = wr.sum(dim=(2, 3))                              # (32, 64)
            out[f"{mod}.output_blocks.3.weight"] = (w1 @ w_eff).unsqueeze(-1).unsqueeze(-1).contiguous()
            out[f"{mod}.output_blocks.3.bias"] = (w1 @ br + b1).contiguous()

    # ---- prompt stem is unchanged; the prompt pyramid is rebuilt by the distiller ---------------
    for k, v in sd.items():
        if k.startswith("prompt_stem."):
            out[k] = v

    # ---- refiner -------------------------------------------------------------------------------
    C = a.refiner_channels[-1]
    ref = {k[len("refiner."):]: v for k, v in sd.items() if k.startswith("refiner.")}
    for k in ("input_proj.weight", "input_proj.bias", "prompt_proj.weight", "prompt_proj.bias",
              "out_proj.weight", "out_proj.bias"):
        if k in ref: out["refiner." + k] = ref[k]
    for k, v in ref.items():
        if k.startswith("down_stages.") and int(k.split(".")[1]) <= 3:
            out["refiner." + k] = v
        elif k.startswith("downsample_blocks.") and int(k.split(".")[1]) <= 2:
            out["refiner." + k] = v
        elif k.startswith("upsample_blocks.") and 1 <= int(k.split(".")[1]) <= 3:
            i = int(k.split(".")[1]) - 1
            out["refiner." + ".".join(["upsample_blocks", str(i)] + k.split(".")[2:])] = v
        elif k.startswith("up_stages.") and 1 <= int(k.split(".")[1]) <= 3:
            i = int(k.split(".")[1]) - 1
            out["refiner." + ".".join(["up_stages", str(i)] + k.split(".")[2:])] = v
    # bottleneck <- the old level-3 decoder block (same 256 width, same job: refine at the coarsest level)
    for k, v in ref.items():
        if k.startswith("up_stages.0."):
            out["refiner." + "bottleneck_stage." + k[len("up_stages.0."):]] = v
    # encoder_fuse: prune 512 -> C outputs by row norm
    W = ref["encoder_fuse.weight"].float(); b = ref["encoder_fuse.bias"].float()
    sel = torch.sort(torch.topk(W.pow(2).sum(1).sqrt(), C).indices).values
    W, b = W[sel], b[sel]
    if a.uv_fold:
        out["refiner.encoder_fuse.weight"] = W[:, :-2].contiguous()
        out["refiner_uv_proj.weight"] = W[:, -2:].contiguous().view(C, 2, 1, 1)
        mk["refiner"]["encoder_channels"] = int(mc["refiner"]["encoder_channels"])   # v3 subtracts 2 itself
        cfg["refiner_uv_fold"] = True
    else:
        out["refiner.encoder_fuse.weight"] = W.contiguous()
    out["refiner.encoder_fuse.bias"] = b.contiguous()
    # fuse_proj is genuinely new: initialise it as "pass the level-3 features through, ignore the encoder
    # feature", i.e. [I | 0] then I, so the refiner starts as the transferred UNet with an inert conditioning
    # path instead of a random one.
    I = torch.eye(C)
    out["refiner.fuse_proj.0.weight"] = torch.cat([I, torch.zeros(C, C)], dim=1).contiguous()
    out["refiner.fuse_proj.0.bias"] = torch.zeros(C)
    out["refiner.fuse_proj.2.weight"] = I.contiguous()
    out["refiner.fuse_proj.2.bias"] = torch.zeros(C)

    cfg["neck_hidden"] = None
    cfg["conf_head"] = False          # not deployed, and 18.4 GFLOP per training step
    cfg.pop("refiner_channels", None)
    ck2 = {"lidar_prompt": cfg, "model_kwargs": mk, "state_dict": {k: v.contiguous() for k, v in out.items()},
           "step": 0, "args": ck.get("args"), "compress": {"model_b": {"from": a.ckpt, "refiner_channels": list(a.refiner_channels),
                                                                 "uv_fold": bool(a.uv_fold)}}}
    torch.save(ck2, a.out)
    print(f"wrote {a.out}: {len(out)} tensors")
    print("model_kwargs:", json.dumps({k: (v if k != 'refiner' else v) for k, v in mk.items()}, indent=1)[:900])


if __name__ == "__main__":
    main()
