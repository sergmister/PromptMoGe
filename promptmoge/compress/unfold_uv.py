"""Inverse of the UV fold: merge `refiner_uv_proj` back into `encoder_fuse` (exact).

Needed only to build a *reference* refiner for compression distillation: the teacher is run on the full
[tokens | uv] conditioning map, so it must be the 1026-wide form.
"""
import argparse, torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    sd = dict(ck["state_dict"]); cfg = dict(ck["lidar_prompt"])
    assert cfg.get("refiner_uv_fold"), "checkpoint is not UV-folded"
    W = sd.pop("refiner.encoder_fuse.weight")
    U = sd.pop("refiner_uv_proj.weight").reshape(W.shape[0], 2)
    sd["refiner.encoder_fuse.weight"] = torch.cat([W, U], dim=1).contiguous()
    cfg["refiner_uv_fold"] = False
    cfg.pop("refiner_qat", None)
    ck["state_dict"] = sd; ck["lidar_prompt"] = cfg
    torch.save(ck, a.out)
    print(f"wrote {a.out}: encoder_fuse {tuple(W.shape)} + uv -> {tuple(sd['refiner.encoder_fuse.weight'].shape)}")


if __name__ == "__main__":
    main()
