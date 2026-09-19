"""Assemble a deployable checkpoint from a trained compression run and report exactly what it contains.

The transforms, in order:
  1. PromptNeck re-scheduled and distilled from the stock pyramid   (already in the run's init)
  2. refiner level 4 narrowed 512 -> 256                             (already in the run's init, fine-tuned)
  3. UV folded out of encoder_fuse                                   (already in the run's init)
  4. int8 QAT of the refiner                                         (already in the run, if enabled)
  5. exact rescaling of the neck + heads                             (applied here, last, because the
     activations it measures are the trained ones)
plus a deployment flag (`deploy_drop_conf_head`): the conf head is computed and discarded by
`use_learned_conf=False`, so the exported program should not contain it.

    python -m promptmoge.compress.make_deliverable --arm runs/model_a_qat/final.pt --out model_a_int8.pt
"""
import argparse, json, subprocess, sys, torch

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, help="final checkpoint of the trained run"); ap.add_argument("--out", required=True)
    ap.add_argument("--target_max", type=float, default=8.0)
    ap.add_argument("--drop_conf_head", type=int, default=1)
    ap.add_argument("--base", default=None, help="base MoGe-3 ViT-L weights (default: $MOGE3_WEIGHTS or checkpoints/moge-3-vitl/model.pt)")
    a = ap.parse_args()
    cmd = [sys.executable, "-m", "promptmoge.compress.rescale_tail", "--ckpt", a.arm, "--out", a.out, "--target_max", str(a.target_max)]
    rc = subprocess.run(cmd + (["--base", a.base] if a.base else [])).returncode
    if rc != 0:
        raise SystemExit(f"rescale_tail failed ({rc})")
    ck = torch.load(a.out, map_location="cpu", weights_only=False)
    cfg = ck["lidar_prompt"]
    cfg["deploy_drop_conf_head"] = bool(a.drop_conf_head)
    ck["lidar_prompt"] = cfg
    ck.setdefault("compress", {})["summary"] = {
        "neck_hidden": cfg.get("neck_hidden"),
        "refiner_channels": cfg.get("refiner_channels"),
        "refiner_qat": bool(cfg.get("refiner_qat", False)),
        "tail_rescale": ck.get("compress", {}).get("tail_rescale", {}).get("alphas"),
        "refiner_uv_fold": bool(cfg.get("refiner_uv_fold", False)),
        "drop_conf_head": bool(a.drop_conf_head),
    }
    torch.save(ck, a.out)
    print("\nDELIVERABLE", a.out)
    print(json.dumps(ck["compress"]["summary"], indent=1))
    print("tensors:", len(ck["state_dict"]), " step:", ck.get("step"))

if __name__ == "__main__":
    main()
