"""Build the compressed refiner variant of a checkpoint: narrowed level 4 + UV fold (+ the int8 QAT flag).

The narrowing halves the coarsest refiner level (512 -> 256). Level 4's channel identity is preserved across
`downsample_blocks[3] -> down_stages[4] -> fuse_proj -> bottleneck -> upsample_blocks[0]`, so the cut is a
structured prune of one channel set (A); `encoder_fuse`'s output is a second, independent set (B) that only
feeds the second half of `fuse_proj[0]`. Channels are ranked by how much they reach the decoder
(||upsample_blocks[0].linear.weight[:, c]||) resp. how much they reach the fusion (||fuse_proj[0].weight[:, 512+c]||),
so the surviving subnetwork is the strongest one available before fine-tuning.

The UV fold splits `encoder_fuse` (1026 -> C) into a 1024-wide GEMM plus a 1x1 projection of the two UV planes,
which is exactly equal and removes the Neural Engine's 1026 -> 1152 padding.

    python -m promptmoge.compress.make_refiner --ckpt teacher_neck.pt --out init_a.pt --channels 32 64 128 256 256 --uv_fold --qat
"""
import argparse, json, torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--channels", type=int, nargs="*", default=[32, 64, 128, 256, 256])
    ap.add_argument("--uv_fold", action="store_true")
    ap.add_argument("--qat", action="store_true")
    a = ap.parse_args()

    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    sd = dict(ck["state_dict"]); cfg = dict(ck["lidar_prompt"])
    C_old = sd["refiner.encoder_fuse.weight"].shape[0]
    C_new = a.channels[-1]
    n_enc = sd["refiner.encoder_fuse.weight"].shape[1]
    print(f"level-4 width {C_old} -> {C_new}; encoder_channels {n_enc}" + (" -> %d + uv" % (n_enc - 2) if a.uv_fold else ""))

    if C_new != C_old:
        # channel importance
        imp_A = sd["refiner.upsample_blocks.0.linear.weight"].float().pow(2).sum(0).sqrt()      # [C_old]
        imp_B = sd["refiner.fuse_proj.0.weight"].float()[:, C_old:].pow(2).sum(0).sqrt()        # [C_old]
        selA = torch.sort(torch.topk(imp_A, C_new).indices).values
        selB = torch.sort(torch.topk(imp_B, C_new).indices).values
        print(f"kept A: {float(imp_A[selA].sum() / imp_A.sum()):.3f} of the outgoing norm; "
              f"B: {float(imp_B[selB].sum() / imp_B.sum()):.3f}")

        def blk(prefix, sel):
            sd[prefix + ".norm1.weight"] = sd[prefix + ".norm1.weight"][sel]
            sd[prefix + ".norm1.bias"] = sd[prefix + ".norm1.bias"][sel]
            for c in ("conv1", "conv2"):
                w = sd[f"{prefix}.{c}.weight"]                      # (Co, kd, kh, kw, Ci)
                sd[f"{prefix}.{c}.weight"] = w[sel][..., sel].contiguous()
                sd[f"{prefix}.{c}.bias"] = sd[f"{prefix}.{c}.bias"][sel]

        sd["refiner.downsample_blocks.3.linear.weight"] = sd["refiner.downsample_blocks.3.linear.weight"][selA]
        sd["refiner.downsample_blocks.3.linear.bias"] = sd["refiner.downsample_blocks.3.linear.bias"][selA]
        blk("refiner.down_stages.4.0", selA)
        blk("refiner.bottleneck_stage.0", selA)
        sd["refiner.encoder_fuse.weight"] = sd["refiner.encoder_fuse.weight"][selB]
        sd["refiner.encoder_fuse.bias"] = sd["refiner.encoder_fuse.bias"][selB]
        w0 = sd["refiner.fuse_proj.0.weight"]
        sd["refiner.fuse_proj.0.weight"] = torch.cat([w0[selA][:, selA], w0[selA][:, C_old + selB]], dim=1).contiguous()
        sd["refiner.fuse_proj.0.bias"] = sd["refiner.fuse_proj.0.bias"][selA]
        sd["refiner.fuse_proj.2.weight"] = sd["refiner.fuse_proj.2.weight"][selA][:, selA].contiguous()
        sd["refiner.fuse_proj.2.bias"] = sd["refiner.fuse_proj.2.bias"][selA]
        sd["refiner.upsample_blocks.0.linear.weight"] = sd["refiner.upsample_blocks.0.linear.weight"][:, selA].contiguous()
        cfg["refiner_channels"] = list(a.channels)

    if a.uv_fold:
        W = sd["refiner.encoder_fuse.weight"]
        sd["refiner.encoder_fuse.weight"] = W[:, :-2].contiguous()
        sd["refiner_uv_proj.weight"] = W[:, -2:].contiguous().view(W.shape[0], 2, 1, 1)
        cfg["refiner_uv_fold"] = True
    if a.qat:
        cfg["refiner_qat"] = True

    ck["state_dict"] = sd; ck["lidar_prompt"] = cfg
    ck.setdefault("compress", {})["refiner"] = {"channels": list(a.channels), "uv_fold": bool(a.uv_fold), "qat": bool(a.qat)}
    torch.save(ck, a.out)
    print("wrote", a.out, "| lidar_prompt:", json.dumps({k: v for k, v in cfg.items() if k.startswith(("refiner", "neck_h", "qat"))}))


if __name__ == "__main__":
    main()
