"""Warm-start a re-scheduled PromptNeck by distilling the stock one's injections.

Re-scheduling the pyramid's widths changes every tensor's shape, so a checkpoint cannot be transferred and
the module would restart from its zero-gate initialisation. Measured cost of that on the dev subset:
confident AbsRel 0.0121 -> 0.0289, i.e. the neck pyramid, not the ViT token injection, carries most of the
LiDAR conditioning. Instead the new pyramid is fitted directly to the old one's five injection tensors on
real prompt maps (a 2.6 M-parameter regression, minutes), so fine-tuning starts from the behaviour
the teacher already has.

    python -m promptmoge.compress.distill_prompt_neck --ckpt teacher.pt --out teacher_neck.pt --hidden 512 256 128 64 32
"""
import argparse, json, os, random, time
import cv2, numpy as np, torch, torch.nn.functional as F
from pathlib import Path
from moge.model.modules.prompt_stem import PromptNeck, build_prompt

DIMS = [1024, 256, 128, 64, 32]
FINE = (480, 640)
TOKEN = (420, 560)          # build_prompt runs at base*14, the pyramid at base*16


def sample_prompts(n, STAGE, seed=0):
    rng = random.Random(seed)
    vids = sorted(p.name for p in STAGE.iterdir() if p.is_dir() and (p / "conf").is_dir())
    out = []
    while len(out) < n:
        v = rng.choice(vids)
        fs = sorted((STAGE / v / "conf").glob("*.png"))
        if not fs: continue
        f = rng.choice(fs); ts = f.stem
        d = cv2.imread(str(STAGE / v / "lidar" / f"{ts}.png"), cv2.IMREAD_UNCHANGED)
        c = cv2.imread(str(f), cv2.IMREAD_UNCHANGED)
        if d is None or c is None: continue
        out.append((d.astype(np.float32) / 1000.0, c.astype(np.float32)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--hidden", type=int, nargs="*", default=[512, 256, 128, 64, 32])
    ap.add_argument("--levels", type=int, default=5, help="4 = Model B's x8 ladder: fit the old pyramid's levels 0-3 (30x40 ... 240x320) with a 4-level pyramid driven by a 240x320 prompt")
    ap.add_argument("--steps", type=int, default=1500); ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--ref", default=None, help="checkpoint holding the reference (5-level) pyramid; defaults to --ckpt")
    ap.add_argument("--lr", type=float, default=2e-3); ap.add_argument("--pool", type=int, default=3000)
    ap.add_argument("--arkit_root", default="data/arkit_stage", help="real training stage root (the prompt maps are sampled from it)")
    a = ap.parse_args()

    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    sd = ck["state_dict"]; cfg = dict(ck["lidar_prompt"])
    ref = torch.load(a.ref or a.ckpt, map_location="cpu", weights_only=False)["state_dict"]
    in_ch = ref["prompt_neck.enc0.0.weight"].shape[1]
    old = PromptNeck(dims=DIMS, in_channels=in_ch).cuda().eval()
    old.load_state_dict({k[len("prompt_neck."):]: v for k, v in ref.items() if k.startswith("prompt_neck.")})
    for p in old.parameters(): p.requires_grad_(False)
    nl = a.levels
    fine = (FINE[0] // 2 ** (5 - nl), FINE[1] // 2 ** (5 - nl))
    new = PromptNeck(dims=DIMS[:nl], in_channels=in_ch, hidden=tuple(a.hidden)).cuda()
    with torch.no_grad():                                   # start from the trained gate scales
        for gn, go in zip(new.gates, old.gates): gn.copy_(go)
    n_par = sum(p.numel() for p in new.parameters())
    print(f"reference 5-level pyramid @480x640 -> new {nl}-level pyramid @{fine}, hidden {tuple(a.hidden)}; {n_par/1e6:.2f} M params")

    STAGE = Path(a.arkit_root)
    pool = sample_prompts(a.pool, STAGE)
    print(f"{len(pool)} real prompt maps sampled from {STAGE}")
    opt = torch.optim.AdamW([p for p in new.parameters()], lr=a.lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.steps)
    rng = random.Random(1); t0 = time.time()
    for step in range(1, a.steps + 1):
        idx = [rng.randrange(len(pool)) for _ in range(a.batch)]
        d = torch.from_numpy(np.stack([pool[i][0] for i in idx]))[:, None].cuda()
        c = torch.from_numpy(np.stack([pool[i][1] for i in idx]))[:, None].cuda()
        with torch.no_grad():
            pr, _ = build_prompt(d, c, TOKEN)
            tgt = old(F.interpolate(pr, FINE, mode="nearest"))[:nl]
            pf_new = F.interpolate(pr, fine, mode="nearest")
        prd = new(pf_new)
        loss = sum(F.mse_loss(p, t) / (t.detach().pow(2).mean() + 1e-8) for p, t in zip(prd, tgt)) / len(tgt)
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(new.parameters(), 1.0)
        opt.step(); sched.step()
        if step % 250 == 0 or step == 1:
            with torch.no_grad():
                per = [float((p - t).pow(2).mean().sqrt() / (t.pow(2).mean().sqrt() + 1e-12)) for p, t in zip(prd, tgt)]
            print(f"  step {step:5d} nmse {float(loss):.4f} rel-rms/level " + " ".join(f"{x:.3f}" for x in per)
                  + f"  ({time.time()-t0:.0f}s)", flush=True)

    sd = {k: v for k, v in sd.items() if not k.startswith("prompt_neck.")}
    sd.update({f"prompt_neck.{k}": v.detach().cpu() for k, v in new.state_dict().items()})
    cfg["neck_hidden"] = list(a.hidden)
    ck["state_dict"] = sd; ck["lidar_prompt"] = cfg
    ck.setdefault("compress", {})["neck_distill"] = {"hidden": list(a.hidden), "steps": a.steps,
                                                     "final_rel_rms": per, "nmse": float(loss)}
    torch.save(ck, a.out); print("wrote", a.out)


if __name__ == "__main__":
    main()
