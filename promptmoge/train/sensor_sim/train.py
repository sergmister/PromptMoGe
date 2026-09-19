"""Train the ARKit sensor simulator on the staged training videos (real ARKit depth + conf vs Faro GT).
Loss: Gaussian NLL of log(arkit/gt) on pixels with valid GT (all confidences -- the sensor is dense), CE on confidence.
Holdout: last 10 % of videos. Saves <out_dir>/model.pt + a realism report (real vs simulated statistics)."""
import argparse, json, random, time
from pathlib import Path
import numpy as np, cv2, torch, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from promptmoge.train.train_data import _list_frames, STAGE
from promptmoge.train.sensor_sim.model import SensorUNet, make_inputs, simulate, mixture_nll
cv2.setNumThreads(1)


class SensorPairs(Dataset):
    def __init__(self, frames, flip=True, root=None):
        self.frames = frames; self.flip = flip; self.root = Path(root) if root else STAGE
    def __len__(self): return len(self.frames)
    def __getitem__(self, i):
        for _ in range(8):
            try:
                return self._load(i)
            except Exception:
                i = random.randrange(len(self.frames))
        raise RuntimeError("SensorPairs: repeated read failures")

    def _load(self, i):
        vid, ts = self.frames[i]; d = self.root / vid
        gt = cv2.imread(str(d / "gt" / f"{ts}.png"), cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000
        lr = cv2.imread(str(d / "lidar" / f"{ts}.png"), cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000
        cf = cv2.imread(str(d / "conf" / f"{ts}.png"), cv2.IMREAD_UNCHANGED).astype(np.int64)
        rgb = cv2.cvtColor(cv2.imread(str(d / "rgb" / f"{ts}.jpg"), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        assert gt.shape == (1008, 1344) and lr.shape == (192, 256) and cf.shape == (192, 256)
        g = cv2.resize(gt, (256, 192), interpolation=cv2.INTER_NEAREST)
        im = cv2.resize(rgb, (256, 192), interpolation=cv2.INTER_AREA).astype(np.float32) / 255
        if self.flip and random.random() < 0.5:
            g, lr, cf, im = g[:, ::-1], lr[:, ::-1], cf[:, ::-1], im[:, ::-1]
        return (torch.from_numpy(np.ascontiguousarray(g))[None], torch.from_numpy(np.ascontiguousarray(im)).permute(2, 0, 1),
                torch.from_numpy(np.ascontiguousarray(lr))[None], torch.from_numpy(np.ascontiguousarray(cf)))


def realism(model, dl, device, n_batches=40):
    """Compare real vs simulated: AbsRel vs GT by depth bin, P(conf), conf_wrong rate, edge far-bias."""
    model.eval(); bins = [0.25, 0.5, 1, 2, 3, 5]
    acc = {k: {"real": [], "sim": []} for k in ["absrel_by_bin", "P_conf", "conf_wrong", "edge_bias"]}
    with torch.no_grad():
        for bi, (gt, rgb, lr, cf) in enumerate(dl):
            if bi >= n_batches: break
            gt, rgb, lr, cf = gt.to(device), rgb.to(device), lr.to(device), cf.to(device)
            sd, sc, _ = simulate(model, gt, rgb)
            for name, dep, con in (("real", lr, cf.float().unsqueeze(1)), ("sim", sd, sc)):
                v = (gt > 0) & (dep > 0)
                rel = ((dep - gt).abs() / gt.clamp_min(1e-3))
                acc["absrel_by_bin"][name].append([rel[v & (gt >= lo) & (gt < hi)].mean().item() if (v & (gt >= lo) & (gt < hi)).any() else float('nan') for lo, hi in zip(bins[:-1], bins[1:])])
                acc["P_conf"][name].append([(con == k).float().mean().item() for k in range(3)])
                c2 = v & (con == 2); acc["conf_wrong"][name].append(((rel > 0.15) & c2).sum().item() / max(c2.sum().item(), 1))
                lg = torch.log(gt.clamp_min(1e-3)); jump = torch.zeros_like(v)
                jx = ((lg[..., 1:] - lg[..., :-1]).abs() > 0.1) & (gt[..., 1:] > 0) & (gt[..., :-1] > 0); jump[..., 1:] |= jx; jump[..., :-1] |= jx
                gmax = F.max_pool2d(gt, 5, 1, 2); near = v & jump & (gt < 0.9 * gmax)
                acc["edge_bias"][name].append((torch.log(dep.clamp_min(1e-3)) - lg)[near].median().item() if near.any() else float('nan'))
    model.train()
    return {k: {n: (np.nanmean(np.array(vals), axis=0).tolist() if k in ("absrel_by_bin", "P_conf") else float(np.nanmean(vals))) for n, vals in d.items()} for k, d in acc.items()}


@torch.no_grad()
def rim_table(model, dl, device, n_batches=25):
    """median log(sensor/GT) at pixels adjacent to a >10% depth jump, by side (near: background behind; far: foreground in
    front) and |log(neighbour/this)| bin, real vs simulated -- the rim-bias realism check."""
    model.eval(); agg = {}
    for bi, (gt, rgb, lr, cf) in enumerate(dl):
        if bi >= n_batches: break
        gt, rgb, lr, cf = gt.to(device), rgb.to(device), lr.to(device), cf.to(device)
        sd, sc, _ = simulate(model, gt, rgb)
        lg = torch.log(gt.clamp_min(1e-3))
        gmax = F.max_pool2d(gt, 3, 1, 1); gmin = -F.max_pool2d(-torch.where(gt > 0, gt, torch.full_like(gt, 1e3)), 3, 1, 1)
        jump = ((gmax / gmin.clamp_min(1e-3)) > 1.1) & (gt > 0)
        for name, dep in (("real", lr), ("sim", sd)):
            v = (gt > 0) & (dep > 0) & (gt > 0.3) & (gt < 4)
            for side, m, ref in (("near", jump & (gt < 0.95 * gmax), gmax), ("far", jump & (gt > 1.05 * gmin), gmin)):
                r = torch.log(ref.clamp_min(1e-3)) - lg; d = (torch.log(dep.clamp_min(1e-3)) - lg)
                for lo, hi in ((0.1, 0.3), (0.3, 0.7), (0.7, 3.0)):
                    sel = m & v & (r.abs() >= lo) & (r.abs() < hi)
                    if sel.sum() > 50:
                        agg.setdefault(f"{side}|{lo}-{hi}|{name}", []).append((d[sel].median().item(), int(sel.sum())))
    model.train()
    return {k: float(np.average([x[0] for x in v], weights=[x[1] for x in v])) for k, v in agg.items()}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--steps", type=int, default=6000); ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4); ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--edge_weight", type=float, default=1.0, help="NLL weight on pixels near GT edges (rim-bias emphasis)"); ap.add_argument("--init", default=None)
    ap.add_argument("--laplace", type=int, default=0, help="Laplace (median-fitting, L1) likelihood instead of Gaussian")
    ap.add_argument("--mixed_pixel", type=int, default=0, help="v4: outlier mean = learned fraction towards the neighbouring surface across a jump")
    ap.add_argument("--tag", default="", help="save as model_<tag>.pt / realism_<tag>.json")
    ap.add_argument("--out_dir", default="runs/sensor_sim"); ap.add_argument("--arkit_root", default=None, help="real training stage root (default: data/arkit_stage)")
    a = ap.parse_args()
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    root = Path(a.arkit_root) if a.arkit_root else STAGE
    frames = _list_frames(root); vids = sorted({v for v, _ in frames}); hold = set(vids[int(0.9 * len(vids)):])
    tr = [f for f in frames if f[0] not in hold]; va = [f for f in frames if f[0] in hold]
    print(f"train frames {len(tr)} | holdout frames {len(va)} ({len(hold)} videos)", flush=True)
    dl = DataLoader(SensorPairs(tr, root=root), batch_size=a.batch, shuffle=True, num_workers=a.workers, drop_last=True, persistent_workers=True)
    dv = DataLoader(SensorPairs(va, flip=False, root=root), batch_size=a.batch, shuffle=False, num_workers=2)
    dev = "cuda"; model = SensorUNet(mixed_pixel=int(a.mixed_pixel)).to(dev)
    if a.init: model.load_state_dict(torch.load(a.init, map_location=dev))
    model.laplace = bool(a.laplace)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.steps, pct_start=0.05)
    print(f"params {sum(p.numel() for p in model.parameters())/1e6:.2f} M", flush=True)
    it = iter(dl); t0 = time.time(); log = open(out / "log.jsonl", "a")
    for step in range(1, a.steps + 1):
        try: gt, rgb, lr, cf = next(it)
        except StopIteration: it = iter(dl); gt, rgb, lr, cf = next(it)
        gt, rgb, lr, cf = gt.to(dev), rgb.to(dev), lr.to(dev), cf.to(dev)
        x = make_inputs(gt, rgb); o = model(x); mu, log_sigma, logits = o["mu1"], o["log_b1"], o["conf_logits"]
        v = (gt[:, 0] > 0) & (lr[:, 0] > 0)
        tgt = torch.log(lr[:, 0].clamp_min(1e-3)) - x[:, 0]                 # log(arkit / hole-filled gt)
        nll = mixture_nll(o, tgt)                                            # 2-component Laplace mixture (core + outlier)
        w = v.float()
        if a.edge_weight != 1.0:
            near_edge = F.max_pool2d(x[:, 6:7], 5, 1, 2)[:, 0] > 0      # dilated GT edge mask (input channel 6)
            w = w * (1.0 + (a.edge_weight - 1.0) * near_edge.float())
        l_nll = (w * nll).sum() / w.sum().clamp_min(1)
        l_ce = F.cross_entropy(logits, cf, reduction="none"); l_ce = torch.where(gt[:, 0] > 0, l_ce, torch.zeros_like(l_ce)).sum() / (gt[:, 0] > 0).sum().clamp_min(1)
        loss = l_nll + l_ce
        opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step()
        if step % 100 == 0:
            with torch.no_grad():
                mae = torch.where(v, (tgt - mu).abs(), torch.zeros_like(mu)).sum() / v.sum().clamp_min(1)
                acc = torch.where(gt[:, 0] > 0, (logits.argmax(1) == cf).float(), torch.zeros_like(mu)).sum() / (gt[:, 0] > 0).sum().clamp_min(1)
            rec = {"step": step, "nll": l_nll.item(), "ce": l_ce.item(), "mae_log": mae.item(), "conf_acc": acc.item(), "sigma_mean": log_sigma.exp().mean().item(), "t": round(time.time() - t0)}
            print(json.dumps(rec), flush=True); log.write(json.dumps(rec) + "\n"); log.flush()
        if step % 2000 == 0 or step == a.steps:
            sfx = f"_{a.tag}" if a.tag else ""
            torch.save(model.state_dict(), out / f"model{sfx}.pt")
            rep = realism(model, dv, dev); rep["rim"] = rim_table(model, dv, dev)
            (out / f"realism{sfx}.json").write_text(json.dumps(rep, indent=1)); print("REALISM", json.dumps(rep), flush=True)


if __name__ == "__main__":
    main()
