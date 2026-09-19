"""Command line inference: RGB + LiDAR depth in, metric depth (and optionally a point cloud) out.

    python -m promptmoge.infer --model A --image rgb.jpg --depth lidar_mm.png --conf conf.png --out out/
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
import torch

from . import infer, load_model


def read_depth(path: str) -> np.ndarray:
    if path.endswith(".npy"):
        return np.load(path).astype(np.float32)                       # metres
    return cv2.imread(path, cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000.0   # 16-bit PNG, millimetres


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="A", help="L, A, B or a checkpoint path")
    ap.add_argument("--base", default=None, help="MoGe-3 ViT-L weights (default: downloaded from Hugging Face)")
    ap.add_argument("--image", required=True)
    ap.add_argument("--depth", required=True, help="LiDAR depth: 16-bit PNG in millimetres or .npy in metres")
    ap.add_argument("--conf", default=None, help="ARKit confidence map {0,1,2} (PNG or .npy)")
    ap.add_argument("--steps", type=int, default=1, help="refinement steps K")
    ap.add_argument("--gauge_poly", type=int, default=2)
    ap.add_argument("--out", default="out")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()

    model = load_model(a.model, base=a.base, device=a.device)
    image = cv2.cvtColor(cv2.imread(a.image), cv2.COLOR_BGR2RGB)
    conf = None if a.conf is None else (np.load(a.conf) if a.conf.endswith(".npy") else cv2.imread(a.conf, cv2.IMREAD_UNCHANGED))
    out = infer(model, image, read_depth(a.depth), conf, refine_steps=a.steps, gauge_poly=a.gauge_poly)

    dst = Path(a.out); dst.mkdir(parents=True, exist_ok=True)
    depth = out["depth"].float().cpu().numpy()
    np.save(dst / "depth.npy", depth)
    valid = np.isfinite(depth) & (depth > 0)
    cv2.imwrite(str(dst / "depth_mm.png"), np.where(valid, depth * 1000, 0).clip(0, 65535).astype(np.uint16))
    pts = out["points"].float().cpu().numpy()[valid]
    rgb = image[valid]
    with open(dst / "points.ply", "wb") as f:
        f.write(f"ply\nformat binary_little_endian 1.0\nelement vertex {len(pts)}\nproperty float x\nproperty float y\n"
                "property float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n".encode())
        rec = np.empty(len(pts), dtype=[("p", "<f4", 3), ("c", "u1", 3)]); rec["p"] = pts; rec["c"] = rgb
        f.write(rec.tobytes())
    s, t = out["metric_fit"].flatten().tolist()
    print(f"depth {depth.shape}, median {np.median(depth[valid]):.3f} m, metric fit s={s:.4f} t={t:.4f} -> {dst}")


if __name__ == "__main__":
    main()
