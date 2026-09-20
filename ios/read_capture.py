"""Read a capture saved by the demo app, and optionally export it as PNGs and point clouds.

    from read_capture import load
    c = load("captures/20260920_101500_123")
    c["rgb"], c["lidar_depth"], c["lidar_conf"], c["A"]["depth"][1], c["A"]["mask"], c["A"]["points"](1)

    python ios/read_capture.py captures/20260920_101500_123 --export out/
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def load(path):
    d = Path(path)
    meta = json.loads((d / "meta.json").read_text())
    dh, dw = meta["lidar"]
    out = {"meta": meta, "intrinsics": np.array(meta["intrinsics"], np.float32),
           "rgb": cv2.cvtColor(cv2.imread(str(d / "rgb.jpg")), cv2.COLOR_BGR2RGB),
           "lidar_depth": np.fromfile(d / "lidar_depth.bin", np.float32).reshape(dh, dw),      # metres, 0 = no return
           "lidar_conf": np.fromfile(d / "lidar_conf.bin", np.uint8).reshape(dh, dw)}
    for name, m in meta["models"].items():
        h, w = m["height"], m["width"]
        depth = {r["steps"]: np.fromfile(d / r["file"], np.float32).reshape(h, w) for r in m["results"]}   # metres, unmasked
        mask = np.fromfile(d / f"{name}_mask.bin", np.float32).reshape(h, w) > 0
        rays = np.fromfile(d / f"{name}_rays.bin", np.float32).reshape(2, h, w)                # x/z, y/z
        points = lambda k, depth=depth, rays=rays: np.stack([rays[0] * depth[k], rays[1] * depth[k], depth[k]], -1)
        out[name] = {"depth": depth, "mask": mask, "rays": rays, "points": points, "meta": m}
    return out


def save_depth_png(path, depth, valid):
    cv2.imwrite(str(path), np.where(valid, depth * 1000, 0).clip(0, 65535).astype(np.uint16))     # 16-bit millimetres


def save_ply(path, points, colors, valid):
    rec = np.empty(int(valid.sum()), dtype=[("p", "<f4", 3), ("c", "u1", 3)])
    rec["p"], rec["c"] = points[valid], colors[valid]
    with open(path, "wb") as f:
        f.write(f"ply\nformat binary_little_endian 1.0\nelement vertex {len(rec)}\nproperty float x\nproperty float y\n"
                "property float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n".encode())
        f.write(rec.tobytes())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("capture")
    ap.add_argument("--export", default=None, help="write rgb, 16-bit depth PNGs (mm), confidence and .ply point clouds here")
    a = ap.parse_args()
    c = load(a.capture)
    print(f"{c['meta']['date']}  {c['meta']['device']}  image {c['rgb'].shape[1]}x{c['rgb'].shape[0]}  "
          f"LiDAR {c['lidar_depth'].shape[1]}x{c['lidar_depth'].shape[0]}, {(c['lidar_depth'] > 0).mean():.0%} valid")
    for name in c["meta"]["models"]:
        m = c[name]
        for r in m["meta"]["results"]:
            d = m["depth"][r["steps"]][m["mask"]]
            print(f"  Model {name} K={r['steps']}: {r['ms']:.0f} ms, median depth {np.median(d):.3f} m, scale {r['scale']:.4f} shift {r['shift']:.4f}")
    if a.export:
        out = Path(a.export); out.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out / "rgb.jpg"), cv2.cvtColor(c["rgb"], cv2.COLOR_RGB2BGR))
        save_depth_png(out / "lidar_depth_mm.png", c["lidar_depth"], c["lidar_depth"] > 0)
        cv2.imwrite(str(out / "lidar_conf.png"), c["lidar_conf"])
        for name in c["meta"]["models"]:
            m = c[name]; h, w = m["mask"].shape
            colors = cv2.resize(c["rgb"], (w, h), interpolation=cv2.INTER_AREA)
            for k, depth in m["depth"].items():
                valid = m["mask"] & np.isfinite(depth) & (depth > 0)
                save_depth_png(out / f"{name}_depth_k{k}_mm.png", depth, valid)
                save_ply(out / f"{name}_k{k}.ply", m["points"](k), colors, valid)
        print("exported to", out)


if __name__ == "__main__":
    main()
