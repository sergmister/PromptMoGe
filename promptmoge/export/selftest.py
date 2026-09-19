"""Write the on-device self-test data for an exported model: one frame and the PyTorch reference depth after every
refinement step. The demo app runs its whole pipeline on that frame and reports the error against this reference.

    python -m promptmoge.export.selftest --image rgb.jpg --depth lidar_mm.png --conf conf.png --out ios/models
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from ..infer import read_depth
from ..model import BASE_WEIGHTS, load_model
from .coreml import COLS, ROWS

STEPS = 3


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--image", required=True)
    ap.add_argument("--depth", required=True, help="LiDAR depth: 16-bit PNG in millimetres or .npy in metres")
    ap.add_argument("--conf", required=True, help="confidence map {0,1,2}")
    ap.add_argument("--out", default="ios/models")
    ap.add_argument("--models", nargs="*", default=["A", "B"])
    ap.add_argument("--base", default=BASE_WEIGHTS)
    a = ap.parse_args()

    image = torch.from_numpy(cv2.cvtColor(cv2.imread(a.image), cv2.COLOR_BGR2RGB)).float().div(255).permute(2, 0, 1)
    depth = torch.from_numpy(read_depth(a.depth))[None, None]
    conf = torch.from_numpy(cv2.imread(a.conf, cv2.IMREAD_UNCHANGED).astype(np.float32))[None, None]
    for name in a.models:
        out = Path(a.out) / Path(name).stem / "selftest"
        out.mkdir(parents=True, exist_ok=True)
        model = load_model(name, base=a.base)
        ref = model.infer(image, num_tokens=ROWS * COLS, refine_steps=STEPS, return_per_step=True, lidar_depth=depth,
                          lidar_conf=conf, metric_from="lidar_ls", gauge_poly=0)
        # the network input, resized exactly as the model does it
        small = F.interpolate(image[None], (ROWS * 14, COLS * 14), mode="bilinear", align_corners=False, antialias=True)
        small[0].numpy().astype(np.float32).tofile(out / "image.bin")
        depth[0, 0].numpy().astype(np.float32).tofile(out / "lidar_depth.bin")
        conf[0, 0].numpy().astype(np.uint8).tofile(out / "lidar_conf.bin")
        torch.stack(ref["depth_per_step"]).numpy().astype(np.float32).tofile(out / "depth.bin")
        json.dump({"image_shape": list(image.shape[-2:]), "lidar_shape": list(depth.shape[-2:]), "steps": STEPS},
                  open(out / "selftest.json", "w"))
        print(f"{name}: wrote {out}")


if __name__ == "__main__":
    main()
