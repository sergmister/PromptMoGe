"""Frame loading for the ARKitScenes dev scenes used for evaluation (RGB wide1008, LiDAR lowres, ARKit conf, Faro GT).

Layout under $MOGE3_DATA_ROOT (default: the working directory):
  data/dev_stage/{wide1008/<scene>/<ts>.jpg, lowres/<scene>/<ts>.png, conf/<scene>/<ts>.png}
  data/dev_gt/<scene>/{highres_depth/<scene>_<ts>.png, wide_intrinsics/<scene>_<ts>.pincam, lowres_wide.traj}
  data/dev_manifest.json      {"scenes": {<scene>: {"frames": [{"ts": "<ts>"}, ...]}}}

Conventions:
  * lowres LiDAR is DENSE (ARKit smoothed sceneDepth); validity == confidence, not depth>0.
  * confidence: 0 = low / no-return, 1 = medium, 2 = high.
  * GT highres_depth is 1440x1920 uint16 mm; 0 == no Faro coverage.
  * wide1008 RGB is 1008x1344 (box-downscaled from 1440x1920), same physical camera as the
    192x256 LiDAR (pure 7.5x scale between intrinsics), so pure resizing aligns all streams.
"""
from __future__ import annotations
import os

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional

import cv2
import numpy as np
from PIL import Image

ROOT = Path(os.environ.get("MOGE3_DATA_ROOT", "."))
STAGE = ROOT / "data/dev_stage"
GT = ROOT / "data/dev_gt"
MANIFEST = ROOT / "data/dev_manifest.json"
SCENES = ["41069021", "41069042", "41069046"]


@dataclass
class Frame:
    scene: str
    ts: str

    @property
    def key(self) -> str:
        return f"{self.scene}/{self.ts}"

    def rgb(self) -> np.ndarray:
        """uint8 HxWx3 (1008x1344)."""
        return np.asarray(Image.open(STAGE / "wide1008" / self.scene / f"{self.ts}.jpg").convert("RGB"))

    def lidar(self) -> np.ndarray:
        """float32 192x256 metres."""
        return np.asarray(Image.open(STAGE / "lowres" / self.scene / f"{self.ts}.png")).astype(np.float32) / 1000.0

    def conf(self) -> np.ndarray:
        """uint8 192x256 in {0,1,2}."""
        return np.asarray(Image.open(STAGE / "conf" / self.scene / f"{self.ts}.png")).astype(np.uint8)

    def gt(self) -> np.ndarray:
        """float32 1440x1920 metres, 0 == invalid."""
        return np.asarray(Image.open(GT / self.scene / "highres_depth" / f"{self.scene}_{self.ts}.png")).astype(np.float32) / 1000.0

    def intrinsics(self) -> Optional[np.ndarray]:
        p = GT / self.scene / "wide_intrinsics" / f"{self.scene}_{self.ts}.pincam"
        if not p.exists():
            return None
        w, h, fx, fy, cx, cy = [float(x) for x in p.read_text().split()]
        return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64), (int(w), int(h))


def load_manifest() -> Dict[str, List[Frame]]:
    man = json.loads(MANIFEST.read_text())
    return {vid: [Frame(vid, fr["ts"]) for fr in sc["frames"]] for vid, sc in man["scenes"].items()}


def iter_frames(scenes: Optional[List[str]] = None, stride: int = 1, limit: int = 0) -> Iterator[Frame]:
    man = load_manifest()
    n = 0
    for vid in (scenes or SCENES):
        for fr in man[vid][::stride]:
            yield fr
            n += 1
            if limit and n >= limit:
                return


def resize(a: np.ndarray, h: int, w: int, interp: int) -> np.ndarray:
    return a if a.shape[:2] == (h, w) else cv2.resize(a, (w, h), interpolation=interp)


def save_depth_png(path: Path, depth_m: np.ndarray):
    """uint16 millimetres; non-finite / negative -> 0 (invalid). cv2 with low compression (fast)."""
    d = np.where(np.isfinite(depth_m), depth_m, 0.0)
    d = np.clip(d * 1000.0, 0, 65535).astype(np.uint16)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), d, [cv2.IMWRITE_PNG_COMPRESSION, 1])


class AsyncWriter:
    """Thread pool for PNG writes so GPU inference is not stalled by encoding."""

    def __init__(self, workers: int = 8):
        from concurrent.futures import ThreadPoolExecutor
        self.ex = ThreadPoolExecutor(workers); self.futs = []

    def save(self, path: Path, depth_m: np.ndarray):
        self.futs.append(self.ex.submit(save_depth_png, path, depth_m))
        if len(self.futs) > 64:
            self.futs = [f for f in self.futs if not f.done()]

    def close(self):
        for f in self.futs:
            f.result()
        self.ex.shutdown()


def load_depth_png(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path)).astype(np.float32) / 1000.0


def load_traj(scene: str):
    """ARKit trajectory (ts, world-to-camera 4x4). If LF_TRAJ_DIR is set and holds <scene>.npz (arrays `ts`, `w2c`),
    that trajectory is used instead."""
    tdir = os.environ.get("LF_TRAJ_DIR")
    if tdir and (Path(tdir) / f"{scene}.npz").exists():
        z = np.load(Path(tdir) / f"{scene}.npz"); return z["ts"], z["w2c"]
    rows = [l.split() for l in open(GT / scene / "lowres_wide.traj") if l.strip()]
    ts = np.array([float(r[0]) for r in rows])
    w2c = []
    for r in rows:
        R, _ = cv2.Rodrigues(np.array([float(r[1]), float(r[2]), float(r[3])]))
        T = np.eye(4); T[:3, :3] = R; T[:3, 3] = [float(r[4]), float(r[5]), float(r[6])]
        w2c.append(T)
    return ts, np.stack(w2c)


def pose_for(ts_all, w2c_all, ts: float, tol: float = 0.05):
    i = int(np.argmin(np.abs(ts_all - ts)))
    return (w2c_all[i] if abs(ts_all[i] - ts) <= tol else None), abs(ts_all[i] - ts)


def intrinsics_at(fr: Frame, hw):
    K, (W0, H0) = fr.intrinsics()
    h, w = hw
    S = np.diag([w / W0, h / H0, 1.0])
    return S @ K
