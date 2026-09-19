"""Training data for LiDAR-prompted MoGe-3.

ArkitTrainDataset: staged ARKitScenes Training frames (rgb 1008x1344 jpg, gt 1008x1344 uint16 mm, lidar/conf 192x256).
  Augmentations (all geometric ops applied consistently to rgb / gt / lidar / conf):
    * horizontal flip
    * random 4:3 crop with scale in [crop_min, 1] -> resized to train_hw (rgb: area, gt: nearest, lidar/conf: nearest)
  Prompt corruption (simulated sensor failure, applied on the 192x256 LiDAR):
    * hole blobs: random rectangles set to depth=0 (true missing)  -- teaches completion from RGB
    * low-conf blobs: random rectangles set conf=0 with depth kept (ARKit-style smoothed junk) -- teaches conf gating
    * uncertainty-aware perturbation is applied on GPU in the trainer (perturb_prompt_depth).
ReplayDataset: RGB-only images (COCO val2017) for teacher distillation; returns gt/lidar as zeros with has_lidar=False.
"""
from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

cv2.setNumThreads(1)
STAGE = Path("data/arkit_stage")      # default stage root: <video>/{rgb/<ts>.jpg, gt/<ts>.png, lidar/<ts>.png, conf/<ts>.png}


def _frame_stats(root: Path) -> dict:
    """{video/ts: GT median depth in metres} for every frame of a stage (computed once at 1/8 resolution, cached)."""
    import json
    cache = root / "_frame_stats.json"
    if cache.exists():
        return json.loads(cache.read_text())
    out = {}
    for vid, ts in _scan_frames(root):
        g = cv2.imread(str(root / vid / "gt" / f"{ts}.png"), cv2.IMREAD_UNCHANGED)
        if g is None:
            continue
        g = g[::8, ::8].astype(np.float32) / 1000.0; v = g[g > 0]
        out[f"{vid}/{ts}"] = float(np.median(v)) if v.size else 0.0
    cache.write_text(json.dumps(out))
    return out


def _list_frames(root: Path) -> List[tuple]:
    import json
    cache = root / "_frames_cache.json"
    if cache.exists():
        return [tuple(x) for x in json.loads(cache.read_text())]
    out = _scan_frames(root)
    cache.write_text(json.dumps(out))
    return out


def _scan_frames(root: Path) -> List[tuple]:
    out = []
    for vd in sorted(p for p in root.iterdir() if p.is_dir()):
        for p in sorted((vd / "conf").glob("*.png")):
            ts = p.stem
            if (vd / "rgb" / f"{ts}.jpg").exists() and (vd / "gt" / f"{ts}.png").exists() and (vd / "lidar" / f"{ts}.png").exists():
                out.append((vd.name, ts))
    return out


def _photometric(rgb: np.ndarray, rng: random.Random) -> np.ndarray:
    """Photometric jitter on uint8 RGB: brightness/contrast (±20 %), saturation (±30 %), gamma (0.8-1.25),
    Gaussian blur (p=0.2, sigma 0.5-1.5), sensor noise (p=0.3, sigma 2-6). Geometry untouched."""
    x = rgb.astype(np.float32) / 255.0
    x = np.clip((x - 0.5) * rng.uniform(0.8, 1.2) + 0.5 + rng.uniform(-0.2, 0.2), 0, 1)
    g = x.mean(axis=2, keepdims=True); x = np.clip(g + (x - g) * rng.uniform(0.7, 1.3), 0, 1)
    x = x ** rng.uniform(0.8, 1.25)
    if rng.random() < 0.2:
        x = cv2.GaussianBlur(x, (0, 0), rng.uniform(0.5, 1.5))
    if rng.random() < 0.3:
        x = np.clip(x + np.random.default_rng(rng.randrange(1 << 30)).normal(0, rng.uniform(2, 6) / 255.0, x.shape).astype(np.float32), 0, 1)
    return (x * 255.0 + 0.5).astype(np.uint8)


def _mask_edge_band(gt: np.ndarray, rgb: np.ndarray, r: int, misaligned_only: bool) -> np.ndarray:
    """Set GT to 0 (missing) within r px of its own log-depth edges; with misaligned_only, only around edges farther than
    1.5 px from a Canny edge of the (un-augmented) RGB."""
    v = gt > 0; lg = np.log(np.where(v, gt, 1.0))
    ex = (np.abs(np.diff(lg, axis=1)) > 0.1) & v[:, 1:] & v[:, :-1]; ey = (np.abs(np.diff(lg, axis=0)) > 0.1) & v[1:] & v[:-1]
    e = np.zeros_like(v); e[:, 1:] |= ex; e[:, :-1] |= ex; e[1:] |= ey; e[:-1] |= ey          # both sides of each jump
    if misaligned_only:
        canny = cv2.Canny(cv2.cvtColor(np.ascontiguousarray(rgb), cv2.COLOR_RGB2GRAY), 50, 150) > 0
        dist = cv2.distanceTransform((~canny).astype(np.uint8), cv2.DIST_L2, 3)
        e &= dist > 1.5
    band = cv2.dilate(e.astype(np.uint8), np.ones((2 * r + 1, 2 * r + 1), np.uint8)) > 0
    out = gt.copy(); out[band] = 0.0
    return out


class ArkitTrainDataset(Dataset):
    def __init__(self, root: Path = STAGE, train_hw=(504, 672), crop_min: float = 0.6, flip: bool = True,
                 hole_p: float = 0.5, hole_n: int = 3, hole_frac: float = 0.25,
                 lowconf_p: float = 0.3, lowconf_n: int = 2, lowconf_frac: float = 0.2,
                 videos: Optional[List[str]] = None, seed: int = 0, load_normals: bool = False, synthetic: bool = False,
                 photo_aug: bool = False, gt_max_range: float = 0.0, max_median_depth: float = 0.0,
                 edge_band_px: int = 0, edge_band_misaligned_only: bool = False, prompt_shift_px: int = 0,
                 range_drop_p: float = 0.0, range_drop_min: float = 2.0, range_drop_max: float = 4.5,
                 fov_drop_p: float = 0.0, fov_drop_max: float = 0.25):
        self.root = Path(root)
        # range-falloff hole augmentation: with probability range_drop_p every prompt pixel beyond a random range in
        # [range_drop_min, range_drop_max] m becomes a hole (the real far-hole geometry; forces completion from RGB + near prompt)
        self.range_drop_p, self.range_drop_min, self.range_drop_max = float(range_drop_p), float(range_drop_min), float(range_drop_max)
        # field-of-view band augmentation: with probability fov_drop_p the prompt is removed in a band of random width
        # (3 %..fov_drop_max of the frame) along 1-2 random sides — the real border-fringe hole geometry (LiDAR FOV < camera FOV)
        self.fov_drop_p, self.fov_drop_max = float(fov_drop_p), float(fov_drop_max)
        # >0: invalidate the GT in a band of this radius (train-res px) around its own log-depth edges (jump > 0.1) — real Faro
        # renders are misaligned with the image by a median 4.9 px at 504x672, so boundary supervision there teaches hedging;
        # misaligned_only keeps GT edges that lie within 1.5 px of a Canny image edge (those are trustworthy)
        self.edge_band_px, self.edge_band_misaligned_only = int(edge_band_px), bool(edge_band_misaligned_only)
        self.prompt_shift_px = int(prompt_shift_px)      # >0: random +-shift (sensor px) of the LiDAR prompt relative to RGB/GT (device LiDAR-RGB calibration spread)
        if max_median_depth > 0:      # keep only frames whose GT median depth lies inside the sensor envelope (synthetic halls out)
            stats = _frame_stats(self.root)
            self._frames_pre = None
        self.gt_max_range = gt_max_range    # >0: GT beyond this range is treated as missing (synthetic halls: no far-range supervision)
        self.photo_aug = photo_aug          # brightness/contrast/saturation/gamma jitter, occasional blur and noise (RGB only)
        self.load_normals, self.synthetic = load_normals, synthetic   # synthetic stages carry exact half-res normals (MoGe head convention)
        self.frames = _list_frames(self.root)
        if max_median_depth > 0:
            self.frames = [f for f in self.frames if stats.get(f"{f[0]}/{f[1]}", 0.0) <= max_median_depth and stats.get(f"{f[0]}/{f[1]}", 0.0) > 0]
        if videos is not None:
            vs = set(videos); self.frames = [f for f in self.frames if f[0] in vs]
        self.train_hw = tuple(train_hw); self.crop_min = crop_min; self.flip = flip
        self.hole_p, self.hole_n, self.hole_frac = hole_p, hole_n, hole_frac
        self.lowconf_p, self.lowconf_n, self.lowconf_frac = lowconf_p, lowconf_n, lowconf_frac
        self.rng = random.Random(seed)

    def __len__(self):
        return len(self.frames)

    @staticmethod
    def _rand_rects(rng, h, w, n, frac):
        rects = []
        for _ in range(rng.randint(1, n)):
            rh = int(h * rng.uniform(0.05, frac)); rw = int(w * rng.uniform(0.05, frac))
            y = rng.randint(0, h - rh); x = rng.randint(0, w - rw)
            rects.append((y, x, rh, rw))
        return rects

    def __getitem__(self, i):
        for attempt in range(8):
            try:
                return self._load(i)
            except Exception as e:      # staging may still be writing / pruning videos concurrently
                i = random.randrange(len(self.frames))
        raise RuntimeError("ArkitTrainDataset: repeated read failures")

    def _load(self, i):
        vid, ts = self.frames[i]
        d = self.root / vid
        rgb = cv2.cvtColor(cv2.imread(str(d / "rgb" / f"{ts}.jpg"), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        gt = cv2.imread(str(d / "gt" / f"{ts}.png"), cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000.0
        lidar = cv2.imread(str(d / "lidar" / f"{ts}.png"), cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000.0
        conf = cv2.imread(str(d / "conf" / f"{ts}.png"), cv2.IMREAD_UNCHANGED).astype(np.float32)
        assert rgb.shape[:2] == gt.shape[:2] == (1008, 1344) and lidar.shape == conf.shape == (192, 256)
        if self.gt_max_range > 0:
            gt = np.where(gt > self.gt_max_range, 0.0, gt).astype(np.float32)
        nrm = None
        if self.load_normals and (d / "normal" / f"{ts}.npz").exists():
            nrm = np.load(d / "normal" / f"{ts}.npz")["n"].astype(np.float32) / 127.0        # [504,672,3]
        rng = random.Random(hash((vid, ts, self.rng.random())) & 0xFFFFFFFF)
        H, W = rgb.shape[:2]; h, w = lidar.shape
        # crop (4:3 preserved since both source and target are 4:3)
        s = rng.uniform(self.crop_min, 1.0)
        ch, cw = int(round(H * s)), int(round(W * s))
        y0 = rng.randint(0, H - ch); x0 = rng.randint(0, W - cw)
        fy, fx = h / H, w / W
        ly0, lx0 = int(round(y0 * fy)), int(round(x0 * fx)); lh, lw = max(int(round(ch * fy)), 8), max(int(round(cw * fx)), 8)
        th, tw = self.train_hw
        rgb = cv2.resize(rgb[y0:y0 + ch, x0:x0 + cw], (tw, th), interpolation=cv2.INTER_AREA)
        gt = cv2.resize(gt[y0:y0 + ch, x0:x0 + cw], (tw, th), interpolation=cv2.INTER_NEAREST)
        lidar = cv2.resize(lidar[ly0:ly0 + lh, lx0:lx0 + lw], (w, h), interpolation=cv2.INTER_NEAREST)
        conf = cv2.resize(conf[ly0:ly0 + lh, lx0:lx0 + lw], (w, h), interpolation=cv2.INTER_NEAREST)
        th, tw = self.train_hw
        if nrm is not None:
            ny0, nx0, nh, nw = y0 // 2, x0 // 2, max(ch // 2, 2), max(cw // 2, 2)
            nrm = cv2.resize(nrm[ny0:ny0 + nh, nx0:nx0 + nw], (tw, th), interpolation=cv2.INTER_NEAREST)
        if self.edge_band_px > 0:
            gt = _mask_edge_band(gt, rgb, self.edge_band_px, self.edge_band_misaligned_only)
        if self.photo_aug:
            rgb = _photometric(rgb, rng)
        if self.flip and rng.random() < 0.5:
            rgb, gt, lidar, conf = rgb[:, ::-1], gt[:, ::-1], lidar[:, ::-1], conf[:, ::-1]
            if nrm is not None:
                nrm = nrm[:, ::-1] * np.array([-1, 1, 1], dtype=np.float32)              # mirror flips the x component
        lidar = np.ascontiguousarray(lidar); conf = np.ascontiguousarray(conf)
        if self.prompt_shift_px > 0:                       # shift the prompt (depth + conf together) by a random integer offset, edge-replicated
            dy = rng.randint(-self.prompt_shift_px, self.prompt_shift_px); dx = rng.randint(-self.prompt_shift_px, self.prompt_shift_px)
            if dy or dx:
                M = np.float32([[1, 0, dx], [0, 1, dy]])
                lidar = cv2.warpAffine(lidar, M, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REPLICATE)
                conf = cv2.warpAffine(conf, M, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REPLICATE)
        # simulated sensor failures
        if rng.random() < self.hole_p:
            for (y, x, rh, rw) in self._rand_rects(rng, h, w, self.hole_n, self.hole_frac):
                lidar[y:y + rh, x:x + rw] = 0.0; conf[y:y + rh, x:x + rw] = 0.0
        if rng.random() < self.lowconf_p:
            for (y, x, rh, rw) in self._rand_rects(rng, h, w, self.lowconf_n, self.lowconf_frac):
                conf[y:y + rh, x:x + rw] = 0.0
        if self.range_drop_p > 0 and rng.random() < self.range_drop_p:      # sensor range falloff: everything beyond a random range is a hole
            far = lidar > rng.uniform(self.range_drop_min, self.range_drop_max)
            lidar[far] = 0.0; conf[far] = 0.0
        if self.fov_drop_p > 0 and rng.random() < self.fov_drop_p:        # border band(s): prompt removed along 1-2 random sides
            for side in rng.sample(range(4), rng.randint(1, 2)):
                f = rng.uniform(0.03, self.fov_drop_max); bh, bw = max(1, int(round(h * f))), max(1, int(round(w * f)))
                if side == 0: lidar[:bh] = 0.0; conf[:bh] = 0.0
                elif side == 1: lidar[-bh:] = 0.0; conf[-bh:] = 0.0
                elif side == 2: lidar[:, :bw] = 0.0; conf[:, :bw] = 0.0
                else: lidar[:, -bw:] = 0.0; conf[:, -bw:] = 0.0
        return {
            "image": torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1).float() / 255.0,
            "gt": torch.from_numpy(np.ascontiguousarray(gt)).unsqueeze(0),
            "lidar": torch.from_numpy(lidar).unsqueeze(0),
            "conf": torch.from_numpy(conf).unsqueeze(0),
            "has_lidar": torch.tensor(True), "has_gt": torch.tensor(True), "key": f"{vid}/{ts}",
            "normal": torch.from_numpy(np.ascontiguousarray(nrm)).permute(2, 0, 1).float() if nrm is not None else torch.zeros(3, th, tw),
            "has_normal": torch.tensor(nrm is not None), "is_synth": torch.tensor(self.synthetic),
        }


class ReplayDataset(Dataset):
    """RGB-only replay images; resized (area) to train_hw with a random 4:3 crop."""

    def __init__(self, root: Path, train_hw=(504, 672), seed: int = 0, limit: int = 0):
        self.paths = sorted(Path(root).glob("*.jpg"))
        if limit:
            self.paths = self.paths[:limit]
        self.train_hw = tuple(train_hw); self.rng = random.Random(seed)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = cv2.cvtColor(cv2.imread(str(self.paths[i]), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        H, W = img.shape[:2]; th, tw = self.train_hw
        # largest 4:3 crop
        if W / H > tw / th:
            cw = int(H * tw / th); ch = H
        else:
            cw = W; ch = int(W * th / tw)
        rng = random.Random(i + int(self.rng.random() * 1e9))
        y0 = rng.randint(0, H - ch); x0 = rng.randint(0, W - cw)
        img = cv2.resize(img[y0:y0 + ch, x0:x0 + cw], (tw, th), interpolation=cv2.INTER_AREA)
        if rng.random() < 0.5:
            img = img[:, ::-1]
        h, w = 192, 256
        return {
            "image": torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float() / 255.0,
            "gt": torch.zeros(1, th, tw), "lidar": torch.zeros(1, h, w), "conf": torch.zeros(1, h, w),
            "normal": torch.zeros(3, th, tw), "has_normal": torch.tensor(False), "is_synth": torch.tensor(False),
            "has_lidar": torch.tensor(False), "has_gt": torch.tensor(False), "key": self.paths[i].stem,
        }


def collate(batch: List[Dict]) -> Dict:
    out = {}
    for k in batch[0]:
        if k == "key":
            out[k] = [b[k] for b in batch]
        else:
            out[k] = torch.stack([b[k] for b in batch])
    return out
