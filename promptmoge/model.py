"""Load a PromptMoGe checkpoint and run it.

A checkpoint holds only the tensors that differ from MoGe-3 ViT-L (prompt stem and pyramid, neck, heads, refiner);
the frozen DINOv2 backbone comes from the original MoGe-3 release, which is downloaded on first use.
"""
from __future__ import annotations

import importlib.util
import os
import sys
import warnings
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch

HF_REPO = os.environ.get("PROMPTMOGE_HF_REPO", "sergmister/PromptMoGe")
BASE_WEIGHTS = os.environ.get("MOGE3_WEIGHTS", "Ruicheng/moge-3-vitl")
MODELS = {
    "L": "promptmoge_l.pt",   # full-resolution teacher, fp16/bf16 refiner
    "A": "promptmoge_a.pt",   # 480x640 point map, distilled prompt pyramid, int8-QAT refiner
    "B": "promptmoge_b.pt",   # 240x320 point map, one pyramid level fewer everywhere
}


# The sparse refiner runs on FlexGEMM (CUDA). Without it, fall back to the bundled pure-PyTorch equivalent, which is
# slow but exact and runs anywhere -- enough for CPU inference and for the exporters.
if importlib.util.find_spec("flex_gemm") is None:
    sys.path.insert(0, str(Path(__file__).parent / "_fallback"))


def _resolve(name_or_path: str) -> str:
    if Path(name_or_path).exists():
        return name_or_path
    if name_or_path.upper() not in MODELS:
        raise ValueError(f"unknown model {name_or_path!r}: expected a checkpoint path or one of {sorted(MODELS)}")
    from huggingface_hub import hf_hub_download
    return hf_hub_download(HF_REPO, MODELS[name_or_path.upper()])


def load_model(name_or_path: str = "A", base: Optional[str] = None, device: Union[str, torch.device] = "cpu"):
    """`name_or_path`: "L", "A", "B" or a local checkpoint. `base`: MoGe-3 ViT-L weights (path or Hugging Face id)."""
    from moge.model.v3 import MoGeModel

    ck = torch.load(_resolve(name_or_path), map_location="cpu", weights_only=False)
    kwargs = dict(ck.get("model_kwargs") or {})        # Model B carries its 4-level neck / heads / refiner here
    kwargs["lidar_prompt"] = ck["lidar_prompt"]
    with warnings.catch_warnings():
        # The base release cannot fill the modules PromptMoGe adds or reshapes; the overlay does, checked below.
        warnings.simplefilter("ignore")
        model = MoGeModel.from_pretrained(base or BASE_WEIGHTS, model_kwargs=kwargs)
    report = model.load_trainable_state(ck["state_dict"])
    uncovered = [k for k in report.missing_keys if not k.startswith(("encoder.", "scale_head."))]
    if report.unexpected_keys or uncovered or model._dropped_on_load:
        raise RuntimeError(f"checkpoint does not match the model code: unexpected {report.unexpected_keys[:3]}, "
                           f"uncovered {uncovered[:3]}, shape mismatch {model._dropped_on_load[:3]}")
    return model.eval().to(device)


def nearest_fill(depth: np.ndarray, conf: np.ndarray):
    """Replace every missing LiDAR pixel (depth <= 0) by its nearest valid neighbour, with confidence 1.

    The models are trained on ARKit's dense `sceneDepth`; a prompt with scattered holes (raw AVFoundation depth,
    sparse sensors) degrades them sharply, and this fill restores the accuracy. It is a no-op on a dense prompt.
    """
    valid = depth > 0
    if valid.all() or not valid.any():
        return depth, conf
    from scipy import ndimage
    idx = ndimage.distance_transform_edt(~valid, return_distances=False, return_indices=True)
    return depth[tuple(idx)], np.where(valid, conf, 1).astype(conf.dtype)


@torch.inference_mode()
def infer(model, image: np.ndarray, lidar_depth: np.ndarray, lidar_conf: Optional[np.ndarray] = None,
          refine_steps: int = 1, num_tokens: int = 1200, gauge_poly: int = 2, fill: bool = True,
          calibration_lut: Optional[dict] = None):
    """Metric depth for one frame.

    image        HxWx3 uint8 RGB, any resolution
    lidar_depth  hxw float32 metres, <= 0 where missing (ARKit sceneDepth is 192x256)
    lidar_conf   hxw in {0, 1, 2} (ARKit confidence); all-high if omitted
    refine_steps sparse 3-D refinement steps K. 1 is the deployed setting, 0 skips the refiner, never more than 3.
    gauge_poly   2 fits a per-frame quadratic gauge to the sensor (dense ARKit-grade depth); use 0 for sparse sensors

    Returns the dict of `MoGeModel.infer`: `depth` [H, W] metres, `points` [H, W, 3], `normal`, `mask`, `intrinsics`,
    `metric_fit`.
    """
    device = next(model.parameters()).device
    depth = np.asarray(lidar_depth, dtype=np.float32)
    conf = np.full(depth.shape, 2, np.float32) if lidar_conf is None else np.asarray(lidar_conf, dtype=np.float32)
    if fill:
        depth, conf = nearest_fill(depth, conf)
    img = torch.from_numpy(np.ascontiguousarray(image)).float().div(255).permute(2, 0, 1).to(device)
    return model.infer(img, num_tokens=num_tokens, refine_steps=refine_steps, use_fp16=device.type == "cuda",
                       lidar_depth=torch.from_numpy(depth)[None, None].to(device),
                       lidar_conf=torch.from_numpy(conf)[None, None].to(device),
                       metric_from="lidar_ls", calibration_lut=calibration_lut,
                       calibrate_input=calibration_lut is not None, gauge_poly=gauge_poly)
