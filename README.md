# PromptMoGe

**Metric depth from one RGB frame and a phone LiDAR — in real time, on the device.**

PromptMoGe prompts [MoGe-3](https://github.com/microsoft/MoGe) ViT-L with the low-resolution LiDAR depth that every
iPhone Pro and iPad Pro already produces (256×192, with confidence), and returns a dense, sharp, metric point map.
Two compressed variants run end to end on an iPad Pro: the dense network on the Neural Engine and GPU through
Core ML, the sparse 3-D refiner on custom int8 Metal kernels.

| | output | refiner | iPad Pro 11-inch (M5), end to end |
|---|---|---|---|
| **PromptMoGe-L** (teacher) | any resolution | fp16 | — (CUDA / CPU) |
| **Model A** | 480×640 | int8 | **152 ms** |
| **Model B** | 240×320 | int8 | **129 ms** |

**[Project page](https://sergmister.github.io/PromptMoGe/)** — method, architecture diagram and all benchmark results.
Latencies are measured on an iPad Pro 11-inch (M5) (iPad17,1, iOS 27 beta), warm, one refinement step.
*Real-world captures coming soon.*

## Why

- **More accurate than the sensor and than the state of the art.** On ARKitScenes against laser-scan ground truth
  (3 held-out captures, 1 094 frames), at the same 1 200-token budget:

  | AbsRel ↓ | all pixels | confident pixels |
  |---|---|---|
  | raw ARKit LiDAR | 0.0211 | 0.0192 |
  | MoGe-3 (RGB only), scale and shift fitted to the LiDAR | — | 0.0349 |
  | PromptDA-L, native 756×1008 (3 888 tokens) | 0.0155 | 0.0138 |
  | PromptDA-L, 420×560 (1 200 tokens) | 0.0149 | 0.0133 |
  | **PromptMoGe-L** | **0.0137** | **0.0122** |
  | **Model A** (on-device) | 0.0138 | **0.0122** |
  | **Model B** (on-device) | 0.0142 | 0.0125 |

  PromptDA-L is our re-run of the public model under the same protocol. Zero-shot, PromptMoGe-L is ahead of it
  on 6 of 7 further benchmarks (the exception is outdoor).
- **Compression that costs nothing.** Model A is distilled from the teacher — a re-scheduled prompt pyramid
  (431 → 29 GFLOPs), exact activation rescaling for fp16 accelerators, a narrowed and int8 quantisation-aware
  refiner — and stays within 1 % of it.
- **The whole phone, used properly.** Self-attention runs on the GPU and everything else dense on the Neural
  Engine (the engine alone is 2× slower at 1 200 tokens). The sparse refiner —
  voxelisation, neighbour maps, int8 submanifold convolutions, a robust metric fit — is one Metal command buffer
  per step, with no host round trips. On-device depth is within 0.2 % of the PyTorch reference.
- **Geometry from the image, scale from the sensor.** The prompt enters a frozen backbone through
  zero-initialised gates, so training starts from exactly MoGe-3 and keeps its priors: holes and low-confidence
  regions are filled from the image, and thin structures the sensor cannot resolve are recovered. Metric scale comes from a closed-form robust fit to the confident LiDAR —
  no intrinsics and no calibration are needed.

## Install

```bash
git clone https://github.com/sergmister/PromptMoGe && cd PromptMoGe
pip install -e .            # add ".[export]" for the Core ML exporters
```

Weights are on Hugging Face and download on first use: [`sergmister/PromptMoGe`](https://huggingface.co/sergmister/PromptMoGe)
holds the three PyTorch checkpoints and the ready-to-run iOS models. A checkpoint carries only what differs from
MoGe-3; the frozen backbone comes from the original [`Ruicheng/moge-3-vitl`](https://huggingface.co/Ruicheng/moge-3-vitl).

On CUDA, install [FlexGEMM](https://github.com/JeffreyXiang/FlexGEMM) for the sparse refiner. Without it a bundled
pure-PyTorch equivalent is used, which is exact and runs anywhere (CPU, Apple silicon), just slower.

## Use

```python
import cv2
from promptmoge import load_model, infer

model = load_model("A", device="cuda")                      # "L", "A", "B" or a checkpoint path
image = cv2.cvtColor(cv2.imread("rgb.jpg"), cv2.COLOR_BGR2RGB)
lidar = cv2.imread("lidar_mm.png", -1) / 1000.0             # ARKit sceneDepth, metres, 0 = missing
conf = cv2.imread("conf.png", -1)                           # ARKit confidence {0, 1, 2}

out = infer(model, image, lidar, conf, refine_steps=1)
out["depth"]        # [H, W] metres        out["points"]   # [H, W, 3] camera space
out["normal"]       # [H, W, 3]            out["mask"], out["intrinsics"], out["metric_fit"]
```

or `python -m promptmoge.infer --model A --image rgb.jpg --depth lidar_mm.png --conf conf.png --out out/`, which
also writes a coloured point cloud.

`refine_steps` is the number of sparse 3-D refinement steps K: 1 is the recommended setting, 0 skips the refiner
(1.4 % less accurate, sharper on thin structures), more than 3 is never useful. Missing LiDAR pixels are
nearest-filled before inference (`fill=True`) — the models expect a dense prompt, and sparse or holey depth
otherwise costs up to 2× in error. For sparse single-pulse sensors pass `gauge_poly=0`.

## iOS demo

`ios/PromptMoGeDemo` captures a frame with the LiDAR camera, runs Model A and Model B with K = 0, 1 and 3, and
shows the metric point clouds next to the raw sensor, with tap-to-measure. It needs a LiDAR device on iOS 26+
(Metal 4 tensors) and Xcode 26+.

```bash
hf download sergmister/PromptMoGe --include "ios/*" --local-dir .     # ready-made models -> ios/models, or build them:
python -m promptmoge.export.coreml  --out ios/models                  #   Core ML: shared ViT + per-model prompt, neck, heads
python -m promptmoge.export.refiner --out ios/models                  #   int8 refiner weights for the Metal engine

open ios/PromptMoGeDemo/PromptMoGeDemo.xcodeproj                      # set your team, build and run once
ios/push_models.sh <device-id>                                        # copies ios/models into the app's Documents
```

Every capture is saved on the device under `Documents/captures/<timestamp>/`: the camera frame, the raw LiDAR depth and
confidence, and each model's depth after K = 0, 1 and 3 with its mask and rays. `ios/pull_captures.sh <device-id>` copies
them off, and `python ios/read_capture.py <capture> --export out/` turns one into PNGs and point clouds.

To check an export on the device against PyTorch, write a reference with
`python -m promptmoge.export.selftest --image … --depth … --conf …`, push, and put a file `SELFTEST` containing `1`
into the app's Documents: the app then reports per-step depth error and warm latency instead of opening the camera.

## Training

[`promptmoge/train`](promptmoge/train/README.md) trains the teacher (frozen DINOv2, losses against laser-scan and
synthetic ground truth, distillation from stock MoGe-3 for everything the sensor does not supervise, simulated
sensor failures). [`promptmoge/compress`](promptmoge/compress/README.md) derives Model A and Model B from it.

## Layout

```
moge/                 MoGe-3 with the LiDAR prompt path (prompt stem + pyramid, gated injections, QAT refiner)
promptmoge/           load_model / infer, training, compression, Core ML + Metal exporters
ios/PromptMoGeDemo/   SwiftUI demo: Core ML pipeline (ANE + GPU) and the Metal sparse refiner
```

## Limitations

Trained on indoor scenes within the LiDAR's range (~5 m); outdoors the sensor adds little. With no sensor at all
the model falls back to RGB-only MoGe-3 geometry, which the prompt training erodes by a few percent. The
refinement step trades a little thin-structure accuracy for overall accuracy.

## License and acknowledgements

MIT. Built on [MoGe](https://github.com/microsoft/MoGe) (Microsoft, MIT) and [DINOv2](https://github.com/facebookresearch/dinov2)
(Meta AI, Apache 2.0); the refiner uses [FlexGEMM](https://github.com/JeffreyXiang/FlexGEMM) on CUDA. Compared against
[PromptDA](https://github.com/DepthAnything/PromptDA); evaluated on [ARKitScenes](https://github.com/apple/ARKitScenes).
