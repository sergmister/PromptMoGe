"""Export the int8 sparse refiner of Model A / Model B for the Metal engine.

The refiner is trained quantisation-aware: every sparse convolution and linear layer except the input, prompt and
output projections carries a per-output-channel int8 weight quantiser and a per-tensor int8 quantiser on its input
with a learned clip. The Metal kernels execute exactly that arithmetic (int8 x int8 -> int32 accumulation, fp16
bias, LayerNorm and residuals), so the export is the quantised representation itself, not an approximation of it:

    refiner.bin    named blocks -- int8 weights [taps][Cin][Cout], per-channel scales, fp16 biases, activation clips
    refiner.json   geometry and the block index

QAT stores its master weights under `parametrizations.weight.original`; everything is therefore read from the
constructed modules and re-quantised with the training rule, then checked against the module's own fake-quantised
weight.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from ..model import BASE_WEIGHTS, load_model
from .coreml import ASPECT, COLS, ROWS

QMAX = 127.0


class BlockWriter:
    def __init__(self, path):
        self.f, self.index, self.offset = open(path, "wb"), {}, 0

    def put(self, name, tensor, dtype):
        assert name not in self.index, name
        x = np.ascontiguousarray(tensor.detach().float().cpu().numpy(), dtype=dtype)
        self.f.write(x.tobytes())
        self.index[name] = {"offset": self.offset, "count": int(x.size), "shape": list(x.shape), "dtype": np.dtype(dtype).name}
        self.offset += x.nbytes


def quantise(module, channel_dims):
    """The training quantiser on the fp32 master weight -> (int8 values, per-output-channel scale)."""
    w = module.parametrizations.weight.original.detach().float()
    s = (w.abs().amax(dim=channel_dims, keepdim=True) / QMAX).float().clamp_min(1e-12)
    q = torch.clamp(torch.round(w / s), -QMAX, QMAX)
    assert torch.equal((q * s).to(w.dtype), module.weight.detach().float()), "weight quantiser mismatch"
    return q, s.flatten()


@torch.inference_mode()
def export(name, base, out: Path):
    from moge.utils.geometry_torch import normalized_view_plane_uv
    out.mkdir(parents=True, exist_ok=True)
    model = load_model(name, base=base)
    unet = model.refiner
    assert getattr(unet, "_qat_layers", None) and getattr(model, "refiner_uv_fold", False), \
        "the Metal engine runs the compressed refiner (int8 QAT, UV planes folded out); use Model A or B"
    levels = len(unet.down_stages)
    channels = [int(unet.down_stages[k][0].channels) for k in range(levels)]
    w = BlockWriter(out / "refiner.bin")

    def clip(layer):
        aq = getattr(unet, "aq_" + layer.replace(".", "_"))
        assert int(aq.observed) > 0, f"{layer}: activation observer never ran"
        return aq.clip.detach().float().reshape(1)

    def linear(tag, lin):
        w.put(f"{tag}.w", lin.weight.t(), np.float16)
        w.put(f"{tag}.b", lin.bias, np.float16)

    def qlinear(tag, layer, lin):
        q, s = quantise(lin, (1,))
        w.put(f"{tag}.w8", q.t(), np.int8)
        w.put(f"{tag}.ws", s, np.float32)
        w.put(f"{tag}.b", lin.bias, np.float16)
        w.put(f"{tag}.clip", clip(layer), np.float32)

    def resblock(tag, layer, blk):
        w.put(f"{tag}.n1w", blk.norm1.weight, np.float16)
        w.put(f"{tag}.n1b", blk.norm1.bias, np.float16)
        for c, conv in (("c1", blk.conv1), ("c2", blk.conv2)):
            q, s = quantise(conv, (1, 2, 3, 4))                       # [Cout, 3, 3, 3, Cin]
            w.put(f"{tag}.{c}w8", q.reshape(q.shape[0], 27, -1).permute(1, 2, 0), np.int8)
            w.put(f"{tag}.{c}ws", s, np.float32)
            w.put(f"{tag}.{c}b", conv.bias, np.float16)
            w.put(f"{tag}.{c}clip", clip(f"{layer}.conv{c[1]}"), np.float32)

    linear("input_proj", unet.input_proj)
    linear("prompt_proj", unet.prompt_proj)
    linear("out_proj", unet.out_proj)
    for k in range(levels):
        resblock(f"down{k}", f"down_stages.{k}.0", unet.down_stages[k][0])
        if k < levels - 1:
            qlinear(f"pool{k}", f"downsample_blocks.{k}.linear", unet.downsample_blocks[k].linear)
    qlinear("encoder_fuse", "encoder_fuse", unet.encoder_fuse)
    qlinear("fuse0", "fuse_proj.0", unet.fuse_proj[0])
    qlinear("fuse2", "fuse_proj.2", unet.fuse_proj[2])
    resblock("bott", "bottleneck_stage.0", unet.bottleneck_stage[0])
    for i in range(levels - 1):
        qlinear(f"up{i}", f"upsample_blocks.{i}.linear", unet.upsample_blocks[i].linear)
        resblock(f"dec{i}", f"up_stages.{i}.0", unet.up_stages[i][0])
    # The UV half of the encoder conditioning is a fixed function of the token position: bake it.
    uv = normalized_view_plane_uv(width=COLS, height=ROWS, aspect_ratio=ASPECT, dtype=torch.float32)
    w.put("uv_bias", model.refiner_uv_proj(uv.permute(2, 0, 1).unsqueeze(0))[0], np.float32)
    w.f.close()

    down = int(unet.encoder_downsample)
    json.dump({"H": ROWS * down, "W": COLS * down, "R": ROWS, "C": COLS, "levels": levels, "channels": channels,
               "enc_channels": int(unet.encoder_fuse.in_features), "enc_out": int(unet.encoder_fuse.out_features),
               "depth_resolution": float(model.refiner_depth_resolution), "weights": w.index},
              open(out / "refiner.json", "w"), indent=1)
    print(f"  saved refiner.bin ({w.offset / 1e6:.1f} MB): {levels} levels, channels {channels}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="ios/models")
    ap.add_argument("--models", nargs="*", default=["A", "B"], help="A, B or checkpoint paths")
    ap.add_argument("--base", default=BASE_WEIGHTS)
    a = ap.parse_args()
    for name in a.models:
        print("refiner", name); export(name, a.base, Path(a.out) / Path(name).stem)


if __name__ == "__main__":
    main()
