"""Export the dense half of PromptMoGe to Core ML, partitioned for Apple silicon.

The Neural Engine retires softmax elements at a fixed rate, so self-attention at 1200 tokens costs it more than
half of the whole ViT, while the GPU's fused attention never materialises the score matrix. Core ML's own planner
keeps the whole encoder on the engine, so the partition is made explicit: every transformer block becomes two
models -- attention for the GPU, the MLP half for the engine -- and everything else runs on the engine.

    vit/     stem, blk{k}_attn, blk{k}_rest, blk{0,4,8}_attn_inj, head     shared by every model (frozen DINOv2)
    <model>/ prompt_stem, prompt_neck, neck_heads                           per model (A, B)

The sparse refiner is not a Core ML model; see `promptmoge.export.refiner`.
"""
import argparse
import types
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..model import BASE_WEIGHTS, load_model

ROWS, COLS = 30, 40                       # 1200 tokens at 4:3; the network input is 420x560
ASPECT = COLS / ROWS
INJECT_BLOCKS = [0, 4, 8]                 # ViT blocks that receive a prompt injection


class PatchEmbedS2D(nn.Module):
    """Conv2d(3, D, 14, stride 14) as pixel_unshuffle + 1x1 convolution: exact, and ~18x faster on the engine."""
    def __init__(self, conv: nn.Conv2d):
        super().__init__()
        self.k = conv.kernel_size[0]
        d, c = conv.out_channels, conv.in_channels
        self.proj = nn.Conv2d(c * self.k * self.k, d, 1)
        with torch.no_grad():
            self.proj.weight.copy_(conv.weight.reshape(d, c * self.k * self.k, 1, 1))
            self.proj.bias.copy_(conv.bias)

    def forward(self, x):
        return self.proj(F.pixel_unshuffle(x, self.k))


def prepare_vit(model):
    """Make the encoder exportable without changing its function at the fixed 30x40 grid."""
    vit = model.encoder.backbone
    h, w = ROWS * 14, COLS * 14
    with torch.no_grad():
        tok = torch.cat((vit.cls_token.expand(1, -1, -1), vit.patch_embed(torch.zeros(1, 3, h, w))), dim=1)
        vit.onnx_compatible_mode = False
        pos = vit.interpolate_pos_encoding(tok, h, w).detach().clone()
    # DINOv2 resamples its position embedding bicubically, which Core ML cannot express: bake the result.
    vit.interpolate_pos_encoding = types.MethodType(lambda self, x, h, w: pos.to(x.dtype), vit)
    vit.patch_embed.proj = PatchEmbedS2D(vit.patch_embed.proj)
    return vit


class Stem(nn.Module):
    """image [1, 3, 420, 560] in 0..1 -> tokens [1, 1201, 1024]."""
    def __init__(self, model):
        super().__init__(); self.enc = model.encoder

    def forward(self, image):
        x = (image - self.enc.image_mean) / self.enc.image_std
        return self.enc.backbone.prepare_tokens_with_masks(x)


class Attention(nn.Module):
    """x -> x + ls1(attn(norm1(x))). The residual add is folded in so the engine model takes one tensor."""
    def __init__(self, block):
        super().__init__(); self.b = block

    def forward(self, x):
        return x + self.b.ls1(self.b.attn(self.b.norm1(x)))


class AttentionInjected(Attention):
    """The same with the LiDAR prompt injection added to the patch tokens first (the class token is left alone).
    Taking it as a model input keeps every buffer Core ML's own; adding it on the host costs a copy path."""
    def forward(self, x, inj):
        return super().forward(x + torch.cat([torch.zeros_like(inj[:, :1]), inj], dim=1))


class MLP(nn.Module):
    """y -> y + ls2(mlp(norm2(y)))."""
    def __init__(self, block):
        super().__init__(); self.b = block

    def forward(self, y):
        return y + self.b.ls2(self.b.mlp(self.b.norm2(y)))


class Head(nn.Module):
    """The four tapped layers -> the 1024-channel encoder feature at the token grid. Summed pairwise: the stock
    stack-and-sum builds a rank-5 tensor, which is ~15x slower on the engine."""
    def __init__(self, model):
        super().__init__(); self.enc = model.encoder

    def forward(self, t0, t1, t2, t3):
        acc = None
        for proj, t in zip(self.enc.output_projections, (t0, t1, t2, t3)):
            f = self.enc.backbone.norm(t)[:, 1:].permute(0, 2, 1).unflatten(2, (ROWS, COLS)).contiguous()
            acc = proj(f) if acc is None else acc + proj(f)
        return acc


class PromptStem(nn.Module):
    """prompt [1, 4, 420, 560] -> one token injection per injected block."""
    def __init__(self, model):
        super().__init__(); self.m = model.prompt_stem

    def forward(self, p):
        inj = self.m(p)[1]
        return tuple(inj[k] for k in sorted(inj))


class PromptNeck(nn.Module):
    """prompt at the point-map resolution -> one additive injection per neck level."""
    def __init__(self, model):
        super().__init__(); self.m = model.prompt_neck

    def forward(self, p):
        return tuple(self.m(p))


class NeckHeads(nn.Module):
    """encoder feature + neck injections -> (x/z, y/z, log z) point map and mask logit."""
    def __init__(self, model, levels):
        super().__init__()
        from moge.utils.geometry_torch import normalized_view_plane_uv
        self.m, self.levels = model, levels
        for lv in range(levels):
            uv = normalized_view_plane_uv(width=COLS << lv, height=ROWS << lv, aspect_ratio=ASPECT, dtype=torch.float32)
            self.register_buffer(f"uv{lv}", uv.permute(2, 0, 1).unsqueeze(0))

    def forward(self, enc, *injections):
        feats = [torch.cat([enc, self.uv0], dim=1)] + [getattr(self, f"uv{lv}") for lv in range(1, self.levels)]
        feats = self.m.neck(feats, injections=list(injections))
        return self.m.points_head(feats)[-1], self.m.mask_head(feats)[-1]


def convert(module, inputs, outputs, path, int8):
    """fp16 I/O throughout (the engine's datapath). int8 weights only for engine-resident models: they are a
    speed-up there and a small loss on the GPU."""
    import coremltools as ct
    from coremltools.optimize.coreml import OpLinearQuantizerConfig, OptimizationConfig, linear_quantize_weights
    module = module.eval()
    traced = torch.jit.trace(module, tuple(inputs.values()))
    ml = ct.convert(traced,
                    inputs=[ct.TensorType(name=n, shape=t.shape, dtype=np.float16) for n, t in inputs.items()],
                    outputs=[ct.TensorType(name=n, dtype=np.float16) for n in outputs],
                    minimum_deployment_target=ct.target.iOS18, compute_precision=ct.precision.FLOAT16,
                    convert_to="mlprogram")
    if int8:
        ml = linear_quantize_weights(ml, config=OptimizationConfig(global_config=OpLinearQuantizerConfig(
            mode="linear_symmetric", dtype="int8", granularity="per_channel")))
    ml.save(str(path))
    print("  saved", path.name, flush=True)


@torch.inference_mode()
def export_vit(base, out: Path):
    from moge.model.v3 import MoGeModel
    out.mkdir(parents=True, exist_ok=True)
    model = MoGeModel.from_pretrained(base).eval()
    vit = prepare_vit(model)
    taps = list(model.encoder.intermediate_layers)
    x = Stem(model)(torch.rand(1, 3, ROWS * 14, COLS * 14))
    convert(Stem(model), {"image": torch.rand(1, 3, ROWS * 14, COLS * 14)}, ["x"], out / "stem.mlpackage", int8=True)
    inj = torch.randn(1, ROWS * COLS, x.shape[-1]) * 0.01
    tapped = []
    for k, blk in enumerate(vit.blocks):
        convert(Attention(blk), {"x": x}, ["a"], out / f"blk{k}_attn.mlpackage", int8=False)
        if k in INJECT_BLOCKS:
            convert(AttentionInjected(blk), {"x": x, "inj": inj}, ["a"], out / f"blk{k}_attn_inj.mlpackage", int8=False)
        a = Attention(blk)(x)
        convert(MLP(blk), {"x": a}, ["y"], out / f"blk{k}_rest.mlpackage", int8=True)
        x = MLP(blk)(a)
        if k in taps:
            tapped.append(x)
    convert(Head(model), dict(zip(["t0", "t1", "t2", "t3"], tapped)), ["enc"], out / "head.mlpackage", int8=True)


@torch.inference_mode()
def export_model(name, base, out: Path):
    out.mkdir(parents=True, exist_ok=True)
    model = load_model(name, base=base)
    levels = len(model.prompt_neck.gates)             # 5 for Model A (x16 ladder), 4 for Model B (x8)
    H, W = ROWS << (levels - 1), COLS << (levels - 1)
    assert sorted(model.prompt_stem.gates.keys()) == [str(k) for k in INJECT_BLOCKS]
    convert(PromptStem(model), {"p": torch.rand(1, 4, ROWS * 14, COLS * 14)}, [f"i{k}" for k in INJECT_BLOCKS],
            out / "prompt_stem.mlpackage", int8=False)
    p = torch.rand(1, 4, H, W)
    injections = PromptNeck(model)(p)
    convert(PromptNeck(model), {"p": p}, [f"n{i}" for i in range(levels)], out / "prompt_neck.mlpackage", int8=False)
    neck = NeckHeads(model, levels)
    inputs = {"enc": torch.randn(1, 1024, ROWS, COLS), **{f"n{i}": t for i, t in enumerate(injections)}}
    coord, _ = neck(*inputs.values())
    # A neck with fewer levels than injections silently ignores the extras and computes a different function.
    assert tuple(coord.shape[-2:]) == (H, W), f"point map {tuple(coord.shape[-2:])} is not the {levels}-level {H}x{W}"
    convert(neck, inputs, ["coord", "mask"], out / "neck_heads.mlpackage", int8=True)



def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="ios/models")
    ap.add_argument("--models", nargs="*", default=["A", "B"], help="A, B or checkpoint paths")
    ap.add_argument("--base", default=BASE_WEIGHTS, help="MoGe-3 ViT-L weights (path or Hugging Face id)")
    ap.add_argument("--skip_vit", action="store_true")
    a = ap.parse_args()
    out = Path(a.out)
    if not a.skip_vit:
        print("ViT"); export_vit(a.base, out / "vit")
    for name in a.models:
        print("model", name); export_model(name, a.base, out / Path(name).stem)


if __name__ == "__main__":
    main()
