"""Train the LiDAR prompt path of MoGe-3 ViT-L (frozen DINOv2 backbone by default, frozen RGB-only teacher against forgetting).

Protocol (per batch):
  * ARKit frames with prompt (prob 1-p_drop): anchor_metric (differentiable LS-to-LiDAR fit, truncated log-L1 vs Faro,
    completion-weighted) + ssi + grad + geometric-normal (+ optional RPNL local, boundary-jump, conf-head) losses;
    uv / normal / mask distilled from the teacher; per-step losses on refined steps (refiner inputs detached as in stock).
  * ARKit frames with prompt dropped (prob p_drop) and replay images: L1 to the frozen teacher's (x/z, y/z, log z), normals, mask.
  * uncertainty-aware perturbation of the prompt depth before the prompt is built; optional input-side sensor calibration.
  * fp16 autocast + GradScaler, or bf16 per-module precision (encoder + refiner bf16, neck/heads fp32) for trainable sparse convs.
Checkpoints: only the trainable parameters + config (small).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

os.environ.setdefault("FLEX_GEMM_AUTOTUNE_MODE", "always")

from moge.model.v3 import MoGeModel
from moge.model.modules.prompt_stem import perturb_prompt_depth, fit_scale_shift
from promptmoge.train.train_data import ArkitTrainDataset, ReplayDataset, collate
from promptmoge.train.losses import (teacher_edge_grad_loss, exact_normal_loss, anchor_metric_loss, prompt_fidelity_loss, ssi_log_loss, grad_loss, distill_coord_loss, distill_normal_mask, lidar_fit,
                                 gauge_loss, geometric_normal_loss, edge_jump_loss, rpnl_loss, head_normal_loss)
from promptmoge.train.evaluate import quick_eval, DEFAULT_LUT
from promptmoge.train.data import MANIFEST as DEV_MANIFEST

DEFAULT_BASE = os.environ.get("MOGE3_WEIGHTS", "checkpoints/moge-3-vitl/model.pt")


def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True, help="run name: checkpoints and logs go to <out_dir>/<name>")
    ap.add_argument("--base", default=DEFAULT_BASE, help="base MoGe-3 ViT-L weights (default: $MOGE3_WEIGHTS or checkpoints/moge-3-vitl/model.pt)")
    ap.add_argument("--out_dir", default="runs", help="root directory of training runs")
    ap.add_argument("--replay_root", default="data/replay", help="flat directory of RGB-only replay .jpg images (e.g. COCO val2017); used when --replay_frac > 0 and it exists")
    ap.add_argument("--inject_blocks", type=int, nargs="*", default=[0, 4, 8])
    ap.add_argument("--neck", type=int, default=1)
    ap.add_argument("--neck_hidden", type=int, nargs="*", default=None,
                    help="PromptNeck encoder widths coarse->fine (stock 16 32 64 128 256; inverted e.g. 512 256 128 64 32)")
    ap.add_argument("--refiner_residual", type=int, default=0)
    ap.add_argument("--output_gate", type=int, default=0, help="uncertainty-gated refiner output")
    ap.add_argument("--soft_residual_gate", type=int, default=0, help="residual gate floor 0.15 at conf 0 instead of hard zero")
    ap.add_argument("--lora_rank", type=int, default=0, help="LoRA rank on ViT attention (0 = off)")
    ap.add_argument("--lora_blocks", type=int, nargs="*", default=list(range(12)))
    ap.add_argument("--lr_lora", type=float, default=2e-5)
    ap.add_argument("--lr_backbone", type=float, default=0.0, help="full DINOv2 fine-tuning rate (0 = frozen); use with --l2sp")
    ap.add_argument("--prompt_teacher", default=None, help="two-pass (mono-prompt) checkpoint used as a frozen PROMPTED teacher: its base output on the same prompt is distilled into the single-pass student")
    ap.add_argument("--prompt_distill_align", type=int, default=0, help="align the prompted teacher's z to the student's gauge (global LS scale+shift, detached) before the distillation L1")
    ap.add_argument("--prompt_distill_synth_only", type=int, default=0, help="apply the prompted-teacher distillation only to SYNTHETIC prompted samples (off-domain self-anchor of the prompted path)")
    ap.add_argument("--w_prompt_distill", type=float, default=0.0, help="weight of the prompted-teacher distillation on prompted samples")
    ap.add_argument("--profile_steps", type=int, default=0, help=">0: time the phases of this many steps (data wait, teacher fwd, student fwd, losses, backward, optimizer) and exit")
    ap.add_argument("--teacher_on_dropped_only", type=int, default=0, help="1: run the frozen teacher only on samples that need it (prompt-dropped / replay), not on prompted ones")
    ap.add_argument("--channels_last", type=int, default=0, help="1: channels_last memory format for the conv neck/heads")
    ap.add_argument("--compile_neck", type=int, default=0, help="1: torch.compile the neck and heads")
    ap.add_argument("--photo_aug", type=int, default=0, help="photometric jitter on the RGB of LiDAR samples (real and synthetic)")
    ap.add_argument("--arkit_videos", default=None, help="file with one ARKit video id per line: restrict the real training set (controls for data-size ablations)")
    ap.add_argument("--calib_jitter_scale", type=float, default=0.0, help="per-sample sensor calibration jitter: scale ~ 1 + U(-x, x), applied to LiDAR AND GT (teaches transparency to device calibration, not correction)")
    ap.add_argument("--calib_jitter_offset", type=float, default=0.0, help="per-sample range-offset jitter in metres ~ U(-x, x), applied to LiDAR AND GT")
    ap.add_argument("--w_feat_distill", type=float, default=0.0, help="anchor for backbone unlocking: MSE between student and frozen-teacher backbone features on RGB-only samples (normalised by teacher feature variance)")
    ap.add_argument("--l2sp", type=float, default=0.0, help="L2-SP: penalty weight pulling backbone weights toward the pretrained values")
    ap.add_argument("--conf_head", type=int, default=0, help="learned sensor-error head")
    ap.add_argument("--w_conf", type=float, default=0.5)
    ap.add_argument("--err_head", type=int, default=0, help="learned model-error head (|z_pred - z_gt| / z_gt of the base prediction), exported for fusion weighting")
    ap.add_argument("--w_err", type=float, default=0.5)
    ap.add_argument("--conf_target_seen", type=int, default=1, help="supervise the confidence heads against the prompt the network was actually handed (dropout / perturbation / beam-lattice redraw included) rather than the clean sensor; 0 supervises against the clean sensor")
    ap.add_argument("--aux_only", type=int, default=0, help="train ONLY the confidence heads; everything else frozen and run forward-only")
    ap.add_argument("--aux_heads_detached", type=int, default=1, help="detach the neck features feeding the conf/err heads: a confidence head reads the representation, it must not reshape what the depth head depends on")
    ap.add_argument("--aux_head_levels", type=int, default=0, help="ladder depth of the conf/err heads (0 = the neck's). 4 makes them read the levels Model A and Model B share and emit 240x320 in both, so one sensor-confidence head serves either model")
    ap.add_argument("--err_after_refine", type=int, default=1, help="supervise the model-error head from the LAST refinement step rather than the base prediction (needs --refine_steps_train > 0)")
    ap.add_argument("--w_edge", type=float, default=0.0, help="boundary-jump loss weight")
    ap.add_argument("--video_balance", type=float, default=0.0, help="real stream: sample frames with weight (frames in its video)^-alpha (0 = plain shuffle, 1 = every video equally likely); dense raw stages vary 13-1000 frames per video")
    ap.add_argument("--fidelity_gt_verified", type=float, default=0.0, help=">0 applies the fidelity loss only on sensor pixels whose calibrated depth agrees with GT within this relative error (confidently-wrong returns stop teaching copying)")
    ap.add_argument("--grad_ckpt", type=int, default=0, help="1: gradient checkpointing of the student ViT blocks, neck and heads (exact recompute; fits the full recipe at batch 4 into 24 GB)")
    ap.add_argument("--keep_every", type=int, default=0, help=">0: also keep runs/<name>/step_<N>.pt every N steps")
    ap.add_argument("--w_teacher_edge", type=float, default=0.0, help="edge-aware log-gradient loss on prompted samples against the frozen RGB-only teacher (LS-aligned), inside a band around its edges")
    ap.add_argument("--teacher_edge_band", type=int, default=3, help="band radius (output px) for --w_teacher_edge")
    ap.add_argument("--neck_zero_levels", type=int, nargs="*", default=[], help="PromptNeck levels whose gates are zeroed and frozen (0 = coarsest)")
    ap.add_argument("--edge_refined", type=int, default=0, help="also apply the boundary loss to refined steps (it trades holes for edges)")
    ap.add_argument("--mono_prompt", type=int, default=0, help="RGB-only pass as prompt channels (aligned mono log depth + disagreement)")
    ap.add_argument("--w_rpnl", type=float, default=0.0, help="patch-normalised local loss weight")
    ap.add_argument("--mono_prompt_mode", default="aligned", choices=["aligned"], help="how the RGB-only pass enters the prompt (with --mono_prompt)")
    ap.add_argument("--synth_root", default=None, help="synthetic stage root (Hypersim etc. with simulated LiDAR + exact normals)")
    ap.add_argument("--synth_unprompted", type=float, default=0.0, help="fraction of synthetic samples trained WITHOUT the simulated sensor prompt, supervised by scale-invariant GT losses (isolates synthetic geometry knowledge from the simulated sensor)")
    ap.add_argument("--synth_calib_input", default=None, help="calibration LUT json applied to SYNTHETIC LiDAR instead of --calib_input (the simulated sensor has its own bias)")
    ap.add_argument("--synth_sensor_jitter", type=float, default=0.0, help="per-frame log-scale jitter (std) applied to the SYNTHETIC sensor only, GT untouched: mimics the real per-frame sensor/GT gauge spread (0.0075 measured)")
    ap.add_argument("--synth_max_median_depth", type=float, default=0.0, help="drop synthetic frames whose GT median depth exceeds this (m): scene selection inside the sensor envelope")
    ap.add_argument("--synth_gt_max_range", type=float, default=0.0, help="synthetic GT beyond this range (m) is treated as missing, so far-range pixels (4x-weighted off-sensor) do not dominate the metric loss")
    ap.add_argument("--synth_frac", type=float, default=0.5, help="fraction of the LiDAR batch slots drawn from the synthetic stage")
    ap.add_argument("--w_exact_normal", type=float, default=0.0, help="normal-head cosine loss vs exact renderer normals (synthetic samples only; scales 1 and 4)")
    ap.add_argument("--calib_input", default=None, help="LUT json: calibrate the LiDAR input before losses / prompt (input-side calibration)")
    ap.add_argument("--crop_min", type=float, default=0.6, help="FoV/crop jitter lower bound (1.0 = no crop jitter)")
    ap.add_argument("--no_flip", action="store_true", help="disable horizontal flips")
    ap.add_argument("--arkit_root", default=None, help="real training stage root (default: data/arkit_stage)")
    ap.add_argument("--conf_phase", type=int, default=0, help="confidence-as-phase prompt channels (nlog*cos/sin(2*pi*conf/3))")
    ap.add_argument("--prompt_shift_px", type=int, default=0, help="random +-shift (sensor px) of the real LiDAR prompt relative to RGB (device calibration spread augmentation)")
    ap.add_argument("--sim_prompt", nargs="*", default=None, help="one or more v7 densifier checkpoints: with probability --sim_prompt_p, replace a sample's prompt by a single-pulse draw of that beam lattice (one checkpoint is chosen at random per step). The device's sensor delivers 112 beams; ARKitScenes' integrated map behaves like ~690, so passing both spans the sparsity range the model must survive")
    ap.add_argument("--sim_prompt_p", type=float, default=0.0)
    ap.add_argument("--sim_prompt_gain", type=float, default=1.0)
    ap.add_argument("--range_drop_p", type=float, default=0.0, help="range-falloff hole augmentation: probability that every prompt pixel beyond a random range becomes a hole")
    ap.add_argument("--range_drop_min", type=float, default=2.0); ap.add_argument("--range_drop_max", type=float, default=4.5)
    ap.add_argument("--fov_drop_p", type=float, default=0.0, help="border-band prompt dropout: probability of removing the prompt in a band along 1-2 random sides")
    ap.add_argument("--fov_drop_max", type=float, default=0.25, help="max band width as a fraction of the frame")
    ap.add_argument("--loss_gauge_poly", type=int, default=0, help="anchor loss supervises up to a per-frame polynomial prompt gauge (2 = quadratic; pair with infer gauge_poly=2)")
    ap.add_argument("--w_fidelity", type=float, default=0.0, help="soft prompt-fidelity loss weight on confident prompt pixels")
    ap.add_argument("--fidelity_sigma", type=float, default=0.0, help="fidelity loss on the low-pass (masked Gaussian, sensor px) deviation instead of per pixel")
    ap.add_argument("--fidelity_margin", type=float, default=0.01, help="free deviation from the calibrated prompt (log units) before the fidelity loss applies")
    ap.add_argument("--real_edge_band", type=int, default=0, help="real frames: invalidate GT within this many px of its own depth edges (Faro edge misalignment)")
    ap.add_argument("--real_edge_band_misaligned", action="store_true", help="... only around GT edges farther than 1.5 px from an image edge")
    ap.add_argument("--e2e_refiner", type=int, default=0, help="let refiner losses flow into the encoder feature path (refiner_detach_backbone=False); coords stay detached")
    ap.add_argument("--amp_dtype", default="fp16", choices=["fp16", "bf16"], help="bf16 is required for trainable refiner sparse convs")
    ap.add_argument("--anchor", type=float, default=0.0)
    ap.add_argument("--metric_calib", type=int, default=0, help="H3-lite: learn a global (log_s, t) calibration after the LiDAR LS fit")
    ap.add_argument("--train_refiner", type=int, default=0, help="1: train refiner prompt_proj (+ convs if --refiner_lr>0)")
    ap.add_argument("--refiner_channels", type=int, nargs="*", default=None,
                    help="refiner model_channels (stock 32 64 128 256 512; narrowed 32 64 128 256 256)")
    ap.add_argument("--refiner_uv_fold", type=int, default=0, help="fold the 2 UV planes out of encoder_fuse (enc_channels 1026 -> 1024)")
    ap.add_argument("--refiner_qat", type=int, default=0, help="int8 QAT of the refiner (per-channel weights, per-tensor activations, 99.95 pct clip)")
    ap.add_argument("--qat_pct", type=float, default=0.9995)
    ap.add_argument("--qat_freeze_at", type=int, default=0, help=">0: freeze the activation-clip observers after this step")
    ap.add_argument("--refiner_teacher", default=None, help="checkpoint whose (uncompressed) refiner is the distillation target for a narrowed / quantised refiner")
    ap.add_argument("--w_refiner_distill", type=float, default=0.0, help="weight of the L1 between the student and teacher refiner log-depth updates")
    ap.add_argument("--refine_steps_train", type=int, default=0)
    ap.add_argument("--refiner_every", type=int, default=1, help="run/train the refiner only every N-th step (other steps are base-only: ~2.5x cheaper)")
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--train_hw", type=int, nargs=2, default=[504, 672])
    ap.add_argument("--num_tokens", type=int, default=1200)
    ap.add_argument("--lr_prompt", type=float, default=2e-5)
    ap.add_argument("--lr_gates", type=float, default=2e-4, help="lr for the zero-init injection gates (low-dim, need to grow)")
    ap.add_argument("--lr_decoder", type=float, default=1e-5)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--gn_skip", type=float, default=50.0)
    ap.add_argument("--refiner_lr", type=float, default=0.0)
    ap.add_argument("--warmup", type=int, default=300)
    ap.add_argument("--p_drop", type=float, default=0.4)
    ap.add_argument("--replay_frac", type=float, default=0.25, help="fraction of batch slots drawn from replay (RGB-only) images")
    ap.add_argument("--perturb_lam", type=float, default=0.15)
    ap.add_argument("--w_anchor", type=float, default=1.0)
    ap.add_argument("--w_ssi", type=float, default=0.5)
    ap.add_argument("--w_grad", type=float, default=0.5)
    ap.add_argument("--w_distill", type=float, default=1.0)
    ap.add_argument("--w_uv", type=float, default=1.0)
    ap.add_argument("--w_gauge", type=float, default=0.1)
    ap.add_argument("--w_gnormal", type=float, default=0.0, help="geometric-normal loss (normals of aligned depth vs Faro normals, scales 1 and 4)")
    ap.add_argument("--offsensor_weight", type=float, default=1.0, help="anchor-loss weight for conf<2 / simulated-hole pixels")
    ap.add_argument("--hole_p", type=float, default=0.5)
    ap.add_argument("--hole_frac", type=float, default=0.25)
    ap.add_argument("--lowconf_p", type=float, default=0.3)
    ap.add_argument("--w_normal", type=float, default=0.5)
    ap.add_argument("--w_head_normal", type=float, default=0.0, help="supervise the normal HEAD from Faro-derived normals on prompted samples (teacher distillation then only on RGB-only)")
    ap.add_argument("--w_mask", type=float, default=0.2)
    ap.add_argument("--eval_every", type=int, default=500)
    ap.add_argument("--eval_stride", type=int, default=10)
    ap.add_argument("--save_every", type=int, default=500)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--init_from", type=str, default=None)
    ap.add_argument("--freeze_decoder", type=int, default=0)
    ap.add_argument("--refiner_only", type=int, default=0,
                    help="train ONLY the refiner (base frozen and run without grad): the refiner already takes detached inputs, so no backward through the ViT / neck / heads is needed -- much cheaper per step")
    ap.add_argument("--freeze_decoder_until", type=int, default=500, help="decoder lr = 0 for the first N steps (prompt path warms up alone)")
    return ap.parse_args()


def set_trainable(model, a):
    if a.aux_only:
        # Confidence heads as a cheap post-hoc addition to a finished checkpoint. With
        # `--aux_heads_detached` they receive detached neck features, so nothing upstream needs a gradient:
        # the base runs forward-only and a head-only pass is a fraction of a full step. Use this to give any
        # model its own heads -- they do NOT transfer between models (measured: a head trained on Model A's
        # neck reads rho +0.013 on Model B's, against +0.236 on its own).
        for p in model.parameters():
            p.requires_grad_(False)
        aux = [p for m in (getattr(model, "conf_head", None), getattr(model, "err_head", None)) if m is not None
               for p in m.parameters()]
        assert aux, "--aux_only needs --conf_head and/or --err_head"
        for p in aux:
            p.requires_grad_(True)
        return [{"params": aux, "lr": a.lr_prompt, "name": "aux_heads"}]
    if a.refiner_only:
        for p in model.parameters():
            p.requires_grad_(False)
        ref = [p for n, p in model.refiner.named_parameters()]
        ref += list(getattr(model, "refiner_uv_proj", torch.nn.Identity()).parameters())
        for p in ref:
            p.requires_grad_(True)
        return [{"params": ref, "lr": a.refiner_lr if a.refiner_lr > 0 else a.lr_prompt, "name": "refiner"}]
    for p in model.parameters():
        p.requires_grad_(False)
    groups = []
    gate_ids = {id(p) for n, p in model.named_parameters() if ".gates." in n or n == "prompt_calib"}
    lora_ids = {id(p) for n, p in model.named_parameters() if "lora_" in n}
    prompt = [p for p in model.prompt_parameters() if id(p) not in gate_ids and id(p) not in lora_ids]
    gates = [p for p in model.prompt_parameters() if id(p) in gate_ids]
    lora = [p for p in model.prompt_parameters() if id(p) in lora_ids]
    for p in prompt + gates + lora:
        p.requires_grad_(True)
    groups.append({"params": prompt, "lr": a.lr_prompt, "name": "prompt"})
    groups.append({"params": gates, "lr": a.lr_gates, "name": "gates"})
    if lora:
        groups.append({"params": lora, "lr": a.lr_lora, "name": "lora"})
    if a.lr_backbone > 0:
        bb = [p for n, p in model.encoder.backbone.named_parameters() if "lora_" not in n]
        for p in bb:
            p.requires_grad_(True)
        groups.append({"params": bb, "lr": a.lr_backbone, "name": "backbone"})
    if not a.freeze_decoder:
        dec = [p for m in [model.neck, model.points_head, model.normal_head, model.mask_head] for p in m.parameters()]
        for p in dec:
            p.requires_grad_(True)
        groups.append({"params": dec, "lr": a.lr_decoder, "name": "decoder"})
    if a.train_refiner and a.refiner_lr > 0:
        ref = [p for n, p in model.refiner.named_parameters() if "prompt_proj" not in n]
        ref += list(getattr(model, "refiner_uv_proj", torch.nn.Identity()).parameters())
        for p in ref:
            p.requires_grad_(True)
        groups.append({"params": ref, "lr": a.refiner_lr, "name": "refiner"})
    if not a.train_refiner and hasattr(model.refiner, "prompt_proj"):
        for p in model.refiner.prompt_proj.parameters():
            p.requires_grad_(False)
    return groups


def save_ckpt(model, a, step, path: Path):
    sd = {k: v.detach().cpu() for k, v in model.state_dict().items()
          if (not k.startswith("refiner_teacher"))       # the frozen distillation reference is not part of the overlay
          and (k.startswith(("prompt_stem", "prompt_neck", "prompt_calib", "conf_head", "err_head", "neck", "points_head", "normal_head", "mask_head", "refiner", "refiner_uv_proj"))
          or "lora_" in k or (getattr(a, "lr_backbone", 0) > 0 and k.startswith("encoder.backbone")))}
    torch.save({"lidar_prompt": model.lidar_prompt_cfg, "model_kwargs": getattr(model, "_arch_model_kwargs", {}) or {},
                "state_dict": sd, "step": step, "args": vars(a)}, path)


def load_trainable(model, path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    missing, unexpected = model.load_trainable_state(ck["state_dict"])
    return ck


def main():
    a = parse()
    torch.manual_seed(a.seed); random.seed(a.seed)
    out = Path(a.out_dir) / a.name; out.mkdir(parents=True, exist_ok=True)
    (out / "args.json").write_text(json.dumps(vars(a), indent=1))
    cfg = {"inject_blocks": a.inject_blocks, "neck": bool(a.neck), "refiner_residual": bool(a.refiner_residual),
           "output_gate": bool(a.output_gate), "anchor": a.anchor, "metric_calib": bool(a.metric_calib),
           "lora_rank": a.lora_rank, "lora_blocks": a.lora_blocks, "soft_residual_gate": bool(a.soft_residual_gate),
           "conf_head": bool(a.conf_head), "err_head": bool(a.err_head), "mono_prompt": bool(a.mono_prompt), "mono_prompt_mode": a.mono_prompt_mode, "conf_phase": bool(a.conf_phase),
           "neck_hidden": list(a.neck_hidden) if a.neck_hidden else None, "aux_head_levels": a.aux_head_levels or None, "aux_heads_detached": bool(a.aux_heads_detached),
           "refiner_channels": list(a.refiner_channels) if a.refiner_channels else None,
           "refiner_uv_fold": bool(a.refiner_uv_fold), "refiner_qat": bool(a.refiner_qat), "qat_pct": a.qat_pct}
    arch = {}
    if a.init_from:
        _ck0 = torch.load(a.init_from, map_location="cpu", weights_only=False)
        arch = dict(_ck0.get("model_kwargs") or {})
        if arch:
            print("architecture overrides from the init checkpoint:", sorted(arch))
        del _ck0
    model = MoGeModel.from_pretrained(a.base, model_kwargs={**arch, "lidar_prompt": cfg}).cuda()
    model._arch_model_kwargs = arch
    if a.init_from:
        print("init from", a.init_from, load_trainable(model, a.init_from)["step"])
    if a.refiner_teacher and a.w_refiner_distill > 0:
        rck = torch.load(a.refiner_teacher, map_location="cpu", weights_only=False)
        # the teacher's own architecture, in this order: its checkpoint's model_kwargs (Model B's 4-level
        # ladder lives there), then its lidar_prompt override, then the base checkpoint's config
        base_ref_cfg = torch.load(a.base, map_location="cpu", weights_only=True)["model_config"]["refiner"]
        rcfg = dict((rck.get("model_kwargs") or {}).get("refiner") or base_ref_cfg)
        if rck["lidar_prompt"].get("refiner_channels"):
            rcfg["model_channels"] = list(rck["lidar_prompt"]["refiner_channels"])
        if rck["lidar_prompt"].get("refiner_uv_fold"):
            raise SystemExit("--refiner_teacher must be an unfolded (1026-channel) reference refiner")
        model.attach_refiner_teacher(rck["state_dict"], rcfg)
        model.refiner_teacher.cuda()
        print(f"refiner teacher attached from {a.refiner_teacher} (channels {rcfg['model_channels']})")
    if a.grad_ckpt:
        model.enable_gradient_checkpointing()
        print("gradient checkpointing enabled on the student (ViT blocks, neck, heads, refiner)", flush=True)
    teacher = MoGeModel.from_pretrained(a.base).cuda().eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    pteacher = None
    if a.prompt_teacher and a.w_prompt_distill > 0:
        pck = torch.load(a.prompt_teacher, map_location="cpu", weights_only=False)
        pteacher = MoGeModel.from_pretrained(a.base, model_kwargs={"lidar_prompt": pck["lidar_prompt"]}).cuda()
        pteacher.load_trainable_state(pck["state_dict"]); pteacher.eval()
        for p in pteacher.parameters():
            p.requires_grad_(False)
        print("prompted teacher:", a.prompt_teacher, "mono two-pass:", getattr(pteacher, "mono_prompt", False))
    groups = set_trainable(model, a)
    if a.neck_zero_levels and hasattr(model, "prompt_neck"):
        for _i in a.neck_zero_levels:
            _g = model.prompt_neck.gates[_i]
            with torch.no_grad():
                _g.zero_()
            _g.register_hook(lambda grad: torch.zeros_like(grad))      # AdamW weight_decay is 0: the gate stays exactly zero
        print(f"PromptNeck levels {a.neck_zero_levels} zeroed and frozen", flush=True)
    if a.w_teacher_edge > 0 and a.teacher_on_dropped_only:
        raise SystemExit("--w_teacher_edge needs the teacher on prompted samples: use --teacher_on_dropped_only 0")
    if getattr(model.refiner, "_qat_layers", None) and not (a.train_refiner and a.refiner_lr > 0):
        # Freezing the refiner's WEIGHTS does not freeze its quantisation observers: they are buffers updated
        # inside forward under no_grad whenever the module is in training mode, so `requires_grad_(False)`
        # has no effect on them. A run that merely *executes* a QAT refiner while training something else
        # silently recalibrates its activation clips -- and if that run's prompt distribution differs (a
        # beam-lattice redraw, say), the clips drift to the wrong scale. Measured: 12.9x on one layer, which
        # coarsens the int8 step by the same factor and produces per-frame blow-ups.
        from moge.model.modules.qat import freeze_observers
        freeze_observers(model.refiner)
        print("QAT observers frozen (the refiner is not being trained)", flush=True)
    n_tr = sum(p.numel() for g in groups for p in g["params"])
    print(f"trainable params: {n_tr/1e6:.2f} M in {[(g['name'], len(g['params'])) for g in groups]}")
    opt = torch.optim.AdamW(groups, betas=(0.9, 0.999), weight_decay=0.0)
    bb_ref = None
    if a.lr_backbone > 0 and a.l2sp > 0:   # snapshot of the pretrained backbone weights for the L2-SP penalty
        bb_ref = [(p, p.detach().clone()) for n, p in model.encoder.backbone.named_parameters() if "lora_" not in n]
    base_lrs = [g["lr"] for g in opt.param_groups]
    amp_dtype = torch.float16 if a.amp_dtype == "fp16" else torch.bfloat16
    scaler = torch.amp.GradScaler("cuda", enabled=(amp_dtype == torch.float16))
    if amp_dtype == torch.bfloat16:
        # MoGe-3's own recipe: encoder (+ sparse refiner) in bf16, norm-free neck / heads in fp32 (bf16 there is too coarse:
        # the RGB-only distillation term jumped 10x under a blanket bf16 autocast). No outer autocast in this mode.
        model.enable_mixed_precision(torch.bfloat16); teacher.enable_mixed_precision(torch.bfloat16)
        if pteacher is not None:
            pteacher.enable_mixed_precision(torch.bfloat16)
        if hasattr(model, "refiner"):
            model.refiner.enable_mixed_precision(torch.bfloat16)
    outer_autocast = lambda: torch.autocast("cuda", dtype=amp_dtype, enabled=(amp_dtype == torch.float16))
    if a.channels_last:
        for mod in [model.neck, model.points_head, model.normal_head, model.mask_head, teacher.neck, teacher.points_head, teacher.normal_head, teacher.mask_head]:
            mod.to(memory_format=torch.channels_last)
    if a.compile_neck:
        for m_ in (model, teacher):
            m_.neck = torch.compile(m_.neck); m_.points_head = torch.compile(m_.points_head); m_.normal_head = torch.compile(m_.normal_head); m_.mask_head = torch.compile(m_.mask_head)

    vids = [l.strip() for l in open(a.arkit_videos) if l.strip()] if a.arkit_videos else None
    ds_ark = ArkitTrainDataset(**({'root': Path(a.arkit_root)} if a.arkit_root else {}), crop_min=a.crop_min, flip=not a.no_flip, train_hw=a.train_hw, seed=a.seed, hole_p=a.hole_p, hole_frac=a.hole_frac, lowconf_p=a.lowconf_p, videos=vids, photo_aug=bool(a.photo_aug),
                               edge_band_px=a.real_edge_band, edge_band_misaligned_only=a.real_edge_band_misaligned, prompt_shift_px=a.prompt_shift_px,
                               range_drop_p=a.range_drop_p, range_drop_min=a.range_drop_min, range_drop_max=a.range_drop_max, fov_drop_p=a.fov_drop_p, fov_drop_max=a.fov_drop_max)
    replay_root = Path(a.replay_root)
    ds_rep = ReplayDataset(replay_root, train_hw=a.train_hw, seed=a.seed) if replay_root.exists() and a.replay_frac > 0 else None
    n_rep = int(round(a.batch * a.replay_frac)) if ds_rep is not None and len(ds_rep) > 0 else 0
    n_ark = a.batch - n_rep
    ds_syn = ArkitTrainDataset(root=a.synth_root, train_hw=a.train_hw, seed=a.seed + 1, hole_p=a.hole_p, hole_frac=a.hole_frac, lowconf_p=a.lowconf_p,
                               load_normals=True, synthetic=True, photo_aug=bool(a.photo_aug), gt_max_range=a.synth_gt_max_range, max_median_depth=a.synth_max_median_depth,
                               range_drop_p=a.range_drop_p, range_drop_min=a.range_drop_min, range_drop_max=a.range_drop_max, fov_drop_p=a.fov_drop_p, fov_drop_max=a.fov_drop_max) if a.synth_root else None
    n_syn = max(1, int(n_ark * a.synth_frac)) if ds_syn is not None and len(ds_syn) > 0 and a.synth_frac > 0 else 0   # floor: never more synthetic than real slots
    n_ark -= n_syn
    print(f"train frames: arkit {len(ds_ark)} | synth {len(ds_syn) if ds_syn else 0} | replay {len(ds_rep) if ds_rep else 0} | per batch arkit {n_ark} synth {n_syn} replay {n_rep}")
    if a.video_balance > 0:
        from collections import Counter
        from torch.utils.data import WeightedRandomSampler
        _cnt = Counter(v for v, _ in ds_ark.frames)
        _w = torch.tensor([_cnt[v] ** (-a.video_balance) for v, _ in ds_ark.frames], dtype=torch.double)
        print(f"video-balanced sampling alpha={a.video_balance}: {len(_cnt)} videos, frames/video min {min(_cnt.values())} max {max(_cnt.values())}", flush=True)
        dl_ark = DataLoader(ds_ark, batch_size=n_ark, sampler=WeightedRandomSampler(_w, num_samples=len(ds_ark), replacement=True), num_workers=a.workers, collate_fn=collate, drop_last=True, persistent_workers=True)
    else:
        dl_ark = DataLoader(ds_ark, batch_size=n_ark, shuffle=True, num_workers=a.workers, collate_fn=collate, drop_last=True, persistent_workers=True)
    dl_rep = DataLoader(ds_rep, batch_size=n_rep, shuffle=True, num_workers=2, collate_fn=collate, drop_last=True, persistent_workers=True) if n_rep else None
    dl_syn = DataLoader(ds_syn, batch_size=n_syn, shuffle=True, num_workers=max(2, a.workers // 2), collate_fn=collate, drop_last=True, persistent_workers=True) if n_syn else None
    it_ark = iter(dl_ark); it_rep = iter(dl_rep) if dl_rep else None; it_syn = iter(dl_syn) if dl_syn else None

    def nxt(it, dl):
        try:
            return next(it), it
        except StopIteration:
            it = iter(dl); return next(it), it

    sim112 = None
    if a.sim_prompt and a.sim_prompt_p > 0:
        from promptmoge.train.sensor_sim.v7 import load_densifier
        sim112 = [load_densifier(p_) for p_ in a.sim_prompt]
        print("prompt corruption on {:.0%} of prompted samples: ".format(a.sim_prompt_p)
              + ", ".join(f"{m.grid[0]*m.grid[1]} beams ({m.lattice})" for m in sim112), flush=True)
    calib_lut = json.load(open(a.calib_input)) if a.calib_input else None
    synth_lut = json.load(open(a.synth_calib_input)) if a.synth_calib_input else None
    log = open(out / "train_log.jsonl", "a")
    model.train(); model.encoder.eval()
    t0 = time.time(); n_skipped = 0
    prof = {"data": [], "teacher": [], "student": [], "losses": [], "backward": [], "optim": []} if a.profile_steps else None
    def _tick():
        if prof is not None:
            torch.cuda.synchronize(); return time.time()
        return None
    t_prev = _tick()
    for step in range(1, a.steps + 1):
        if a.qat_freeze_at and step == a.qat_freeze_at + 1:
            # freeze the per-tensor activation clips so the last part of training sees the exact deployed quantiser
            from moge.model.modules.qat import freeze_observers
            freeze_observers(model); print(f"[step {step}] QAT activation observers frozen", flush=True)
        # lr schedule: linear warmup, cosine decay; decoder / refiner held at 0 for the first freeze_decoder_until steps
        f = min(1.0, step / a.warmup) * (0.5 * (1 + math.cos(math.pi * min(1.0, step / a.steps))) * 0.9 + 0.1)
        for g, lr in zip(opt.param_groups, base_lrs):
            g["lr"] = lr * f * (0.0 if (g["name"] in ("decoder", "refiner") and step <= a.freeze_decoder_until) else 1.0)
        b_ark, it_ark = nxt(it_ark, dl_ark)
        parts = [b_ark]
        if it_syn is not None:
            b_syn, it_syn = nxt(it_syn, dl_syn); parts.append(b_syn)
        if it_rep is not None:
            b_rep, it_rep = nxt(it_rep, dl_rep); parts.append(b_rep)
        batch = parts[0] if len(parts) == 1 else {k: (torch.cat([b[k] for b in parts]) if k != "key" else sum([b[k] for b in parts], [])) for k in parts[0]}
        if prof is not None:
            t_now = _tick(); prof["data"].append(t_now - t_prev); t_prev = t_now
        img = batch["image"].cuda(non_blocking=True); gt = batch["gt"].cuda(non_blocking=True)
        lidar = batch["lidar"].cuda(non_blocking=True); conf = batch["conf"].cuda(non_blocking=True)
        is_synth_b = batch["is_synth"].cuda().view(-1, 1, 1, 1) if "is_synth" in batch else torch.zeros(img.shape[0], 1, 1, 1, dtype=torch.bool, device=img.device)
        if calib_lut is not None:
            lidar_cal = model.apply_lidar_lut(lidar, calib_lut)      # input-side sensor calibration (loss fits + prompt)
            if synth_lut is not None:
                lidar_cal = torch.where(is_synth_b, model.apply_lidar_lut(lidar, synth_lut), lidar_cal)
            lidar = lidar_cal
        if a.synth_sensor_jitter > 0:
            jf = torch.exp(torch.randn(img.shape[0], 1, 1, 1, device=img.device) * a.synth_sensor_jitter)
            lidar = torch.where(is_synth_b & (lidar > 0), lidar * jf, lidar)
        if a.calib_jitter_scale > 0 or a.calib_jitter_offset > 0:
            # residual device miscalibration: the same (scale, offset) is applied to the sensor and to the target, so the
            # network learns to pass a slightly different calibration through (it is unobservable from RGB), never to undo it
            B = img.shape[0]
            ja = 1.0 + (torch.rand(B, 1, 1, 1, device=img.device) * 2 - 1) * a.calib_jitter_scale
            jt = (torch.rand(B, 1, 1, 1, device=img.device) * 2 - 1) * a.calib_jitter_offset
            lidar = torch.where(lidar > 0, ja * lidar + jt, lidar); gt = torch.where(gt > 0, ja * gt + jt, gt)
        has_lidar = batch["has_lidar"].cuda()
        # prompt dropout: per-sample; dropped samples are trained as RGB-only distillation
        drop = torch.rand(img.shape[0], device=img.device) < a.p_drop
        is_synth = batch["is_synth"].cuda() if "is_synth" in batch else torch.zeros_like(has_lidar)
        if a.synth_unprompted > 0:
            drop = drop | (is_synth & (torch.rand(img.shape[0], device=img.device) < a.synth_unprompted))
        use_prompt = has_lidar & ~drop
        with torch.no_grad():
            lidar_src, conf_src = lidar, conf
            if sim112 is not None and a.sim_prompt_p > 0:
                # Device robustness: ARKitScenes' sceneDepth is temporally integrated and behaves like several
                # hundred effective samples; the shipping device hands over one pulse of a 14x8 = 112-beam
                # lattice. Training on that draw (GT untouched) is what stops the model from copying the prompt.
                sel = torch.rand(lidar.shape[0], device=lidar.device) < a.sim_prompt_p
                if sel.any():
                    from promptmoge.train.sensor_sim.v7 import simulate_v7
                    src = lidar[sel]
                    med = torch.stack([v[v > 0].median() if (v > 0).any() else torch.ones((), device=v.device) for v in src])
                    src = torch.where(src > 0, src, med.view(-1, 1, 1, 1))
                    rgb_lr = F.interpolate(img[sel], src.shape[-2:], mode="area")
                    sim_m = sim112[int(torch.randint(len(sim112), (1,)).item())]
                    d_s, c_s = simulate_v7(sim_m, src.clamp(0, 10), rgb_lr, noise_gain=a.sim_prompt_gain,
                                           seed=int(torch.randint(0, 1 << 30, (1,)).item()))
                    lidar_src = lidar.clone(); conf_src = conf.clone()
                    lidar_src[sel] = d_s.to(lidar.dtype); conf_src[sel] = c_s.to(conf.dtype)
            lidar_in = perturb_prompt_depth(lidar_src, conf_src, lam=a.perturb_lam)
            lidar_in = torch.where(use_prompt.view(-1, 1, 1, 1), lidar_in, torch.zeros_like(lidar_in))
            conf_in = torch.where(use_prompt.view(-1, 1, 1, 1), conf_src, torch.zeros_like(conf))
        # NOTE: samples with use_prompt=False get an all-invalid prompt (valid=0 everywhere, unc=1): the stem sees a
        # "null prompt" (not the bit-identical skip path) -- this is what teaches the network that null prompt == RGB-only.
        with outer_autocast():
            with torch.no_grad():
                if a.teacher_on_dropped_only:
                    # run the frozen teacher only on the samples that distil from it (prompt-dropped / replay); the prompted
                    # samples' uv and gauge terms (which need coord_t) are then disabled (an accuracy trade)
                    Dn = (~use_prompt)
                    tout = {"points": torch.zeros_like(img[:, :1]).permute(0, 2, 3, 1).expand(-1, -1, -1, 3).clone(), "normal": None, "mask": None, "encoder_tokens": None}
                    if Dn.any():
                        tsub = teacher.forward(img[Dn], num_tokens=a.num_tokens, refine_steps=0)
                        pts = torch.zeros((img.shape[0],) + tuple(tsub["points"].shape[1:]), device=img.device, dtype=tsub["points"].dtype); pts[Dn] = tsub["points"]
                        nrm_ = torch.zeros((img.shape[0],) + tuple(tsub["normal"].shape[1:]), device=img.device, dtype=tsub["normal"].dtype); nrm_[Dn] = tsub["normal"]
                        msk_ = torch.zeros((img.shape[0],) + tuple(tsub["mask"].shape[1:]), device=img.device, dtype=tsub["mask"].dtype); msk_[Dn] = tsub["mask"]
                        tout = {"points": pts, "normal": nrm_, "mask": msk_, "encoder_tokens": None}
                    else:
                        tout = None
                else:
                    tout = teacher.forward(img, num_tokens=a.num_tokens, refine_steps=0)
                if prof is not None:
                    t_now = _tick(); prof["teacher"].append(t_now - t_prev); t_prev = t_now
                ptout = pteacher.forward(img, num_tokens=a.num_tokens, refine_steps=0, lidar_depth=lidar_in, lidar_conf=conf_in) if pteacher is not None else None
            rs = a.refine_steps_train if (step % max(a.refiner_every, 1) == 0) else 0
            sout = model.forward(img, num_tokens=a.num_tokens, refine_steps=rs, return_per_step=True,
                                 lidar_depth=lidar_in, lidar_conf=conf_in, refiner_detach_backbone=not a.e2e_refiner)
        if prof is not None:
            t_now = _tick(); prof["student"].append(t_now - t_prev); t_prev = t_now
        # student coords in affine frame from points (exp remap): points [B,H,W,3] = (u z, v z, z) with z = exp(logz)
        def to_coord(points):
            z = points[..., 2].float().clamp_min(1e-6)
            return torch.stack([points[..., 0].float() / z, points[..., 1].float() / z, torch.log(z)], dim=-1)
        coords_s = [to_coord(p) for p in sout["points_per_step"]]
        coord_s = coords_s[0]
        def match_res(c, like):
            # Model B's point map is 240x320 while the frozen 5-level teacher's is 480x640; every teacher term
            # (uv, gauge, distill) compares coords pixel-for-pixel, so bring the teacher onto the student's grid.
            if c.shape[1:3] == like.shape[1:3]:
                return c
            return F.interpolate(c.permute(0, 3, 1, 2).float(), like.shape[1:3], mode="bilinear",
                                 align_corners=False).permute(0, 2, 3, 1)
        if tout is None:      # teacher skipped (no dropped/replay sample in this batch): neutral placeholders, no distillation terms
            tout = {"points": None, "normal": sout["normal"].detach(), "mask": sout["mask"].detach()}; coord_t = coord_s.detach()
        else:
            coord_t = match_res(to_coord(tout["points"]), coord_s)
        losses = {}
        P = use_prompt; D = ~use_prompt
        # --- prompted samples: metric + structure vs GT, uv/normal/mask distilled
        if P.any():
            lz = coord_s[P][..., 2]
            calib = getattr(model, "prompt_calib", None)
            l_anchor, mon = anchor_metric_loss(lz, gt[P], lidar[P], conf[P], calib=calib, offsensor_weight=a.offsensor_weight, conf_in=conf_in[P], gauge_poly=a.loss_gauge_poly)
            if a.w_fidelity > 0:
                conf_fid = conf_in[P]
                if a.fidelity_gt_verified > 0:
                    gt_lr = F.interpolate(gt[P], lidar.shape[-2:], mode="nearest")
                    wrong = (gt_lr > 0) & ((lidar[P] - gt_lr).abs() / gt_lr.clamp_min(1e-3) > a.fidelity_gt_verified)
                    conf_fid = torch.where(wrong, torch.zeros_like(conf_fid), conf_fid)
                    mon["fid_px_dropped"] = float((wrong & (conf_in[P] >= 2)).float().mean())
                losses["fidelity"] = a.w_fidelity * prompt_fidelity_loss(lz, lidar[P], conf_fid, a.fidelity_margin, sigma_px=a.fidelity_sigma)
            if calib is not None:
                mon["calib_s"] = float(torch.exp(calib[0])); mon["calib_t"] = float(calib[1])
            losses["anchor"] = a.w_anchor * l_anchor
            losses["ssi"] = a.w_ssi * ssi_log_loss(lz, gt[P])
            losses["grad"] = a.w_grad * grad_loss(lz, gt[P])
            if not a.teacher_on_dropped_only:
                losses["uv"] = a.w_uv * (coord_s[P][..., :2] - coord_t[P][..., :2]).abs().mean()
            PA = (P & is_synth) if a.prompt_distill_synth_only else P       # anchor set for the prompted-teacher term
            if ptout is not None and PA.any():
                # prompted-teacher distillation: the teacher's base output on the same sensor prompt as a target (on PA)
                pt_pts = ptout["points"][PA].float()
                if a.prompt_distill_align:
                    # the two models live in different affine gauges (arbitrary per-model scale and z-shift): fit the teacher's
                    # z to the student's z (global LS on the valid pixels, detached) so only relative structure is distilled
                    zs = sout["points"][PA][..., 2].detach().float().flatten(1); zt = pt_pts[..., 2].flatten(1)
                    okm = torch.isfinite(zs) & torch.isfinite(zt) & (zs > 1e-3) & (zt > 1e-3)
                    s_al, t_al = fit_scale_shift(zt, zs, okm, differentiable=False)
                    pt_pts = torch.cat([pt_pts[..., :2] * s_al.view(-1, 1, 1, 1), (s_al.view(-1, 1, 1) * pt_pts[..., 2] + t_al.view(-1, 1, 1)).unsqueeze(-1)], dim=-1)
                losses["prompt_distill"] = a.w_prompt_distill * distill_coord_loss(coord_s[PA], to_coord(pt_pts))
            if not a.teacher_on_dropped_only:
                losses["gauge"] = a.w_gauge * gauge_loss(coord_s[P], coord_t[P])
            if a.w_gnormal > 0:
                losses["gnormal"] = a.w_gnormal * geometric_normal_loss(lz, gt[P], lidar[P], conf[P])
            if a.w_rpnl > 0:
                losses["rpnl"] = a.w_rpnl * rpnl_loss(lz, gt[P])
            if a.w_teacher_edge > 0 and tout.get("points") is not None:
                losses["teacher_edge"] = a.w_teacher_edge * teacher_edge_grad_loss(lz, coord_t[P][..., 2], band_px=a.teacher_edge_band)
            if a.w_edge > 0:
                losses["edge"] = a.w_edge * edge_jump_loss(lz, gt[P])
                if a.edge_refined:
                    for k, cs in enumerate(coords_s[1:], start=1):
                        losses[f"edge_s{k}"] = a.w_edge * edge_jump_loss(cs[P][..., 2], gt[P])
            if sout.get("sensor_err") is not None:
                # target: the relative error vs Faro of the depth map the network was ACTUALLY HANDED, at output
                # resolution, clamped to [0, 0.5] like the head's range.
                # `--conf_target_seen` (default) uses `lidar_in` -- the prompt after dropout/perturbation and, when
                # `--sim_prompt` is on, after the beam-lattice redraw. Supervising against the *clean* sensor
                # instead is why such a head could not tell a 112-beam prompt from ARKit's: it was
                # asked to predict an error the input never showed.
                gts = gt[P].squeeze(1)
                se = F.interpolate(sout["sensor_err"][P].unsqueeze(1), gts.shape[-2:], mode="bilinear", align_corners=False).squeeze(1)
                lid_src = lidar_in if a.conf_target_seen else lidar
                lid_up = F.interpolate(lid_src[P], gts.shape[-2:], mode="nearest").squeeze(1)
                vm = (gts > 0) & (lid_up > 0)
                tgt = torch.where(vm, ((lid_up - gts).abs() / gts.clamp_min(1e-3)).clamp(0, 0.5), torch.zeros_like(gts))
                losses["conf"] = a.w_conf * (torch.where(vm, (se - tgt).abs(), torch.zeros_like(se)).sum() / vm.sum().clamp_min(1))
            if sout.get("model_err") is not None and a.w_err > 0:
                # target: the relative error vs Faro of the depth the model actually delivers, clamped to [0, 0.5].
                # `--err_after_refine` (default) measures the LAST refinement step rather than the base prediction:
                # the refiner is what ships, so a confidence trained on the base output describes a map nobody sees.
                # The gauge is fitted to the prompt the network was handed, as it is at deployment.
                gts = gt[P].squeeze(1)
                me = F.interpolate(sout["model_err"][P].unsqueeze(1), gts.shape[-2:], mode="bilinear", align_corners=False).squeeze(1)
                lz_err = coords_s[-1][P][..., 2] if (a.err_after_refine and len(coords_s) > 1) else lz
                lid_gauge = lidar_in if a.conf_target_seen else lidar
                conf_gauge = conf_in if a.conf_target_seen else conf
                # metric prediction exactly as the anchor loss attaches it: robust LS scale/shift to the calibrated prompt (+ calib)
                with torch.no_grad():
                    s_e, t_e = lidar_fit(lz_err.detach(), lid_gauge[P], conf_gauge[P])
                    if calib is not None:
                        s_e = s_e * torch.exp(calib[0]); t_e = t_e * torch.exp(calib[0]) + calib[1]
                    zp = s_e.view(-1, 1, 1) * torch.exp(lz_err.detach().float()) + t_e.view(-1, 1, 1)
                    if zp.shape[-2:] != gts.shape[-2:]:
                        zp = F.interpolate(zp.unsqueeze(1), gts.shape[-2:], mode="bilinear", align_corners=False).squeeze(1)
                vm = (gts > 0) & torch.isfinite(zp) & (zp > 0)
                tgt = torch.where(vm, ((zp - gts).abs() / gts.clamp_min(1e-3)).clamp(0, 0.5), torch.zeros_like(gts))
                losses["err"] = a.w_err * (torch.where(vm, (me - tgt).abs(), torch.zeros_like(me)).sum() / vm.sum().clamp_min(1))
            if sout.get("refiner_deltas"):
                # compression distillation: match the uncompressed refiner's own log-depth update, per voxel.
                # Normalised by the teacher update's RMS so the weight is scale-free across steps.
                num = sum((ds[P] - dt[P]).abs().mean() for ds, dt in sout["refiner_deltas"])
                den = sum(dt[P].abs().mean().detach() for _, dt in sout["refiner_deltas"]).clamp_min(1e-6)
                losses["refiner_distill"] = a.w_refiner_distill * num / den
                mon["refiner_rel_l1"] = float((num / den).detach())
            for k, cs in enumerate(coords_s[1:], start=1):
                l_k, mon_k = anchor_metric_loss(cs[P][..., 2], gt[P], lidar[P], conf[P], calib=calib, offsensor_weight=a.offsensor_weight, conf_in=conf_in[P], gauge_poly=a.loss_gauge_poly)
                losses[f"anchor_s{k}"] = a.w_anchor * l_k
                mon[f"absrel_anchor_s{k}"] = mon_k["absrel_anchor"]
                if a.w_gnormal > 0:   # keep refined steps as smooth as the base (refiner otherwise re-introduces sensor fuzz)
                    losses[f"gnormal_s{k}"] = a.w_gnormal * geometric_normal_loss(cs[P][..., 2], gt[P], lidar[P], conf[P])
        else:
            mon = {}
        # --- RGB-only samples (dropped prompt or replay): distill to teacher
        # synthetic samples without a prompt: GT-supervised (scale-invariant) instead of teacher-distilled
        U = D & is_synth & batch["has_gt"].cuda() if a.synth_unprompted > 0 else torch.zeros_like(D)
        if U.any():
            losses["ssi_unprompted"] = a.w_ssi * ssi_log_loss(coord_s[U][..., 2], gt[U])
            losses["grad_unprompted"] = a.w_grad * grad_loss(coord_s[U][..., 2], gt[U])
            if a.w_rpnl > 0:
                losses["rpnl_unprompted"] = a.w_rpnl * rpnl_loss(coord_s[U][..., 2], gt[U])
        D = D & ~U
        if D.any():
            losses["distill"] = a.w_distill * distill_coord_loss(coord_s[D], coord_t[D])
            if a.w_feat_distill > 0 and sout.get("encoder_tokens") is not None:
                sf = sout["encoder_tokens"][D].float(); tf = tout["encoder_tokens"][D].float()
                losses["feat_distill"] = a.w_feat_distill * ((sf - tf) ** 2).mean() / (tf.var() + 1e-6)
        if a.w_head_normal > 0:
            # prompted samples -> Faro normals; RGB-only / replay samples -> teacher normals (anti-forgetting)
            if P.any():
                losses["head_normal"] = a.w_head_normal * head_normal_loss(sout["normal"][P], gt[P])
            ln = (sout["normal"][D].float() - tout["normal"][D].float()).abs().mean() if D.any() else torch.zeros((), device=img.device)
            _, lm = distill_normal_mask(None, None, sout["mask"][D], tout["mask"][D]) if (a.teacher_on_dropped_only and D.any()) else (distill_normal_mask(None, None, sout["mask"], tout["mask"]) if not a.teacher_on_dropped_only else (None, torch.zeros((), device=img.device)))
        elif a.teacher_on_dropped_only:
            # teacher outputs exist only on dropped/replay samples: distil normals and masks there only
            if D.any():
                ln, lm = distill_normal_mask(sout["normal"][D], tout["normal"][D], sout["mask"][D], tout["mask"][D])
            else:
                ln = lm = torch.zeros((), device=img.device)
        else:
            ln, lm = distill_normal_mask(sout["normal"], tout["normal"], sout["mask"], tout["mask"])
        losses["normal"] = a.w_normal * ln; losses["mask"] = a.w_mask * lm
        if a.w_exact_normal > 0:
            # synthetic samples carry exact renderer normals -> pixel-scale + 4x head supervision (prompted or not)
            S = batch["has_normal"].cuda()
            if S.any():
                losses["exact_normal"] = a.w_exact_normal * exact_normal_loss(sout["normal"][S], batch["normal"].cuda()[S])
        if bb_ref is not None:
            losses["l2sp"] = a.l2sp * sum(((p - p0) ** 2).sum() for p, p0 in bb_ref)
        loss = sum(losses.values())
        if prof is not None:
            t_now = _tick(); prof["losses"].append(t_now - t_prev); t_prev = t_now
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        if prof is not None:
            t_now = _tick(); prof["backward"].append(t_now - t_prev); t_prev = t_now
        scaler.unscale_(opt)
        gn = torch.nn.utils.clip_grad_norm_([p for g in groups for p in g["params"]], a.clip)
        if not torch.isfinite(loss) or not torch.isfinite(gn) or gn > a.gn_skip:
            n_skipped += 1
            opt.zero_grad(set_to_none=True); scaler.update()
        else:
            scaler.step(opt); scaler.update()
            if prof is not None:
                t_now = _tick(); prof['optim'].append(t_now - t_prev); t_prev = t_now
                if step >= a.profile_steps:
                    import numpy as _np
                    print('PROFILE ms/step (median over steps 6..N): ' + ', '.join(f'{k} {1000*_np.median(v[5:] if len(v) > 6 else v):.0f}' for k, v in prof.items()) + f' | total {1000*sum(_np.median(v[5:] if len(v) > 6 else v) for v in prof.values()):.0f}', flush=True); return
        if step % 20 == 0 or step == 1:
            with torch.no_grad():
                wn = model.prompt_stem.gate_norms()
                if hasattr(model, "prompt_neck"):
                    wn.update(model.prompt_neck.gate_norms())
                wn["logz_absmax"] = float(coord_s[..., 2].abs().max())
            rec = {"step": step, "loss": loss.item(), **{k: v.item() for k, v in losses.items()}, **mon, **wn,
                   "gn": gn.item(), "skipped": n_skipped, "lr": opt.param_groups[0]["lr"], "n_prompt": int(P.sum()), "refined": rs, "time_s": round(time.time() - t0)}
            log.write(json.dumps(rec) + "\n"); log.flush()
            print(json.dumps(rec), flush=True)
        if step % a.save_every == 0 or step == a.steps:
            save_ckpt(model, a, step, out / "latest.pt")
        if a.keep_every and step % a.keep_every == 0:
            save_ckpt(model, a, step, out / f"step_{step}.pt")
        if (step % a.eval_every == 0 or step == a.steps) and DEV_MANIFEST.exists():      # periodic evaluation needs the dev scenes (see data.py)
            model.eval()
            ev = quick_eval(model, teacher, stride=a.eval_stride, refine_steps=3, calib_lut=a.calib_input or DEFAULT_LUT, calibrate_input=True)
            model.train(); model.encoder.eval()
            ev["step"] = step
            (out / "eval_log.jsonl").open("a").write(json.dumps(ev) + "\n")
            print("EVAL", json.dumps(ev), flush=True)
    save_ckpt(model, a, a.steps, out / "final.pt")


if __name__ == "__main__":
    main()
