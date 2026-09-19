from typing import *
from numbers import Number
import warnings

import torch
import torch.nn.functional as F
try:
    import utils3d_moge as utils3d
except ImportError:
    import utils3d

from ..utils.geometry_torch import normalized_view_plane_uv, recover_focal_shift
from .v2 import MoGeModel as MoGeModelV2
from .modules.sparse_unet import Sparse3DUNet
from .modules.prompt_stem import PromptStem, PromptNeck, build_prompt, fit_scale_shift, add_lora_to_dinov2


class MoGeModel(MoGeModelV2):
    def __init__(
        self,
        encoder: Dict[str, Any],
        neck: Dict[str, Any],
        points_head: Dict[str, Any] = None,
        mask_head: Dict[str, Any] = None,
        normal_head: Dict[str, Any] = None,
        scale_head: Dict[str, Any] = None,
        num_tokens_range: List[int] = [1200, 3600],
        refiner: Optional[Dict[str, Any]] = None,
        refiner_depth_resolution: float = 256,
        lidar_prompt: Optional[Dict[str, Any]] = None,
        **deprecated_kwargs,
    ):
        """
        lidar_prompt (optional) enables early LiDAR fusion. Keys:
          - inject_blocks: List[int]  ViT block indices receiving gated prompt tokens (0 == patch-embedding level)
          - dim_hidden: int           prompt stem width (default 256)
          - neck: bool                also inject a gated prompt pyramid into the 5 neck levels
          - refiner_residual: bool    feed conf-gated log-residual (log d_lidar_aligned - logz_t) to the refiner
          - soft_residual_gate: bool  residual gate floor 0.15 at conf 0 instead of a hard zero
          - output_gate: bool         scale the refiner's Δlogz by per-voxel uncertainty (+ output_gate_floor, default 0.1)
          - anchor: float             elastic anchor strength beta in [0,1] applied after each refine step (0 = off)
          - metric_calib: bool        learned global (log_s, t) calibration after the LiDAR LS fit (zero-init)
          - lora_rank / lora_blocks   LoRA (rank r, zero-init) on attention qkv/proj of the listed ViT blocks
          - conf_head: bool           learned sensor-error head on the neck features
          - mono_prompt: bool         two-pass prompt (RGB-only prediction + sensor-vs-mono disagreement channels)
          - mono_prompt_mode: str     'aligned' (the only mode in this release)
        All prompt paths are zero-initialised: with lidar inputs absent OR at step 0 the model is bit-identical to stock MoGe-3.
        """
        super().__init__(
            encoder=encoder,
            neck=neck,
            points_head=points_head,
            mask_head=mask_head,
            normal_head=normal_head,
            scale_head=scale_head,
            remap_output='exp',
            num_tokens_range=num_tokens_range,
            **deprecated_kwargs,
        )

        self.lidar_prompt_cfg = dict(lidar_prompt) if lidar_prompt is not None else None
        if self.lidar_prompt_cfg is not None:
            cfg = self.lidar_prompt_cfg
            # The auxiliary heads may be SHALLOWER than the neck. `aux_head_levels = 4` stops both at the
            # neck's level 3, so they read exactly the tensors that Model A's 5-level and Model B's 4-level
            # ladders share (levels 0-3 are the same weights on the same inputs in both) and emit a 240x320
            # map in both. The sensor-error head's target -- |prompt - GT| / GT -- does not depend on the
            # model at all, so with a shared latent and a shared output grid ONE set of weights serves both.
            # (The model-error head is model-specific by construction and is only made architecturally
            # compatible, not shared.) Consumers upsample the map to the point-map resolution.
            _nl = len(neck['dim_res_blocks'])
            _al = int(cfg.get('aux_head_levels') or _nl)
            assert 2 <= _al <= _nl, f"aux_head_levels {_al} must be between 2 and the neck's {_nl}"
            _aux = dict(dim_in=list(neck['dim_res_blocks'])[:_al], dim_res_blocks=[256, 128, 64, 32, 16][:_al],
                        num_res_blocks=[0, 1, 1, 1, 0][:_al], res_block_in_norm='none', res_block_hidden_norm='none',
                        resamplers=['conv_transpose'] * (_al - 2) + ['bilinear'])
            def _bias_at(head, target_median):
                # Both heads emit 0.5 * sigmoid(x), which starts at 0.25 -- ~36x the median error they have to
                # predict (0.007 for the model's own error). Under an L1 loss the head then spends its budget
                # descending a constant-gradient slope instead of learning structure, and can overshoot into
                # saturation (measured: a head that converged to a flat 0.0000). Start it at the median.
                import math
                b = math.log(target_median / (0.5 - target_median))
                torch.nn.init.zeros_(head.output_blocks[-1].weight)
                torch.nn.init.constant_(head.output_blocks[-1].bias, b)
                return head

            if cfg.get('conf_head', False):
                # learned sensor-error head on the (LiDAR-informed) neck features: predicts the sensor's relative
                # depth error per pixel (sigmoid * 0.5); used as a learned confidence for the refiner gate / anchor.
                from .modules.conv_stack import ConvStack
                self.conf_head = _bias_at(ConvStack(dim_out=[None] * (_al - 1) + [1], **_aux),
                                          float(cfg.get('conf_head_prior', 0.020)))
            if cfg.get('err_head', False):
                # learned MODEL-error head on the neck features: predicts the model's own relative depth error per pixel
                # (sigmoid * 0.5); exported for uncertainty-weighted fusion (not used inside the network).
                from .modules.conv_stack import ConvStack
                self.err_head = _bias_at(ConvStack(dim_out=[None] * (_al - 1) + [1], **_aux),
                                         float(cfg.get('err_head_prior', 0.007)))
            self.mono_prompt = bool(cfg.get('mono_prompt', False))     # two-pass, RGB-only prediction as prompt channels
            self.mono_prompt_mode = cfg.get('mono_prompt_mode', 'aligned')   # 'aligned' | 'poisson' (replace) | 'both' (+1 channel)
            self.conf_phase = bool(cfg.get('conf_phase', False))
            n_prompt_ch = ((7 if self.mono_prompt_mode == 'both' else 6) if self.mono_prompt else 4) + (2 if self.conf_phase else 0)
            self.prompt_stem = PromptStem(
                in_channels=n_prompt_ch,
                dim_out=self.encoder.dim_features,
                dim_hidden=cfg.get('dim_hidden', 256),
                inject_blocks=cfg.get('inject_blocks', [0]),
            )
            if cfg.get('neck', False):
                # `neck_hidden` (coarse -> fine, like dims) re-schedules the prompt pyramid's widths; the stock
                # (16, 32, 64, 128, 256) spends 86 % of the module's FLOPs at the finest level where the neck is
                # narrowest. None keeps the stock schedule.
                self.prompt_neck = PromptNeck(dims=neck['dim_res_blocks'], in_channels=n_prompt_ch,
                                              **({'hidden': tuple(cfg['neck_hidden'])} if cfg.get('neck_hidden') else {}))
            self.anchor_strength = float(cfg.get('anchor', 0.0))
            self.refiner_output_gate = bool(cfg.get('output_gate', False))
            self.output_gate_floor = float(cfg.get('output_gate_floor', 0.1))
            if cfg.get('metric_calib', False):
                # learned global calibration applied after the LiDAR LS fit (zero-init == identity):
                #   depth = exp(log_s) * (s*z + t) + t_c
                self.prompt_calib = torch.nn.Parameter(torch.zeros(2))
            # LoRA on the DINOv2 attention is applied AFTER the pretrained weights are loaded -- see `apply_lora` /
            # `from_pretrained`; wrapping in the constructor would rename the attention keys and leave them random.

        if refiner is not None:
            refiner_cfg = dict(refiner)
            lp = self.lidar_prompt_cfg or {}
            if lp.get('refiner_residual', False):
                refiner_cfg['prompt_channels'] = 2
            # narrow the coarsest refiner level (down4 + bottleneck cost 12.7 ms for 0.8 % of the
            # voxels). `refiner_channels` overrides model_channels, e.g. [32, 64, 128, 256, 256].
            if lp.get('refiner_channels'):
                refiner_cfg['model_channels'] = list(lp['refiner_channels'])
            # fold the 2 UV planes out of the refiner's conditioning GEMM so enc_channels is 1024.
            self.refiner_uv_fold = bool(lp.get('refiner_uv_fold', False))
            if self.refiner_uv_fold:
                refiner_cfg['encoder_channels'] = int(refiner_cfg['encoder_channels']) - 2
            self.refiner_depth_resolution = refiner_depth_resolution
            self.refiner = Sparse3DUNet(**refiner_cfg)
            if self.refiner_uv_fold:
                self.refiner_uv_proj = torch.nn.Conv2d(2, refiner_cfg['model_channels'][-1], 1, bias=False)
                torch.nn.init.zeros_(self.refiner_uv_proj.weight)
        else:
            warnings.warn("Warning: refiner is not enabled.")

    @property
    def has_lidar_prompt(self) -> bool:
        return self.lidar_prompt_cfg is not None

    @staticmethod
    def prompt_gauge_poly(depth: torch.Tensor, lidar: torch.Tensor, conf: Optional[torch.Tensor], degree: int = 2, iters: int = 3) -> torch.Tensor:
        """[B,H,W] multiplicative correction exp(poly(u,v)) fitted per frame to log(prompt / prediction) on confident prompt pixels."""
        B, H, W = depth.shape; h, w = lidar.shape[-2:]
        ok = torch.isfinite(depth) & (depth > 0)                      # masked pixels may hold inf; average only over finite depth
        num = F.interpolate(torch.where(ok, depth, torch.zeros_like(depth)).unsqueeze(1), (h, w), mode='area').squeeze(1)
        den = F.interpolate(ok.float().unsqueeze(1), (h, w), mode='area').squeeze(1)
        z_lr = torch.where(den > 0.5, num / den.clamp_min(1e-6), torch.zeros_like(num))
        ld = lidar.view(B, h, w); cf = conf.view(B, h, w) if conf is not None else torch.full_like(ld, 2.0)
        u = torch.linspace(-1, 1, w, device=depth.device); v = torch.linspace(-1, 1, h, device=depth.device)
        vv, uu = torch.meshgrid(v, u, indexing='ij')
        def basis(uu, vv):
            cols = [torch.ones_like(uu), uu, vv]
            if degree >= 2:
                cols += [uu * uu, uu * vv, vv * vv]
            return torch.stack(cols, -1)
        Bm = basis(uu, vv).view(-1, 3 if degree < 2 else 6)
        out = torch.ones_like(depth)
        for b in range(B):
            m = ((cf[b] >= 2) & (ld[b] > 0) & (z_lr[b] > 0)).view(-1)
            if m.sum() < 200:
                continue
            A = Bm[m]; y = (torch.log(ld[b].clamp_min(1e-3)) - torch.log(z_lr[b].clamp_min(1e-3))).view(-1)[m]
            wgt = torch.ones_like(y)
            for _ in range(iters):
                coef = torch.linalg.lstsq(A * wgt[:, None], (y * wgt)[:, None]).solution[:, 0]
                r = y - A @ coef; sc = 1.4826 * r.abs().median() + 1e-4; wgt = torch.clamp(2 * sc / r.abs().clamp_min(1e-6), max=1.0)
            U = torch.linspace(-1, 1, W, device=depth.device); V = torch.linspace(-1, 1, H, device=depth.device)
            VV, UU = torch.meshgrid(V, U, indexing='ij')
            if not torch.isfinite(coef).all():
                continue
            out[b] = torch.exp(basis(UU, VV) @ coef)
        return out

    @staticmethod
    def apply_lidar_lut(lidar_depth: torch.Tensor, lut: Dict[str, Any]) -> torch.Tensor:
        """Input-side sensor calibration: depth *= exp(interp(log depth)) on valid pixels (piecewise-linear in log depth)."""
        xs = torch.as_tensor(lut['log_depth'], device=lidar_depth.device, dtype=torch.float32)
        ys = torch.as_tensor(lut['log_ratio'], device=lidar_depth.device, dtype=torch.float32)
        valid = torch.isfinite(lidar_depth) & (lidar_depth > 0)
        ld = torch.log(lidar_depth.clamp_min(1e-3))
        idx = torch.bucketize(ld, xs).clamp(1, len(xs) - 1)
        x0, x1, y0, y1 = xs[idx - 1], xs[idx], ys[idx - 1], ys[idx]
        w = ((ld - x0) / (x1 - x0)).clamp(0, 1)
        return torch.where(valid, lidar_depth * torch.exp(y0 + w * (y1 - y0)), lidar_depth)

    def load_trainable_state(self, sd: Dict[str, torch.Tensor]):
        """Load a fine-tuned (partial) state dict; a 4-channel prompt stem / neck pyramid is widened to 6/7 channels with
        the new input channels zero-initialised (bit-identical behaviour at load); a wider checkpoint is narrowed by
        dropping its extra (mono) input channels."""
        sd = dict(sd)
        for k, mod in (('prompt_stem.stem.0.weight', getattr(self, 'prompt_stem', None)), ('prompt_neck.enc0.0.weight', getattr(self, 'prompt_neck', None))):
            if k in sd and mod is not None:
                w_new = mod.stem[0].weight if k.startswith('prompt_stem') else mod.enc0[0].weight
                if sd[k].shape[1] < w_new.shape[1]:
                    w = torch.zeros_like(w_new.detach().cpu()); w[:, :sd[k].shape[1]] = sd[k]; sd[k] = w
                elif sd[k].shape[1] > w_new.shape[1]:
                    # single-pass warm start from a two-pass checkpoint: keep the sensor channels, drop the mono channels
                    sd[k] = sd[k][:, :w_new.shape[1]].clone()
        # Drop tensors whose shape no longer matches the module (a re-scheduled prompt neck, a narrowed refiner):
        # load_state_dict raises on a size mismatch even with strict=False, and those tensors are meant to be
        # re-initialised, not transferred. The dropped keys are reported so a silent partial load is impossible.
        own = self.state_dict()
        # QAT parametrisation renames `X.weight` to `X.parametrizations.weight.original`; translate in both
        # directions so a QAT model can warm-start from a plain checkpoint and vice versa.
        PAR = ".parametrizations.weight.original"
        for k in list(sd):
            if k.endswith(".weight") and k not in own and k[:-len(".weight")] + PAR in own:
                sd[k[:-len(".weight")] + PAR] = sd.pop(k)
            elif k.endswith(PAR) and k not in own and k[:-len(PAR)] + ".weight" in own:
                sd[k[:-len(PAR)] + ".weight"] = sd.pop(k)
        dropped = [k for k, v in sd.items() if k in own and tuple(own[k].shape) != tuple(v.shape)]
        if dropped:
            # A re-initialised gated injection path must also drop its GATES, even though those are shape-compatible:
            # the gate scales a projection that is now random, so keeping a trained gate multiplies noise into the
            # frozen backbone / neck. Zero gates are the module's designed starting point (exactly-zero injection).
            for mod_prefix in ('prompt_neck', 'prompt_stem'):
                if any(k.startswith(mod_prefix + '.') for k in dropped):
                    dropped += [k for k in sd if k.startswith(mod_prefix + '.gates') and k not in dropped]
            warnings.warn(f"load_trainable_state: {len(dropped)} tensor(s) dropped (shape mismatch, and the gates of "
                          f"any re-initialised injection path): {dropped[:8]}" + (" ..." if len(dropped) > 8 else ""))
            sd = {k: v for k, v in sd.items() if k not in dropped}
        self._dropped_on_load = dropped
        return self.load_state_dict(sd, strict=False)

    def apply_lora(self):
        """Wrap DINOv2 attention qkv/proj of the configured blocks with zero-init LoRA (idempotent). Call after the
        pretrained weights are loaded and before loading a fine-tuned state dict that contains `lora_` keys."""
        cfg = self.lidar_prompt_cfg or {}
        if cfg.get('lora_rank', 0) > 0 and not getattr(self, '_lora_params', None):
            self._lora_params = add_lora_to_dinov2(self.encoder.backbone, cfg.get('lora_blocks', list(range(12))),
                                                   r=int(cfg['lora_rank']), alpha=float(cfg.get('lora_alpha', cfg['lora_rank'])))
        return self

    def attach_refiner_teacher(self, state_dict: Dict[str, torch.Tensor], refiner_cfg: Dict[str, Any]):
        """Attach a frozen copy of a reference refiner (the uncompressed one) for compression distillation.

        Narrowing level 4 and quantising to int8 are *compression* changes: the objective that matches them is
        "reproduce what the original refiner did", not "fit the GT again". The real protocol's GT is edge-band
        invalidated (`--real_edge_band 3`), so it carries no boundary supervision at all, and the refiner's whole
        measured value is boundary sharpness (edge F1 0.530 -> 0.562) -- fitting GT alone would let a compressed
        refiner smooth its way to a good AbsRel while losing exactly what it is for.
        """
        cfg = dict(refiner_cfg)
        cfg['prompt_channels'] = 2 if (self.lidar_prompt_cfg or {}).get('refiner_residual', False) else 0
        self.refiner_teacher = Sparse3DUNet(**cfg)
        sd = {k[len('refiner.'):]: v for k, v in state_dict.items() if k.startswith('refiner.')}
        missing, unexpected = self.refiner_teacher.load_state_dict(sd, strict=False)
        assert not [m for m in missing if 'prompt_proj' not in m], f"refiner teacher missing {missing[:6]}"
        self.refiner_teacher.requires_grad_(False)
        self.refiner_teacher.eval()
        return self

    def apply_qat(self):
        """Wrap the refiner's wide GEMMs with the deployment's exact int8 fake quantisation.

        Idempotent, and called AFTER the pretrained weights are loaded: `torch.nn.utils.parametrize` moves
        `weight` to `parametrizations.weight.original`, so registering it earlier would leave the refiner
        holding its random initialisation."""
        cfg = self.lidar_prompt_cfg or {}
        if cfg.get('refiner_qat', False) and hasattr(self, 'refiner') and not getattr(self.refiner, '_qat_layers', None):
            from .modules.qat import enable_refiner_qat
            n = enable_refiner_qat(self.refiner, pct=float(cfg.get('qat_pct', 0.9995)))
            self._qat_layers = n
        return self

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, model_kwargs=None, **hf_kwargs):
        model = super().from_pretrained(pretrained_model_name_or_path, model_kwargs=model_kwargs, **hf_kwargs)
        return model.apply_lora().apply_qat()

    def prompt_parameters(self):
        """Parameters introduced by the LiDAR prompt path (stem, neck pyramid, conf head, calib, LoRA, refiner prompt projection)."""
        mods = [m for m in [getattr(self, 'prompt_stem', None), getattr(self, 'prompt_neck', None), getattr(self, 'conf_head', None), getattr(self, 'err_head', None)] if m is not None]
        params = [p for m in mods for p in m.parameters()]
        if hasattr(self, 'prompt_calib'):
            params.append(self.prompt_calib)
        params += list(getattr(self, '_lora_params', []))
        if hasattr(self, 'refiner') and getattr(self.refiner, 'prompt_channels', 0) > 0:
            params += list(self.refiner.prompt_proj.parameters())
        # `refiner_uv_proj` belongs to the refiner's parameter group, not the prompt path (it is the UV half of
        # `encoder_fuse`); listing it here too would put it in two optimiser groups.
        return params

    def init_weights(self):
        super().init_weights()
        if hasattr(self, 'refiner'):
            self.refiner.init_weights()

    def enable_gradient_checkpointing(self):
        super().enable_gradient_checkpointing()
        if hasattr(self, 'refiner'):
            self.refiner.enable_gradient_checkpointing()

    def _voxelize(
        self,
        point_coord: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Size, torch.Tensor]:
        """
        Convert dense point coordinates to a sparse representation.

        - point_coord: [B, H, W, 3] at (x/z, y/z, logz).

        Returns (feats, coords, shape, logz):
        - feats:  (M, 3) fp32 input features [uv, logz].
        - coords: (M, 4) int32, columns (batch, i, j, z_bin).
        - shape:  Size([B, H, W, z_extent, in_channels]).
        - logz:   [B, H, W] dense fp32 log-depth (for the residual update).

        Everything here is forced to fp32: the z binning is quantized at
        1/refiner_depth_resolution, which is finer than fp16 resolution over the
        usual logz range, so binning in fp16 would collapse/jitter voxels.
        """
        if point_coord.ndim != 4 or point_coord.shape[-1] != 3:
            raise ValueError(f"point_coord must be [B, H, W, 3], got {point_coord.shape}")

        point_coord = point_coord.float()
        bsz, height, width, _ = point_coord.shape
        device = point_coord.device

        logz = point_coord[..., 2]
        zq = torch.round(logz * self.refiner_depth_resolution).long()
        z_offset = zq.amin(dim=(1, 2), keepdim=True)
        z_idx = zq - z_offset
        z_extent = z_idx.amax().item() + 1

        i = torch.arange(height, device=device, dtype=torch.long).view(1, height, 1).expand(bsz, height, width)
        j = torch.arange(width, device=device, dtype=torch.long).view(1, 1, width).expand(bsz, height, width)
        batch = torch.arange(bsz, device=device, dtype=torch.long).view(bsz, 1, 1).expand(bsz, height, width)
        coords = torch.stack([batch, i, j, z_idx], dim=-1).reshape(-1, 4).to(torch.int32)

        feats = point_coord.reshape(-1, 3)
        shape = torch.Size([bsz, height, width, z_extent, feats.shape[-1]])
        return feats, coords, shape, logz

    def _refine_logz(
        self,
        point_coord: torch.Tensor,
        encoder_feature: torch.Tensor,
        prompt_feats: Optional[torch.Tensor] = None,
        encoder_bias_map: Optional[torch.Tensor] = None,
        refiner: Optional[torch.nn.Module] = None,
    ) -> torch.Tensor:
        bsz, height, width, _ = point_coord.shape
        feats, coords, shape, logz = self._voxelize(point_coord)
        out: torch.Tensor = (refiner or self.refiner)(feats, coords, shape, encoder_feature, prompt_feats=prompt_feats,
                                                      encoder_bias_map=encoder_bias_map)
        out_logz = out.float().squeeze(-1).reshape(bsz, height, width)
        refined_logz = logz + out_logz
        return refined_logz

    @staticmethod
    def _lidar_in_affine_frame(logz: torch.Tensor, lidar_depth: torch.Tensor, lidar_conf: torch.Tensor, soft_gate: bool = False):
        """
        Align the metric LiDAR to the network's affine log-depth frame with a robust global (s, t):
            lidar ~= s * exp(logz) + t   (fit on confident pixels, detached)
        Returns (log_lidar_affine [B,H,W], gate [B,H,W] in [0,1] (conf-weighted validity), s [B], t [B]).
        The uncertainty used by the refiner output gate is available via `_lidar_uncertainty`.
        """
        bsz, height, width = logz.shape
        d = F.interpolate(lidar_depth.float(), (height, width), mode='nearest').squeeze(1)
        c = (F.interpolate(lidar_conf.float(), (height, width), mode='nearest').squeeze(1) / 2.0).clamp(0, 1)
        valid = torch.isfinite(d) & (d > 0)
        z = torch.exp(logz.float())
        s, t = fit_scale_shift(z.flatten(1), d.flatten(1), (valid & (c >= 0.999)).flatten(1))
        d_aff = (d - t.view(-1, 1, 1)) / s.view(-1, 1, 1).clamp_min(1e-6)
        ok = valid & (d_aff > 1e-4)
        log_d_aff = torch.where(ok, torch.log(d_aff.clamp_min(1e-4)), logz.float())
        gate = c * ok.float()
        if soft_gate:
            # confidence is a heuristic: keep a floor so the refiner can learn to use conf-0 depth where it is right
            gate = (0.15 + 0.85 * c) * ok.float()
        return log_d_aff, gate, s, t

    @staticmethod
    def _lidar_uncertainty(hw, lidar_depth: torch.Tensor, lidar_conf: torch.Tensor) -> torch.Tensor:
        """Per-pixel uncertainty u = 1 - conf·(1 - range falloff(3→5 m)); 1 where the sensor has no value. [B,H,W]."""
        from .modules.prompt_stem import range_uncertainty
        d = F.interpolate(lidar_depth.float(), hw, mode='nearest').squeeze(1)
        c = (F.interpolate(lidar_conf.float(), hw, mode='nearest').squeeze(1) / 2.0).clamp(0, 1)
        valid = torch.isfinite(d) & (d > 0)
        u = 1.0 - c * (1.0 - range_uncertainty(d))
        return torch.where(valid, u, torch.ones_like(u))

    def forward(
        self,
        image: torch.Tensor,
        num_tokens: Union[int, torch.LongTensor],
        refine_steps: int = 3,
        refiner_detach_backbone: bool = True,
        return_per_step: bool = False,
        lidar_depth: Optional[torch.Tensor] = None,
        lidar_conf: Optional[torch.Tensor] = None,
        anchor_strength: Optional[float] = None,
        use_prompt: bool = True,
        use_learned_conf: bool = False,
        lidar_lut: Optional[Dict[str, Any]] = None,
        mono_z_override: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        lidar_depth: [B,1,h,w] metric depth (<=0 == missing); lidar_conf: [B,1,h,w] ARKit confidence in {0,1,2}.
        When lidar_depth is None the forward is exactly the stock MoGe-3 forward (prompt path skipped).
        use_prompt=False keeps the 2D prompt path off but still allows the refiner-side LiDAR terms (refiner residual, elastic anchor).
        lidar_lut: optional sensor calibration applied to lidar_depth at entry (input-side calibration).
        """
        if refine_steps > 0 and not hasattr(self, 'refiner'):
            raise ValueError("Refiner is not enabled but refine_steps > 0.")

        batch_size, _, img_h, img_w = image.shape
        device, dtype = image.device, image.dtype

        aspect_ratio = img_w / img_h
        base_h, base_w = (num_tokens / aspect_ratio) ** 0.5, (num_tokens * aspect_ratio) ** 0.5
        if isinstance(base_h, torch.Tensor):
            base_h, base_w = base_h.round().long(), base_w.round().long()
        else:
            base_h, base_w = round(base_h), round(base_w)

        # LiDAR prompt (early fusion): build normalised prompt map -> stem -> gated token injections
        has_lidar = lidar_depth is not None
        if has_lidar and lidar_conf is None:
            lidar_conf = torch.full_like(lidar_depth, 2.0)
        if has_lidar and lidar_lut is not None:
            lidar_depth = self.apply_lidar_lut(lidar_depth.float(), lidar_lut)   # calibrate the sensor before anything sees it
        use_prompt = has_lidar and use_prompt and self.has_lidar_prompt
        token_injections, neck_injections, prompt_stats = None, None, None
        if use_prompt:
            mono_z = None
            if getattr(self, 'mono_prompt', False):
                # the model's own RGB-only pass (prompt skipped, no refinement, no grad) supplies the monocular structure.
                # Streaming deployment: `mono_z_override` (e.g. the previous frame's prediction warped by the pose) replaces
                # that pass -- no second backbone run; build_prompt aligns it affinely to the sensor either way.
                with torch.no_grad():
                    if mono_z_override is not None:
                        mono_z = mono_z_override.float()
                    else:
                        mono_out = self.forward(image, num_tokens=num_tokens, refine_steps=0, use_prompt=False)
                        mono_z = mono_out['points'][..., 2].unsqueeze(1).float()
                    prefill = None
                    if getattr(self, 'mono_prompt_mode', 'aligned') != 'aligned':
                        raise NotImplementedError("only mono_prompt_mode='aligned' is part of this release")
            else:
                prefill = None
            prompt, prompt_stats = build_prompt(lidar_depth, lidar_conf, (base_h * 14, base_w * 14), mono_z=mono_z, prefill=prefill, conf_phase=getattr(self, 'conf_phase', False))
            prompt = prompt.to(dtype)
            _, token_injections = self.prompt_stem(prompt)
            if hasattr(self, 'prompt_neck'):
                # the pyramid's finest level is the neck's finest level: token_grid x 2**(levels-1)
                # (x16 -> 480x640 for the 5-level ladder, x8 -> 240x320 for Model B's 4-level one)
                f = 2 ** (len(self.neck.res_blocks) - 1)
                prompt_fine = F.interpolate(prompt, (base_h * f, base_w * f), mode='nearest')
                neck_injections = self.prompt_neck(prompt_fine)
            # Spatial prompt gate. `prompt_gate_map` is a (B, 1, gh, gw) map in [0, 1] on (or coarser
            # than) the token grid; it scales every injection -- the token injections at blocks 0/4/8 and each
            # level of the neck pyramid -- so the prompt can be trusted in one part of the frame and ignored in
            # another. `None` (the default) is exactly the ungated model. A uniform map of ones is bit-identical
            # to it, and a uniform map of tau reproduces `set_prompt_trust(tau)`, so this generalises both.
            gm = getattr(self, 'prompt_gate_map', None)
            if gm is not None:
                gm = gm.to(dtype=dtype, device=device)
                if token_injections is not None:
                    g_tok = F.interpolate(gm, (base_h, base_w), mode='bilinear', align_corners=False)
                    g_tok = g_tok.flatten(2).transpose(1, 2)                      # (B, N, 1)
                    token_injections = {b: t * g_tok for b, t in token_injections.items()}
                if neck_injections is not None:
                    neck_injections = [
                        inj if inj is None else inj * F.interpolate(gm, inj.shape[-2:], mode='bilinear', align_corners=False)
                        for inj in neck_injections
                    ]
        beta = (getattr(self, 'anchor_strength', 0.0) if anchor_strength is None else anchor_strength) if has_lidar else 0.0
        use_refiner_residual = has_lidar and hasattr(self, 'refiner') and getattr(self.refiner, 'prompt_channels', 0) > 0

        # Backbones encoding
        features, cls_token = self.encoder(image, base_h, base_w, return_class_token=True, token_injections=token_injections)
        encoder_tokens = features                       # raw backbone features (for feature-level anchoring / distillation)
        # The neck's depth is the model's output-resolution ladder: 5 levels put the point map at
        # token_grid x 16 (480x640 at 1200 tokens), 4 levels at token_grid x 8 (240x320). Everything below
        # follows `len(self.neck.res_blocks)` so both configurations use the same code path.
        n_levels = len(self.neck.res_blocks)
        features = [features] + [None] * (n_levels - 1)

        # Concat UVs for aspect ratio input
        for level in range(n_levels):
            uv = normalized_view_plane_uv(width=base_w * 2 ** level, height=base_h * 2 ** level, aspect_ratio=aspect_ratio, dtype=dtype, device=device)
            uv = uv.permute(2, 0, 1).unsqueeze(0).expand(batch_size, -1, -1, -1)
            if features[level] is None:
                features[level] = uv
            else:
                features[level] = torch.concat([features[level], uv], dim=1)

        # Shared neck (optionally with gated prompt pyramid injections)
        neck_features = self.neck(features, injections=neck_injections)

        # Heads decoding
        raw_coord = self.points_head(neck_features)[-1] if hasattr(self, 'points_head') else None
        # A confidence head is a READ-OUT of the representation, not a shaper of it. Left attached, a randomly
        # initialised error head whose loss opens ~20x above every other term drags the shared neck with it
        # (measured: holes +119 %, low-conf +60 % on the arm that trained them attached). Detaching costs the
        # heads nothing -- they still see the same features -- and protects the depth path.
        aux_in = [f.detach() for f in neck_features] if (self.lidar_prompt_cfg or {}).get('aux_heads_detached', False) else neck_features
        sensor_err = None
        # `conf_head` only feeds `use_learned_conf`, which the deployment leaves off, so at
        # deployment it is 18.41 GFLOP (~1.9 ms on the ANE) computed and discarded. `deploy_drop_conf_head`
        # skips it; nothing else in the graph reads `sensor_err`.
        if hasattr(self, 'conf_head') and use_prompt and not (self.lidar_prompt_cfg or {}).get('deploy_drop_conf_head', False):
            sensor_err = 0.5 * torch.sigmoid(self.conf_head(aux_in)[-1].float().squeeze(1))   # [B,H,W] predicted |d_lidar-gt|/gt
            if use_learned_conf:
                # learned confidence replaces ARKit's for the refiner-side terms: 2 * (1 - err/0.15) clamped to [0,2]
                lidar_conf = F.interpolate((2.0 * (1.0 - sensor_err.detach() / 0.15).clamp(0, 1)).unsqueeze(1), lidar_depth.shape[-2:], mode='area')
        normal, mask = (
            getattr(self, head)(neck_features)[-1] if hasattr(self, head) else None
            for head in ['normal_head', 'mask_head']
        )
        model_err = 0.5 * torch.sigmoid(self.err_head(aux_in)[-1].float().squeeze(1)) if hasattr(self, 'err_head') else None   # [B,H,W]
        metric_scale = self.scale_head(cls_token) if hasattr(self, 'scale_head') else None

        # Refine point map in factorized coordinate space
        coord_per_step: List[torch.Tensor] = []
        coords: Optional[torch.Tensor] = None
        points: Optional[torch.Tensor] = None
        points_per_step: Optional[List[torch.Tensor]] = None
        if raw_coord is not None: # raw_coord is B3HW at (x/z, y/z, logz)
            current_coord = raw_coord.permute(0, 2, 3, 1).float() # BHW3 at (x/z, y/z, logz)
            if return_per_step:
                coord_per_step.append(current_coord)

            if refine_steps > 0:
                refiner_feature: torch.Tensor = features[0]
                uv_bias_map = None
                refiner_deltas: List[Tuple[torch.Tensor, torch.Tensor]] = []
                if getattr(self, 'refiner_uv_fold', False):
                    # `features[0]` is [encoder tokens | 2 UV planes]; the UV half of encoder_fuse is a
                    # fixed per-position map, so it is computed once here and added inside the refiner.
                    uv_bias_map = self.refiner_uv_proj(refiner_feature[:, -2:].float()).to(refiner_feature.dtype)
                    refiner_feature = refiner_feature[:, :-2]

                for _ in range(refine_steps):
                    feature_for_refiner = refiner_feature.detach() if refiner_detach_backbone else refiner_feature
                    prompt_feats = None
                    if use_refiner_residual:
                        # confidence-gated signed residual between LiDAR (aligned into the affine frame) and current logz
                        log_d_aff, gate, _, _ = self._lidar_in_affine_frame(current_coord[..., 2].detach(), lidar_depth, lidar_conf,
                                                                            soft_gate=bool((self.lidar_prompt_cfg or {}).get('soft_residual_gate', False)))
                        residual = (log_d_aff - current_coord[..., 2].detach()) * gate
                        prompt_feats = torch.stack([residual, gate], dim=-1).reshape(-1, 2)
                    refined_logz = self._refine_logz(current_coord.detach(), feature_for_refiner, prompt_feats=prompt_feats,
                                                     encoder_bias_map=uv_bias_map)
                    if getattr(self, 'refiner_teacher', None) is not None:
                        # same input coords and conditioning: the difference is exactly the compression error
                        with torch.no_grad():
                            t_logz = self._refine_logz(current_coord.detach(), features[0].detach(),
                                                       prompt_feats=prompt_feats, refiner=self.refiner_teacher)
                        refiner_deltas.append((refined_logz - current_coord[..., 2].detach().float(),
                                               t_logz - current_coord[..., 2].detach().float()))
                    if has_lidar and getattr(self, 'refiner_output_gate', False):
                        # confident close-range voxels are only lightly refined; holes / low-conf / far get the full update
                        u = self._lidar_uncertainty(refined_logz.shape[-2:], lidar_depth, lidar_conf)
                        g = self.output_gate_floor + (1.0 - self.output_gate_floor) * u
                        refined_logz = current_coord[..., 2].detach().float() + g * (refined_logz - current_coord[..., 2].detach().float())
                    if has_lidar and beta > 0:
                        # elastic anchor -- relax confident voxels toward the (re-aligned) LiDAR, free elsewhere
                        log_d_aff, gate, _, _ = self._lidar_in_affine_frame(refined_logz.detach(), lidar_depth, lidar_conf)
                        refined_logz = refined_logz + beta * gate * (log_d_aff - refined_logz)
                    current_coord = torch.cat([current_coord[..., :2], refined_logz.unsqueeze(-1)], dim=-1)
                    if return_per_step:
                        coord_per_step.append(current_coord)

            coords = torch.stack(coord_per_step, dim=1) if return_per_step else current_coord.unsqueeze(1)

        # Resize and remap outputs
        resize = lambda x, channel_last=False: F.interpolate(
            x.movedim(-1, -3) if channel_last else x,
            (img_h, img_w),
            mode='bilinear',
            align_corners=False,
            antialias=False,
        ).movedim(-3, -1 if channel_last else -3)

        if coords is not None:
            num_point_steps = coords.shape[1]
            coords = resize(coords.flatten(0, 1), channel_last=True)
            coords = coords.unflatten(0, (batch_size, num_point_steps))
            points_all = self._remap_points(coords)
            points = points_all[:, -1]
            if return_per_step:
                points_per_step = list(points_all.unbind(dim=1))

        if normal is not None:
            normal = resize(normal)
            normal = normal.permute(0, 2, 3, 1)
            normal = F.normalize(normal, dim=-1)
        if mask is not None:
            mask = resize(mask)
            mask = mask.squeeze(1).sigmoid()
        if metric_scale is not None:
            metric_scale = metric_scale.squeeze(1).exp()

        return_dict = {
            'points': points,
            'points_per_step': points_per_step,
            'normal': normal,
            'mask': mask,
            'metric_scale': metric_scale,
            'prompt_mu': prompt_stats['mu'] if prompt_stats is not None else None,
            'sensor_err': sensor_err,
            'model_err': model_err,
            'encoder_tokens': encoder_tokens,
            'refiner_deltas': refiner_deltas if (raw_coord is not None and refine_steps > 0 and getattr(self, 'refiner_teacher', None) is not None) else None,
        }
        return_dict = {k: v for k, v in return_dict.items() if v is not None}

        return return_dict

    @torch.inference_mode()
    def infer(
        self,
        image: torch.Tensor,
        num_tokens: int = None,
        resolution_level: int = 9,
        force_projection: bool = True,
        apply_mask: bool = True,
        fov_x: Optional[Union[Number, torch.Tensor]] = None,
        refine_steps: int = 3,
        return_per_step: bool = False,
        use_fp16: bool = False,
        lidar_depth: Optional[torch.Tensor] = None,
        lidar_conf: Optional[torch.Tensor] = None,
        metric_from: Literal['cls', 'lidar_ls', 'prompt_gauge'] = 'lidar_ls',
        anchor_strength: Optional[float] = None,
        use_prompt: bool = True,
        use_learned_conf: bool = False,
        mono_z_override: Optional[torch.Tensor] = None,
        calibration_lut: Optional[Dict[str, Any]] = None,
        calibrate_input: bool = False,
        gauge_poly: int = 0,
    ) -> Dict[str, torch.Tensor]:
        """
        User-friendly inference function

        LiDAR extensions:
        - `lidar_depth` [B,1,h,w] or [1,h,w] metres (<=0 missing), `lidar_conf` same shape in {0,1,2}.
        - `metric_from`: how metric depth is recovered when lidar is given:
            'cls'          stock: exp(scale_head(cls)) global scale.
            'lidar_ls'     robust global scale + z-shift fitted to confident LiDAR (2-DoF, no GT).
            'prompt_gauge' depth = exp(logz + mu) where mu is the prompt's log-depth centre (no shift recovery).
        - `use_prompt=False` scores the RGB-only path while still using lidar for LS anchoring.
        - `calibration_lut`: optional {'log_depth': [...], 'log_ratio': [...]} sensor-to-reference calibration (fitted once on
          training data). With `calibrate_input=True` it is applied to the LiDAR *input* (prompt, fits, refiner) -- the
          preferred deployment; otherwise it is applied as the final post-processing step to the metric depth / points.
          The returned depth is then the deployed end result.

        ### Parameters
        - `image`: input image tensor of shape (B, 3, H, W) or (3, H, W).
        - `num_tokens`: the number of base ViT tokens to use for inference. If None, it is determined by `resolution_level`.
        - `resolution_level`: inference resolution level from 0 to 9. Higher values use more tokens and preserve finer details. Default: 9.
        - `force_projection`: if True, recompute each point map from its depth map and intrinsics. Default: True.
        - `apply_mask`: if True, mask invalid points and depths using the predicted mask. Default: True.
        - `fov_x`: horizontal camera field of view in degrees. If None, it is inferred from each returned point map. Default: None.
        - `refine_steps`: number of sparse 3D refinement updates. Default: 3.
        - `return_per_step`: if True, return predictions for the initial estimate and every refinement step. Default: False.
        - `use_fp16`: if True, use mixed precision to speed up inference. Default: False.

        ### Returns
        A dictionary containing the following keys when the corresponding outputs are available:
        - `points`: final camera-space point map of shape (B, H, W, 3) or (H, W, 3).
        - `points_per_step`: point maps for the initial prediction and every refinement step, when `return_per_step=True`.
        - `intrinsics`: camera intrinsics associated with the final point map, of shape (B, 3, 3) or (3, 3).
        - `intrinsics_per_step`: camera intrinsics for the initial prediction and every refinement step, when `return_per_step=True`.
        - `depth`: final depth map of shape (B, H, W) or (H, W).
        - `depth_per_step`: depth maps for the initial prediction and every refinement step, when `return_per_step=True`.
        - `mask`: predicted valid-pixel mask of shape (B, H, W) or (H, W).
        - `normal`: predicted normal map of shape (B, H, W, 3) or (H, W, 3).
        - `metric_fit`: (B, 2) the (scale, shift) used to make the final depth metric, when lidar is given.
        """
        if refine_steps > 0 and not hasattr(self, 'refiner'):
            raise ValueError("Refiner is not enabled but refine_steps > 0.")

        if image.dim() == 3:
            omit_batch_dim = True
            image = image.unsqueeze(0)
        else:
            omit_batch_dim = False
        image = image.to(dtype=self.dtype, device=self.device)
        if lidar_depth is not None:
            if lidar_depth.dim() == 3:
                lidar_depth = lidar_depth.unsqueeze(0)
            lidar_depth = lidar_depth.to(device=self.device, dtype=torch.float32)
            if lidar_conf is None:
                lidar_conf = torch.full_like(lidar_depth, 2.0)
            if lidar_conf.dim() == 3:
                lidar_conf = lidar_conf.unsqueeze(0)
            lidar_conf = lidar_conf.to(device=self.device, dtype=torch.float32)
        else:
            metric_from = 'cls'
        if lidar_depth is not None and calibration_lut is not None and calibrate_input:
            lidar_depth = self.apply_lidar_lut(lidar_depth, calibration_lut)
        if metric_from == 'prompt_gauge' and not (use_prompt and self.has_lidar_prompt):
            raise ValueError("metric_from='prompt_gauge' requires the prompt path")

        original_height, original_width = image.shape[-2:]
        aspect_ratio = original_width / original_height

        # Determine the number of base tokens to use
        if num_tokens is None:
            min_tokens, max_tokens = self.num_tokens_range
            num_tokens = int(min_tokens + (resolution_level / 9) * (max_tokens - min_tokens))

        # Forward pass
        with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=use_fp16 and self.dtype != torch.float16):
            output = self.forward(image, num_tokens=num_tokens, refine_steps=refine_steps, return_per_step=return_per_step,
                                  lidar_depth=lidar_depth, lidar_conf=lidar_conf, anchor_strength=anchor_strength, use_prompt=use_prompt,
                                  use_learned_conf=use_learned_conf, mono_z_override=mono_z_override)
        affine_points, normal, mask, metric_scale = (output.get(k, None) for k in ['points', 'normal', 'mask', 'metric_scale'])
        affine_points_per_step = output.get('points_per_step', None)
        prompt_mu = output.get('prompt_mu', None)

        # Always process the output in fp32 precision
        if affine_points_per_step is None:
            affine_points_per_step = [affine_points] if affine_points is not None else None
        affine_points_per_step = [p.float() for p in affine_points_per_step] if affine_points_per_step is not None else None
        normal, mask, metric_scale, fov_x = map(lambda x: x.float() if isinstance(x, torch.Tensor) else x, [normal, mask, metric_scale, fov_x])
        with torch.autocast(device_type=self.device.type, dtype=torch.float32):
            if mask is not None:
                mask_binary = mask > 0.5
            else:
                mask_binary = None

            if affine_points_per_step is not None:
                # Per-step (focal, shift) recovery: refinement modifies logz which changes
                # the 3D shape, so focal recovered from each step's point map differs slightly
                # Jointly solving (focal, shift) on the same point map gives the optimal
                # affine->camera alignment for that step (matches v2_4's single-step logic).
                if fov_x is not None:
                    focal_fixed = aspect_ratio / (1 + aspect_ratio ** 2) ** 0.5 / torch.tan(torch.deg2rad(torch.as_tensor(fov_x, device=self.device, dtype=torch.float32) / 2))
                    if focal_fixed.ndim == 0:
                        focal_fixed = focal_fixed[None].expand(affine_points_per_step[-1].shape[0])
                else:
                    focal_fixed = None

                points_per_step, depth_per_step, intrinsics_per_step, metric_fit_per_step = [], [], [], []
                for affine_points in affine_points_per_step:
                    if focal_fixed is None:
                        focal_i, shift_i = recover_focal_shift(affine_points, mask_binary)
                    else:
                        focal_i = focal_fixed
                        _, shift_i = recover_focal_shift(affine_points, mask_binary, focal=focal_i)
                    fx_i, fy_i = focal_i / 2 * (1 + aspect_ratio ** 2) ** 0.5 / aspect_ratio, focal_i / 2 * (1 + aspect_ratio ** 2) ** 0.5
                    intrinsics_i = utils3d.pt.intrinsics_from_focal_center(fx_i, fy_i, 0.5, 0.5)

                    points = affine_points.clone()
                    if metric_from == 'prompt_gauge':
                        # Network log-depth is interpreted directly in the prompt's gauge: z = exp(logz + mu), no shift.
                        depth = points[..., 2] * torch.exp(prompt_mu)[:, None, None]
                        fit_i = torch.stack([torch.exp(prompt_mu), torch.zeros_like(prompt_mu)], dim=-1)
                    else:
                        points[..., 2] += shift_i[..., None, None]
                        depth = points[..., 2].clone()
                        if metric_from == 'lidar_ls':
                            # Robust global scale + z-shift to confident LiDAR (fit at LiDAR resolution)
                            d_lr = F.interpolate(depth.unsqueeze(1), lidar_depth.shape[-2:], mode='area').squeeze(1)
                            m_lr = (lidar_conf.squeeze(1) >= 2) & (lidar_depth.squeeze(1) > 0) & (d_lr > 0)
                            if mask_binary is not None:
                                m_lr &= F.interpolate(mask_binary.float().unsqueeze(1), lidar_depth.shape[-2:], mode='area').squeeze(1) > 0.99
                            s_i, t_i = fit_scale_shift(d_lr.flatten(1), lidar_depth.squeeze(1).flatten(1), m_lr.flatten(1))
                            if use_prompt and hasattr(self, 'prompt_calib'):
                                s_i = s_i * torch.exp(self.prompt_calib[0]); t_i = t_i * torch.exp(self.prompt_calib[0]) + self.prompt_calib[1]
                            depth = s_i[:, None, None] * depth + t_i[:, None, None]
                            fit_i = torch.stack([s_i, t_i], dim=-1)
                        else:
                            fit_i = torch.stack([metric_scale, torch.zeros_like(metric_scale)], dim=-1) if metric_scale is not None else None

                    if force_projection or metric_from != 'cls':
                        points = utils3d.pt.depth_map_to_point_map(depth, intrinsics=intrinsics_i)

                    if metric_from == 'cls' and metric_scale is not None:
                        points *= metric_scale[:, None, None, None]
                        depth *= metric_scale[:, None, None]

                    if calibration_lut is not None and not calibrate_input and lidar_depth is not None and metric_from != 'cls':
                        # final post-processing: depth-dependent sensor calibration (piecewise-linear in log depth)
                        xs = torch.as_tensor(calibration_lut['log_depth'], device=depth.device, dtype=torch.float32)
                        ys = torch.as_tensor(calibration_lut['log_ratio'], device=depth.device, dtype=torch.float32)
                        ld = torch.log(depth.clamp_min(1e-3))
                        idx = torch.bucketize(ld, xs).clamp(1, len(xs) - 1)
                        x0, x1, y0, y1 = xs[idx - 1], xs[idx], ys[idx - 1], ys[idx]
                        w = ((ld - x0) / (x1 - x0)).clamp(0, 1)
                        ratio = torch.exp(y0 + w * (y1 - y0))
                        depth = depth * ratio
                        points = utils3d.pt.depth_map_to_point_map(depth, intrinsics=intrinsics_i)

                    points_per_step.append(points)
                    depth_per_step.append(depth)
                    intrinsics_per_step.append(intrinsics_i)
                    metric_fit_per_step.append(fit_i)

                # Final intrinsics correspond to the final-step point map (the one used as
                # the canonical output `points` / `depth`).
                intrinsics = intrinsics_per_step[-1]

                # Build per-step masks so each step is self-consistently masked
                # against its own depth>0 (mirrors v2_4's `mask_binary &= points[..., 2] > 0`).
                if mask_binary is not None:
                    mask_per_step = [mask_binary & (d > 0) for d in depth_per_step]
                    mask_binary = mask_per_step[-1]
                else:
                    mask_per_step = None

                points = points_per_step[-1]
                depth = depth_per_step[-1]
            else:
                points_per_step = None
                depth_per_step = None
                intrinsics_per_step = None
                mask_per_step = None
                metric_fit_per_step = None
                points, depth, intrinsics = None, None, None

            if apply_mask:
                if mask_per_step is not None:
                    points_per_step = [torch.where(m[..., None], p, torch.inf) for p, m in zip(points_per_step, mask_per_step)]
                    depth_per_step  = [torch.where(m, d, torch.inf) for d, m in zip(depth_per_step, mask_per_step)]
                    points, depth = points_per_step[-1], depth_per_step[-1]
                if mask_binary is not None and normal is not None:
                    normal = torch.where(mask_binary[..., None], normal, torch.zeros_like(normal))

            if not return_per_step:
                points_per_step = None
                depth_per_step = None
                intrinsics_per_step = None

        if gauge_poly > 0 and lidar_depth is not None:
            # per-frame polynomial prompt gauge: the prediction's smooth deviation from the calibrated prompt on confident pixels
            # (scale, offset, tilt, bow in image coordinates) is fitted robustly and removed everywhere (2026-09-04: -4 % confident AbsRel)
            ld = lidar_depth if (calibration_lut is None or calibrate_input) else self.apply_lidar_lut(lidar_depth, calibration_lut)
            corr = self.prompt_gauge_poly(depth, ld, lidar_conf, degree=gauge_poly)
            depth = depth * corr
            points = points * corr.unsqueeze(-1)
            if return_per_step and depth_per_step is not None:          # the per-step maps (written by the evaluators) get their own fit
                for k in range(len(depth_per_step)):
                    ck = self.prompt_gauge_poly(depth_per_step[k], ld, lidar_conf, degree=gauge_poly)
                    depth_per_step[k] = depth_per_step[k] * ck
                    if points_per_step is not None:
                        points_per_step[k] = points_per_step[k] * ck.unsqueeze(-1)
        return_dict = {
            'points': points,
            'intrinsics': intrinsics,
            'depth': depth,
            'mask': mask_binary,
            'normal': normal,
            'points_per_step': points_per_step,
            'intrinsics_per_step': intrinsics_per_step,
            'depth_per_step': depth_per_step,
            'metric_fit': metric_fit_per_step[-1] if (metric_fit_per_step is not None and metric_fit_per_step[-1] is not None) else None,
            'model_err': output.get('model_err'),
            'sensor_err': output.get('sensor_err'),
        }
        return_dict = {k: v for k, v in return_dict.items() if v is not None}

        if omit_batch_dim:
            return_dict = {
                k: [item.squeeze(0) for item in v] if isinstance(v, list) else v.squeeze(0)
                for k, v in return_dict.items()
            }

        return return_dict
