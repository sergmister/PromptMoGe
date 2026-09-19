# Deriving Model A and Model B

Both on-device models are derived from the trained full-resolution teacher (PromptMoGe-L, `teacher.pt` below:
the `final.pt` of a `promptmoge.train.train` run). All models keep 1200 tokens (a 30x40 grid).
Needs a CUDA GPU, the staged ARKitScenes training frames (`data/arkit_stage`), the dev scenes
(`promptmoge/train/data.py`) and the base weights (`$MOGE3_WEIGHTS` or `checkpoints/moge-3-vitl/model.pt`).

```bash
LUT=promptmoge/train/calibration
COMMON="--inject_blocks 0 4 8 --neck 1 --refiner_residual 1 --soft_residual_gate 1 --amp_dtype bf16 --workers 3 \
  --train_hw 504 672 --num_tokens 1200 --replay_frac 0.0 --perturb_lam 0.15 --hole_p 0.8 --hole_frac 0.35 \
  --lowconf_p 0.3 --offsensor_weight 4 --w_gnormal 0.5 --w_rpnl 0.2 --crop_min 0.6 --real_edge_band 3 --photo_aug 1 \
  --calib_input $LUT/train_lidar_calibration_lut.json --gn_skip 1000 --freeze_decoder_until 0 \
  --eval_every 1000 --eval_stride 40 --save_every 1000"
REFINER_ONLY="--refiner_only 1 --train_refiner 1 --refine_steps_train 1 --refiner_every 1 --w_refiner_distill 1.0 \
  --teacher_on_dropped_only 1 --p_drop 0.0 --batch 4"
```

## Model A (480x640, int8 refiner)

1. Re-schedule the prompt pyramid (wide at the coarse levels) and fit it to the trained one; no training run:
   `python -m promptmoge.compress.distill_prompt_neck --ckpt teacher.pt --out teacher_neck.pt --hidden 512 256 128 64 32`
2. Narrow refiner level 4 (512 -> 256), fold the UV planes out of `encoder_fuse`, flag int8 QAT:
   `python -m promptmoge.compress.make_refiner --ckpt teacher_neck.pt --out init_a.pt --channels 32 64 128 256 256 --uv_fold --qat`
3. Refiner-only QAT, distilled from the uncompressed refiner (5000 steps, observers frozen for the last quarter):
   `python -m promptmoge.train.train --name model_a_qat --init_from init_a.pt --steps 5000 $COMMON $REFINER_ONLY
   --neck_hidden 512 256 128 64 32 --conf_head 1 --refiner_channels 32 64 128 256 256 --refiner_uv_fold 1 --refiner_qat 1
   --qat_freeze_at 3750 --refiner_lr 2e-4 --warmup 200 --refiner_teacher teacher_neck.pt`
   (drop `--refiner_qat 1 --qat_freeze_at` here and `--qat` in step 2 for the fp16-refiner variant)
4. Exact tail rescaling (activation peak 8) and deployment flags:
   `python -m promptmoge.compress.make_deliverable --arm runs/model_a_qat/final.pt --out model_a_int8.pt`
5. Checks: `python -m promptmoge.compress.cert_bound model_a_int8.pt 8 cert_a.json` (every conv bound must stay far
   below 32768) and `python -m promptmoge.compress.verify_int8 --ckpt model_a_int8.pt`.

## Model B (240x320, 4-level ladder, int8 refiner)

1. Transfer levels 0-3 of the neck, heads and refiner to the x8 ladder:
   `python -m promptmoge.compress.make_model_b --ckpt teacher.pt --out init_b.pt --uv_fold`
2. Distil a 4-level prompt pyramid from the teacher's 5-level one:
   `python -m promptmoge.compress.distill_prompt_neck --ckpt init_b.pt --ref teacher.pt --levels 4 --hidden 512 256 128 64 --out init_b_neck.pt`
3. Least-squares fit of the head output projections to the teacher's output at 240x320:
   `python -m promptmoge.compress.fit_model_b_heads --ref teacher.pt --ckpt init_b_neck.pt --out init_b_fit.pt`
4. Main run (9000 steps; real + synthetic stage):
   `python -m promptmoge.train.train --name model_b --init_from init_b_fit.pt --steps 9000 $COMMON
   --neck_hidden 512 256 128 64 --conf_head 0 --refiner_uv_fold 1 --train_refiner 1 --refiner_lr 2e-4 --refine_steps_train 1
   --refiner_every 1 --batch 3 --lr_prompt 5e-5 --lr_gates 2e-4 --lr_decoder 5e-5 --clip 1.0 --warmup 200 --p_drop 0.25
   --w_edge 0.2 --w_normal 0.5 --w_mask 0.2 --w_uv 1.0 --w_fidelity 1.0 --fidelity_sigma 8 --fidelity_margin 0.003
   --synth_root data/synth_stage --synth_frac 0.34 --synth_unprompted 0.5 --synth_sensor_jitter 0.007
   --synth_calib_input $LUT/synth_lidar_calibration_lut.json --synth_max_median_depth 4.0 --synth_gt_max_range 8.0
   --w_exact_normal 1.0`
5. Unfold the trained refiner as the uncompressed reference, then refiner-only QAT (3000 steps):
   `python -m promptmoge.compress.unfold_uv --ckpt runs/model_b/final.pt --out model_b_ref.pt`
   `python -m promptmoge.train.train --name model_b_qat --init_from runs/model_b/final.pt --steps 3000 $COMMON $REFINER_ONLY
   --neck_hidden 512 256 128 64 --conf_head 0 --refiner_uv_fold 1 --refiner_qat 1 --qat_freeze_at 2250 --refiner_lr 1e-4
   --warmup 100 --refiner_teacher model_b_ref.pt`
6. `python -m promptmoge.compress.make_deliverable --arm runs/model_b_qat/final.pt --out model_b_int8.pt`, then the
   step-5 checks of Model A on `model_b_int8.pt`.

Tools that read the base MoGe-3 weights take `--base`; those that run the model also take `--calib_lut`.
Score Model B on its own grid: `python -m promptmoge.train.score --grid 240 320 ...`.
