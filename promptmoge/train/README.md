# Training the teacher (PromptMoGe-L)

`promptmoge.train.train` fine-tunes the LiDAR prompt path of MoGe-3 ViT-L: the DINOv2 backbone stays frozen, a frozen
copy of the stock model is the teacher for everything the sensor does not supervise (RGB-only behaviour, normals,
mask), and only the trainable tensors are saved. It needs a CUDA GPU with [FlexGEMM](https://github.com/JeffreyXiang/FlexGEMM)
(24 GB with `--grad_ckpt 1` at batch 4) and the base weights (`--base`, default `$MOGE3_WEIGHTS`).

## Data

Not included. Real frames come from [ARKitScenes](https://github.com/apple/ARKitScenes) (RGB, ARKit LiDAR depth and
confidence, laser-scan ground truth), synthetic ones from [Hypersim](https://github.com/apple/ml-hypersim) with a
simulated sensor (`sensor_sim/`). Both are staged as one folder per video:

    <root>/<video>/{rgb/<ts>.jpg (1008x1344), gt/<ts>.png (uint16 mm), lidar/<ts>.png (192x256 uint16 mm), conf/<ts>.png {0,1,2}}
    synthetic stages add normal/<ts>.npz (key "n": int8 normals x127 at 504x672)

`data.py` documents the layout of the evaluation scenes; without them training simply skips the periodic evaluation.

## Recipe

The released teacher was trained in stages with one recipe, each stage warm-started (`--init_from`) from the last;
the final stage is:

```bash
LUT=promptmoge/train/calibration
python -m promptmoge.train.train --name teacher --steps 12000 --batch 4 --grad_ckpt 1 --amp_dtype bf16 \
  --inject_blocks 0 4 8 --neck 1 --refiner_residual 1 --soft_residual_gate 1 --conf_head 1 \
  --lr_prompt 5e-5 --lr_decoder 2e-5 --warmup 100 --gn_skip 1000 --freeze_decoder_until 0 \
  --p_drop 0.25 --replay_frac 0.0 --hole_p 0.8 --hole_frac 0.35 --photo_aug 1 --real_edge_band 3 --video_balance 0.5 \
  --offsensor_weight 4 --w_gnormal 0.5 --w_rpnl 0.2 --w_edge 0.2 --w_teacher_edge 0.5 \
  --w_fidelity 1.0 --fidelity_sigma 8 --fidelity_margin 0.003 --calib_input $LUT/train_lidar_calibration_lut.json \
  --arkit_root data/arkit_stage --synth_root data/synth_stage --synth_frac 0.34 --synth_unprompted 0.5 \
  --synth_calib_input $LUT/synth_lidar_calibration_lut.json --synth_sensor_jitter 0.007 \
  --synth_max_median_depth 4.0 --synth_gt_max_range 8.0 --w_exact_normal 1.0
```

Model A and Model B are derived from the result: see [`../compress`](../compress/README.md).
Evaluate a checkpoint with `python -m promptmoge.train.evaluate` and `python -m promptmoge.train.score`.
