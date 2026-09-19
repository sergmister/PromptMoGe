The model code of [MoGe](https://github.com/microsoft/MoGe) (MIT, see `LICENSE`; `model/modules/dinov2` is DINOv2,
Apache 2.0), reduced to what MoGe-3 needs and extended with the LiDAR prompt path: `model/modules/prompt_stem.py`
(prompt construction, stem, pyramid, robust scale/shift fit), `model/modules/qat.py` (int8 quantisation-aware
training of the sparse refiner) and the prompt, gauge, confidence-head and compression hooks in `model/v3.py`.
