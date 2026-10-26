# thesis_exp: specification used

The task referred to "attached instructions" that never arrived. Every choice below is mine and follows
the standard EDM recipe (Karras et al., 2022) as closely as the data allows. Each one is a config value
in `configs/main.yaml` and can be changed. Full citations, with table and equation numbers:
[`REFERENCES.md`](REFERENCES.md).

## Data
- Official fastMRI knee single-coil v2.0 (`knee_singlecoil_train.tar.xz`, `knee_singlecoil_val.tar.xz`;
  Zbontar et al. 2018; Knoll et al. 2020a). The single-coil k-space is emulated from multi-coil data
  (ESC, Tygert and Zbontar 2020).
- Train split for training (973 volumes). Val split for monitoring and evaluation (199 volumes, 7,135 slices).
  The test and challenge splits have no fully sampled ground truth, so they are not used.
- Each slice becomes a complex image: centered orthonormal inverse FFT of the full k-space, center crop
  to 320×320, stored as two channels (real, imaginary).
  Check: |image| matches the official `reconstruction_esc` with a maximum relative error of 3.3e-7 on all
  199 validation volumes.
- Intensity scale (per volume, our choice): the maximum magnitude of a low-resolution image built from
  only the central 4% of k-space columns. Both fastMRI test masks (4× with an 8% center, 8× with a 4% center)
  always sample those columns. The same scale can therefore be computed from undersampled data at test
  time, without the ground truth.
- A global gain then sets the RMS of the training data to σ_data = 0.5 (EDM's σ_data).
- No augmentation.

## Model
- EDM preconditioning and loss (Karras et al. 2022, Eq. 7–8 and Table 1): σ_data 0.5,
  ln σ ~ N(−1.2, 1.2²), λ(σ) = (σ² + σ_data²)/(σ σ_data)², c_skip, c_out, c_in and c_noise = ¼ ln σ as in
  Table 1.
- DDPM++ backbone (Song et al. 2021), implemented as in EDM's `SongUNet` (NVlabs/edm,
  `training/networks.py`: standard encoder and decoder, positional noise embedding, [1,1] resampling).
  Our adaptation to 320×320:
  - 128 base channels, multipliers [1, 1, 2, 2, 2, 2] (320 → 10)
  - 2 residual blocks per level
  - self-attention at 20×20 and in the bottleneck
  - dropout 0.1
  - 54.0M parameters.

## Training
- Adam (Kingma and Ba 2015; 0.9, 0.999, ε 1e-8), learning rate 2e-4 (EDM's FFHQ/AFHQv2 value,
  Karras et al. 2022, Table 7), linear warm-up over 320 kimg (ours; EDM uses 10 Mimg).
- Global batch 64. EMA half-life 500 kimg (Karras et al. 2022, Table 7) with ramp-up ratio 0.05
  (NVlabs/edm `training/training_loop.py`).
- Non-finite gradients are zeroed (as in NVlabs/edm `training/training_loop.py`).
- Duration 10 Mimg, about 290 epochs. Training resumes automatically from the newest state; a longer
  run is the same command with a larger `train.duration_kimg`.
- bf16 compute with fp32 master weights.

## Monitoring (every 1000 kimg, EMA weights)
- Denoising MSE at fixed σ on 256 fixed validation slices, with fixed noise.
- 16 unconditional samples (Heun sampler, Karras et al. 2022, Algorithm 1 with the noise levels of Eq. 5;
  40 steps).
- 4× and 8× reconstruction of the first 4 validation volumes.

## Evaluation
- Single-coil compressed sensing on the cropped 320×320 k-space, retrospectively undersampled.
  This is the simplification used by Chung and Ye 2022 and Song et al. 2022; it is not the raw 640×368
  acquisition grid (Shimron et al. 2022 discuss why this can be optimistic).
- Masks: fastMRI `RandomMaskFunc` (4× with 0.08 center, 8× with 0.04 center; Zbontar et al. 2018,
  Section 4.9), with the per-volume seed fastMRI uses for validation (`tuple(map(ord, fname))` in
  `fastmri/data/transforms.py`).
- Reconstruction: EDM sampling where every denoised estimate gets hard data consistency (measured
  k-space entries replaced; DDNM, Wang et al. 2023, Eq. 13), followed by a step to the next noise level
  with stochasticity η: x_next = x̂₀ + σ_next (√(1−η²) ε̂ + η z) (DDRM, Kawar et al. 2022, Eq. 4; η = 0 is
  DDIM, Song, Meng and Ermon 2021).
  - η and the number of steps are tuned on the first 10 validation volumes.
  - Results are reported on the other 189.
- Metrics: NMSE, PSNR, SSIM against `reconstruction_esc`, as in fastMRI's `evaluate.py` (Zbontar et al.
  2018, Eq. 7–9; SSIM: Wang et al. 2004). Central 8 slices per volume (ours); the data range is the
  volume maximum.
- Baseline: zero-filled reconstruction.
