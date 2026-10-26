# Results

An EDM diffusion model trained from scratch on fastMRI knee single-coil images, used as a prior for
accelerated MRI reconstruction, and compared with zero-filling and the official pretrained fastMRI U-Net.

## Summary

On the 189 test volumes (mean ± standard error over volumes):

| | 4× PSNR / SSIM | 8× PSNR / SSIM |
|---|---|---|
| zero-filled | 29.38 / 0.662 | 26.79 / 0.561 |
| pretrained U-Net (supervised, 4× only) | 32.10 / 0.730 | 27.41 / 0.604 |
| **EDM (ours, η=1, mean of 8)** | **32.10 / 0.723** | **30.28 / 0.638** |

- **4×:** the EDM ties the official supervised U-Net on PSNR (difference 0.00 ± 0.05 dB). It is 0.007
  lower on SSIM. Both are 2.7 dB above zero-filling.
  - The EDM is better on PD scans (+0.33 dB).
  - It is worse on the noisier PDFS scans (−0.32 dB).
- **8×:** the EDM is **2.9 dB ahead** of the U-Net (+2.87 ± 0.17 dB, better on 173 of 189 volumes).
  - The U-Net was trained for 4× only.
  - The same diffusion prior serves every acceleration without retraining.
  - For context, the fastMRI paper reports about 30.1 dB on validation at 8× for its U-Net trained on the
    training set with random 4× and 8× masks (Zbontar et al. 2018, Table 8: PD 31.50, PDFS 28.71 dB).
    That number uses fastMRI's protocol and all slices, so it is not directly comparable, but it is the
    same range.
- **The U-Net has an unfair advantage:** it was trained on the validation volumes used here as the test
  set (see Caveats).
- **Training ended early:** reconstruction quality stopped improving after about 1 Mimg, so training was
  stopped at 4.1 Mimg (16.5 h on 4 GPUs) instead of 10 Mimg.

## Setup

- **Data:** official fastMRI knee single-coil.
  - Training split for training (973 volumes, 34,742 slices).
  - Validation split for evaluation (199 volumes). The knee test set has no public ground truth, and
    the fastMRI leaderboard is offline. As fastMRI recommends (fastMRI GitHub Discussion #293), the
    validation set is therefore split in two:
    - the first 10 volumes for tuning;
    - the other **189 volumes as the test set**.
- **Model:** a 54M-parameter DDPM++ U-Net (Song et al. 2021) with EDM preconditioning and loss
  (Karras et al. 2022), on 2-channel complex 320×320 images (details in `SPEC.md`). It is trained only to
  remove noise from knee images; it never sees undersampled data.
- **Reconstruction:** the sampler starts from noise and runs 100 EDM steps. After every denoising step,
  the measured k-space lines are put back (hard data consistency, DDNM: Wang et al. 2023, Eq. 13),
  followed by a step to the next noise level with stochasticity η (DDRM: Kawar et al. 2022, Eq. 4; η = 0
  is DDIM). The output is the mean magnitude of 8 independent reconstructions (posterior samples).
- **Undersampling:**
  - fastMRI random Cartesian masks (Zbontar et al. 2018, Section 4.9): 4× with 8% center lines, 8× with
    4% center lines.
  - Fixed per-volume seeds, the same as fastMRI validation (fastMRI code).
  - Applied to the k-space of the cropped 320×320 image.
- **Metrics:**
  - PSNR, SSIM and NMSE against `reconstruction_esc`, computed as in fastMRI's `evaluate.py`
    (Zbontar et al. 2018, Section 5; SSIM: Wang et al. 2004).
  - Magnitude images; the data range is the volume maximum.
  - Central 8 slices per volume (our choice).
- **Baselines:**
  - Zero-filling.
  - The official pretrained fastMRI single-coil U-Net (Ronneberger et al. 2015; fastMRI code,
    `knee_sc_leaderboard_state_dict.pt`, 496M parameters). It is a supervised model, trained on pairs of
    undersampled and fully sampled images. It is run on exactly the same inputs as the EDM. Caveats are
    listed at the end.
- **Sources:** every formula, code source and literature number, with table and equation numbers, is in
  [`REFERENCES.md`](REFERENCES.md).

## Training

- 4.11 Mimg seen (118 epochs), in 16.5 hours on 4× RTX A6000.
  - bf16, channels-last and `torch.compile`, at 75 images/s.
- The run was planned for 10 Mimg and stopped early. Reconstruction quality had stopped improving
  after about 1 Mimg:

| checkpoint | 4× PSNR / SSIM | 8× PSNR / SSIM |
|---|---|---|
| zero-filled | 28.08 / 0.628 | 25.44 / 0.515 |
| 1,000 kimg | 30.18 / 0.661 | 27.72 / 0.562 |
| 2,000 kimg | 30.20 / 0.661 | 27.80 / 0.563 |
| 3,600 kimg | 30.19 / 0.659 | 27.81 / 0.561 |

How these were measured:
- Tuning volumes 0–7, with the same η=0.5, mean-of-4 sampler and the same seeds for every checkpoint.
- Per volume, every change from 1,000 to 3,600 kimg is within ±0.1 dB.
- The training loss (0.30) is expected: in EDM's weighting, a loss of 1.0 means the network contributes
  nothing (the training target has unit variance; Karras et al. 2022, Appendix B.6). At small noise levels the loss stays close to 1, because the scans contain their own noise
  (σ ≈ 0.12 in normalized units). That noise cannot be told apart from the added noise.
- The final model is the EMA snapshot at 4,109 kimg.

## Sampler tuning (10 tuning volumes)

| setting | 4× PSNR / SSIM | 8× PSNR / SSIM |
|---|---|---|
| zero-filled | 28.26 / 0.633 | 25.58 / 0.521 |
| pretrained U-Net | 31.06 / 0.699 | 26.26 / 0.562 |
| η=0.5 | 28.75 / 0.593 | 26.28 / 0.496 |
| η=0.5, mean of 4 | 30.35 / 0.668 | 27.92 / 0.570 |
| η=0.5, mean of 8 | 30.68 / 0.683 | 28.28 / 0.586 |
| η=1 | 30.36 / 0.665 | 28.68 / 0.576 |
| η=1, 200 steps | 30.30 / 0.663 | 28.83 / 0.579 |
| η=1, stop at σ=0.15 | 30.42 / 0.670 | 28.66 / 0.575 |
| η=1, stop at σ=0.15, mean of 4 | 30.75 / 0.684 | 29.04 / 0.587 |
| η=1, mean of 4 | 30.92 / 0.690 | 29.16 / 0.596 |
| **η=1, mean of 8** | **31.03 / 0.695** | **29.25 / 0.599** |

- The selection rule, fixed before tuning: the highest SSIM averaged over 4× and 8×.
- Two choices matter most:
  - **Full stochasticity (η=1)** gives +1.6 dB (4×) and +2.4 dB (8×) over η=0.5.
  - **Averaging posterior samples** gives +0.6 dB.
- **Why averaging helps:** a single sample invents its own noise texture. PSNR and SSIM penalize that
  texture against a noisy ground truth. The mean approaches the posterior mean, which is what these
  metrics reward.
- More steps and stopping early do not help.

## Final results (189 test volumes)

### 4×

| contrast | method | PSNR (dB) | SSIM | NMSE |
|---|---|---|---|---|
| all (189) | Zero-filled | 29.379 ± 0.186 | 0.662 ± 0.008 | 0.054 ± 0.002 |
| all (189) | U-Net (pretrained) | 32.101 ± 0.251 | 0.730 ± 0.009 | 0.034 ± 0.002 |
| all (189) | EDM (ours) | 32.104 ± 0.282 | 0.723 ± 0.009 | 0.037 ± 0.002 |
| PD (94) | Zero-filled | 30.349 ± 0.229 | 0.727 ± 0.009 | 0.030 ± 0.001 |
| PD (94) | U-Net (pretrained) | 34.151 ± 0.296 | 0.815 ± 0.009 | 0.013 ± 0.001 |
| PD (94) | EDM (ours) | 34.477 ± 0.331 | 0.807 ± 0.009 | 0.013 ± 0.001 |
| PDFS (95) | Zero-filled | 28.420 ± 0.258 | 0.597 ± 0.010 | 0.079 ± 0.003 |
| PDFS (95) | U-Net (pretrained) | 30.073 ± 0.277 | 0.647 ± 0.010 | 0.055 ± 0.002 |
| PDFS (95) | EDM (ours) | 29.756 ± 0.302 | 0.640 ± 0.010 | 0.061 ± 0.003 |

| contrast | EDM − U-Net PSNR | EDM − U-Net SSIM | EDM better (PSNR) |
|---|---|---|---|
| all | 0.003 ± 0.045 | -0.007 ± 0.001 | 74/189 |
| PD | 0.326 ± 0.069 | -0.007 ± 0.001 | 59/94 |
| PDFS | -0.317 ± 0.036 | -0.007 ± 0.001 | 15/95 |

### 8×

| contrast | method | PSNR (dB) | SSIM | NMSE |
|---|---|---|---|---|
| all (189) | Zero-filled | 26.789 ± 0.156 | 0.561 ± 0.009 | 0.090 ± 0.003 |
| all (189) | U-Net (pretrained) | 27.409 ± 0.156 | 0.604 ± 0.009 | 0.077 ± 0.002 |
| all (189) | EDM (ours) | 30.279 ± 0.243 | 0.638 ± 0.010 | 0.051 ± 0.003 |
| PD (94) | Zero-filled | 26.958 ± 0.217 | 0.630 ± 0.010 | 0.064 ± 0.002 |
| PD (94) | U-Net (pretrained) | 27.498 ± 0.226 | 0.675 ± 0.009 | 0.057 ± 0.002 |
| PD (94) | EDM (ours) | 32.167 ± 0.288 | 0.732 ± 0.011 | 0.021 ± 0.001 |
| PDFS (95) | Zero-filled | 26.622 ± 0.224 | 0.493 ± 0.011 | 0.116 ± 0.003 |
| PDFS (95) | U-Net (pretrained) | 27.321 ± 0.216 | 0.534 ± 0.011 | 0.098 ± 0.002 |
| PDFS (95) | EDM (ours) | 28.412 ± 0.280 | 0.544 ± 0.012 | 0.081 ± 0.004 |

| contrast | EDM − U-Net PSNR | EDM − U-Net SSIM | EDM better (PSNR) |
|---|---|---|---|
| all | 2.870 ± 0.166 | 0.034 ± 0.003 | 173/189 |
| PD | 4.669 ± 0.178 | 0.058 ± 0.003 | 94/94 |
| PDFS | 1.090 ± 0.105 | 0.010 ± 0.002 | 79/95 |

Reading the tables:
- **Same volumes, masks, targets and metric code for all three methods.** The zero-filled numbers are
  checked to be identical between the two evaluation runs.
- **"EDM − U-Net"** is the paired difference per volume.
- **PD vs PDFS:** PDFS (fat-suppressed) scans are much noisier. The U-Net, trained to predict the target
  directly, learns to smooth that noise. The mean of 8 EDM samples still keeps some sample noise; more
  samples would likely narrow the PDFS gap. That is untested.
- **SSIM** favours the smoother U-Net output slightly, even where PSNR is equal.

## Acceleration sweep (6 test volumes)

The same EDM, the same pretrained U-Net and zero-filling, from 2× to 16× (`results/acceleration_sweep.json`,
`figures/acceleration_sweep.png`).

Setup:
- Volumes: the first 3 PD and first 3 PDFS test volumes, central 4 slices.
- EDM: η=1, mean of 4 samples.
- Center fraction: 0.32/R (fastMRI's values at 4× and 8×), never below 4%.

PSNR (dB) / SSIM:

| acceleration | zero-filled | U-Net (trained for 4×) | EDM |
|---|---|---|---|
| 2× | 31.94 / 0.786 | 32.31 / 0.792 | **33.23 / 0.804** |
| 4× | 28.41 / 0.630 | **31.25 / 0.694** | 31.07 / 0.686 |
| 6× | 26.53 / 0.555 | 27.64 / 0.606 | **30.15 / 0.631** |
| 8× | 25.71 / 0.516 | 26.34 / 0.556 | **29.56 / 0.598** |
| 10× | 25.50 / 0.496 | 25.84 / 0.527 | **28.97 / 0.570** |
| 12× | 25.39 / 0.484 | 25.62 / 0.510 | **28.64 / 0.552** |
| 16× | 25.18 / 0.467 | 25.21 / 0.486 | **27.83 / 0.522** |

- The supervised U-Net is only competitive at the acceleration it was trained for.
- The diffusion prior degrades gracefully. At 12× it still matches zero-filling at 4×.

## Figures

All figures are in `figures/`; `../README.md` lists them, and `PRESENTATION.md` orders them into slides.
Charts are rebuilt from `results/` by `scripts/make_plots.py`. Image figures are built by
`scripts/make_figures.py`.

For the image figures, the volumes are fixed by rule: the first PD and the first PDFS volume of the test split. They were not
picked by eye.

- `figures/comparison_4x.png` and `figures/comparison_8x.png`, in columns:
  - ground truth;
  - zero-filled;
  - pretrained U-Net;
  - EDM (mean of 8);
  - 5× absolute error maps and the sampling mask.
- `figures/uncertainty_4x.png` and `figures/uncertainty_8x.png`, in columns:
  - one posterior sample;
  - the mean of 8;
  - the pixel-wise standard deviation over the 8 samples.
  
  The diffusion model gives an uncertainty map for free. It lights up at edges and fine structures,
  where the error of the mean is also largest.

## Caveats

- **The pretrained U-Net was trained on the training and validation sets** (`combine_train_val=True` in
  fastMRI's leaderboard script, `fastmri_examples/unet/unet_reproduce_20201111.py/unet_knee_sc_leaderboard.py`).
  It has therefore seen the test volumes, and its numbers here are optimistic. On all 199 validation
  volumes, under fastMRI's own protocol, it reaches 4× PD 34.61 dB and PDFS 30.14 dB. The fastMRI paper
  reports 34.00 dB and 29.95 dB for its U-Net trained without the validation set (Zbontar et al. 2018,
  Table 8, 256 channels).
- **The pretrained U-Net was trained for 4× only** (the same leaderboard script uses 4× masks with a 0.08
  center only). At 8× it is outside its training distribution. So the 8× result shows that the diffusion
  prior **adapts to a new acceleration without retraining**. It does not show that it beats supervised
  methods at 8×: the fastMRI paper's U-Net, trained on the training set with random 4× and 8× masks,
  reaches about 30.1 dB at 8× on validation (Zbontar et al. 2018, Table 8 and Section 6.3; fastMRI
  protocol, all slices). A U-Net trained on the training split only, at both accelerations, would be the
  fully fair comparison.
- **Protocol:**
  - The undersampling is applied to the k-space of the cropped 320×320 image, as in Chung and Ye 2022 and
    Song et al. 2022, not to the raw 640×368 acquisition. Shimron et al. 2022 show that such processed
    data can make results optimistic.
  - The U-Net loses 0.15–0.3 dB under this protocol compared with its own.
  - Metrics use the central 8 slices.
- **One trained model and one sampling seed per volume.** Standard errors are over volumes.

## Reproduce

```bash
# sampler tuning on the first 10 validation volumes
CKPT=runs/main/ema-00004108.pt GPUS=0,1,2,3 GRID="$(cat runs/main/tune_grid.txt)" bash scripts/tune_recon.sh
# final evaluation on the other 189
GPUS=0,1,2,3 bash scripts/torchrun.sh scripts/evaluate.py --ckpt runs/main/ema-00004108.pt \
    --out runs/main/eval_final --skip 10 train.channels_last=false recon.eta=1.0 recon.num_samples=8
# pretrained U-Net (weights from dl.fbaipublicfiles.com/fastMRI/trained_models/unet/)
GPUS=0,1,2,3 bash scripts/torchrun.sh scripts/evaluate_unet.py --out runs/baselines/unet
# table and figures
python scripts/compare_results.py --out docs/results_table.md
CUDA_VISIBLE_DEVICES=0 python scripts/make_figures.py
```
