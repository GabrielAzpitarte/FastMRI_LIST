# Progress notes

`python scripts/progress_table.py runs/main` prints the up-to-date tables from the run logs.

## Check at 1000 kimg (Wed 10-21, 13:2x)
- **Unconditional samples** (`runs/main/samples/samples-00001000.png`) already look like realistic coronal
  knee slices, in both contrasts (with and without fat suppression), including scanner noise texture.
- **Held-out denoising MSE** (data RMS 0.5, so data variance 0.25) is well below the noise variance at every σ.
  For example: σ=0.2 gives 0.0118 against σ²=0.04; σ=5 gives 0.039 against a data variance of 0.25.
- **Probe reconstructions** (4 val volumes, η=0.5, 1 sample) were *below* zero-filling: 4× SSIM 0.445 vs 0.519,
  8× SSIM 0.352 vs 0.394.

## Why the first probe was below zero-filling
Example: `runs/main/diag_1000/eta0.5/example_4x_file1000000.png`. Columns are ground truth | zero-filled |
EDM | 5× zero-filled error | 5× EDM error.
- The EDM reconstruction removes the aliasing and recovers sharp anatomy, while the zero-filled image is
  blurry and ghosted.
- But a single posterior sample carries new random noise texture that matches the scanner noise of the target
  statistically, not pixel by pixel. Against a noisy ground truth this roughly doubles the noise error, which
  PSNR and SSIM punish. Zero-filling is smooth and so avoids it.
- Averaging posterior samples (an approximation of the posterior mean) removes this penalty.

Sampler comparison, 1000-kimg EMA, first 2 validation volumes (part of the tuning set), central 8 slices:

| setting | 4× PSNR | 4× SSIM | 8× PSNR | 8× SSIM |
|---|---|---|---|---|
| zero-filled | 25.72 | 0.504 | 23.88 | 0.392 |
| η=0 (deterministic) | 25.08 | 0.423 | 23.50 | 0.332 |
| η=0.5 | 25.49 | 0.438 | 24.19 | 0.355 |
| η=0.5, end at σ=0.05 | 25.58 | 0.443 | 24.27 | 0.360 |
| η=0.5, end at σ=0.15 | 25.93 | 0.469 | 24.47 | 0.378 |
| η=1 | 26.44 | 0.502 | 25.46 | 0.413 |
| η=0.5, mean of 4 samples | **26.81** | **0.512** | 25.28 | **0.416** |

Main points:
- At only 1 Mimg, the prior already beats zero-filling once it is sampled with η=1 or averaged.
- The final tuning grid (`scripts/tune_recon.sh`) combines η=1, averaging and a later stop.
- It only uses the tuning volumes. The final numbers are reported on the other 189 volumes.
