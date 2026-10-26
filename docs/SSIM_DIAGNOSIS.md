# SSIM diagnosis: why SSIM is about 0.72 at 4×

Question (2026-10-26): the 4× SSIM (EDM 0.723, U-Net 0.730) looked too low; the expectation was about 0.9 at 4×
and at least 0.95 at 2×. Full tables: `SSIM_DIAGNOSIS_TABLES.md`. Figure: `figures/ssim_diagnosis.png`.

## Answer

- **The metric is the fastMRI metric.** The official `fastmri/evaluate.py` functions give the same NMSE, PSNR and
  SSIM to 6 decimals. Fully sampled images score 1.000.
- **0.9 at 4× is a multi-coil number.** Single-coil knee, 4× SSIM: fastMRI U-Net 0.754 (test), PD 0.815 / PDFS 0.634
  (val); 2019 challenge winner i-RIM 0.751 (val) / 0.781 (test). Multi-coil: U-Net 0.904 (test), i-RIM 0.928 (test).
- **The single-coil ground truth is noisy** (median SNR 20 for PD, 8 for PDFS). The noise in unmeasured k-space
  lines cannot be predicted, so SSIM against the noisy target has a ceiling. Oracle (exact anatomy, simulated):
  0.847 (2×), 0.756 (4×), 0.701 (8×).
- **At 2× the EDM is the best method and reaches 98 % of the ceiling** (EDM 0.832, U-Net 0.815, zero-filled 0.807;
  EDM better than the U-Net on 187/189 volumes). Only 27/189 volumes reach 0.95 even for the oracle.
- **Slices, normalization and SSIM conventions do not explain the values** (details below).

## Main table (189 test volumes, central 8 slices, SSIM)

| method | 2× | 4× | 8× |
|---|---|---|---|
| zero-filled | 0.807 | 0.662 | 0.561 |
| U-Net (pretrained, 4× only) | 0.815 | 0.730 | 0.604 |
| **EDM (η=1, mean of 8)** | **0.832** | **0.723** | **0.638** |
| oracle ceiling | 0.847 | 0.756 | 0.701 |
| EDM / ceiling | 98 % | 96 % | 91 % |

PD (94): EDM 0.890 / 0.807 / 0.732, ceiling 0.904 / 0.843 / 0.804.
PDFS (95): EDM 0.774 / 0.640 / 0.544, ceiling 0.790 / 0.670 / 0.599.
2× PSNR: zero-filled 32.91, U-Net 33.32, EDM 34.47 dB (`runs/main/eval_2x`).

## Checks

| check | result |
|---|---|
| ours vs official `evaluate.py` | identical to 6 decimals |
| slices: all / without first+last / central 8 (U-Net 4×) | 0.732 / 0.731 / 0.730 |
| same for the EDM (6 volumes, 4×) | 0.677 / 0.679 / 0.690 |
| least-squares intensity factor (4×) | U-Net 1.003, EDM 1.02, zero-filled 1.025 |
| other conventions (U-Net 4×): Gaussian 11×11 / slice max / min-max [0,1] / 8-bit / knee mask | 0.728 / 0.635 / 0.626 / 0.563 / 0.748 |
| target blurred 0.7 px vs target | 0.839 (PD 0.893, PDFS 0.786) |
| noise-free anatomy vs its noisy target (simulated) | 0.622 (PD 0.741, PDFS 0.504) |

## More posterior samples (20 volumes)

Mean of 1 / 8 / 32 samples: 2× 0.771 / 0.808 / 0.812, 4× 0.658 / 0.689 / 0.692 (U-Net 0.694, ceiling 0.730).
8 → 32 samples adds +0.004 on every volume and saturates after 16 samples.

## Method of the ceiling

Per volume: noise σ from a robust Immerkær (Laplacian, MAD) estimate on the complex image (agrees with outer
k-space). Anatomy = NLM-denoised complex image. Noisy target = |anatomy + complex Gaussian noise(σ)|. Oracle at R× =
|anatomy + measured part of the noise| with the evaluation mask. The simulated zero-filled SSIM matches the real one
within 0.01–0.03 (slightly optimistic), so the ceiling is approximate.

## Reproduce

```
GPUS=6,7 bash scripts/torchrun.sh scripts/ssim_diagnostics.py classic --out runs/diag_ssim --workers 24
GPUS=0,1,2,3 bash scripts/torchrun.sh scripts/ssim_diagnostics.py edm --out runs/diag_ssim --per-contrast 10 --slices 8
GPUS=0,1,2,3 bash scripts/torchrun.sh scripts/ssim_diagnostics.py edm --out runs/diag_ssim --per-contrast 10 --slices 8 --samples 32 --accs 4 2
GPUS=0,1,2,3 bash scripts/torchrun.sh scripts/ssim_diagnostics.py edm --out runs/diag_ssim --per-contrast 3 --slices 0
GPUS=0,1,2,3 bash scripts/torchrun.sh scripts/evaluate.py --ckpt runs/main/ema-00004108.pt --out runs/main/eval_2x --skip 10 \
  recon.accelerations=[2] recon.center_fractions=[0.16] recon.eta=1.0 recon.num_samples=8
.venv/bin/python scripts/ssim_report.py --dir runs/diag_ssim --out docs/SSIM_DIAGNOSIS_TABLES.md
.venv/bin/python scripts/ssim_figure.py --pd runs/diag_ssim/edm_s8/file1000107 --pdfs runs/diag_ssim/edm_s8/file1000114
```

## Sources

- Zbontar et al., fastMRI, arXiv:1811.08839 (Tables 8–12; SSIM definition in Section 5.3).
- fastMRI evaluation code: github.com/facebookresearch/fastMRI, `fastmri/evaluate.py`.
- Knoll et al., Overview of the 2019 fastMRI challenge, MRM 84(6), 2020, arXiv:2001.02518.
- i-RIM results: github.com/pputzky/irim_fastMRI.
