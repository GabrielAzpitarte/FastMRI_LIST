# References and sources

Where every formula, design choice, piece of borrowed code and literature number in this repository comes
from. **Ours** marks choices made for this project; they do not come from a paper.

Each entry below was checked against the source itself (2026-10-26). Table, equation and section numbers
refer to the arXiv version listed in the bibliography.

## 1. Data and evaluation protocol

| what | source |
|---|---|
| fastMRI knee single-coil data: 973 training and 199 validation volumes | Zbontar et al. 2018, Section 4 and Table 4; Knoll et al. 2020a |
| Emulated single-coil (ESC) k-space; target `reconstruction_esc` = inverse FFT cropped to the central 320×320 | Zbontar et al. 2018, Sections 4.4 and 4.7, Appendix A; Tygert and Zbontar 2020 |
| Random Cartesian masks: 8 % central lines at 4×, 4 % at 8×, other lines uniformly at random | Zbontar et al. 2018, Section 4.9; `RandomMaskFunc` in `fastmri/data/subsample.py` (fastMRI code) |
| Fixed mask per validation volume, seed `tuple(map(ord, fname))` | `fastmri/data/transforms.py` (fastMRI code) |
| NMSE over the volume | Zbontar et al. 2018, Eq. 7, Section 5.1 |
| PSNR over the volume, peak = maximum of the target volume | Zbontar et al. 2018, Eq. 8, Section 5.2 |
| SSIM: 7×7 window, k1 = 0.01, k2 = 0.03, L = maximum of the target volume, mean over slices | Zbontar et al. 2018, Eq. 9, Section 5.3 (SSIM itself: Wang et al. 2004); `fastmri/evaluate.py` (fastMRI code), which calls scikit-image |
| Validation set split in two (tuning / test), because the knee test ground truth is not public and the leaderboard is offline | fastMRI GitHub Discussion #293 |
| Undersampling the k-space of the cropped 320×320 image, not the raw acquisition | Same simplification as Chung and Ye 2022 (fastMRI knee, 320×320) and Song et al. 2022 (k-space simulated by FFT of image slices). Shimron et al. 2022 and Jalal et al. 2021 explain why it can give optimistic numbers. |
| Central 8 slices per volume; 189-volume test split (validation volumes 10–198) | ours |
| Complex 2-channel images; per-volume scale from the central 4 % of k-space columns; global gain to RMS 0.5 | ours (the target RMS 0.5 is EDM's σ_data, Karras et al. 2022) |
| Center fraction 0.32/R in the acceleration sweep | ours (it gives fastMRI's 0.08 at 4× and 0.04 at 8×) |

## 2. Diffusion model (EDM)

| what | source |
|---|---|
| Denoiser D(x; σ) = c_skip(σ) x + c_out(σ) F(c_in(σ) x; c_noise(σ)) | Karras et al. 2022, Eq. 7 and Table 1 |
| c_skip = σ_d² / (σ² + σ_d²), c_out = σ σ_d / √(σ² + σ_d²), c_in = 1 / √(σ² + σ_d²), c_noise = ¼ ln σ, σ_d = 0.5 | Karras et al. 2022, Table 1 ("Ours" column); derivation in Appendix B.6 |
| Loss weight λ(σ) = (σ² + σ_d²) / (σ σ_d)²; training noise ln σ ~ N(−1.2, 1.2²) | Karras et al. 2022, Table 1 and Eq. 8 |
| "A loss of 1.0 means the network adds nothing": c_in, c_out and λ make the effective training target unit-variance | Karras et al. 2022, Section 5 and Appendix B.6 |
| Noise levels σ_i = (σ_max^(1/ρ) + i/(N−1) (σ_min^(1/ρ) − σ_max^(1/ρ)))^ρ, σ_min = 0.002, σ_max = 80, ρ = 7 | Karras et al. 2022, Eq. 5 and Table 1 |
| Heun 2nd-order deterministic sampler (unconditional samples) | Karras et al. 2022, Algorithm 1 |
| Network: DDPM++ (`SongUNet`: positional noise embedding, standard encoder and decoder, [1,1] resampling filter) | Song et al. 2021 (DDPM++), which improves the DDPM U-Net of Ho et al. 2020; configuration and code as in NVlabs/edm (`training/networks.py`, `train.py --arch=ddpmpp`) |
| 320×320 adaptation: multipliers [1,1,2,2,2,2], attention at 20×20, 2 blocks per level, dropout 0.1 | ours |
| EMA of the weights, half-life 500 kimg, ramp-up ratio 0.05 | Karras et al. 2022, Appendix F and Table 7; NVlabs/edm `training/training_loop.py` (ramp-up ratio) |
| Learning rate 2e-4 | Karras et al. 2022, Table 7 (FFHQ and AFHQv2 setting) |
| Adam optimizer (β = 0.9, 0.999; ε = 1e-8) | Kingma and Ba 2015 |
| Non-finite gradients set to zero | NVlabs/edm `training/training_loop.py` |
| Warm-up 320 kimg, global batch 64, bf16, stop at 4.1 Mimg | ours |

## 3. Reconstruction with the diffusion prior

| what | source |
|---|---|
| Idea: train the prior on images only; add the measurements at reconstruction time with k-space data consistency | Chung and Ye 2022; Song et al. 2022; Jalal et al. 2021 |
| Hard data consistency on each denoised estimate: x̂₀ = A⁺y + (I − A⁺A) x₀ with A = mask · FFT. With an orthonormal FFT, this replaces the measured k-space entries | Wang et al. 2023 (DDNM), Eq. 13 and Algorithm 1 |
| Step to the next noise level: x_next = x̂₀ + σ_next (√(1−η²) ε̂ + η z), with ε̂ = (x − x̂₀)/σ_cur | Kawar et al. 2022 (DDRM), Eq. 4–5, case of an unobserved component (s_i = 0), written in EDM's σ form (Karras et al. 2022). η = 0 is the deterministic DDIM step (Song, Meng and Ermon 2021). |
| Mean of several posterior samples, because the posterior mean minimizes the expected squared error | standard estimation result |
| Standard deviation over posterior samples as an uncertainty map | as in Chung and Ye 2022 |
| 100 steps, η = 1, mean of 8; tuning grid and selection rule | ours |

## 4. Baselines

| what | source |
|---|---|
| U-Net architecture | Ronneberger et al. 2015; fastMRI version in `fastmri/models/unet.py` (fastMRI code, MIT license). `thesis_exp/baselines/unet.py` re-creates it to load the official weights. |
| Pretrained weights `knee_sc_leaderboard_state_dict.pt` (256 channels, 4 pooling layers) | `fastmri_examples/unet/run_pretrained_unet_inference.py` (fastMRI code) |
| How those weights were trained: `combine_train_val=True`; masks 4× with 0.08 center only (script defaults); RMSProp, learning rate 1e-3, 50 epochs | `fastmri_examples/unet/unet_reproduce_20201111.py/unet_knee_sc_leaderboard.py` (fastMRI code) |
| Zero-filled reconstruction: inverse FFT of the undersampled k-space | Zbontar et al. 2018, Section 6.3 (the U-Net input) |

## 5. Literature numbers quoted in the docs

| number | source |
|---|---|
| fastMRI U-Net, single-coil knee, **validation**. Trained on the training set only, with random 4× and 8× masks. 4×: PD 34.00 dB / 0.815, PDFS 29.95 dB / 0.636. 8×: PD 31.50 dB / 0.762, PDFS 28.71 dB / 0.559 (256 channels; the 128-channel row is within 0.06 dB and 0.002 SSIM) | Zbontar et al. 2018, Table 8; training in Sections 6.3 and 4.9 |
| "About 30.1 dB at 8×" for that U-Net | mean of the PD and PDFS values of Table 8 (256 channels) |
| fastMRI U-Net, single-coil knee, **test**: 4× 32.22 dB / 0.754; 8× 29.45 dB / 0.651 | Zbontar et al. 2018, Table 9 |
| fastMRI U-Net, multi-coil knee, validation 4×: PD 0.927–0.929, PDFS 0.864–0.866 | Zbontar et al. 2018, Table 10 |
| fastMRI U-Net, multi-coil knee, test 4×: 35.91 dB / 0.904 | Zbontar et al. 2018, Table 12 |
| i-RIM, single-coil: 4× validation 0.751, test 0.781; 8× validation 0.692, test 0.687. Multi-coil 4× test 0.928 | Putzky et al. 2019, Table 1 |
| i-RIM (team AImsterdam) won the single-coil 4× track of the 2019 fastMRI challenge | Knoll et al. 2020b, Table 2c |

## 6. SSIM diagnosis methods

| what | source |
|---|---|
| SSIM with an 11×11 Gaussian window (σ = 1.5) | Wang et al. 2004; scikit-image `structural_similarity(gaussian_weights=True, sigma=1.5, use_sample_covariance=False)` |
| Noise σ from the Laplacian-difference filter | Immerkær 1996. We use the median absolute deviation instead of the mean absolute value, to be robust to edges (ours). |
| "Known anatomy" image: non-local-means denoising | Buades et al. 2005; scikit-image `denoise_nl_means` (fast mode: Darbon et al. 2008) |
| Knee foreground mask: Otsu threshold | Otsu 1979 |
| Oracle ceiling, SNR definition, blurred-target test, other SSIM conventions | ours |

## Bibliography

### Papers

- **Buades et al. 2005.** A. Buades, B. Coll, J.-M. Morel. A non-local algorithm for image denoising. CVPR 2005, vol. 2,
  pp. 60–65. doi:10.1109/CVPR.2005.38
- **Chung and Ye 2022.** H. Chung, J. C. Ye. Score-based diffusion models for accelerated MRI. Medical Image Analysis 80,
  102479, 2022. arXiv:2110.05243
- **Darbon et al. 2008.** J. Darbon, A. Cunha, T. F. Chan, S. Osher, G. J. Jensen. Fast nonlocal filtering applied to
  electron cryomicroscopy. ISBI 2008, pp. 1331–1334. doi:10.1109/ISBI.2008.4541250
- **Ho et al. 2020.** J. Ho, A. Jain, P. Abbeel. Denoising diffusion probabilistic models. NeurIPS 2020. arXiv:2006.11239
- **Immerkær 1996.** J. Immerkær. Fast noise variance estimation. Computer Vision and Image Understanding 64(2), 300–302,
  1996. doi:10.1006/cviu.1996.0060
- **Jalal et al. 2021.** A. Jalal, M. Arvinte, G. Daras, E. Price, A. G. Dimakis, J. I. Tamir. Robust compressed sensing
  MRI with deep generative priors. NeurIPS 2021. arXiv:2108.01368
- **Karras et al. 2022.** T. Karras, M. Aittala, T. Aila, S. Laine. Elucidating the design space of diffusion-based
  generative models. NeurIPS 2022. arXiv:2206.00364
- **Kawar et al. 2022.** B. Kawar, M. Elad, S. Ermon, J. Song. Denoising diffusion restoration models. NeurIPS 2022.
  arXiv:2201.11793
- **Kingma and Ba 2015.** D. P. Kingma, J. Ba. Adam: a method for stochastic optimization. ICLR 2015. arXiv:1412.6980
- **Knoll et al. 2020a.** F. Knoll, J. Zbontar, A. Sriram, M. J. Muckley, et al. fastMRI: a publicly available raw
  k-space and DICOM dataset of knee images for accelerated MR image reconstruction using machine learning. Radiology:
  Artificial Intelligence 2(1), e190007, 2020. doi:10.1148/ryai.2020190007
- **Knoll et al. 2020b.** F. Knoll, T. Murrell, A. Sriram, N. Yakubova, J. Zbontar, M. Rabbat, A. Defazio,
  M. J. Muckley, D. K. Sodickson, C. L. Zitnick, M. P. Recht. Advancing machine learning for MR image reconstruction
  with an open competition: overview of the 2019 fastMRI challenge. Magnetic Resonance in Medicine 84(6), 3054–3070,
  2020. arXiv:2001.02518
- **Otsu 1979.** N. Otsu. A threshold selection method from gray-level histograms. IEEE Transactions on Systems, Man,
  and Cybernetics 9(1), 62–66, 1979. doi:10.1109/TSMC.1979.4310076
- **Putzky et al. 2019.** P. Putzky, D. Karkalousos, J. Teuwen, N. Miriakov, B. Bakker, M. Caan, M. Welling. i-RIM
  applied to the fastMRI challenge. arXiv:1910.08952, 2019
- **Ronneberger et al. 2015.** O. Ronneberger, P. Fischer, T. Brox. U-Net: convolutional networks for biomedical image
  segmentation. MICCAI 2015. arXiv:1505.04597
- **Shimron et al. 2022.** E. Shimron, J. I. Tamir, K. Wang, M. Lustig. Implicit data crimes: machine learning bias
  arising from misuse of public data. PNAS 119(13), e2117203119, 2022. arXiv:2109.08237
- **Song et al. 2021.** Y. Song, J. Sohl-Dickstein, D. P. Kingma, A. Kumar, S. Ermon, B. Poole. Score-based generative
  modeling through stochastic differential equations. ICLR 2021. arXiv:2011.13456
- **Song, Meng and Ermon 2021.** J. Song, C. Meng, S. Ermon. Denoising diffusion implicit models. ICLR 2021.
  arXiv:2010.02502
- **Song et al. 2022.** Y. Song, L. Shen, L. Xing, S. Ermon. Solving inverse problems in medical imaging with
  score-based generative models. ICLR 2022. arXiv:2111.08005
- **Tygert and Zbontar 2020.** M. Tygert, J. Zbontar. Simulating single-coil MRI from the responses of multiple coils.
  Communications in Applied Mathematics and Computational Science 15(1), 1–13, 2020. arXiv:1811.08026
- **Wang et al. 2004.** Z. Wang, A. C. Bovik, H. R. Sheikh, E. P. Simoncelli. Image quality assessment: from error
  visibility to structural similarity. IEEE Transactions on Image Processing 13(4), 600–612, 2004.
  doi:10.1109/TIP.2003.819861
- **Wang et al. 2023.** Y. Wang, J. Yu, J. Zhang. Zero-shot image restoration using denoising diffusion null-space
  model. ICLR 2023. arXiv:2212.00490
- **Zbontar et al. 2018.** J. Zbontar, F. Knoll, A. Sriram, et al. fastMRI: an open dataset and benchmarks for
  accelerated MRI. arXiv:1811.08839 (v2, 2019)

### Code, data and web pages

- **fastMRI code** (MIT license): https://github.com/facebookresearch/fastMRI. Files used: `fastmri/evaluate.py`,
  `fastmri/data/subsample.py`, `fastmri/data/transforms.py`, `fastmri/models/unet.py`,
  `fastmri_examples/unet/run_pretrained_unet_inference.py`,
  `fastmri_examples/unet/unet_reproduce_20201111.py/unet_knee_sc_leaderboard.py`.
- **Pretrained U-Net weights:** https://dl.fbaipublicfiles.com/fastMRI/trained_models/unet/knee_sc_leaderboard_state_dict.pt
- **fastMRI GitHub Discussion #293** (leaderboard shutdown; split the validation set):
  https://github.com/facebookresearch/fastMRI/discussions/293
- **fastMRI data:** https://fastmri.med.nyu.edu/
- **NVlabs/edm** (CC BY-NC-SA 4.0): https://github.com/NVlabs/edm. `thesis_exp/edm/` re-implements its `SongUNet`,
  `EDMPrecond`, `EDMLoss`, EMA schedule and `edm_sampler`.
- **i-RIM code:** https://github.com/pputzky/irim_fastMRI
- **scikit-image** (`structural_similarity`, `peak_signal_noise_ratio`, `denoise_nl_means`, `threshold_otsu`):
  https://scikit-image.org
