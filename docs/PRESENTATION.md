# Slide outline (about 10–15 minutes)

Every figure is in `docs/figures/`. Each slide has one figure and 2–3 points to say.

**1. Title.** A diffusion prior for accelerated knee MRI, trained from scratch on fastMRI.
- Goal: train a generative model of knee MRI images and use it to reconstruct scans acquired 4× and 8×
  faster.

**2. The problem.** `kspace_and_masks.png`
- MRI measures k-space (frequencies) line by line. Skipping lines makes the scan 4× or 8× faster.
- Filling the missing lines with zeros ("zero-filled") gives blurry, aliased images.
- The task is to recover the missing information.

**3. The idea.** `trajectory_8x.png`
- **Training:** a diffusion model (EDM) learns only to remove noise from knee images, so it learns what
  knees look like. It never sees undersampled scans.
- **Reconstruction:** start from pure noise and denoise step by step. After every step, put the measured
  k-space lines back (data consistency).
- Because the measurements enter only at test time, the same model works for any acceleration or mask.

**4. Training.** `generated_vs_real.png` (or `samples_during_training.png`)
- **Data:** fastMRI knee single-coil, 973 training volumes (34,742 slices), complex 320×320 images.
- **Model and run:** a 54M-parameter U-Net with EDM preconditioning. 4.1 M images seen (118 epochs), 16.5 h
  on 4 GPUs.
- Samples generated from pure noise look like real knees, in both contrasts (PD and fat-suppressed).

**5. Is a loss of 0.3 bad?** `training_loss.png`, `loss_per_noise_level.png`
- In EDM's scaling, 1.0 means the network does nothing. 0.3 means it removes about 70% of that error.
- At small noise levels the loss cannot go down, because the scans contain their own noise
  (σ ≈ 0.12). That noise is indistinguishable from the noise we add, and those levels dominate the
  average.

**6. When to stop training.** `checkpoint_plateau.png`
- Reconstruction quality was flat from 1 to 3.6 M images (within ±0.1 dB on the same volumes and seeds).
- So training stopped at 4.1 M instead of the planned 10 M, saving about 20 GPU-hours.
- The measurements do most of the work. The prior only has to be good enough.

**7. Sampler tuning.** `sampler_tuning.png`
- Tuned on 10 separate volumes. The selection rule was fixed before tuning.
- Full stochasticity (η=1) gives +1.6–2.4 dB. Averaging 8 posterior samples gives +0.6 dB.
- More steps and stopping early do not help.

**8. The baseline.** The official pretrained fastMRI U-Net.
- It is a supervised model: it learns the mapping from undersampled to fully sampled images directly from
  pairs.
- It was trained for 4× only, and on the training plus validation sets. So it has seen our test volumes,
  which is an advantage for it.

**9. Main result.** `final_results.png`
- **4×:** 32.10 dB for both, a tie (0.00 ± 0.05 dB). The U-Net is 0.007 higher on SSIM.
- **8×:** EDM 30.28 dB vs U-Net 27.41 dB, **+2.9 dB**.
- 189 test volumes. Both models get exactly the same inputs, masks and metrics.

**10. Per volume.** `per_volume_scatter.png` and `final_by_contrast.png`
- **8×:** EDM is better on 173 of 189 volumes, and on all 94 PD volumes.
- **4×:** EDM wins on PD scans (+0.33 dB) and loses on the noisier PDFS scans (−0.32 dB). The U-Net
  learned to smooth that noise away.

**11. What the images look like.** `comparison_8x.png`, `zoom_8x.png`
- At 8× the U-Net leaves aliasing streaks. It is outside its training setting.
- EDM recovers bone edges, cartilage and texture.
- At 4× (`comparison_4x.png`, `zoom_4x.png`) both are good. The U-Net is smoother; EDM is sharper.

**12. One model, any acceleration.** `acceleration_sweep.png`
- The same EDM, from 2× to 16×, compared with the U-Net trained for 4×.
- EDM beats the U-Net at every acceleration **except 4×**, the one the U-Net was trained for. That includes
  2× (+0.9 dB).
- At 16×, EDM still reaches 27.8 dB. The U-Net (25.2 dB) is no better than zero-filling there.
- This is a quick sweep (6 test volumes, central 4 slices, mean of 4 samples), not the full evaluation.

**13. Bonus: uncertainty for free.** `uncertainty_8x.png`
- Every reconstruction is a sample from the posterior. Their spread is a pixel-wise uncertainty map.
- It lights up at edges and fine structures, where the error is largest. A supervised U-Net cannot
  provide this.

**14. Caveats and next steps.**
- The U-Net saw the test volumes in training, and was trained for 4× only. The fair comparison is a
  U-Net trained on the training split at both accelerations (the next experiment).
- One trained model and one sampling seed per volume. The undersampling is applied to the cropped 320×320
  k-space, as in most diffusion-prior papers.
- **Ideas:** more samples on PDFS, faster samplers (fewer steps), multi-coil data, other anatomies.

**15. What I learned.**
- Diffusion training, and what the loss does and does not mean.
- Data consistency for inverse problems.
- Evaluating fairly: tuning vs test split, paired comparisons, checking a baseline's training data.
- PSNR rewards averaged (posterior-mean) outputs. Individual samples look sharper but score lower.
