# Throughput (RTX A6000 48 GB, PyTorch 2.14.1, CUDA 13.0)

Single GPU, full training step (loss, backward, Adam, EMA), 320×320, 54.0M parameters. Per image, one
forward plus backward pass costs 2.49 TFLOP.

| precision | channels-last | compile | micro-batch | img/s | memory |
|---|---|---|---|---|---|
| fp32 (TF32) | no | no | 8 | 9.0 | 35.3 GB |
| fp32 (TF32) | no | no | 16 | OOM | |
| bf16 | no | no | 8 / 16 / 20 | 16.7 / 17.1 / 17.3 | 18 / 35 / 44 GB |
| bf16 | yes | no | 8 / 16 / 20 | 9.2 / 9.3 / 9.3 | 18 / 35 / 44 GB |
| bf16 | no | yes | 8 / 16 | 19.1 / 19.7 | 14 / 27 GB |
| **bf16** | **yes** | **yes** | 8 / **16** / 20 / 24 | 20.1 / **20.2** / 20.3 / 20.2 | 14 / **27** / 33 / 40 GB |
| bf16, max-autotune | yes | yes | 16 | 20.5 | 27 GB (21 min compile) |

## Profile
- Eager bf16 (NCHW): 17% of GPU time goes to cuDNN layout conversions (NCHW↔NHWC) and about 30% to
  elementwise kernels (GroupNorm, SiLU, adds, dropout).
- Channels-last in eager mode is slower overall, because of the GroupNorm kernels.
- With channels-last plus `torch.compile`, the conversions disappear and GroupNorm, SiLU and dropout
  fuse into Triton kernels. Convolutions then take about 45% of the time, at about 50 TFLOP/s effective.

## Multi-GPU (DDP, real data loader)
- 3 GPUs: 56.4 img/s (18.8 per GPU, 93% of single-GPU speed).
- Compiling the whole DDP model (to overlap gradient all-reduce with the backward pass) gave
  56.5–57.0 img/s. No gain, so only the U-Net is compiled.

## Chosen settings
- bf16, channels-last, `torch.compile` (default mode), micro-batch 16 per GPU.
- 4 GPUs × 16 = global batch 64, with no gradient accumulation.
- 8 data-loader workers per GPU reading a memory-mapped float32 cache (28.5 GB), which fits in page cache.

## Rejected
- fp32: 2.2× slower.
- Channels-last without compile: slower.
- `max-autotune`: +1.4% for a 21-minute compile at every (re)start.
- Larger micro-batches: no gain.

## Data preparation
The archives are multi-block xz. The system `xz` (5.2.5) decompresses on a single thread at about
15 MB/s. `thesis_exp/data/xz_parallel.py` decompresses the blocks in parallel with the standard
library and pipes into `tar -x`:
- about 300 MB/s
- output byte-identical to `xz -dc`
- train (94 GB) unpacks in about 6 minutes instead of about 1.5 hours.
