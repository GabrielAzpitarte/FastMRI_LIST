# FastMRI-EDM

An EDM diffusion prior trained from scratch on fastMRI knee single-coil data and evaluated on accelerated
reconstruction. Choices are in `docs/SPEC.md`, throughput in `docs/BENCHMARK.md`.

```bash
# environment
uv venv .venv && VIRTUAL_ENV=.venv uv pip install -e . pytest

# data: resumable download, parallel unpacking (links file from the fastMRI access email)
LINKS=~/fastmri_links.txt bash scripts/download_data.sh
python scripts/preprocess.py data/raw/singlecoil_val data/cache/val
python scripts/preprocess.py data/raw/singlecoil_train data/cache/train

# tests (GPU tests run when a GPU is visible)
python -m pytest -q

# throughput
bash scripts/benchmark.sh
python scripts/profile_step.py --config configs/main.yaml --micro-batch 16

# training (resumes from the newest state in the run directory; touch runs/main/STOP to stop cleanly)
# scripts/torchrun.sh keeps compiler caches and temp files under .cache/ (the root disk is full)
GPUS=0,1,2,3 bash scripts/torchrun.sh scripts/train.py --config configs/main.yaml --run-dir runs/main

# evaluation of an EMA snapshot (first 10 validation volumes are the tuning set)
GPUS=0,1,2,3 bash scripts/torchrun.sh scripts/evaluate.py --ckpt runs/main/ema-00010000.pt \
    --out runs/main/eval --skip 10
```

Layout: `fastmri_edm/mri` (FFTs, masks, operator, scaling), `data` (HDF5, cache, dataset,
xz unpacking), `edm` (network, preconditioning, loss, sampler, EMA), `train` (loop, checkpoints,
distributed helpers), `eval` (metrics, reconstruction, progress checks).
