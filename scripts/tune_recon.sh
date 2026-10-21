#!/usr/bin/env bash
# Sampler settings evaluated on the tuning set (first 10 validation volumes), ranked by mean SSIM.
#   CKPT=runs/main/ema-00010000.pt GPUS=0,1,2,3 bash scripts/tune_recon.sh
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=${OUT:-runs/main/tune}
LIMIT=${LIMIT:-10}
GRID=${GRID:-"eta0.5 recon.eta=0.5
eta1.0 recon.eta=1.0
eta1.0_smin0.15 recon.eta=1.0 recon.sigma_min=0.15
eta1.0_avg4 recon.eta=1.0 recon.num_samples=4
eta1.0_smin0.15_avg4 recon.eta=1.0 recon.sigma_min=0.15 recon.num_samples=4
eta1.0_steps200 recon.eta=1.0 recon.num_steps=200"}
while read -r name overrides; do
  [ -f "$OUT/$name/metrics.json" ] && continue
  # shellcheck disable=SC2086
  bash scripts/torchrun.sh scripts/evaluate.py --ckpt "$CKPT" --out "$OUT/$name" --limit "$LIMIT" \
    train.channels_last=false $overrides < /dev/null
done <<< "$GRID"
.venv/bin/python scripts/summarize_tuning.py "$OUT"
