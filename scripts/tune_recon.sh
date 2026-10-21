#!/usr/bin/env bash
# Grid over sampler steps and stochasticity (eta) on the tuning set (first 10 validation volumes).
#   CKPT=runs/main/ema-00010000.pt GPUS=0,1,2,3 bash scripts/tune_recon.sh
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=${OUT:-runs/main/tune}
for steps in ${STEPS:-50 100 200}; do
  for eta in ${ETAS:-0.0 0.5 1.0}; do
    bash scripts/torchrun.sh scripts/evaluate.py --ckpt "$CKPT" --out "$OUT/steps${steps}_eta${eta}" \
      --limit 10 recon.num_steps="$steps" recon.eta="$eta" train.channels_last=false
  done
done
.venv/bin/python scripts/summarize_tuning.py "$OUT"
