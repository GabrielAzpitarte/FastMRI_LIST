#!/usr/bin/env bash
# Single-GPU throughput sweep: one variant per GPU, micro-batches in sequence.
set -u
cd "$(dirname "$0")/.."
OUT=${OUT:-logs/benchmark.jsonl}
read -r -a GPUS <<< "${GPUS:-0 1 2 3}"
VARIANTS=(
  "train.precision=fp32 train.channels_last=false train.compile=false"
  "train.precision=bf16 train.channels_last=false train.compile=false"
  "train.precision=bf16 train.channels_last=true train.compile=false"
  "train.precision=bf16 train.channels_last=true train.compile=true"
)
for i in "${!VARIANTS[@]}"; do
  (
    for mb in ${MICRO_BATCHES:-8 16 20 24}; do
      # shellcheck disable=SC2086
      CUDA_VISIBLE_DEVICES=${GPUS[$i]} .venv/bin/python scripts/benchmark.py --config configs/main.yaml \
        --micro-batch "$mb" ${VARIANTS[$i]} >> "$OUT" 2>> logs/benchmark.err
    done
  ) &
done
wait
