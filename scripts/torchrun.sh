#!/usr/bin/env bash
# torchrun on the GPUs in $GPUS, with compiler caches and temp files kept under the project (not /tmp).
#   GPUS=0,1,2,3 bash scripts/torchrun.sh scripts/train.py --config configs/main.yaml --run-dir runs/main
set -euo pipefail
cd "$(dirname "$0")/.."
export TORCHINDUCTOR_CACHE_DIR=$PWD/.cache/inductor
export TRITON_CACHE_DIR=$PWD/.cache/triton
export TMPDIR=$PWD/.cache/tmp
mkdir -p "$TMPDIR"
GPUS=${GPUS:-0,1,2,3}
NPROC=$(( $(tr -cd , <<< "$GPUS" | wc -c) + 1 ))
CUDA_VISIBLE_DEVICES=$GPUS exec .venv/bin/torchrun --standalone --nproc_per_node="$NPROC" "$@"
