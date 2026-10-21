#!/usr/bin/env bash
# Resumable download (curl -L -C -) and parallel extraction of fastMRI knee single-coil val + train.
# Extraction equals tar -xJf, but decompresses the archive's xz blocks on many cores.
set -euo pipefail

LINKS=${LINKS:-$HOME/fastmri_links.txt}
RAW=${RAW:-$(cd "$(dirname "$0")/.." && pwd)/data/raw}
BUDGET=${BUDGET:-200000000000}
MARGIN=${MARGIN:-15000000000}
PY=${PY:-$(cd "$(dirname "$0")/.." && pwd)/.venv/bin/python}

url_for() { grep -o "https://[^[:space:]]*/$1\.tar\.xz?[^[:space:]]*" "$LINKS" | head -1; }

remote_size() {
  curl -sS -r 0-0 -D - -o /dev/null "$(url_for "$1")" | tr -d '\r' \
    | awk -F/ 'tolower($0) ~ /^content-range/ {print $NF}'
}

local_size() { stat -c %s "$1.tar.xz" 2>/dev/null || echo 0; }

gb() { echo $(( $1 / 1000000000 )); }

fetch() {
  local total=""
  until [ -n "$total" ]; do
    total=$(remote_size "$1" || true)
    [ -n "$total" ] || sleep 30
  done
  echo "$(date +%T) fetch $1 ($(gb "$total") GB)"
  while [ "$(local_size "$1")" -lt "$total" ]; do
    curl -fL -C - --retry 10 --retry-delay 30 -sS -o "$1.tar.xz" "$(url_for "$1")" || sleep 60
  done
  [ "$(local_size "$1")" -eq "$total" ] || { echo "size mismatch for $1"; return 1; }
  echo "$(date +%T) fetched $1"
}

extract() {
  echo "$(date +%T) extract $1"
  "$PY" -m thesis_exp.data.xz_parallel "$1.tar.xz" --workers 48 | tar -x
  touch "$1.extracted"
  echo "$(date +%T) extracted $1"
}

extracted_bytes() { du -sb --exclude='*.tar.xz' . | cut -f1; }

train_fits() {
  local val_bytes=$1 val_tar=$2 train_tar pred free
  train_tar=$(local_size knee_singlecoil_train)
  pred=$(( train_tar * val_bytes / val_tar ))
  free=$(df -B1 --output=avail . | tail -1)
  echo "val extracted $(gb "$val_bytes") GB; train tar $(gb "$train_tar") GB;" \
       "predicted train extracted $(gb "$pred") GB; free $(gb "$free") GB"
  [ $(( val_bytes + train_tar + pred )) -le "$BUDGET" ] && [ $(( pred + MARGIN )) -le "$free" ]
}

main() {
  mkdir -p "$RAW" && cd "$RAW"
  local val_pid train_pid val_tar val_bytes
  if [ ! -f knee_singlecoil_val.extracted ]; then fetch knee_singlecoil_val & val_pid=$!; fi
  if [ ! -f knee_singlecoil_train.extracted ]; then fetch knee_singlecoil_train & train_pid=$!; fi
  [ -z "${val_pid:-}" ] || wait "$val_pid"
  [ -z "${train_pid:-}" ] || wait "$train_pid"
  if [ ! -f knee_singlecoil_val.extracted ]; then
    val_tar=$(local_size knee_singlecoil_val)
    extract knee_singlecoil_val
    echo "$val_tar" > knee_singlecoil_val.tarbytes
    rm -f knee_singlecoil_val.tar.xz
  fi
  val_bytes=$(extracted_bytes)
  if [ ! -f knee_singlecoil_train.extracted ]; then
    if train_fits "$val_bytes" "$(cat knee_singlecoil_val.tarbytes)"; then
      extract knee_singlecoil_train
    else
      echo "$(date +%T) train extraction would exceed the disk budget; stopping"
      touch NEEDS_STREAM
      exit 2
    fi
  fi
  rm -f knee_singlecoil_train.tar.xz
  for d in */; do echo "$d $(ls "$d" | wc -l) files"; done
  df -h . | tail -1
  touch DONE
  echo "$(date +%T) done"
}

main "$@"
