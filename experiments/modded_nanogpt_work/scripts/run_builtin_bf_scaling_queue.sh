#!/usr/bin/env bash
set -uo pipefail

mode="${1:-}"
if [[ "$mode" != "pair" && "$mode" != "single" ]]; then
  echo "usage: $0 pair|single" >&2
  exit 2
fi

mkdir -p logs

gpu_mem_mb() {
  local gpu="$1"
  nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$gpu" | tr -d ' '
}

wait_gpu_free() {
  local gpu="$1"
  local threshold="${2:-10000}"
  while true; do
    local mem
    mem="$(gpu_mem_mb "$gpu" || echo 999999)"
    if [[ "$mem" =~ ^[0-9]+$ && "$mem" -lt "$threshold" ]]; then
      echo "$(date -Is) gpu${gpu} free mem=${mem}MiB"
      return 0
    fi
    echo "$(date -Is) waiting gpu${gpu} mem=${mem}MiB"
    sleep 60
  done
}

run_baseline_bf() {
  local gpu="$1"
  local bf="$2"
  local ts
  ts="$(date +%Y%m%d_%H%M%S)"
  local run="builtin_bigram_bf${bf}_1500_${ts}"
  echo "$(date -Is) launch gpu${gpu} ${run}"
  (
    export CUDA_VISIBLE_DEVICES="$gpu"
    export BIGRAM_FACTOR="$bf"
    export RUN_ID="$run"
    export NUM_SCHEDULED_ITERATIONS=1500
    export NUM_EXTENSION_ITERATIONS=0
    export VAL_LOSS_EVERY=500
    export SAVE_CHECKPOINT=0
    export WANDB=1
    bash scripts/run_builtin_bigram_baseline.sh
  ) > "logs/${run}.launch.txt" 2>&1
  echo "$(date -Is) finished ${run} exit=$?"
}

if [[ "$mode" == "pair" ]]; then
  (
    for bf in 5 20 80; do
      wait_gpu_free 0
      run_baseline_bf 0 "$bf"
    done
  ) &
  (
    for bf in 10 40; do
      wait_gpu_free 1
      run_baseline_bf 1 "$bf"
    done
  ) &
  wait
else
  for bf in 5 10 20 40 80; do
    wait_gpu_free 0
    run_baseline_bf 0 "$bf"
  done
fi
