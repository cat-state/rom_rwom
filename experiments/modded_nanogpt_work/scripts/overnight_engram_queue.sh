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
    sleep 120
  done
}

run_engram_job() {
  local gpu="$1"
  local run="$2"
  shift 2
  if [[ -f "logs/${run}.txt" || -f "logs/${run}.console.txt" ]]; then
    echo "$(date -Is) skip existing ${run}"
    return 0
  fi
  echo "$(date -Is) launch gpu${gpu} ${run}"
  (
    export CUDA_VISIBLE_DEVICES="$gpu"
    export RUN_ID="$run"
    export WANDB=1
    export HIT_LR_EXPONENT=0
    export ENGRAM_UNTIED_PROJ=1
    export NUM_EXTENSION_ITERATIONS=0
    export SAVE_CHECKPOINT="${SAVE_CHECKPOINT:-0}"
    export SAVE_CHECKPOINT_EVERY="${SAVE_CHECKPOINT_EVERY:-0}"
    export WANDB_HIST_EVERY="${WANDB_HIST_EVERY:-250}"
    export WANDB_HIST_ROWS="${WANDB_HIST_ROWS:-131072}"
    for kv in "$@"; do
      export "$kv"
    done
    bash scripts/run_engram_bf80_hitlr_exponent.sh
  ) > "logs/${run}.launch.txt" 2>&1
  echo "$(date -Is) finished ${run} exit=$?"
}

latest_val_loss() {
  local file="$1"
  local py="${PYTHON_BIN:-python3}"
  "$py" - "$file" <<'PY'
import re, sys
path = sys.argv[1]
last = ""
try:
    with open(path, errors="replace") as f:
        for line in f:
            m = re.search(r"step:(\d+)/(\d+) val_loss:([0-9.]+)", line)
            if m:
                last = m.group(3)
except FileNotFoundError:
    pass
print(last)
PY
}

run_h1_gain_diag() {
  local gpu="$1"
  local gain_label="$2"
  local gain_value="$3"
  local ts
  ts="$(date +%Y%m%d_%H%M%S)"
  run_engram_job "$gpu" "diagnostic_h1_gain${gain_label}_bf80_hitlrexp0_500_${ts}" \
    ENGRAM_HEADS=1 ENGRAM_ATTNRES_MERGE_GAIN="$gain_value" \
    NUM_SCHEDULED_ITERATIONS=500 VAL_LOSS_EVERY=250 SAVE_CHECKPOINT=0
}

run_h3_gain_diag() {
  local gpu="$1"
  local gain_label="$2"
  local gain_value="$3"
  local ts
  ts="$(date +%Y%m%d_%H%M%S)"
  run_engram_job "$gpu" "diagnostic_h3_gain${gain_label}_bf80_hitlrexp0_500_${ts}" \
    ENGRAM_HEADS=3 ENGRAM_ATTNRES_MERGE_GAIN="$gain_value" \
    NUM_SCHEDULED_ITERATIONS=500 VAL_LOSS_EVERY=250 SAVE_CHECKPOINT=0
}

pair_gpu0() {
  wait_gpu_free 0
  run_h3_gain_diag 0 1p25 1.25
  run_h3_gain_diag 0 1p75 1.75
}

pair_gpu1() {
  wait_gpu_free 1
  local bf160_log
  bf160_log="$(ls -t logs/diagnostic_h1_gain1p5_bf160_hitlrexp0_500_*.txt 2>/dev/null | grep -vE '\.(console|launch)\.txt$' | head -1 || true)"
  local bf160_loss
  bf160_loss="$(latest_val_loss "$bf160_log")"
  echo "$(date -Is) bf160 diagnostic log=${bf160_log:-none} final_loss=${bf160_loss:-none}"
  if [[ -n "$bf160_loss" ]] && "${PYTHON_BIN:-python3}" - "$bf160_loss" <<'PY'
import sys
sys.exit(0 if float(sys.argv[1]) <= 3.5305 else 1)
PY
  then
    local ts
    ts="$(date +%Y%m%d_%H%M%S)"
    run_engram_job 1 "full_h1_gain1p5_bf160_hitlrexp0_1500_ckpt_${ts}" \
      BIGRAM_FACTOR=160 ENGRAM_HEADS=1 ENGRAM_ATTNRES_MERGE_GAIN=1.5 \
      ENGRAM_ATTNRES_METRICS=0 \
      NUM_SCHEDULED_ITERATIONS=1500 VAL_LOSS_EVERY=250 SAVE_CHECKPOINT=1 SAVE_CHECKPOINT_EVERY=500
  else
    run_h1_gain_diag 1 1p25 1.25
    run_h1_gain_diag 1 1p75 1.75
  fi
}

single_gpu0() {
  wait_gpu_free 0
  run_h1_gain_diag 0 1p25 1.25
  run_h1_gain_diag 0 1p75 1.75
}

if [[ "$mode" == "pair" ]]; then
  pair_gpu0 &
  pair_gpu1 &
  wait
else
  single_gpu0
fi
