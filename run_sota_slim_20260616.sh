#!/usr/bin/env bash
set -euo pipefail

cd /root/modded-nanogpt
mkdir -p logs

export UV_BIN="${UV_BIN:-/root/.local/bin/uv}"
export PYTHON_BIN="${PYTHON_BIN:-/root/.venvs/modded-nanogpt/bin/python}"
source scripts/env_cuda_uv.sh

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export SOTA_PATHWAY="${SOTA_PATHWAY:-sota_k2}"
export BIGRAM_FACTOR="${BIGRAM_FACTOR:-300}"
export MODEL_SEED="${MODEL_SEED:-5}"
export TRAIN_DATA_SEED="${TRAIN_DATA_SEED:-0}"
export ENGRAM_HASH_SEED="${ENGRAM_HASH_SEED:-0}"
export NUM_SCHEDULED_ITERATIONS="${NUM_SCHEDULED_ITERATIONS:-1500}"
export VAL_LOSS_EVERY="${VAL_LOSS_EVERY:-250}"

export WANDB="${WANDB:-1}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_PROJECT="${WANDB_PROJECT:-rom-rwom}"
export WANDB_ENTITY="${WANDB_ENTITY:-uwu1}"
export WANDB_GROUP="${WANDB_GROUP:-sota-slim-20260616}"
export WANDB_TAGS="${WANDB_TAGS:-engram,sota,slim}"
export WANDB_HISTOGRAMS="${WANDB_HISTOGRAMS:-1}"
export WANDB_HIST_EVERY="${WANDB_HIST_EVERY:-250}"
export WANDB_HIST_ROWS="${WANDB_HIST_ROWS:-131072}"

if [[ -z "${RUN_ID:-}" ]]; then
  if [[ "$SOTA_PATHWAY" == "sota_k2" ]]; then
    export RUN_ID="bf${BIGRAM_FACTOR}_sota_k2_headmix_layerdelta_norowsigns_hashseed${ENGRAM_HASH_SEED}_seed${MODEL_SEED}_${NUM_SCHEDULED_ITERATIONS}_sotafile"
  else
    export RUN_ID="bf${BIGRAM_FACTOR}_${SOTA_PATHWAY}_seed${MODEL_SEED}_${NUM_SCHEDULED_ITERATIONS}_sotafile"
  fi
fi
export WANDB_NAME="${WANDB_NAME:-$RUN_ID}"

if grep -q "step:${NUM_SCHEDULED_ITERATIONS}/${NUM_SCHEDULED_ITERATIONS} val_loss" "logs/${RUN_ID}.txt" 2>/dev/null; then
  echo "$(date -Is) skip complete ${RUN_ID}"
  exit 0
fi

exec "$UV_BIN" run --python "$PYTHON_BIN" python -m torch.distributed.run \
  --standalone --nproc_per_node=1 train_gpt_sota.py \
  --pathway "$SOTA_PATHWAY" \
  > "logs/${RUN_ID}.console.txt" 2>&1
