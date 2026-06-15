#!/usr/bin/env bash
set -euo pipefail

# Built-in modded-nanogpt bigram baseline. This intentionally leaves
# ROM_LAYERS unset so the default speedrun path injects bigram features
# through every layer.
BF="${BIGRAM_FACTOR:-5}"
export RUN_ID="${RUN_ID:-builtin_bigram_bf${BF}_${NUM_SCHEDULED_ITERATIONS:-1500}_$(date +%Y%m%d_%H%M%S)}"
export GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-8}"
export BIGRAM_FACTOR="${BIGRAM_FACTOR:-5}"
export NUM_SCHEDULED_ITERATIONS="${NUM_SCHEDULED_ITERATIONS:-1500}"
export NUM_EXTENSION_ITERATIONS="${NUM_EXTENSION_ITERATIONS:-0}"
export VAL_LOSS_EVERY="${VAL_LOSS_EVERY:-500}"
export SAVE_CHECKPOINT="${SAVE_CHECKPOINT:-0}"
export SAVE_CHECKPOINT_EVERY="${SAVE_CHECKPOINT_EVERY:-0}"

unset ENGRAM_BIGRAM
unset ROM_BIGRAM
unset ROM_TOKEN
unset ROM_LAYERS
unset ROM_LAYER_ONLY
unset ENGRAM_SPARSE_ADAM
unset ENGRAM_SPARSE_VECTOR_ADAM
unset ENGRAM_ADAM_EVERY_STEP
unset ENGRAM_ATTNRES_MERGE
unset ENGRAM_SHORT_CONV

export WANDB="${WANDB:-1}"
export WANDB_GROUP="${WANDB_GROUP:-builtin-bigram-bf-scaling}"
export WANDB_NAME="${WANDB_NAME:-$RUN_ID}"
export WANDB_TAGS="${WANDB_TAGS:-builtin_bigram baseline bf_sweep}"
export COMPILE_DENSE_LAYER_BODY="${COMPILE_DENSE_LAYER_BODY:-1}"
export COMPILE_MODEL="${COMPILE_MODEL:-0}"
export COMPILE_LAYER_MODULES="${COMPILE_LAYER_MODULES:-0}"
export ENGRAM_UPDATE_METRICS=0
export ENGRAM_HIT_HIST=0
export WANDB_HISTOGRAMS="${WANDB_HISTOGRAMS:-0}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/env_cuda_uv.sh"

mkdir -p logs
exec "${UV_BIN:-uv}" run --python "${PYTHON_BIN:-python}" python -m torch.distributed.run --standalone --nproc_per_node=1 train_gpt.py 2>&1 | tee "logs/${RUN_ID}.console.txt"
