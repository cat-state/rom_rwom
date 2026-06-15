#!/usr/bin/env bash
set -euo pipefail

RUN_DIR="${RUN_DIR:-logs/bf120_h1_layerpart_tail1_origsched_ckpt_1500_20260519_0840}"
CKPT="${CKPT:-${RUN_DIR}/state_step001500.pt}"
VAL_TOKENS="${VAL_TOKENS:-1048576}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-1048576}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export CUDA_VISIBLE_DEVICES

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/env_cuda_uv.sh"

"${UV_BIN:-uv}" run --python "${PYTHON_BIN:-python}" python scripts/make_punctuation_mask_hists.py \
  --run-dir "${RUN_DIR}" \
  --analysis engram_analysis_step001500_top1000.json \
  --hist engram_hit_hist_step001500.pt \
  --limits 25,50

"${UV_BIN:-uv}" run --python "${PYTHON_BIN:-python}" python scripts/make_semantic_mask_hists.py \
  --run-dir "${RUN_DIR}" \
  --analysis engram_analysis_step001500_top1000.json \
  --hist engram_hit_hist_step001500.pt \
  --limits 25,50

run_eval() {
  local label="$1"
  local hist="$2"
  local run_id="eval_bf120_${label}_$(date +%Y%m%d_%H%M%S)"
  echo "=== eval ${label} hist=${hist}"
  RUN_ID="${run_id}" \
  WANDB=0 \
  ENGRAM_EVAL_CKPT="${CKPT}" \
  ENGRAM_EVAL_OUT="reports/engram_bf120_counterfactuals/${label}.json" \
  ENGRAM_HIT_HIST=1 \
  ENGRAM_HIT_HIST_LOAD="${hist}" \
  ENGRAM_MASK_UNHIT_EVAL=1 \
  BIGRAM_FACTOR=120 \
  GRAD_ACCUM_STEPS=16 \
  ENGRAM_BIGRAM=1 \
  ROM_LAYERS=2,8 \
  NUM_SCHEDULED_ITERATIONS=1460 \
  NUM_EXTENSION_ITERATIONS=40 \
  HIT_LR_EXPONENT=0 \
  ENGRAM_DIM=768 \
  ENGRAM_HEADS=1 \
  ENGRAM_PER_HEAD=1 \
  ENGRAM_LAYER_HASHES=1 \
  ENGRAM_LAYER_READOUTS=1 \
  ENGRAM_LAYER_PARTITIONS=1 \
  ENGRAM_CANONICALIZE=1 \
  ENGRAM_UNTIED_PROJ=1 \
  ENGRAM_SHORT_CONV=1 \
  ENGRAM_NORMALIZE_READOUT=1 \
  ENGRAM_ATTNRES_MERGE=1 \
  ENGRAM_ATTNRES_MERGE_GAIN=1.5 \
  ENGRAM_SPARSE_ADAM=1 \
  ENGRAM_SPARSE_HIT_LR=0 \
  ENGRAM_SPARSE_ADAM_TAIL_STEPS=1 \
  ENGRAM_LR_MUL=6.34615384615 \
  ENGRAM_INIT_STD=0.01 \
  VAL_TOKENS="${VAL_TOKENS}" \
  VAL_BATCH_SIZE="${VAL_BATCH_SIZE}" \
  COMPILE_MODEL=0 \
  COMPILE_LAYER_MODULES=0 \
  COMPILE_DENSE_LAYER_BODY=0 \
  "${UV_BIN:-uv}" run --python "${PYTHON_BIN:-python}" python -m torch.distributed.run --standalone --nproc_per_node=1 train_gpt.py
}

mkdir -p reports/engram_bf120_counterfactuals

run_eval base_mask_unhit "${RUN_DIR}/engram_hit_hist_step001500.pt"
run_eval punct_top25 "${RUN_DIR}/engram_hit_hist_step001500_mask_punct_top25.pt"
run_eval punct_top50 "${RUN_DIR}/engram_hit_hist_step001500_mask_punct_top50.pt"
run_eval semantic_top25 "${RUN_DIR}/engram_hit_hist_step001500_mask_semantic_top25.pt"
run_eval semantic_top50 "${RUN_DIR}/engram_hit_hist_step001500_mask_semantic_top50.pt"
