#!/usr/bin/env bash
set -euo pipefail

# BF80 Engram with 4 identity-HC streams and the old inverse hit-count LR rule.
# Override RUN_ID, BIGRAM_FACTOR, GRAD_ACCUM_STEPS, or VAL_LOSS_EVERY from the environment if needed.
export RUN_ID="${RUN_ID:-engram_bf80_identityhc_inversehit_canon_layers2_8_dim768_h6_ng3_ga8_1500_$(date +%Y%m%d_%H%M%S)}"
export GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-8}"
export BIGRAM_FACTOR="${BIGRAM_FACTOR:-80}"
export NUM_SCHEDULED_ITERATIONS="${NUM_SCHEDULED_ITERATIONS:-1460}"
export NUM_EXTENSION_ITERATIONS="${NUM_EXTENSION_ITERATIONS:-40}"
export VAL_LOSS_EVERY="${VAL_LOSS_EVERY:-250}"

export ENGRAM_BIGRAM=1
export ROM_LAYERS="${ROM_LAYERS:-2,8}"
export ENGRAM_DIM="${ENGRAM_DIM:-768}"
export ENGRAM_HEADS="${ENGRAM_HEADS:-6}"
export ENGRAM_MAX_NGRAM="${ENGRAM_MAX_NGRAM:-3}"
export ENGRAM_LAYER_HASHES=1
export ENGRAM_LAYER_READOUTS=1
export ENGRAM_PER_HEAD=1
export ENGRAM_CANONICALIZE=1

export ENGRAM_MHC=1
export ENGRAM_MHC_STREAMS=4
export ENGRAM_MHC_IDENTITY=1
export ENGRAM_MHC_DYNAMIC=1
export ENGRAM_MHC_DELTA=0

export ENGRAM_SPARSE_ADAM=1
export ENGRAM_ADAM_EVERY_STEP=1
export ENGRAM_LR_MUL="${ENGRAM_LR_MUL:-6.34615384615}"
export ENGRAM_LR_FLOOR=0
export ENGRAM_SPARSE_HIT_LR=1
export ENGRAM_SPARSE_HIT_LR_EXPONENT=-0.5
export ENGRAM_SPARSE_HIT_LR_MIN=0
export ENGRAM_SPARSE_HIT_LR_MAX=1000000000

export ENGRAM_UPDATE_METRICS="${ENGRAM_UPDATE_METRICS:-1}"
export ENGRAM_UPDATE_METRICS_EVERY="${ENGRAM_UPDATE_METRICS_EVERY:-125}"
export COMPILE_MODEL="${COMPILE_MODEL:-0}"
export COMPILE_LAYER_MODULES="${COMPILE_LAYER_MODULES:-0}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/env_cuda_uv.sh"

mkdir -p logs
exec "${UV_BIN:-uv}" run --python "${PYTHON_BIN:-python}" python -m torch.distributed.run --standalone --nproc_per_node=1 train_gpt.py 2>&1 | tee "logs/${RUN_ID}.console.txt"
