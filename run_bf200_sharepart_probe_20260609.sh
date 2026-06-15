#!/usr/bin/env bash
set -euo pipefail

cd /root/modded-nanogpt
mkdir -p logs

export UV_BIN=/root/.local/bin/uv
export PYTHON_BIN=/root/.venvs/modded-nanogpt/bin/python
source scripts/env_cuda_uv.sh

export WANDB=1
export WANDB_MODE=${WANDB_MODE:-offline}
export WANDB_PROJECT=${WANDB_PROJECT:-rom-rwom}
export WANDB_ENTITY=${WANDB_ENTITY:-uwu1}
export WANDB_GROUP=${WANDB_GROUP:-bf200-structural-after-sketch-20260609}
export WANDB_TAGS=${WANDB_TAGS:-engram,research,bf200,sharepart,param-efficiency}
export WANDB_HISTOGRAMS=${WANDB_HISTOGRAMS:-1}
export WANDB_HIST_EVERY=${WANDB_HIST_EVERY:-250}
export WANDB_HIST_ROWS=${WANDB_HIST_ROWS:-131072}

export ENGRAM_BIGRAM=1
export BIGRAM_FACTOR=200
export GRAD_ACCUM_STEPS=16
export ROM_LAYERS=2,8
export ENGRAM_DIM=768
export ENGRAM_HEADS=1
export ENGRAM_MAX_NGRAM=3
export ENGRAM_NGRAM_ROW_FACTORS=0.5,1.5
export ENGRAM_SHORT_CONV=1
export ENGRAM_LAYER_HASHES=1
export ENGRAM_LAYER_READOUTS=1
export ENGRAM_LAYER_PARTITIONS=0
export ENGRAM_LAYER_PARTITION_GROUPS=0
export ENGRAM_PER_HEAD=1
export ENGRAM_CANONICALIZE=1
export ENGRAM_NORMALIZE_READOUT=1
export ENGRAM_NORMALIZE_MEMORY_HEADS=1
export ENGRAM_INIT_STD=0.01
export ENGRAM_UNTIED_PROJ=1
export ENGRAM_ATTNRES_MERGE=1
export ENGRAM_ATTNRES_MERGE_GAIN=1.5
export ENGRAM_READ_HIT_SCALE_EXPONENT=0.25
export ENGRAM_READ_HIT_SCALE_OFFSET=1.0
export ENGRAM_READ_HIT_SCALE_MIN=0.25
export ENGRAM_READ_HIT_SCALE_MAX=4.0
export ENGRAM_READ_HIT_SCALE_NORM_MEAN=1
export ENGRAM_LR_MUL=5.0
export ENGRAM_LR_FLOOR=0
export ENGRAM_SPARSE_ADAM=0
export ENGRAM_SPARSE_VECTOR_ADAM=0
export ENGRAM_SPARSE_SCALAR_ADAM=1
export ENGRAM_SPARSE_ROW_ADAGRAD=0
export ENGRAM_SPARSE_ADAM_TAIL_STEPS=0
export ENGRAM_ADAM_EVERY_STEP=1
export ENGRAM_HIT_HIST=1
export ENGRAM_UPDATE_METRICS=1
export ENGRAM_UPDATE_METRICS_EVERY=250

export NUM_SCHEDULED_ITERATIONS=1500
export NUM_EXTENSION_ITERATIONS=0
export VAL_LOSS_EVERY=250
export SAVE_CHECKPOINT=0
export SAVE_CHECKPOINT_EVERY=0
export COMPILE_MODEL=0
export COMPILE_LAYER_MODULES=0
export COMPILE_DENSE_LAYER_BODY=1
export TRAIN_DATA_SEED=0
export MODEL_SEED=5

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export RUN_ID=bf200_sota_sharepart_seed5_1500_20260609
export WANDB_NAME="$RUN_ID"

if [[ -f "logs/${RUN_ID}.console.txt" ]]; then
  echo "$(date -Is) skip existing ${RUN_ID}"
  exit 0
fi

exec "$UV_BIN" run --python "$PYTHON_BIN" python -m torch.distributed.run --standalone --nproc_per_node=1 train_gpt.py 2>&1 | tee "logs/${RUN_ID}.console.txt"
