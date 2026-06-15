#!/usr/bin/env bash
set -euo pipefail

cd /root/modded-nanogpt
mkdir -p logs

export UV_BIN=/root/.local/bin/uv
export PYTHON_BIN=/root/.venvs/modded-nanogpt/bin/python
source scripts/env_cuda_uv.sh

export WANDB=1
export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_PROJECT="${WANDB_PROJECT:-rom-rwom}"
export WANDB_ENTITY="${WANDB_ENTITY:-uwu1}"
export WANDB_GROUP="${WANDB_GROUP:-bf300-optimizer-probes-20260610}"
export WANDB_TAGS="${WANDB_TAGS:-engram,research,optimizer,row-frequency}"
export WANDB_HISTOGRAMS="${WANDB_HISTOGRAMS:-1}"
export WANDB_HIST_EVERY="${WANDB_HIST_EVERY:-250}"
export WANDB_HIST_ROWS="${WANDB_HIST_ROWS:-131072}"

export ENGRAM_BIGRAM=1
export BIGRAM_FACTOR=300
export GRAD_ACCUM_STEPS=16
export ROM_LAYERS=2,8
export ENGRAM_DIM=768
export ENGRAM_HEADS=1
export ENGRAM_MAX_NGRAM=3
export ENGRAM_NGRAM_ROW_FACTORS=0.5,1.5
export ENGRAM_SHORT_CONV=1
export ENGRAM_LAYER_HASHES=1
export ENGRAM_LAYER_READOUTS=0
export ENGRAM_LAYER_READOUT_DELTA=1
export ENGRAM_LAYER_PARTITIONS=1
export ENGRAM_LAYER_PARTITION_GROUPS=1
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
export ENGRAM_HEAD_MIX=1
export ENGRAM_HEAD_MIX_INIT=0.5,-0.5
export ENGRAM_SUPERPOSE_K=2
export ENGRAM_SUPERPOSE_INCLUDE_BASE=1
export ENGRAM_SUPERPOSE_AUX_SCALE=0.5
export ENGRAM_SUPERPOSE_NORMALIZE=1
export ENGRAM_LR_MUL=5.0
export ENGRAM_LR_FLOOR=0
export ENGRAM_SPARSE_ADAM=0
export ENGRAM_SPARSE_VECTOR_ADAM=0
export ENGRAM_SPARSE_SCALAR_ADAM=1
export ENGRAM_SPARSE_ROW_ADAGRAD=0
export ENGRAM_SPARSE_ADAM_TAIL_STEPS=0
export ENGRAM_SPARSE_HIT_LR=0
export ENGRAM_SPARSE_FAL=0
export ENGRAM_SPARSE_IFAL=0
export ENGRAM_SPARSE_HIT_LR_MIN=0.25
export ENGRAM_SPARSE_HIT_LR_MAX=4.0
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

gpu_mem_mb() {
  nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$1" | tr -d " "
}

wait_gpu_free() {
  local gpu="$1"
  local threshold="${GPU_FREE_THRESHOLD_MB:-10000}"
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

run_one() {
  local gpu="$1"
  local run="$2"
  local variant="$3"
  if grep -q "step:1500/1500 val_loss" "logs/${run}.txt" 2>/dev/null; then
    echo "$(date -Is) skip complete ${run}"
    return 0
  fi
  wait_gpu_free "$gpu"
  echo "$(date -Is) launch gpu${gpu} ${run}"
  (
    export CUDA_VISIBLE_DEVICES="$gpu"
    export RUN_ID="$run"
    export WANDB_NAME="$run"
    case "$variant" in
      rowadagrad)
        export ENGRAM_SPARSE_SCALAR_ADAM=0
        export ENGRAM_SPARSE_ROW_ADAGRAD=1
        ;;
      ifal)
        export ENGRAM_SPARSE_SCALAR_ADAM=1
        export ENGRAM_SPARSE_IFAL=1
        ;;
      *)
        echo "unknown variant: ${variant}" >&2
        exit 2
        ;;
    esac
    exec "$UV_BIN" run --python "$PYTHON_BIN" python -m torch.distributed.run --standalone --nproc_per_node=1 train_gpt.py
  ) 2>&1 | tee "logs/${run}.console.txt"
  local status=${PIPESTATUS[0]}
  echo "$(date -Is) finished ${run} exit=${status}" | tee -a "logs/${run}.console.txt"
  return "$status"
}

run_one 0 bf300_sota_k2_headmix_rowadagrad_seed5_1500_20260610 rowadagrad &
run_one 1 bf300_sota_k2_headmix_ifal025to4_seed5_1500_20260610 ifal &
wait
