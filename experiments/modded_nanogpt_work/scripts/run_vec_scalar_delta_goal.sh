#!/usr/bin/env bash
set -euo pipefail

mode="${1:-pair-initial}"
if [[ "$mode" != "pair-initial" && "$mode" != "scalar-only" && "$mode" != "scalar-bf" && "$mode" != "row-adagrad" && "$mode" != "fal-quality" && "$mode" != "baseline-tail1" && "$mode" != "bank-scalar" && "$mode" != "bank-fullwidth" && "$mode" != "dense-embed-vecadam" && "$mode" != "dense-embed-scalaradam" ]]; then
  echo "usage: $0 pair-initial|scalar-only|scalar-bf|row-adagrad|fal-quality|baseline-tail1|bank-scalar|bank-fullwidth|dense-embed-vecadam|dense-embed-scalaradam" >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/.."
source "${SCRIPT_DIR}/env_cuda_uv.sh"
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

run_train() {
  local gpu="$1"
  local run="$2"
  shift 2
  local suffix="${RUN_SUFFIX:-}"
  if [[ -n "${MODEL_SEED:-}" ]]; then
    suffix="${suffix:+${suffix}_}seed${MODEL_SEED}"
  fi
  if [[ -n "${TRAIN_DATA_SEED:-}" ]]; then
    suffix="${suffix:+${suffix}_}data${TRAIN_DATA_SEED}"
  fi
  if [[ -n "$suffix" ]]; then
    run="${run}_${suffix}"
  fi
  if [[ -f "logs/${run}.console.txt" ]]; then
    echo "$(date -Is) skip existing ${run}"
    return 0
  fi
  echo "$(date -Is) launch gpu${gpu} ${run}"
  (
    export CUDA_VISIBLE_DEVICES="$gpu"
    export RUN_ID="$run"
    export WANDB="${WANDB:-1}"
    export WANDB_GROUP="${WANDB_GROUP:-engram-vec-scalar-delta}"
    export WANDB_NAME="$run"
    export WANDB_TAGS="${WANDB_TAGS:-engram scalar_adam delta_attnres}"
    export WANDB_HISTOGRAMS="${WANDB_HISTOGRAMS:-1}"
    export WANDB_HIST_EVERY="${WANDB_HIST_EVERY:-250}"
    export WANDB_HIST_ROWS="${WANDB_HIST_ROWS:-131072}"

    export NUM_SCHEDULED_ITERATIONS="${NUM_SCHEDULED_ITERATIONS:-1500}"
    export NUM_EXTENSION_ITERATIONS="${NUM_EXTENSION_ITERATIONS:-0}"
    export VAL_LOSS_EVERY="${VAL_LOSS_EVERY:-500}"
    export SAVE_CHECKPOINT="${SAVE_CHECKPOINT:-0}"
    export SAVE_CHECKPOINT_EVERY="${SAVE_CHECKPOINT_EVERY:-0}"
    export COMPILE_DENSE_LAYER_BODY="${COMPILE_DENSE_LAYER_BODY:-1}"
    export COMPILE_MODEL="${COMPILE_MODEL:-0}"
    export COMPILE_LAYER_MODULES="${COMPILE_LAYER_MODULES:-0}"

    export ENGRAM_BIGRAM=1
    export ROM_LAYERS="${ROM_LAYERS:-2,8}"
    export BIGRAM_FACTOR="${BIGRAM_FACTOR:-120}"
    export GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-16}"
    export ENGRAM_DIM="${ENGRAM_DIM:-768}"
    export ENGRAM_HEADS="${ENGRAM_HEADS:-1}"
    export ENGRAM_MAX_NGRAM="${ENGRAM_MAX_NGRAM:-3}"
    export ENGRAM_SHORT_CONV="${ENGRAM_SHORT_CONV:-1}"
    export ENGRAM_LAYER_HASHES=1
    export ENGRAM_LAYER_READOUTS=1
    export ENGRAM_LAYER_PARTITIONS=1
    export ENGRAM_PER_HEAD=1
    export ENGRAM_CANONICALIZE=1
    export ENGRAM_NORMALIZE_READOUT=1
    export ENGRAM_INIT_STD="${ENGRAM_INIT_STD:-0.01}"
    export ENGRAM_UNTIED_PROJ="${ENGRAM_UNTIED_PROJ:-1}"

    export ENGRAM_ATTNRES_MERGE=1
    export ENGRAM_ATTNRES_MERGE_GAIN="${ENGRAM_ATTNRES_MERGE_GAIN:-1.5}"
    export ENGRAM_LR_MUL="${ENGRAM_LR_MUL:-6.34615384615}"
    export ENGRAM_LR_FLOOR=0
    export ENGRAM_UPDATE_METRICS="${ENGRAM_UPDATE_METRICS:-1}"
    export ENGRAM_UPDATE_METRICS_EVERY="${ENGRAM_UPDATE_METRICS_EVERY:-500}"
    export ENGRAM_HIT_HIST="${ENGRAM_HIT_HIST:-1}"

    for kv in "$@"; do
      export "$kv"
    done

    exec "${UV_BIN:-uv}" run --python "${PYTHON_BIN:-python}" python -m torch.distributed.run --standalone --nproc_per_node=1 train_gpt.py
  ) 2>&1 | tee "logs/${run}.console.txt"
  echo "$(date -Is) finished ${run} exit=${PIPESTATUS[0]}"
}

if [[ "$mode" == "pair-initial" ]]; then
  ts="$(date +%Y%m%d_%H%M%S)"
  (
    wait_gpu_free 0
    run_train 0 "bf120_h1_layerpart_scalaradam_1500_${ts}" \
      ENGRAM_SPARSE_ADAM=0 \
      ENGRAM_SPARSE_VECTOR_ADAM=0 \
      ENGRAM_SPARSE_SCALAR_ADAM=1 \
      ENGRAM_SPARSE_ROW_ADAGRAD=0
  ) &
  (
    wait_gpu_free 1
    run_train 1 "bf120_h1_layerpart_tail1_deltaattnres_1500_${ts}" \
      ENGRAM_SPARSE_ADAM=1 \
      ENGRAM_SPARSE_VECTOR_ADAM=0 \
      ENGRAM_SPARSE_SCALAR_ADAM=0 \
      ENGRAM_SPARSE_ROW_ADAGRAD=0 \
      ENGRAM_SPARSE_ADAM_TAIL_STEPS=1 \
      ENGRAM_ADAM_EVERY_STEP=0 \
      ENGRAM_ATTNRES_DELTA=1
  ) &
  wait
elif [[ "$mode" == "scalar-only" ]]; then
  ts="$(date +%Y%m%d_%H%M%S)"
  wait_gpu_free "${GPU:-0}" "${GPU_FREE_THRESHOLD_MB:-10000}"
  run_train "${GPU:-0}" "bf120_h1_layerpart_scalaradam_1500_${ts}" \
    ENGRAM_SPARSE_ADAM=0 \
    ENGRAM_SPARSE_VECTOR_ADAM=0 \
    ENGRAM_SPARSE_SCALAR_ADAM=1 \
    ENGRAM_SPARSE_ROW_ADAGRAD=0
elif [[ "$mode" == "scalar-bf" ]]; then
  ts="$(date +%Y%m%d_%H%M%S)"
  bf="${BIGRAM_FACTOR:-160}"
  heads="${ENGRAM_HEADS:-1}"
  dim="${ENGRAM_DIM:-768}"
  lr_tag="$(printf '%s' "${ENGRAM_LR_MUL:-6.34615384615}" | tr -c '0-9A-Za-z' 'p')"
  wait_gpu_free "${GPU:-0}" "${GPU_FREE_THRESHOLD_MB:-10000}"
  run_train "${GPU:-0}" "bf${bf}_h${heads}_dim${dim}_lrmul${lr_tag}_layerpart_scalaradam_1500_${ts}" \
    BIGRAM_FACTOR="${bf}" \
    ENGRAM_HEADS="${heads}" \
    ENGRAM_DIM="${dim}" \
    ENGRAM_SPARSE_ADAM=0 \
    ENGRAM_SPARSE_VECTOR_ADAM=0 \
    ENGRAM_SPARSE_SCALAR_ADAM=1 \
    ENGRAM_SPARSE_ROW_ADAGRAD=0
elif [[ "$mode" == "row-adagrad" ]]; then
  ts="$(date +%Y%m%d_%H%M%S)"
  bf="${BIGRAM_FACTOR:-120}"
  heads="${ENGRAM_HEADS:-1}"
  dim="${ENGRAM_DIM:-768}"
  lr_tag="$(printf '%s' "${ENGRAM_LR_MUL:-6.34615384615}" | tr -c '0-9A-Za-z' 'p')"
  wait_gpu_free "${GPU:-0}" "${GPU_FREE_THRESHOLD_MB:-10000}"
  run_train "${GPU:-0}" "bf${bf}_h${heads}_dim${dim}_lrmul${lr_tag}_layerpart_rowadagrad_1500_${ts}" \
    BIGRAM_FACTOR="${bf}" \
    ENGRAM_HEADS="${heads}" \
    ENGRAM_DIM="${dim}" \
    ENGRAM_SPARSE_ADAM=0 \
    ENGRAM_SPARSE_VECTOR_ADAM=0 \
    ENGRAM_SPARSE_SCALAR_ADAM=0 \
    ENGRAM_SPARSE_ROW_ADAGRAD=1
elif [[ "$mode" == "fal-quality" ]]; then
  ts="$(date +%Y%m%d_%H%M%S)"
  bf="${BIGRAM_FACTOR:-120}"
  heads="${ENGRAM_HEADS:-1}"
  dim="${ENGRAM_DIM:-768}"
  lr_tag="$(printf '%s' "${ENGRAM_LR_MUL:-12.6923076923}" | tr -c '0-9A-Za-z' 'p')"
  wait_gpu_free "${GPU:-0}" "${GPU_FREE_THRESHOLD_MB:-10000}"
  run_train "${GPU:-0}" "bf${bf}_h${heads}_dim${dim}_lrmul${lr_tag}_layerpart_fal_1500_${ts}" \
    BIGRAM_FACTOR="${bf}" \
    ENGRAM_HEADS="${heads}" \
    ENGRAM_DIM="${dim}" \
    ENGRAM_LR_MUL="${ENGRAM_LR_MUL:-12.6923076923}" \
    ENGRAM_SPARSE_ADAM=1 \
    ENGRAM_SPARSE_VECTOR_ADAM=0 \
    ENGRAM_SPARSE_SCALAR_ADAM=0 \
    ENGRAM_SPARSE_ROW_ADAGRAD=0 \
    ENGRAM_ADAM_EVERY_STEP=1 \
    ENGRAM_SPARSE_ADAM_TAIL_STEPS=0 \
    ENGRAM_SPARSE_FAL=1 \
    ENGRAM_SPARSE_IFAL=0 \
    ENGRAM_SPARSE_HIT_LR=0
elif [[ "$mode" == "baseline-tail1" ]]; then
  ts="$(date +%Y%m%d_%H%M%S)"
  wait_gpu_free "${GPU:-0}" "${GPU_FREE_THRESHOLD_MB:-10000}"
  run_train "${GPU:-0}" "bf120_h1_layerpart_tail1_baseline_1500_${ts}" \
    ENGRAM_SPARSE_ADAM=1 \
    ENGRAM_SPARSE_VECTOR_ADAM=0 \
    ENGRAM_SPARSE_SCALAR_ADAM=0 \
    ENGRAM_SPARSE_ROW_ADAGRAD=0 \
    ENGRAM_SPARSE_ADAM_TAIL_STEPS=1 \
    ENGRAM_ADAM_EVERY_STEP=0 \
    ADAM_EMBED_VECTOR_ADAM=0 \
    ADAM_EMBED_SCALAR_ADAM=0
elif [[ "$mode" == "bank-scalar" ]]; then
  ts="$(date +%Y%m%d_%H%M%S)"
  bf="${BIGRAM_FACTOR:-80}"
  heads="${ENGRAM_HEADS:-2}"
  dim="${ENGRAM_DIM:-768}"
  store_dim="${ENGRAM_STORE_DIM:-384}"
  wait_gpu_free "${GPU:-0}" "${GPU_FREE_THRESHOLD_MB:-10000}"
  run_train "${GPU:-0}" "bf${bf}_h${heads}_store${store_dim}_bankattnres_scalaradam_1500_${ts}" \
    BIGRAM_FACTOR="${bf}" \
    ENGRAM_HEADS="${heads}" \
    ENGRAM_DIM="${dim}" \
    ENGRAM_STORE_DIM="${store_dim}" \
    ENGRAM_BANK_ATTNRES=1 \
    ENGRAM_SPARSE_ADAM=0 \
    ENGRAM_SPARSE_VECTOR_ADAM=0 \
    ENGRAM_SPARSE_SCALAR_ADAM=1 \
    ENGRAM_SPARSE_ROW_ADAGRAD=0
elif [[ "$mode" == "bank-fullwidth" ]]; then
  ts="$(date +%Y%m%d_%H%M%S)"
  wait_gpu_free "${GPU:-0}" "${GPU_FREE_THRESHOLD_MB:-10000}"
  run_train "${GPU:-0}" "bf80_h4_bankattnres_store384_scalaradam_1500_${ts}" \
    BIGRAM_FACTOR=80 \
    GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-16}" \
    ENGRAM_HEADS=4 \
    ENGRAM_STORE_DIM=384 \
    ENGRAM_BANK_ATTNRES=1 \
    ENGRAM_SPARSE_ADAM=0 \
    ENGRAM_SPARSE_VECTOR_ADAM=0 \
    ENGRAM_SPARSE_SCALAR_ADAM=1 \
    ENGRAM_SPARSE_ROW_ADAGRAD=0
elif [[ "$mode" == "dense-embed-vecadam" ]]; then
  ts="$(date +%Y%m%d_%H%M%S)"
  wait_gpu_free "${GPU:-0}" "${GPU_FREE_THRESHOLD_MB:-10000}"
  run_train "${GPU:-0}" "bf120_h1_layerpart_tail1_dense_embed_vecadam_1500_${ts}" \
    ENGRAM_SPARSE_ADAM=1 \
    ENGRAM_SPARSE_VECTOR_ADAM=0 \
    ENGRAM_SPARSE_SCALAR_ADAM=0 \
    ENGRAM_SPARSE_ROW_ADAGRAD=0 \
    ENGRAM_SPARSE_ADAM_TAIL_STEPS=1 \
    ENGRAM_ADAM_EVERY_STEP=0 \
    ADAM_EMBED_VECTOR_ADAM=1 \
    ADAM_EMBED_SCALAR_ADAM=0
else
  ts="$(date +%Y%m%d_%H%M%S)"
  wait_gpu_free "${GPU:-0}" "${GPU_FREE_THRESHOLD_MB:-10000}"
  run_train "${GPU:-0}" "bf120_h1_layerpart_tail1_dense_embed_scalaradam_1500_${ts}" \
    ENGRAM_SPARSE_ADAM=1 \
    ENGRAM_SPARSE_VECTOR_ADAM=0 \
    ENGRAM_SPARSE_SCALAR_ADAM=0 \
    ENGRAM_SPARSE_ROW_ADAGRAD=0 \
    ENGRAM_SPARSE_ADAM_TAIL_STEPS=1 \
    ENGRAM_ADAM_EVERY_STEP=0 \
    ADAM_EMBED_VECTOR_ADAM=0 \
    ADAM_EMBED_SCALAR_ADAM=1
fi
