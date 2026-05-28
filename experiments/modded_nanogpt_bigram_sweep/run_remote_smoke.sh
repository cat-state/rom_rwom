#!/usr/bin/env bash
# Remote helper: run a tiny job to verify CUDA, data, patching, and logging
# before launching full speedrun sweeps.

set -euo pipefail

cd "${REPO:-$HOME/modded-nanogpt}"
mkdir -p logs

venv="${VENV:-$HOME/.venvs/modded-nanogpt}"
if [ -z "${CUDA_HOME:-}" ]; then
  export CUDA_HOME="$venv/lib/python3.10/site-packages/nvidia/cuda_runtime"
fi

factor="${BIGRAM_FACTOR:-5}"
gpu_label="${GPU_LABEL:-b200}"
run_id="rom_smoke_bigram${factor}_${gpu_label}_$(date -u +%Y%m%d_%H%M%S)"

BIGRAM_FACTOR="$factor" \
RUN_ID="$run_id" \
NPROC_PER_NODE=1 \
NUM_SCHEDULED_ITERATIONS="${NUM_SCHEDULED_ITERATIONS:-4}" \
NUM_EXTENSION_ITERATIONS="${NUM_EXTENSION_ITERATIONS:-0}" \
VAL_LOSS_EVERY="${VAL_LOSS_EVERY:-0}" \
$HOME/.local/bin/uv run --python "$venv/bin/python" \
  torchrun --standalone --nproc_per_node=1 train_gpt.py 2>&1 | tee "logs/${run_id}.console.txt"
