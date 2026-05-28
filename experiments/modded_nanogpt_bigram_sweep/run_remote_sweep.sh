#!/usr/bin/env bash
# Remote helper: run the full bigram-factor sweep.

set -euo pipefail

cd "${REPO:-$HOME/modded-nanogpt}"
mkdir -p logs

venv="${VENV:-$HOME/.venvs/modded-nanogpt}"
if [ -z "${CUDA_HOME:-}" ]; then
  export CUDA_HOME="$venv/lib/python3.10/site-packages/nvidia/cuda_runtime"
fi

status=0
gpu_label="${GPU_LABEL:-b200}"
for factor in 5 25 100; do
  run_id="rom_bigram${factor}_${gpu_label}_$(date -u +%Y%m%d_%H%M%S)"
  echo "Starting $run_id"
  set +e
  BIGRAM_FACTOR="$factor" RUN_ID="$run_id" NPROC_PER_NODE=1 \
    $HOME/.local/bin/uv run --python "$venv/bin/python" \
    torchrun --standalone --nproc_per_node=1 train_gpt.py 2>&1 | tee "logs/${run_id}.console.txt"
  rc=${PIPESTATUS[0]}
  set -e
  echo "$run_id exit_code=$rc" | tee -a logs/rom_bigram_sweep_status.txt
  if [ "$rc" -ne 0 ]; then
    status=$rc
  fi
done

exit "$status"
