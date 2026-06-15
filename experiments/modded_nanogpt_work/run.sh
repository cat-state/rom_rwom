#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/scripts/env_cuda_uv.sh"

NPROC_PER_NODE=${NPROC_PER_NODE:-8}
"${UV_BIN:-uv}" run --python "${PYTHON_BIN:-python}" torchrun --standalone --nproc_per_node="$NPROC_PER_NODE" train_gpt.py
