#!/usr/bin/env bash
# Shared runtime setup for uv-managed PyTorch CUDA wheels.
#
# The fused CUDA kernels need CUDA_HOME/include for NVRTC compilation. In the
# uv/pip setup this lives inside the nvidia-cuda-runtime package, not in a
# system CUDA install.

if [[ -z "${UV_BIN:-}" && -x /home/ubuntu/.local/bin/uv ]]; then
  export UV_BIN=/home/ubuntu/.local/bin/uv
fi

if [[ -z "${PYTHON_BIN:-}" && -x /home/ubuntu/.venvs/modded-nanogpt/bin/python ]]; then
  export PYTHON_BIN=/home/ubuntu/.venvs/modded-nanogpt/bin/python
fi

if [[ -z "${CUDA_HOME:-}" ]]; then
  cuda_runtime_dir=""
  cuda_nvrtc_dir=""

  if [[ -n "${PYTHON_BIN:-}" && -x "${PYTHON_BIN}" ]]; then
    py_prefix="$(dirname "$(dirname "${PYTHON_BIN}")")"
    py_version="$("${PYTHON_BIN}" - <<'PY'
import sys
print(f"python{sys.version_info.major}.{sys.version_info.minor}")
PY
)"
    candidate_runtime="${py_prefix}/lib/${py_version}/site-packages/nvidia/cuda_runtime"
    candidate_nvrtc="${py_prefix}/lib/${py_version}/site-packages/nvidia/cuda_nvrtc"
    if [[ -d "${candidate_runtime}" ]]; then
      cuda_runtime_dir="${candidate_runtime}"
    fi
    if [[ -d "${candidate_nvrtc}" ]]; then
      cuda_nvrtc_dir="${candidate_nvrtc}"
    fi
  fi

  fallback_runtime=/home/ubuntu/.venvs/modded-nanogpt/lib/python3.10/site-packages/nvidia/cuda_runtime
  fallback_nvrtc=/home/ubuntu/.venvs/modded-nanogpt/lib/python3.10/site-packages/nvidia/cuda_nvrtc
  if [[ -z "${cuda_runtime_dir}" && -d "${fallback_runtime}" ]]; then
    cuda_runtime_dir="${fallback_runtime}"
  fi
  if [[ -z "${cuda_nvrtc_dir}" && -d "${fallback_nvrtc}" ]]; then
    cuda_nvrtc_dir="${fallback_nvrtc}"
  fi

  if [[ -n "${cuda_runtime_dir}" ]]; then
    export CUDA_HOME="${cuda_runtime_dir}"
    ld_prefix="${CUDA_HOME}/lib"
    if [[ -n "${cuda_nvrtc_dir}" ]]; then
      ld_prefix="${ld_prefix}:${cuda_nvrtc_dir}/lib"
    fi
    export LD_LIBRARY_PATH="${ld_prefix}:${LD_LIBRARY_PATH:-}"
  fi
fi
