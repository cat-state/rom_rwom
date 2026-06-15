#!/usr/bin/env bash
set -euo pipefail

RUN_ID=${1:-bf360_dynsketchk4_dimsigns_ckpt_skipwarm_s4_1500_20260526_1252}
GPU=${GPU:-1}
RUN_DIR="logs/${RUN_ID}"
CKPT="${RUN_DIR}/state_step001500.pt"
HIST="${RUN_DIR}/engram_hit_hist_step001500.pt"
OUT_DIR="${RUN_DIR}/prune_evals"

source scripts/env_cuda_uv.sh

echo "waiting_for ${CKPT} and ${HIST}"
while [[ ! -s "${CKPT}" || ! -s "${HIST}" ]]; do
  sleep 60
done

mkdir -p "${OUT_DIR}"

"${PYTHON_BIN}" scripts/make_hit_count_mask_hists.py \
  --run-dir "${RUN_DIR}" \
  --hist "$(basename "${HIST}")" \
  --thresholds "1024,256,64" \
  --below-thresholds "2,4,8,16,32" \
  --buckets "" \
  --step "001500" | tee "${OUT_DIR}/mask_generation.json"

common_env=(
  CUDA_VISIBLE_DEVICES="${GPU}"
  BIGRAM_FACTOR=360
  GRAD_ACCUM_STEPS=16
  NUM_SCHEDULED_ITERATIONS=1500
  NUM_EXTENSION_ITERATIONS=0
  VAL_LOSS_EVERY=250
  SAVE_CHECKPOINT=0
  COMPILE_MODEL=0
  COMPILE_DENSE_LAYER_BODY=1
  ROM_LAYERS=2,8
  ENGRAM_BIGRAM=1
  ENGRAM_HIT_HIST=1
  ENGRAM_DIM=768
  ENGRAM_INIT_ZERO=1
  ENGRAM_INIT_STD=0.01
  ENGRAM_HEADS=1
  ENGRAM_MAX_NGRAM=3
  ENGRAM_SHORT_CONV=1
  ENGRAM_NORMALIZE_READOUT=1
  ENGRAM_NORMALIZE_MEMORY_HEADS=1
  ENGRAM_SKETCH_K=4
  ENGRAM_SKETCH_DIM_SIGNS=1
  ENGRAM_SKETCH_DIM_SIGN_MODE=random
  ENGRAM_LAYER_HASHES=1
  ENGRAM_LAYER_PARTITIONS=1
  ENGRAM_LAYER_PARTITION_GROUPS=1
  ENGRAM_PER_HEAD=1
  ENGRAM_CANONICALIZE=1
  ENGRAM_ATTNRES_MERGE=1
  ENGRAM_ATTNRES_MERGE_GAIN=1.5
  ENGRAM_UNTIED_PROJ=1
  ENGRAM_LR_MUL=5.0
  ENGRAM_SPARSE_SCALAR_ADAM=1
  ENGRAM_SPARSE_SANITIZE=1
  ENGRAM_SPARSE_GRAD_COALESCE_HOOK=1
  WANDB=1
  WANDB_PROJECT=rom-rwom
)

run_eval() {
  local label="$1"
  local hist_path="$2"
  local mask="$3"
  local mode="$4"
  local eval_run="${RUN_ID}_eval_${label}_${mode}"
  local out="${OUT_DIR}/${label}_${mode}.json"
  echo "eval_start ${label} mode=${mode} mask=${mask} hist=${hist_path}"
  env "${common_env[@]}" \
    RUN_ID="${eval_run}" \
    WANDB_NAME="${eval_run}" \
    ENGRAM_EVAL_CKPT="${CKPT}" \
    ENGRAM_HIT_HIST_LOAD="${hist_path}" \
    ENGRAM_MASK_UNHIT_EVAL="${mask}" \
    ENGRAM_MASK_UNHIT_EVAL_MODE="${mode}" \
    ENGRAM_EVAL_OUT="${out}" \
    "${UV_BIN}" run --python "${PYTHON_BIN}" python -m torch.distributed.run --standalone --nproc_per_node=1 train_gpt.py --run_id "${eval_run}" \
    > "${OUT_DIR}/${label}_${mode}.txt" 2>&1
  cat "${out}"
}

run_eval "base" "${HIST}" 0 "zero"
run_eval "unhit" "${HIST}" 1 "zero"
run_eval "unhit" "${HIST}" 1 "random"

for label in hit_lt_2 hit_lt_4 hit_lt_8 hit_lt_16 hit_lt_32 hit_ge_64 hit_ge_256 hit_ge_1024; do
  mask_hist="${RUN_DIR}/engram_hit_hist_step001500_mask_${label}.pt"
  if [[ -s "${mask_hist}" ]]; then
    run_eval "${label}" "${mask_hist}" 1 "zero"
    run_eval "${label}" "${mask_hist}" 1 "random"
  fi
done

"${PYTHON_BIN}" - <<'PY' "${OUT_DIR}"
import json, pathlib, sys
out = pathlib.Path(sys.argv[1])
rows = []
for path in sorted(out.glob("*.json")):
    if path.name == "mask_generation.json":
        continue
    data = json.loads(path.read_text())
    rows.append({"name": path.stem, "val_loss": data.get("val_loss"), "hist": data.get("engram_hit_hist")})
summary = {"results": rows}
(out / "summary.json").write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
PY
