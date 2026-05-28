#!/usr/bin/env bash
# Idempotent setup for the H200 modded-nanogpt bigram sweep.
# Run on the remote VM as user ubuntu after SSH/console access is restored.

set -euxo pipefail

repo="${REPO:-$HOME/modded-nanogpt}"
uv_bin="${UV_BIN:-$HOME/.local/bin/uv}"
venv="${VENV:-$HOME/.venvs/modded-nanogpt}"

if [ ! -x "$uv_bin" ]; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi

if [ ! -d "$repo/.git" ]; then
  git clone https://github.com/KellerJordan/modded-nanogpt "$repo"
else
  git -C "$repo" fetch origin
  git -C "$repo" checkout master
  git -C "$repo" pull --ff-only
fi

cd "$repo"

if [ ! -f "$venv/pyvenv.cfg" ]; then
  "$uv_bin" venv "$venv" --python 3.10
fi
"$uv_bin" pip install --python "$venv/bin/python" -r requirements.txt

python3 - <<'PY'
from pathlib import Path

train = Path("train_gpt.py")
text = train.read_text()
text = text.replace(
    'run_id: str = f"{uuid.uuid4()}"',
    'run_id: str = os.environ.get("RUN_ID", f"{uuid.uuid4()}")',
)
text = text.replace(
    "bigram_vocab_size: int = 50304 * 5",
    'bigram_vocab_size: int = 50304 * int(os.environ.get("BIGRAM_FACTOR", "5"))',
)
text = text.replace(
    "get_kernel('varunneal/flash-attention-3').flash_attn_interface",
    "get_kernel('kernels-community/flash-attn3', trust_remote_code=True).flash_attn_interface",
)
text = text.replace(
    "get_kernel('varunneal/flash-attention-3', trust_remote_code=True).flash_attn_interface",
    "get_kernel('kernels-community/flash-attn3', trust_remote_code=True).flash_attn_interface",
)
text = text.replace(
    "num_scheduled_iterations: int = 1440  # number of steps to complete lr and ws schedule",
    'num_scheduled_iterations: int = int(os.environ.get("NUM_SCHEDULED_ITERATIONS", "1440"))  # number of steps to complete lr and ws schedule',
)
text = text.replace(
    "num_extension_iterations: int = 40  # number of steps to continue training at final lr and ws",
    'num_extension_iterations: int = int(os.environ.get("NUM_EXTENSION_ITERATIONS", "40"))  # number of steps to continue training at final lr and ws',
)
text = text.replace(
    "val_loss_every: int = 250  # every how many steps to evaluate val loss? 0 for only at the end",
    'val_loss_every: int = int(os.environ.get("VAL_LOSS_EVERY", "250"))  # every how many steps to evaluate val loss? 0 for only at the end',
)
needle = "print0(nvidia_smi())\nprint0(\"=\"*100)"
replacement = (
    "print0(nvidia_smi())\n"
    "print0(f\"Experiment config: world_size={world_size} "
    "grad_accum_steps={grad_accum_steps} "
    "bigram_vocab_size={args.bigram_vocab_size} "
    "run_id={args.run_id}\", console=True)\n"
    "print0(\"=\"*100)"
)
if needle in text:
    text = text.replace(needle, replacement)
train.write_text(text)

run = Path("run.sh")
run.write_text(
    'NPROC_PER_NODE=${NPROC_PER_NODE:-8}\n'
    'torchrun --standalone --nproc_per_node="$NPROC_PER_NODE" train_gpt.py\n'
)
PY

if [ ! -f data/fineweb10B/fineweb_val_000000.bin ]; then
  "$uv_bin" run --python "$venv/bin/python" python data/cached_fineweb10B.py 9
fi

git diff -- train_gpt.py run.sh
nvidia-smi
