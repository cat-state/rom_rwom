# Fresh VM Quickstart

Use this if `ubuntu@89.169.121.171` cannot be repaired from the provider
console. Start from a fresh Ubuntu H200 image with a CUDA 12.8-compatible NVIDIA
driver. Then run:

```bash
sudo apt-get update
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y \
  curl git python3.10 python3.10-venv openssh-server

curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

git clone https://github.com/KellerJordan/modded-nanogpt ~/modded-nanogpt
cd ~/modded-nanogpt

uv venv ~/.venvs/modded-nanogpt --python 3.10
uv pip install --python ~/.venvs/modded-nanogpt/bin/python -r requirements.txt
uv run --python ~/.venvs/modded-nanogpt/bin/python python data/cached_fineweb10B.py 9
```

Patch `train_gpt.py`:

```bash
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

Path("run.sh").write_text(
    'NPROC_PER_NODE=${NPROC_PER_NODE:-8}\n'
    'torchrun --standalone --nproc_per_node="$NPROC_PER_NODE" train_gpt.py\n'
)
PY
```

Run the sweep:

```bash
mkdir -p logs
for factor in 5 25 100; do
  run_id="rom_bigram${factor}_h200_$(date -u +%Y%m%d_%H%M%S)"
  BIGRAM_FACTOR="$factor" RUN_ID="$run_id" NPROC_PER_NODE=1 \
    uv run --python ~/.venvs/modded-nanogpt/bin/python \
    torchrun --standalone --nproc_per_node=1 train_gpt.py 2>&1 | tee "logs/${run_id}.console.txt"
done
```

Expected comparison fields:

- final `val_loss`
- final `train_time`
- final `step_avg`
- `peak memory allocated`
- `peak memory reserved`

