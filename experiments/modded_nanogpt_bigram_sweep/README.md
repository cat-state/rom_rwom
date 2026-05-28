# Modded NanoGPT Bigram Sweep

Goal: run `KellerJordan/modded-nanogpt` on the single H200 at
`ubuntu@89.169.121.171` and compare current bigram hash embedding size
`5 * 50304` against `25 * 50304` and `100 * 50304`.

## Remote State Reached

- Repo cloned at `~/modded-nanogpt`, commit
  `ab2ec892cfc9ceeadb8942f610c6169ea92ef199`.
- `uv` installed at `~/.local/bin/uv`.
- Venv created at `~/.venvs/modded-nanogpt`.
- Requirements installed via:

```bash
~/.local/bin/uv venv ~/.venvs/modded-nanogpt --python 3.10
~/.local/bin/uv pip install --python ~/.venvs/modded-nanogpt/bin/python -r requirements.txt
```

- FineWeb cache downloaded with:

```bash
~/.local/bin/uv run --python ~/.venvs/modded-nanogpt/bin/python \
  python data/cached_fineweb10B.py 9
```

- Data observed:
  - `data/fineweb10B/fineweb_train_000001.bin` through
    `fineweb_train_000009.bin`
  - `data/fineweb10B/fineweb_val_000000.bin`

## Remote Patch

The remote repo was patched to make the run configurable:

```diff
diff --git a/run.sh b/run.sh
--- a/run.sh
+++ b/run.sh
@@
-torchrun --standalone --nproc_per_node=8 train_gpt.py
+NPROC_PER_NODE=${NPROC_PER_NODE:-8}
+torchrun --standalone --nproc_per_node="$NPROC_PER_NODE" train_gpt.py

diff --git a/train_gpt.py b/train_gpt.py
--- a/train_gpt.py
+++ b/train_gpt.py
@@
-    run_id: str = f"{uuid.uuid4()}"
+    run_id: str = os.environ.get("RUN_ID", f"{uuid.uuid4()}")
@@
-    bigram_vocab_size: int = 50304 * 5
+    bigram_vocab_size: int = 50304 * int(os.environ.get("BIGRAM_FACTOR", "5"))
@@
 print0(nvidia_smi())
+print0(f"Experiment config: world_size={world_size} grad_accum_steps={grad_accum_steps} bigram_vocab_size={args.bigram_vocab_size} run_id={args.run_id}", console=True)
 print0("="*100)
```

## Blocker

Initial run failed because the VM had NVIDIA driver `550.163.01` / CUDA driver
`12.4`, while repo requirements install `torch==2.10.0+cu128`.
`torch.cuda._compile_kernel` failed with:

```text
CUDA error: the provided PTX was compiled with an unsupported toolchain.
```

The NVIDIA stack was upgraded to driver `570.211.01`, including
`nvidia-fabricmanager-570` and `libnvidia-nscq-570`, then rebooted. After the
second reboot, the VM is pingable but port 22 refuses connections, so SSH is
not currently available.

Current network state:

```text
ping 89.169.121.171: replies OK
nc -vz 89.169.121.171 22: Connection refused
ssh ubuntu@89.169.121.171: Connection refused
```

Provider console recovery is needed to restore SSH. Likely actions:

```bash
sudo bash recovery_console.sh
```

If the GPU is still broken, inspect:

```bash
sudo dmesg | grep -iE 'nvrm|nvidia|gsp|rm_init|nvswitch' | tail -200
dkms status | grep nvidia
dpkg -l | grep -E 'nvidia|fabric|nscq' | sort
```

## Intended Runs

After SSH/GPU are restored, copy or run `setup_remote.sh` on the VM if the
remote repo state is uncertain. Then:

```bash
cd ~/modded-nanogpt
for factor in 5 25 100; do
  run_id="rom_bigram${factor}_h200_$(date -u +%Y%m%d_%H%M%S)"
  BIGRAM_FACTOR="$factor" RUN_ID="$run_id" NPROC_PER_NODE=1 \
    ~/.local/bin/uv run --python ~/.venvs/modded-nanogpt/bin/python \
    torchrun --standalone --nproc_per_node=1 train_gpt.py 2>&1 | tee "logs/${run_id}.console.txt"
done
```

Compare for each factor:

- final `val_loss`
- final `train_time`
- `step_avg`
- `peak memory allocated`
- whether the run reaches `<=3.28`

The setup script also adds debug-only overrides for
`NUM_SCHEDULED_ITERATIONS`, `NUM_EXTENSION_ITERATIONS`, and `VAL_LOSS_EVERY`.
Use `run_remote_smoke.sh` for a tiny post-recovery CUDA/data check before the
full sweep.

For manual patching on a recovered or fresh checkout:

```bash
git apply modded_nanogpt_bigram_sweep.patch
```

Memory estimate for the bigram table alone:

```bash
python experiments/modded_nanogpt_bigram_sweep/estimate_bigram_memory.py
```

After copying logs locally:

```bash
python experiments/modded_nanogpt_bigram_sweep/summarize_logs.py logs/*.txt logs/*.console.txt
python experiments/modded_nanogpt_bigram_sweep/verify_completion.py experiments/modded_nanogpt_bigram_sweep/summary.csv
```

From this local repo, once provider-console recovery is done:

```bash
experiments/modded_nanogpt_bigram_sweep/wait_upload_run.sh
```

If the current VM cannot be repaired, use `fresh_vm_quickstart.md` on a new H200
instance.

For provider-console recovery, use `operator_handoff.md`.

Nebius serial-log documentation is linked there; it is the first place to check
when the VM is pingable but SSH refuses connections.
