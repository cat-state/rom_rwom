#!/usr/bin/env bash
# Local helper: wait for SSH to return, upload helpers, run setup + sweep, and
# pull logs back. Run from this repo root.

set -euo pipefail

host="${HOST:-ubuntu@89.169.121.171}"
remote_dir="${REMOTE_DIR:-~/rom_bigram_sweep}"
local_dir="experiments/modded_nanogpt_bigram_sweep"
poll_seconds="${POLL_SECONDS:-15}"
max_wait_seconds="${MAX_WAIT_SECONDS:-0}" # 0 means forever
gpu_label="${GPU_LABEL:-b200}"

start="$(date +%s)"
while true; do
  if ssh -o BatchMode=yes -o ConnectTimeout=5 "$host" 'echo ssh-ready' >/dev/null 2>&1; then
    break
  fi
  now="$(date +%s)"
  elapsed="$((now - start))"
  if [ "$max_wait_seconds" -gt 0 ] && [ "$elapsed" -ge "$max_wait_seconds" ]; then
    echo "Timed out waiting for SSH after ${elapsed}s" >&2
    exit 1
  fi
  echo "Waiting for SSH on $host (${elapsed}s elapsed)"
  sleep "$poll_seconds"
done

ssh "$host" "mkdir -p $remote_dir"
scp \
  "$local_dir/setup_remote.sh" \
  "$local_dir/run_remote_sweep.sh" \
  "$local_dir/run_remote_smoke.sh" \
  "$local_dir/summarize_logs.py" \
  "$local_dir/verify_completion.py" \
  "$host:$remote_dir/"

ssh "$host" "chmod +x $remote_dir/*.sh $remote_dir/summarize_logs.py"
ssh "$host" "$remote_dir/setup_remote.sh"
ssh "$host" "GPU_LABEL=$gpu_label $remote_dir/run_remote_smoke.sh"
set +e
ssh "$host" "GPU_LABEL=$gpu_label $remote_dir/run_remote_sweep.sh"
sweep_status=$?
set -e

mkdir -p "$local_dir/logs"
scp "$host:~/modded-nanogpt/logs/rom_bigram*_${gpu_label}_*" "$local_dir/logs/" || true
scp "$host:~/modded-nanogpt/logs/rom_bigram_sweep_status.txt" "$local_dir/logs/" || true
python3 "$local_dir/summarize_logs.py" "$local_dir"/logs/rom_bigram*_"$gpu_label"_* | tee "$local_dir/summary.csv"
python3 "$local_dir/verify_completion.py" "$local_dir/summary.csv"
exit "$sweep_status"
