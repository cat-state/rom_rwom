# Operator Handoff

The experiment is ready, but the current H200 VM at `89.169.121.171` refuses
SSH on port 22 while still responding to ping. Recovery requires provider
console access or a replacement H200.

## Console Recovery

Official Nebius docs:

- VM serial logs: https://docs.nebius.com/compute/monitoring/serial-logs
- VM SSH connection/user-data reference: https://docs.nebius.com/compute/virtual-machines/connect
- Nebius CLI log queries: https://docs.nebius.com/observability/logs/nebius-cli

In the Nebius web console, open Compute -> Virtual machines -> the target VM ->
Serial logs. Nebius keeps Compute serial logs for 14 days, which should show
whether the current boot is stuck in NVIDIA driver initialization, SSH startup,
or cloud-init.

With configured Nebius CLI credentials, query serial logs from a terminal:

```bash
nebius logging query '{}' --bucket sp_serial --since 2h
nebius logging query '{} |= "ssh"' --bucket sp_serial --since 2h
nebius logging query '{} |= "nvidia"' --bucket sp_serial --since 2h
nebius logging query '{} |= "RmInitAdapter"' --bucket sp_serial --since 2h
nebius logging query '{} |= "cloud-init"' --bucket sp_serial --since 2h
```

From the provider console on the existing VM:

```bash
sudo apt-get update
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y openssh-server curl git
sudo systemctl enable ssh || sudo systemctl enable sshd || true
sudo systemctl restart ssh || sudo systemctl restart sshd
ss -ltnp | grep ':22'
nvidia-smi
```

If `nvidia-smi` fails, collect:

```bash
cat /proc/driver/nvidia/version || true
dkms status | grep nvidia || true
dpkg -l | grep -E 'nvidia|fabric|nscq' | sort || true
sudo dmesg | grep -iE 'nvrm|nvidia|gsp|rm_init|nvswitch' | tail -200 || true
```

If the 570 driver stack remains broken, rollback to the original 550 stack to
restore access, then decide whether to use a fresh CUDA 12.8-compatible image:

```bash
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y \
  nvidia-driver-550 nvidia-fabricmanager-550 libnvidia-nscq-550
sudo reboot
```

## Run After SSH Recovery

From this local repo:

```bash
experiments/modded_nanogpt_bigram_sweep/wait_upload_run.sh
```

Expected output artifacts:

- `experiments/modded_nanogpt_bigram_sweep/logs/rom_smoke_bigram5_h200_*.console.txt`
- `experiments/modded_nanogpt_bigram_sweep/logs/rom_bigram5_h200_*.console.txt`
- `experiments/modded_nanogpt_bigram_sweep/logs/rom_bigram25_h200_*.console.txt`
- `experiments/modded_nanogpt_bigram_sweep/logs/rom_bigram100_h200_*.console.txt`
- `experiments/modded_nanogpt_bigram_sweep/summary.csv`

## Success Criteria

The experiment is complete only when `summary.csv` contains completed rows for
factors `5`, `25`, and `100` with final validation loss, train time, step
average, and peak memory.

Machine-check it with:

```bash
python experiments/modded_nanogpt_bigram_sweep/verify_completion.py \
  experiments/modded_nanogpt_bigram_sweep/summary.csv
```
