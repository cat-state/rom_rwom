#!/usr/bin/env bash
# Run these commands from the provider VM console, not from this local machine.
# The VM is pingable but SSH currently refuses connections after the NVIDIA
# driver/Fabric Manager upgrade.

set -euxo pipefail

systemctl status ssh --no-pager || true
systemctl restart ssh || systemctl restart sshd || true
systemctl enable ssh || systemctl enable sshd || true
systemctl status ssh --no-pager || true
ss -ltnp | grep ':22' || true
ufw status || true

apt-get update
DEBIAN_FRONTEND=noninteractive apt-get install -y openssh-server curl git
systemctl restart ssh || systemctl restart sshd || true
ss -ltnp | grep ':22' || true

nvidia-smi || true
cat /proc/driver/nvidia/version || true
dkms status | grep nvidia || true
dpkg -l | grep -E 'nvidia|fabric|nscq' | sort || true
dmesg | grep -iE 'nvrm|nvidia|gsp|rm_init|nvswitch' | tail -200 || true

# If nvidia-smi reports no devices and dmesg shows GSP/RmInitAdapter failures,
# the safest rollback is to reinstall the original 550 stack, re-enable SSH, and
# reboot. This gets the machine reachable again, even if Torch 2.10/cu128 still
# needs a different driver strategy.
#
# DEBIAN_FRONTEND=noninteractive apt-get install -y \
#   nvidia-driver-550 nvidia-fabricmanager-550 libnvidia-nscq-550
# systemctl enable ssh
# reboot
