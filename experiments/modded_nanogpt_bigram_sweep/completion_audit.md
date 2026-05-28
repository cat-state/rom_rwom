# Completion Audit

Objective: measure how scaling the current modded-nanogpt bigram hash embedding
table from `5 * 50304` rows to `25 * 50304` and `100 * 50304` changes speedrun
performance on the single H200 at `ubuntu@89.169.121.171`.

## Success Criteria

| Requirement | Evidence Needed | Current Evidence | Status |
| --- | --- | --- | --- |
| Access H200 host | Successful SSH to `ubuntu@89.169.121.171` | Host pings, but SSH port 22 returns `Connection refused` | Blocked |
| Run on single H200 | `nvidia-smi` shows H200 and training launches with `NPROC_PER_NODE=1` | Earlier pre-upgrade evidence showed NVIDIA H200; current SSH unavailable after driver reboot | Blocked |
| Use modded-nanogpt speedrun | Remote repo cloned and patched | Repo was cloned at commit `ab2ec892cfc9ceeadb8942f610c6169ea92ef199`; patch documented | Partially done |
| Patch applies to upstream | `git apply --check` succeeds on fresh checkout | Verified locally against fresh clone in `/tmp/modded-nanogpt-patchcheck` | Done |
| Adjust for 1 GPU | Launch uses `torchrun --nproc_per_node=1`; world size gives `grad_accum_steps=8` | Remote patch changed `run.sh` to respect `NPROC_PER_NODE`; run command prepared | Partially done |
| Baseline `5 * 50304` | Completed log with final val loss/train time/memory | First run failed before training due CUDA driver/PTX mismatch | Missing |
| `25 * 50304` | Completed log with final val loss/train time/memory | Run command prepared only | Missing |
| `100 * 50304` | Completed log with final val loss/train time/memory | Run command prepared only | Missing |
| Compare performance | Summary table from logs | `summarize_logs.py` exists, no completed logs yet | Missing |
| Check bigram memory feasibility | Parameter/optimizer memory estimate for 5/25/100 | `estimate_bigram_memory.py` reports ~0.36/1.80/7.20 GiB params and ~1.08/5.40/21.59 GiB with rough param+grad+optimizer multiplier | Done |
| Portable experiment bundle | Tarball containing recovery/setup/run/summarizer artifacts | `experiments/modded_nanogpt_bigram_sweep.tar.gz` created and archive contents verified | Done |
| Artifact integrity | Checksums for portable bundle and source files | `TARBALL.SHA256` and `SHA256SUMS` generated; see `TARBALL.SHA256` for current tarball hash | Done |
| Provider recovery references | Official Nebius docs for serial logs / SSH config linked | `operator_handoff.md` links Nebius serial logs and VM SSH docs | Done |
| Completion gate | Script fails unless all required factors have final loss/time/memory | `verify_completion.py` added and wired into `wait_upload_run.sh`; current local run fails with `missing summary: experiments/modded_nanogpt_bigram_sweep/summary.csv`, as expected | Done |
| Local control-flow mock | Remote sweep/summarize/verify path works with fake `uv`/`torchrun` | Mocked `run_remote_sweep.sh` generated 5/25/100 logs; `summarize_logs.py` and `verify_completion.py` accepted mock summary | Done |

## 2026-05-11 B200 Attempt

Prime pod `f706e40939d944f3911c34d0b10d0935` was polled with
`prime pods status`. It stayed `PROVISIONING` with `Installation Status:
PENDING`, never exposed IP/SSH, then moved to `TERMINATED`. No benchmark ran on
that pod, and no driver changes were attempted.

## Blocker

Last access check: `2026-05-11 17:12:24 UTC`.

The remote VM is ICMP-reachable but refuses TCP/22. The last known state:

- Torch `2.10.0+cu128` installed via `uv`.
- Initial training failed because driver `550.163.01` exposed CUDA 12.4 while
  Torch required CUDA 12.8-compatible PTX loading.
- NVIDIA driver/Fabric Manager/NSCQ were upgraded to 570.
- After reboot, SSH stopped listening/reachable.

Provider console recovery is required before the actual sweep can run.
Local checks did not find Nebius/Yandex CLI credentials or shell history entries
for this VM; only unrelated `gcloud` configuration is present. No control-plane
recovery path is available from this shell.

## Ready-To-Run Artifacts

Once SSH is restored, run this from the local repo root:

```bash
experiments/modded_nanogpt_bigram_sweep/wait_upload_run.sh
```

That script will:

1. Wait until SSH is available.
2. Upload the remote setup/smoke/sweep helpers.
3. Reapply the `BIGRAM_FACTOR`/`NPROC_PER_NODE` patch if needed.
4. Verify setup with a tiny smoke run.
5. Run `BIGRAM_FACTOR=5`, `25`, and `100`.
6. Pull logs back and write `summary.csv`.

If console recovery is needed first, run or paste:

```bash
sudo bash recovery_console.sh
```

If the VM cannot be recovered, use `fresh_vm_quickstart.md` on a replacement
H200 instance.
