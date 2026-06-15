#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn.functional as F


class ParamConfig:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs


def rms(x: torch.Tensor) -> torch.Tensor:
    return x.float().square().mean().sqrt()


def coalesce(rows: torch.Tensor, grad: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    unique, inverse = torch.unique(rows, sorted=False, return_inverse=True)
    out = torch.zeros((unique.numel(),) + tuple(grad.shape[1:]), device=grad.device, dtype=torch.float32)
    counts = torch.zeros((unique.numel(),) + (1,) * (grad.ndim - 1), device=grad.device, dtype=torch.float32)
    out.index_add_(0, inverse, grad.float())
    counts.index_add_(0, inverse, torch.ones((rows.numel(),) + (1,) * (grad.ndim - 1), device=grad.device))
    return unique, out / counts.clamp_min(1.0)


def find_engram_weight(state: dict[str, torch.Tensor]) -> torch.Tensor:
    candidates = []
    for key, value in state.items():
        if not torch.is_tensor(value) or value.ndim != 2:
            continue
        if "bigram_embed" in key and "embedding.weight" in key:
            candidates.append((key, value))
    if not candidates:
        raise KeyError("Could not find bigram_embed embedding weight in checkpoint")
    candidates.sort(key=lambda kv: kv[1].numel(), reverse=True)
    print(f"teacher_weight_key={candidates[0][0]} shape={tuple(candidates[0][1].shape)} dtype={candidates[0][1].dtype}", flush=True)
    return candidates[0][1]


def load_teacher(path: Path) -> torch.Tensor:
    ckpt = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    return find_engram_weight(state)


def extract_tensor_payload(obj: object, rows: int, name: str) -> torch.Tensor:
    if torch.is_tensor(obj):
        if obj.numel() != rows:
            raise ValueError(f"{name} rows {obj.numel()} != teacher rows {rows}")
        return obj.reshape(rows)
    if isinstance(obj, dict):
        preferred = ("hit_hist", "hist", "engram_hit_hist", "counts", "values")
        for key in preferred:
            value = obj.get(key)
            if torch.is_tensor(value) and value.numel() == rows:
                return value.reshape(rows)
        for value in obj.values():
            if torch.is_tensor(value) and value.numel() == rows:
                return value.reshape(rows)
    if isinstance(obj, (list, tuple)):
        for value in obj:
            if torch.is_tensor(value) and value.numel() == rows:
                return value.reshape(rows)
    raise TypeError(f"Could not find {name} tensor with {rows} elements")


def make_keys(items: int, group: int, key_dim: int, seed: int, device: torch.device) -> torch.Tensor:
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    state_rows = math.ceil(items / group)
    keys = torch.empty((items, key_dim), device=device)
    for state_id in range(state_rows):
        start = state_id * group
        end = min(items, start + group)
        n = end - start
        if n <= key_dim:
            raw = torch.randn((key_dim, n), generator=gen, device=device)
            q, _ = torch.linalg.qr(raw, mode="reduced")
            keys[start:end] = q.transpose(0, 1)
        else:
            keys[start:end] = F.normalize(torch.randn((n, key_dim), generator=gen, device=device), dim=-1)
    return keys


def least_squares_mse(target: torch.Tensor, keys: torch.Tensor, group: int) -> float:
    items, dim = target.shape
    state_rows = math.ceil(items / group)
    total = 0.0
    count = 0
    for state_id in range(state_rows):
        start = state_id * group
        end = min(items, start + group)
        k = keys[start:end].float()
        y = target[start:end].float()
        sol = torch.linalg.lstsq(k, y).solution
        pred = k @ sol
        total += F.mse_loss(pred, y, reduction="sum").item()
        count += y.numel()
    return total / max(count, 1)


@dataclass
class Snapshot:
    step: int
    table_mse: float
    gdn_mse: float
    table_seen_mse: float
    gdn_seen_mse: float
    table_rms: float
    gdn_rms: float


def evaluate(table: torch.Tensor, state: torch.Tensor, keys: torch.Tensor, buckets: torch.Tensor, target: torch.Tensor, seen: torch.Tensor) -> Snapshot:
    with torch.no_grad():
        table_pred = table.float()
        gdn_pred = torch.einsum("ik,ikd->id", keys.float(), state.index_select(0, buckets).float())
        table_mse = F.mse_loss(table_pred, target.float()).item()
        gdn_mse = F.mse_loss(gdn_pred, target.float()).item()
        if seen.any():
            table_seen = F.mse_loss(table_pred[seen], target.float()[seen]).item()
            gdn_seen = F.mse_loss(gdn_pred[seen], target.float()[seen]).item()
        else:
            table_seen = float("nan")
            gdn_seen = float("nan")
        return Snapshot(
            step=-1,
            table_mse=table_mse,
            gdn_mse=gdn_mse,
            table_seen_mse=table_seen,
            gdn_seen_mse=gdn_seen,
            table_rms=float(rms(table).item()),
            gdn_rms=float(rms(state).item()),
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--hit-hist", type=Path)
    parser.add_argument("--items", type=int, default=262144)
    parser.add_argument("--group", type=int, default=8)
    parser.add_argument("--key-dim", type=int, default=8)
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.03)
    parser.add_argument("--beta1", type=float, default=0.75)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--out-dir", type=Path, default=Path("reports/engram_table_gdn_fit"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    gen_cpu = torch.Generator()
    gen_cpu.manual_seed(args.seed)
    teacher = load_teacher(args.checkpoint)
    rows, dim = teacher.shape
    items = min(args.items, rows)
    if args.hit_hist and args.hit_hist.exists():
        hit_obj = torch.load(args.hit_hist, map_location="cpu", weights_only=False)
        hit = extract_tensor_payload(hit_obj, rows, "hit hist").float().clamp_min(0)
        probs = (hit + 1).double()
        sampled = torch.multinomial(probs / probs.sum(), items, replacement=False, generator=gen_cpu)
        sample_mode = "hit_weighted"
    else:
        sampled = torch.randperm(rows, generator=gen_cpu)[:items]
        sample_mode = "uniform"
    target = teacher.index_select(0, sampled).float().to(device)
    keys = make_keys(items, args.group, args.key_dim, args.seed + 17, device)
    buckets = (torch.arange(items, device=device) // args.group).long()
    state_rows = int(buckets.max().item()) + 1

    ls_mse = least_squares_mse(target, keys, args.group)
    print(
        f"sample_mode={sample_mode} items={items} dim={dim} group={args.group} "
        f"key_dim={args.key_dim} state_rows={state_rows} compression={items/state_rows:.2f} ls_mse={ls_mse:.6g}",
        flush=True,
    )

    table = torch.zeros_like(target)
    table_m = torch.zeros_like(target)
    table_v = torch.zeros_like(target)
    state = torch.zeros((state_rows, args.key_dim, dim), device=device)
    state_m = torch.zeros_like(state)
    state_v = torch.zeros_like(state)
    seen = torch.zeros((items,), device=device, dtype=torch.bool)
    gen = torch.Generator(device=device)
    gen.manual_seed(args.seed + 29)
    snapshots: list[Snapshot] = []

    for step in range(1, args.steps + 1):
        batch = torch.randint(items, (args.batch_size,), generator=gen, device=device)
        seen[batch] = True
        y = target.index_select(0, batch).float()

        pred = table.index_select(0, batch).detach().float().requires_grad_(True)
        loss = F.mse_loss(pred, y)
        grad = torch.autograd.grad(loss, pred)[0].detach()
        unique, g = coalesce(batch, grad)
        m = table_m.index_select(0, unique).float().mul(args.beta1).add(g, alpha=1 - args.beta1)
        v = table_v.index_select(0, unique).float().mul(args.beta2).addcmul(g, g, value=1 - args.beta2)
        upd = -args.lr * m / v.sqrt().clamp_min(1e-8)
        with torch.no_grad():
            table_m.index_copy_(0, unique, m)
            table_v.index_copy_(0, unique, v)
            table.index_add_(0, unique, upd.to(table.dtype))

        b = buckets.index_select(0, batch)
        k = keys.index_select(0, batch)
        state_batch = state.index_select(0, b).detach().float().requires_grad_(True)
        mem = torch.einsum("bk,bkd->bd", k.float(), state_batch)
        loss = F.mse_loss(mem, y)
        state_grad = torch.autograd.grad(loss, state_batch)[0].detach()
        unique_b, g_state = coalesce(b, state_grad)
        m = state_m.index_select(0, unique_b).float().mul(args.beta1).add(g_state, alpha=1 - args.beta1)
        v = state_v.index_select(0, unique_b).float().mul(args.beta2).addcmul(g_state, g_state, value=1 - args.beta2)
        state_upd = -args.lr * m / v.sqrt().clamp_min(1e-8)
        with torch.no_grad():
            state_m.index_copy_(0, unique_b, m)
            state_v.index_copy_(0, unique_b, v)
            state.index_add_(0, unique_b, state_upd.to(state.dtype))

        if step == 1 or step % args.eval_every == 0 or step == args.steps:
            snap = evaluate(table, state, keys, buckets, target, seen)
            snap.step = step
            snapshots.append(snap)
            print(
                f"step={step} table_mse={snap.table_mse:.6g} gdn_mse={snap.gdn_mse:.6g} "
                f"table_seen={snap.table_seen_mse:.6g} gdn_seen={snap.gdn_seen_mse:.6g}",
                flush=True,
            )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows_out = [asdict(s) for s in snapshots]
    meta = {
        "checkpoint": str(args.checkpoint),
        "hit_hist": str(args.hit_hist) if args.hit_hist else "",
        "sample_mode": sample_mode,
        "items": items,
        "teacher_rows": rows,
        "dim": dim,
        "group": args.group,
        "key_dim": args.key_dim,
        "state_rows": state_rows,
        "least_squares_mse": ls_mse,
        "gdn_update": "direct_state_adam",
    }
    (args.out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    with (args.out_dir / "history.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
        writer.writeheader()
        writer.writerows(rows_out)
    (args.out_dir / "history.json").write_text(json.dumps(rows_out, indent=2))


if __name__ == "__main__":
    main()
