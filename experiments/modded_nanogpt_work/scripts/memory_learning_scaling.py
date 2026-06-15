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


def rms_norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return x * torch.rsqrt(x.float().square().mean(dim=-1, keepdim=True) + eps).to(x.dtype)


def row_rms(x: torch.Tensor) -> torch.Tensor:
    return x.float().square().mean(dim=tuple(range(1, x.ndim))).sqrt()


def make_zipf_probs(rows: int, zipf: float, device: torch.device) -> torch.Tensor:
    ranks = torch.arange(1, rows + 1, device=device, dtype=torch.float32)
    probs = ranks.pow(-zipf)
    return probs / probs.sum()


def make_targets(rows: int, dim: int, kind: str, gen: torch.Generator, device: torch.device) -> torch.Tensor:
    if kind == "gaussian":
        z = torch.randn((rows, dim), generator=gen, device=device)
    elif kind == "rademacher":
        z = torch.empty((rows, dim), device=device).bernoulli_(0.5, generator=gen).mul_(2).sub_(1)
    elif kind == "mixed":
        gauss = torch.randn((rows, dim), generator=gen, device=device)
        signs = torch.empty((rows, dim), device=device).bernoulli_(0.5, generator=gen).mul_(2).sub_(1)
        mask = torch.empty((rows, 1), device=device).bernoulli_(0.5, generator=gen)
        z = mask * gauss + (1 - mask) * signs
    else:
        raise ValueError(f"unknown target kind {kind!r}")
    return rms_norm(z.float())


def make_item_keys(rows: int, contexts_per_row: int, key_dim: int, gen: torch.Generator, device: torch.device) -> torch.Tensor:
    if contexts_per_row <= 1:
        return F.normalize(torch.randn((rows, key_dim), generator=gen, device=device), dim=-1)
    if contexts_per_row > key_dim:
        raise ValueError("contexts_per_row must be <= key_dim for orthogonal per-row context keys")
    raw = torch.randn((rows, key_dim, contexts_per_row), generator=gen, device=device)
    q, _ = torch.linalg.qr(raw, mode="reduced")
    return q.transpose(1, 2).reshape(rows * contexts_per_row, key_dim).contiguous()


class FrozenReadout:
    def __init__(self, dim: int, hidden: int, out_dim: int, mode: str, gen: torch.Generator, device: torch.device):
        self.mode = mode
        self.dim = dim
        self.out_dim = out_dim if mode == "mlp" else dim
        if mode == "linear":
            self.w1 = None
            self.w2 = None
        elif mode == "mlp":
            self.w1 = torch.randn((dim, hidden), generator=gen, device=device) / math.sqrt(dim)
            self.w2 = torch.randn((hidden, out_dim), generator=gen, device=device) / math.sqrt(hidden)
        else:
            raise ValueError(f"unknown readout mode {mode!r}")

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if self.mode == "linear":
            return x
        h = F.silu(rms_norm(x).matmul(self.w1))
        return h.matmul(self.w2)


@dataclass
class Result:
    target: str
    readout: str
    rows: int
    contexts_per_row: int
    dim: int
    key_dim: int
    steps: int
    batch_size: int
    zipf: float
    noise: float
    method: str
    final_mse: float
    final_seen_mse: float
    final_rare_mse: float | None
    final_one_hit_mse: float | None
    hit_frac: float
    mean_hits_seen: float
    row_rms: float
    first_write_mse: float | None
    grad_rms_mean: float
    update_rms_mean: float
    update_grad_ratio: float


def evaluate_vector(table: torch.Tensor, readout: FrozenReadout, target_y: torch.Tensor, row_ids: torch.Tensor, item_ids: torch.Tensor) -> float | None:
    if item_ids.numel() == 0:
        return None
    pred = readout(table.index_select(0, row_ids.index_select(0, item_ids)).float())
    tgt = target_y.index_select(0, item_ids).float()
    return float(F.mse_loss(pred, tgt).item())


def evaluate_gdn(state: torch.Tensor, key: torch.Tensor, readout: FrozenReadout, target_y: torch.Tensor, row_ids: torch.Tensor, item_ids: torch.Tensor) -> float | None:
    if item_ids.numel() == 0:
        return None
    s = state.index_select(0, row_ids.index_select(0, item_ids)).float()
    k = key.index_select(0, item_ids).float()
    mem = torch.einsum("bk,bkd->bd", k, s)
    pred = readout(mem)
    tgt = target_y.index_select(0, item_ids).float()
    return float(F.mse_loss(pred, tgt).item())


def coalesce(rows: torch.Tensor, grad: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    unique, inverse = torch.unique(rows, sorted=False, return_inverse=True)
    summed = torch.zeros((unique.numel(),) + tuple(grad.shape[1:]), device=grad.device, dtype=torch.float32)
    counts = torch.zeros((unique.numel(),) + (1,) * (grad.ndim - 1), device=grad.device, dtype=torch.float32)
    summed.index_add_(0, inverse, grad.float())
    counts.index_add_(0, inverse, torch.ones((rows.numel(),) + (1,) * (grad.ndim - 1), device=grad.device))
    return unique, summed / counts.clamp_min(1.0)


def hit_rms(args: argparse.Namespace, hits_before: torch.Tensor, default_rms: float) -> torch.Tensor:
    if args.gdn_hit_rms_low <= 0 or args.gdn_hit_rms_high <= 0:
        return torch.full_like(hits_before.float(), default_rms)
    t = (hits_before.float() / max(args.gdn_hit_rms_knee, 1e-12)).clamp(0.0, 1.0)
    return args.gdn_hit_rms_low + (args.gdn_hit_rms_high - args.gdn_hit_rms_low) * t


def run_one(args: argparse.Namespace, target_kind: str, readout_mode: str, rows: int) -> tuple[list[dict], list[Result]]:
    device = torch.device(args.device)
    gen = torch.Generator(device=device)
    gen.manual_seed(args.seed + rows * 17 + hash((target_kind, readout_mode)) % 100000)
    contexts_per_row = max(1, args.contexts_per_row)
    items = rows * contexts_per_row
    probs = make_zipf_probs(rows, args.zipf, device)
    row_ids = torch.arange(rows, device=device).repeat_interleave(contexts_per_row)
    z = make_targets(items, args.dim, target_kind, gen, device)
    readout = FrozenReadout(args.dim, args.hidden_dim, args.out_dim, readout_mode, gen, device)
    with torch.no_grad():
        target_y = readout(z)

    key = make_item_keys(rows, contexts_per_row, args.key_dim, gen, device)
    methods = {
        "sgd": torch.zeros((rows, args.dim), device=device),
        "adam": torch.zeros((rows, args.dim), device=device),
        "adam_extra": torch.zeros((rows, args.dim), device=device),
        "normwrite": torch.zeros((rows, args.dim), device=device),
    }
    adam_m = torch.zeros_like(methods["adam"])
    adam_v = torch.zeros_like(methods["adam"])
    adam_extra_m = torch.zeros_like(methods["adam_extra"])
    adam_extra_v = torch.zeros_like(methods["adam_extra"])
    gdn = torch.zeros((rows, args.key_dim, args.dim), device=device)
    gdn_oracle = torch.zeros((rows, args.key_dim, args.dim), device=device)
    gdn_perfect = torch.zeros((rows, args.key_dim, args.dim), device=device)
    gdn_rec_adam = torch.zeros((rows, args.key_dim, args.dim), device=device)
    gdn_rec_adam_m = torch.zeros((rows, args.dim), device=device)
    gdn_rec_adam_v = torch.zeros((rows, args.dim), device=device)
    hit_count = torch.zeros((items,), device=device, dtype=torch.long)
    all_items = torch.arange(items, device=device)
    history: list[dict] = []
    gdn_recovered = torch.zeros((rows, args.key_dim, args.dim), device=device)
    if contexts_per_row > 1:
        perfect_state = torch.einsum("ik,id->ikd", key.float(), z.float()).view(rows, contexts_per_row, args.key_dim, args.dim).sum(dim=1)
        gdn_perfect.copy_(perfect_state.to(gdn_perfect.dtype))
    stat_names = ("sgd", "adam", "adam_extra", "normwrite", "gdn", "gdn_recovered", "gdn_rec_adam", "gdn_oracle", "gdn_perfect")
    grad_sums = {name: 0.0 for name in stat_names}
    update_sums = {name: 0.0 for name in stat_names}
    grad_counts = {name: 0 for name in stat_names}
    first_write = {name: [] for name in stat_names}

    for step in range(1, args.steps + 1):
        sampled_rows = torch.multinomial(probs, args.batch_size, replacement=True, generator=gen)
        if contexts_per_row == 1:
            batch_items = sampled_rows
        else:
            ctx = torch.randint(contexts_per_row, (args.batch_size,), generator=gen, device=device)
            batch_items = sampled_rows * contexts_per_row + ctx
        batch_rows = row_ids.index_select(0, batch_items)
        if args.noise > 0:
            noise = torch.randn((args.batch_size, readout.out_dim), generator=gen, device=device) * args.noise
        else:
            noise = 0.0
        y = target_y.index_select(0, batch_items).float() + noise
        hits_before = hit_count.index_select(0, batch_items)
        first = hits_before == 0
        hit_count.index_add_(0, batch_items, torch.ones_like(batch_items, dtype=hit_count.dtype))

        for name, table in methods.items():
            local_steps = max(1, args.adam_extra_steps) if name == "adam_extra" else 1
            for _ in range(local_steps):
                mem = table.index_select(0, batch_rows).detach().float().requires_grad_(True)
                loss = F.mse_loss(readout(mem), y)
                grad = torch.autograd.grad(loss, mem)[0].detach()
                unique, g = coalesce(batch_rows, grad)
                if name == "sgd":
                    upd = -args.sgd_lr * g
                elif name in ("adam", "adam_extra"):
                    if name == "adam":
                        m_store, v_store = adam_m, adam_v
                    else:
                        m_store, v_store = adam_extra_m, adam_extra_v
                    m = m_store.index_select(0, unique).float().mul(args.beta1).add(g, alpha=1 - args.beta1)
                    v = v_store.index_select(0, unique).float().mul(args.beta2).addcmul(g, g, value=1 - args.beta2)
                    upd = -args.adam_lr * m / v.sqrt().clamp_min(args.eps)
                    m_store.index_copy_(0, unique, m)
                    v_store.index_copy_(0, unique, v)
                else:
                    upd = -grad.sign() if args.normwrite_sign else -g
                    upd = rms_norm(upd) * args.normwrite_rms
                with torch.no_grad():
                    table.index_add_(0, unique, upd.to(table.dtype))
                    if args.row_cap > 0:
                        cur = table.index_select(0, unique).float()
                        scale = (args.row_cap / row_rms(cur).clamp_min(1e-12)).clamp_max(1.0).view(-1, 1)
                        table.index_copy_(0, unique, (cur * scale).to(table.dtype))
                grad_sums[name] += float(g.square().mean().sqrt().item())
                update_sums[name] += float(upd.square().mean().sqrt().item())
                grad_counts[name] += 1
            if first.any():
                first_write[name].append(evaluate_vector(table, readout, target_y, row_ids, batch_items[first]))

        k_batch = key.index_select(0, batch_items)
        state_batch = gdn.index_select(0, batch_rows).detach().float().requires_grad_(True)
        mem = torch.einsum("bk,bkd->bd", k_batch.float(), state_batch)
        loss = F.mse_loss(readout(mem), y)
        state_grad = torch.autograd.grad(loss, state_batch)[0].detach()
        if args.gdn_normwrite and args.gdn_hit_rms_low > 0 and args.gdn_hit_rms_high > 0:
            sample_rms = hit_rms(args, hits_before, args.gdn_write_rms).view(-1, 1, 1)
            sample_upd = -rms_norm(state_grad.flatten(start_dim=1)).view_as(state_grad) * sample_rms
            unique, upd_state = coalesce(batch_rows, sample_upd)
            _, g_state = coalesce(batch_rows, state_grad)
        else:
            unique, g_state = coalesce(batch_rows, state_grad)
            if args.gdn_normwrite:
                upd_state = -rms_norm(g_state.flatten(start_dim=1)).view_as(g_state) * args.gdn_write_rms
            else:
                upd_state = -args.gdn_lr * g_state
        with torch.no_grad():
            gdn.index_add_(0, unique, upd_state.to(gdn.dtype))
            if args.gdn_cap > 0:
                cur = gdn.index_select(0, unique).float()
                scale = (args.gdn_cap / row_rms(cur).clamp_min(1e-12)).clamp_max(1.0).view(-1, 1, 1)
                gdn.index_copy_(0, unique, (cur * scale).to(gdn.dtype))
        grad_sums["gdn"] += float(g_state.square().mean().sqrt().item())
        update_sums["gdn"] += float(upd_state.square().mean().sqrt().item())
        grad_counts["gdn"] += 1
        if first.any():
            first_write["gdn"].append(evaluate_gdn(gdn, key, readout, target_y, row_ids, batch_items[first]))

        rec_state_batch = gdn_recovered.index_select(0, batch_rows).detach().float().requires_grad_(True)
        rec_mem = torch.einsum("bk,bkd->bd", k_batch.float(), rec_state_batch)
        loss = F.mse_loss(readout(rec_mem), y)
        rec_state_grad = torch.autograd.grad(loss, rec_state_batch)[0].detach()
        key_norm2 = k_batch.float().square().sum(dim=-1, keepdim=True).clamp_min(1e-12)
        rec_mem_grad = torch.einsum("bk,bkd->bd", k_batch.float(), rec_state_grad.float()) / key_norm2
        if args.gdn_recovered_normwrite:
            rec_delta_mem = -rms_norm(rec_mem_grad) * hit_rms(args, hits_before, args.gdn_recovered_write_rms).view(-1, 1)
        else:
            rec_delta_mem = -args.gdn_recovered_lr * rec_mem_grad
        rec_outer = torch.einsum("bk,bd->bkd", k_batch.float() / key_norm2, rec_delta_mem.float())
        unique, rec_update = coalesce(batch_rows, rec_outer)
        with torch.no_grad():
            gdn_recovered.index_add_(0, unique, rec_update.to(gdn_recovered.dtype))
            if args.gdn_cap > 0:
                cur = gdn_recovered.index_select(0, unique).float()
                scale = (args.gdn_cap / row_rms(cur).clamp_min(1e-12)).clamp_max(1.0).view(-1, 1, 1)
                gdn_recovered.index_copy_(0, unique, (cur * scale).to(gdn_recovered.dtype))
        grad_sums["gdn_recovered"] += float(rec_mem_grad.square().mean().sqrt().item())
        update_sums["gdn_recovered"] += float(rec_delta_mem.square().mean().sqrt().item())
        grad_counts["gdn_recovered"] += 1
        if first.any():
            first_write["gdn_recovered"].append(evaluate_gdn(gdn_recovered, key, readout, target_y, row_ids, batch_items[first]))

        rec_adam_state_batch = gdn_rec_adam.index_select(0, batch_rows).detach().float().requires_grad_(True)
        rec_adam_mem = torch.einsum("bk,bkd->bd", k_batch.float(), rec_adam_state_batch)
        loss = F.mse_loss(readout(rec_adam_mem), y)
        rec_adam_state_grad = torch.autograd.grad(loss, rec_adam_state_batch)[0].detach()
        rec_adam_mem_grad = torch.einsum("bk,bkd->bd", k_batch.float(), rec_adam_state_grad.float()) / key_norm2
        unique, rec_adam_g = coalesce(batch_rows, rec_adam_mem_grad)
        m = gdn_rec_adam_m.index_select(0, unique).float().mul(args.beta1).add(rec_adam_g, alpha=1 - args.beta1)
        v = gdn_rec_adam_v.index_select(0, unique).float().mul(args.beta2).addcmul(rec_adam_g, rec_adam_g, value=1 - args.beta2)
        rec_adam_delta_mem = -args.gdn_rec_adam_lr * m / v.sqrt().clamp_min(args.eps)
        unique_key = key.index_select(0, unique).float()
        unique_key_norm2 = unique_key.square().sum(dim=-1, keepdim=True).clamp_min(1e-12)
        rec_adam_update = torch.einsum("bk,bd->bkd", unique_key / unique_key_norm2, rec_adam_delta_mem.float())
        with torch.no_grad():
            gdn_rec_adam_m.index_copy_(0, unique, m)
            gdn_rec_adam_v.index_copy_(0, unique, v)
            gdn_rec_adam.index_add_(0, unique, rec_adam_update.to(gdn_rec_adam.dtype))
            if args.gdn_cap > 0:
                cur = gdn_rec_adam.index_select(0, unique).float()
                scale = (args.gdn_cap / row_rms(cur).clamp_min(1e-12)).clamp_max(1.0).view(-1, 1, 1)
                gdn_rec_adam.index_copy_(0, unique, (cur * scale).to(gdn_rec_adam.dtype))
        grad_sums["gdn_rec_adam"] += float(rec_adam_g.square().mean().sqrt().item())
        update_sums["gdn_rec_adam"] += float(rec_adam_delta_mem.square().mean().sqrt().item())
        grad_counts["gdn_rec_adam"] += 1
        if first.any():
            first_write["gdn_rec_adam"].append(evaluate_gdn(gdn_rec_adam, key, readout, target_y, row_ids, batch_items[first]))

        oracle_state_batch = gdn_oracle.index_select(0, batch_rows).detach().float()
        oracle_mem = torch.einsum("bk,bkd->bd", k_batch.float(), oracle_state_batch).requires_grad_(True)
        loss = F.mse_loss(readout(oracle_mem), y)
        mem_grad = torch.autograd.grad(loss, oracle_mem)[0].detach()
        if args.gdn_oracle_normwrite:
            delta_mem = -rms_norm(mem_grad) * hit_rms(args, hits_before, args.gdn_oracle_write_rms).view(-1, 1)
        else:
            delta_mem = -args.gdn_oracle_lr * mem_grad
        oracle_outer = torch.einsum("bk,bd->bkd", k_batch.float(), delta_mem.float())
        unique, oracle_update = coalesce(batch_rows, oracle_outer)
        with torch.no_grad():
            gdn_oracle.index_add_(0, unique, oracle_update.to(gdn_oracle.dtype))
            if args.gdn_cap > 0:
                cur = gdn_oracle.index_select(0, unique).float()
                scale = (args.gdn_cap / row_rms(cur).clamp_min(1e-12)).clamp_max(1.0).view(-1, 1, 1)
                gdn_oracle.index_copy_(0, unique, (cur * scale).to(gdn_oracle.dtype))
        grad_sums["gdn_oracle"] += float(mem_grad.square().mean().sqrt().item())
        update_sums["gdn_oracle"] += float(delta_mem.square().mean().sqrt().item())
        grad_counts["gdn_oracle"] += 1
        if first.any():
            first_write["gdn_oracle"].append(evaluate_gdn(gdn_oracle, key, readout, target_y, row_ids, batch_items[first]))

        if contexts_per_row == 1:
            perfect_z = z.index_select(0, batch_items).float()
            perfect_key_norm2 = k_batch.float().square().sum(dim=-1, keepdim=True).clamp_min(1e-12)
            perfect_outer = torch.einsum("bk,bd->bkd", k_batch.float() / perfect_key_norm2, perfect_z)
            unique, perfect_update = coalesce(batch_rows, perfect_outer)
            with torch.no_grad():
                before = gdn_perfect.index_select(0, unique).float()
                gdn_perfect.index_copy_(0, unique, perfect_update.to(gdn_perfect.dtype))
            update_sums["gdn_perfect"] += float((perfect_update.float() - before).square().mean().sqrt().item())
        grad_counts["gdn_perfect"] += 1
        if first.any():
            first_write["gdn_perfect"].append(evaluate_gdn(gdn_perfect, key, readout, target_y, row_ids, batch_items[first]))

        if step == 1 or step % args.eval_every == 0 or step == args.steps:
            seen = all_items[hit_count > 0]
            rare = all_items[(hit_count > 0) & (hit_count <= args.rare_hits)]
            one = all_items[hit_count == 1]
            row = {
                "target": target_kind,
                "readout": readout_mode,
                "rows": rows,
                "contexts_per_row": contexts_per_row,
                "step": step,
                "hit_frac": float((hit_count > 0).float().mean().item()),
                "mean_hits_seen": float(hit_count[hit_count > 0].float().mean().item()) if seen.numel() else 0.0,
            }
            for name, table in methods.items():
                row[f"{name}_mse"] = evaluate_vector(table, readout, target_y, row_ids, all_items)
                row[f"{name}_seen_mse"] = evaluate_vector(table, readout, target_y, row_ids, seen)
                row[f"{name}_rare_mse"] = evaluate_vector(table, readout, target_y, row_ids, rare)
                row[f"{name}_one_hit_mse"] = evaluate_vector(table, readout, target_y, row_ids, one)
                row[f"{name}_rms"] = float(row_rms(table).mean().item())
            row["gdn_mse"] = evaluate_gdn(gdn, key, readout, target_y, row_ids, all_items)
            row["gdn_seen_mse"] = evaluate_gdn(gdn, key, readout, target_y, row_ids, seen)
            row["gdn_rare_mse"] = evaluate_gdn(gdn, key, readout, target_y, row_ids, rare)
            row["gdn_one_hit_mse"] = evaluate_gdn(gdn, key, readout, target_y, row_ids, one)
            row["gdn_rms"] = float(row_rms(gdn).mean().item())
            row["gdn_recovered_mse"] = evaluate_gdn(gdn_recovered, key, readout, target_y, row_ids, all_items)
            row["gdn_recovered_seen_mse"] = evaluate_gdn(gdn_recovered, key, readout, target_y, row_ids, seen)
            row["gdn_recovered_rare_mse"] = evaluate_gdn(gdn_recovered, key, readout, target_y, row_ids, rare)
            row["gdn_recovered_one_hit_mse"] = evaluate_gdn(gdn_recovered, key, readout, target_y, row_ids, one)
            row["gdn_recovered_rms"] = float(row_rms(gdn_recovered).mean().item())
            row["gdn_rec_adam_mse"] = evaluate_gdn(gdn_rec_adam, key, readout, target_y, row_ids, all_items)
            row["gdn_rec_adam_seen_mse"] = evaluate_gdn(gdn_rec_adam, key, readout, target_y, row_ids, seen)
            row["gdn_rec_adam_rare_mse"] = evaluate_gdn(gdn_rec_adam, key, readout, target_y, row_ids, rare)
            row["gdn_rec_adam_one_hit_mse"] = evaluate_gdn(gdn_rec_adam, key, readout, target_y, row_ids, one)
            row["gdn_rec_adam_rms"] = float(row_rms(gdn_rec_adam).mean().item())
            row["gdn_oracle_mse"] = evaluate_gdn(gdn_oracle, key, readout, target_y, row_ids, all_items)
            row["gdn_oracle_seen_mse"] = evaluate_gdn(gdn_oracle, key, readout, target_y, row_ids, seen)
            row["gdn_oracle_rare_mse"] = evaluate_gdn(gdn_oracle, key, readout, target_y, row_ids, rare)
            row["gdn_oracle_one_hit_mse"] = evaluate_gdn(gdn_oracle, key, readout, target_y, row_ids, one)
            row["gdn_oracle_rms"] = float(row_rms(gdn_oracle).mean().item())
            row["gdn_perfect_mse"] = evaluate_gdn(gdn_perfect, key, readout, target_y, row_ids, all_items)
            row["gdn_perfect_seen_mse"] = evaluate_gdn(gdn_perfect, key, readout, target_y, row_ids, seen)
            row["gdn_perfect_rare_mse"] = evaluate_gdn(gdn_perfect, key, readout, target_y, row_ids, rare)
            row["gdn_perfect_one_hit_mse"] = evaluate_gdn(gdn_perfect, key, readout, target_y, row_ids, one)
            row["gdn_perfect_rms"] = float(row_rms(gdn_perfect).mean().item())
            history.append(row)

    final = history[-1]
    results = []
    for method in stat_names:
        grad_mean = grad_sums[method] / max(grad_counts[method], 1)
        upd_mean = update_sums[method] / max(grad_counts[method], 1)
        vals = [v for v in first_write[method] if v is not None]
        results.append(Result(
            target=target_kind,
            readout=readout_mode,
            rows=rows,
            contexts_per_row=contexts_per_row,
            dim=args.dim,
            key_dim=args.key_dim,
            steps=args.steps,
            batch_size=args.batch_size,
            zipf=args.zipf,
            noise=args.noise,
            method=method,
            final_mse=float(final[f"{method}_mse"]),
            final_seen_mse=float(final[f"{method}_seen_mse"]) if final[f"{method}_seen_mse"] is not None else float("nan"),
            final_rare_mse=final[f"{method}_rare_mse"],
            final_one_hit_mse=final[f"{method}_one_hit_mse"],
            hit_frac=float(final["hit_frac"]),
            mean_hits_seen=float(final["mean_hits_seen"]),
            row_rms=float(final[f"{method}_rms"]),
            first_write_mse=sum(vals) / len(vals) if vals else None,
            grad_rms_mean=grad_mean,
            update_rms_mean=upd_mean,
            update_grad_ratio=upd_mean / max(grad_mean, 1e-12),
        ))
    return history, results


def write_report(summary: list[Result], history: list[dict], out_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    rows = [asdict(r) for r in summary]
    with (out_dir / "summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    (out_dir / "summary.json").write_text(json.dumps(rows, indent=2))
    with (out_dir / "history.json").open("w") as f:
        json.dump(history, f, indent=2)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4), dpi=150)
    methods = ["sgd", "adam", "adam_extra", "normwrite", "gdn", "gdn_recovered", "gdn_rec_adam", "gdn_oracle", "gdn_perfect"]
    markers = ["o", "s", "p", "^", "D", "X", "v", "P", "*"]
    for readout in sorted({r.readout for r in summary}):
        for method, marker in zip(methods, markers):
            xs = [r.rows for r in summary if r.readout == readout and r.method == method and r.target == "mixed"]
            ys = [r.final_seen_mse for r in summary if r.readout == readout and r.method == method and r.target == "mixed"]
            if xs:
                axes[0].plot(xs, ys, marker=marker, label=f"{readout}:{method}")
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("rows")
    axes[0].set_ylabel("final seen MSE, mixed targets")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(frameon=False, fontsize=7)

    for method, marker in zip(methods, markers):
        xs = [r.rows for r in summary if r.readout == "mlp" and r.method == method and r.target == "mixed"]
        ys = [r.update_grad_ratio for r in summary if r.readout == "mlp" and r.method == method and r.target == "mixed"]
        if xs:
            axes[1].plot(xs, ys, marker=marker, label=method)
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("rows")
    axes[1].set_ylabel("mean update RMS / grad RMS")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(frameon=False, fontsize=8)

    for target in sorted({r.target for r in summary}):
        vals = [r for r in summary if r.readout == "mlp" and r.target == target]
        xs = list(range(len(vals)))
        axes[2].bar([x + 0.25 * i for i, x in enumerate(xs)], [r.final_seen_mse for r in vals], width=0.2)
    axes[2].axis("off")
    fig.tight_layout()
    fig.savefig(out_dir / "scaling.png")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", default="4096,16384,65536")
    parser.add_argument("--contexts-per-row", type=int, default=1)
    parser.add_argument("--targets", default="gaussian,rademacher,mixed")
    parser.add_argument("--readouts", default="linear,mlp")
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--out-dim", type=int, default=128)
    parser.add_argument("--key-dim", type=int, default=16)
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--zipf", type=float, default=1.05)
    parser.add_argument("--noise", type=float, default=0.03)
    parser.add_argument("--rare-hits", type=int, default=4)
    parser.add_argument("--sgd-lr", type=float, default=4.0)
    parser.add_argument("--adam-lr", type=float, default=0.04)
    parser.add_argument("--adam-extra-steps", type=int, default=4)
    parser.add_argument("--beta1", type=float, default=0.75)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--normwrite-rms", type=float, default=0.02)
    parser.add_argument("--normwrite-sign", action="store_true")
    parser.add_argument("--row-cap", type=float, default=4.0)
    parser.add_argument("--gdn-lr", type=float, default=16.0)
    parser.add_argument("--gdn-normwrite", action="store_true")
    parser.add_argument("--gdn-write-rms", type=float, default=0.02)
    parser.add_argument("--gdn-recovered-lr", type=float, default=16.0)
    parser.add_argument("--gdn-recovered-normwrite", action="store_true")
    parser.add_argument("--gdn-recovered-write-rms", type=float, default=0.02)
    parser.add_argument("--gdn-rec-adam-lr", type=float, default=0.04)
    parser.add_argument("--gdn-oracle-lr", type=float, default=16.0)
    parser.add_argument("--gdn-oracle-normwrite", action="store_true")
    parser.add_argument("--gdn-oracle-write-rms", type=float, default=0.02)
    parser.add_argument("--gdn-cap", type=float, default=2.0)
    parser.add_argument("--gdn-hit-rms-low", type=float, default=0.0)
    parser.add_argument("--gdn-hit-rms-high", type=float, default=0.0)
    parser.add_argument("--gdn-hit-rms-knee", type=float, default=4.0)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out-dir", type=Path, default=Path("reports/memory_learning_scaling"))
    args = parser.parse_args()

    all_history = []
    all_results = []
    for rows in [int(x) for x in args.rows.replace(",", " ").split() if x]:
        for target in [x.strip() for x in args.targets.split(",") if x.strip()]:
            for readout in [x.strip() for x in args.readouts.split(",") if x.strip()]:
                hist, results = run_one(args, target, readout, rows)
                all_history.extend(hist)
                all_results.extend(results)
                best = min(results, key=lambda r: r.final_seen_mse)
                print(f"done rows={rows} target={target} readout={readout} best={best.method} seen_mse={best.final_seen_mse:.4g}")
    write_report(all_results, all_history, args.out_dir)
    print(f"wrote {args.out_dir / 'summary.csv'}")
    print(f"wrote {args.out_dir / 'scaling.png'}")


if __name__ == "__main__":
    main()
