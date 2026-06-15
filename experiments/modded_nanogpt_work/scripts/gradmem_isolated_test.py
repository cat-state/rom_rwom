#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F


def make_distribution(num_rows: int, zipf: float, device: torch.device) -> torch.Tensor:
    ranks = torch.arange(1, num_rows + 1, device=device, dtype=torch.float32)
    probs = ranks.pow(-zipf)
    return probs / probs.sum()


def sample_batch(probs: torch.Tensor, batch_size: int, gen: torch.Generator) -> torch.Tensor:
    return torch.multinomial(probs, batch_size, replacement=True, generator=gen)


def row_rms(x: torch.Tensor) -> torch.Tensor:
    return x.float().square().mean(dim=-1).sqrt()


def update_vector_gradmem(table: torch.Tensor, rows: torch.Tensor, grad: torch.Tensor, lr: float, decay: float, cap: float) -> None:
    with torch.no_grad():
        unique, inverse = torch.unique(rows, sorted=False, return_inverse=True)
        summed = torch.zeros((unique.numel(), grad.size(-1)), device=grad.device, dtype=torch.float32)
        counts = torch.zeros((unique.numel(), 1), device=grad.device, dtype=torch.float32)
        summed.index_add_(0, inverse, grad.float())
        counts.index_add_(0, inverse, torch.ones((rows.numel(), 1), device=grad.device))
        upd = -lr * summed / counts.clamp_min(1.0)
        cur = table.index_select(0, unique).float().mul(decay).add_(upd)
        if cap > 0:
            rms = row_rms(cur).unsqueeze(-1)
            cur.mul_((cap / rms.clamp_min(1e-12)).clamp_max(1.0))
        table.index_copy_(0, unique, cur.to(table.dtype))


def update_gdn_gradmem(state: torch.Tensor, rows: torch.Tensor, keys: torch.Tensor, grad: torch.Tensor, lr: float, decay: float, cap: float) -> None:
    with torch.no_grad():
        unique, inverse = torch.unique(rows, sorted=False, return_inverse=True)
        kd, vd = state.shape[1], state.shape[2]
        outer = torch.einsum("bk,bv->bkv", keys.float(), grad.float())
        summed = torch.zeros((unique.numel(), kd, vd), device=grad.device, dtype=torch.float32)
        counts = torch.zeros((unique.numel(), 1, 1), device=grad.device, dtype=torch.float32)
        summed.index_add_(0, inverse, outer)
        counts.index_add_(0, inverse, torch.ones((rows.numel(), 1, 1), device=grad.device))
        upd = -lr * summed / counts.clamp_min(1.0)
        cur = state.index_select(0, unique).float().mul(decay).add_(upd)
        if cap > 0:
            rms = cur.square().mean(dim=(1, 2), keepdim=True).sqrt()
            cur.mul_((cap / rms.clamp_min(1e-12)).clamp_max(1.0))
        state.index_copy_(0, unique, cur.to(state.dtype))


def evaluate_vector(table: torch.Tensor, target: torch.Tensor, rows: torch.Tensor) -> float:
    pred = table.index_select(0, rows).float()
    tgt = target.index_select(0, rows).float()
    return float(F.mse_loss(pred, tgt).item())


def evaluate_gdn(state: torch.Tensor, target: torch.Tensor, row_key: torch.Tensor, rows: torch.Tensor) -> float:
    s = state.index_select(0, rows).float()
    q = row_key.index_select(0, rows).float()
    pred = torch.einsum("bk,bkv->bv", q, s)
    tgt = target.index_select(0, rows).float()
    return float(F.mse_loss(pred, tgt).item())


def run(args: argparse.Namespace) -> dict:
    device = torch.device(args.device)
    gen = torch.Generator(device=device)
    gen.manual_seed(args.seed)

    probs = make_distribution(args.rows, args.zipf, device)
    target = torch.randn((args.rows, args.dim), generator=gen, device=device)
    target = F.rms_norm(target, (args.dim,))
    key = torch.randn((args.rows, args.key_dim), generator=gen, device=device)
    key = F.normalize(key, dim=-1)

    sgd = torch.zeros((args.rows, args.dim), device=device)
    adam = torch.zeros((args.rows, args.dim), device=device)
    adam_m = torch.zeros_like(adam)
    adam_v = torch.zeros_like(adam)
    gradmem = torch.zeros((args.rows, args.dim), device=device)
    gdn = torch.zeros((args.rows, args.key_dim, args.dim), device=device)
    hit_count = torch.zeros((args.rows,), device=device, dtype=torch.long)

    eval_rows = torch.arange(args.rows, device=device)
    history = []
    first_hit_loss = {"sgd": [], "adam": [], "gradmem": [], "gdn": []}
    post_first_write_loss = {"sgd": [], "adam": [], "gradmem": [], "gdn": []}

    for step in range(1, args.steps + 1):
        rows = sample_batch(probs, args.batch_size, gen)
        tgt = target.index_select(0, rows)
        k = key.index_select(0, rows)
        first = hit_count.index_select(0, rows) == 0
        hit_count.index_add_(0, rows, torch.ones_like(rows, dtype=hit_count.dtype))

        # Squared-error gradient wrt output: d mean((pred-target)^2)/d pred.
        for name, table in (("sgd", sgd), ("adam", adam), ("gradmem", gradmem)):
            pred = table.index_select(0, rows).float()
            if first.any():
                first_hit_loss[name].append(float(F.mse_loss(pred[first], tgt[first].float()).item()))

        gdn_pred = torch.einsum("bk,bkv->bv", k.float(), gdn.index_select(0, rows).float())
        if first.any():
            first_hit_loss["gdn"].append(float(F.mse_loss(gdn_pred[first], tgt[first].float()).item()))

        grad_sgd = 2.0 * (sgd.index_select(0, rows).float() - tgt.float()) / args.dim
        with torch.no_grad():
            unique, inverse = torch.unique(rows, sorted=False, return_inverse=True)
            summed = torch.zeros((unique.numel(), args.dim), device=device)
            counts = torch.zeros((unique.numel(), 1), device=device)
            summed.index_add_(0, inverse, grad_sgd)
            counts.index_add_(0, inverse, torch.ones((rows.numel(), 1), device=device))
            upd = summed / counts.clamp_min(1.0)
            sgd.index_add_(0, unique, -args.sgd_lr * upd)

        grad_adam = 2.0 * (adam.index_select(0, rows).float() - tgt.float()) / args.dim
        with torch.no_grad():
            unique, inverse = torch.unique(rows, sorted=False, return_inverse=True)
            summed = torch.zeros((unique.numel(), args.dim), device=device)
            counts = torch.zeros((unique.numel(), 1), device=device)
            summed.index_add_(0, inverse, grad_adam)
            counts.index_add_(0, inverse, torch.ones((rows.numel(), 1), device=device))
            g = summed / counts.clamp_min(1.0)
            m = adam_m.index_select(0, unique).mul(args.beta1).add(g, alpha=1 - args.beta1)
            v = adam_v.index_select(0, unique).mul(args.beta2).addcmul(g, g, value=1 - args.beta2)
            upd = m / v.sqrt().clamp_min(args.eps)
            adam.index_add_(0, unique, -args.adam_lr * upd)
            adam_m.index_copy_(0, unique, m)
            adam_v.index_copy_(0, unique, v)

        grad_gradmem = 2.0 * (gradmem.index_select(0, rows).float() - tgt.float()) / args.dim
        update_vector_gradmem(gradmem, rows, grad_gradmem, args.gradmem_lr, args.gradmem_decay, args.gradmem_cap)

        grad_gdn = 2.0 * (gdn_pred - tgt.float()) / args.dim
        update_gdn_gradmem(gdn, rows, k, grad_gdn, args.gdn_lr, args.gdn_decay, args.gdn_cap)

        if first.any():
            first_rows = rows[first]
            first_tgt = target.index_select(0, first_rows).float()
            for name, table in (("sgd", sgd), ("adam", adam), ("gradmem", gradmem)):
                pred = table.index_select(0, first_rows).float()
                post_first_write_loss[name].append(float(F.mse_loss(pred, first_tgt).item()))
            first_key = key.index_select(0, first_rows).float()
            first_state = gdn.index_select(0, first_rows).float()
            pred = torch.einsum("bk,bkv->bv", first_key, first_state)
            post_first_write_loss["gdn"].append(float(F.mse_loss(pred, first_tgt).item()))

        if step % args.eval_every == 0 or step == 1 or step == args.steps:
            with torch.no_grad():
                one_touch = eval_rows[hit_count == 1]
                rare_touch = eval_rows[(hit_count > 0) & (hit_count <= 4)]
                one_touch = one_touch if one_touch.numel() > 0 else eval_rows[:0]
                rare_touch = rare_touch if rare_touch.numel() > 0 else eval_rows[:0]
                history.append({
                    "step": step,
                    "hit_frac": float((hit_count > 0).float().mean().item()),
                    "sgd_mse": evaluate_vector(sgd, target, eval_rows),
                    "adam_mse": evaluate_vector(adam, target, eval_rows),
                    "gradmem_mse": evaluate_vector(gradmem, target, eval_rows),
                    "gdn_mse": evaluate_gdn(gdn, target, key, eval_rows),
                    "sgd_one_touch_mse": evaluate_vector(sgd, target, one_touch) if one_touch.numel() else None,
                    "adam_one_touch_mse": evaluate_vector(adam, target, one_touch) if one_touch.numel() else None,
                    "gradmem_one_touch_mse": evaluate_vector(gradmem, target, one_touch) if one_touch.numel() else None,
                    "gdn_one_touch_mse": evaluate_gdn(gdn, target, key, one_touch) if one_touch.numel() else None,
                    "sgd_rare_touch_mse": evaluate_vector(sgd, target, rare_touch) if rare_touch.numel() else None,
                    "adam_rare_touch_mse": evaluate_vector(adam, target, rare_touch) if rare_touch.numel() else None,
                    "gradmem_rare_touch_mse": evaluate_vector(gradmem, target, rare_touch) if rare_touch.numel() else None,
                    "gdn_rare_touch_mse": evaluate_gdn(gdn, target, key, rare_touch) if rare_touch.numel() else None,
                    "sgd_rms": float(row_rms(sgd).mean().item()),
                    "adam_rms": float(row_rms(adam).mean().item()),
                    "gradmem_rms": float(row_rms(gradmem).mean().item()),
                    "gdn_rms": float(gdn.float().square().mean(dim=(1, 2)).sqrt().mean().item()),
                })

    return {
        "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "history": history,
        "first_hit_loss_mean": {
            key_: (sum(vals) / len(vals) if vals else None)
            for key_, vals in first_hit_loss.items()
        },
        "post_first_write_loss_mean": {
            key_: (sum(vals) / len(vals) if vals else None)
            for key_, vals in post_first_write_loss.items()
        },
        "hit_count_quantiles": {
            str(q): float(torch.quantile(hit_count.float(), q).item())
            for q in (0.0, 0.25, 0.5, 0.75, 0.9, 0.99, 1.0)
        },
    }


def write_plot(result: dict, out_png: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    hist = result["history"]
    steps = [p["step"] for p in hist]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), dpi=140)
    for key, label in [
        ("sgd_mse", "row SGD"),
        ("adam_mse", "row Adam"),
        ("gradmem_mse", "vector GradMem"),
        ("gdn_mse", "GDN row"),
    ]:
        axes[0].plot(steps, [p[key] for p in hist], label=label)
    axes[0].set_yscale("log")
    axes[0].set_xlabel("step")
    axes[0].set_ylabel("all-row MSE")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(frameon=False)

    for key, label in [
        ("sgd_rms", "row SGD"),
        ("adam_rms", "row Adam"),
        ("gradmem_rms", "vector GradMem"),
        ("gdn_rms", "GDN row"),
    ]:
        axes[1].plot(steps, [p[key] for p in hist], label=label)
    axes[1].set_xlabel("step")
    axes[1].set_ylabel("mean row/state RMS")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=65536)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--key-dim", type=int, default=16)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--zipf", type=float, default=1.05)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--sgd-lr", type=float, default=2.0)
    parser.add_argument("--adam-lr", type=float, default=0.05)
    parser.add_argument("--beta1", type=float, default=0.75)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--gradmem-lr", type=float, default=16.0)
    parser.add_argument("--gradmem-decay", type=float, default=1.0)
    parser.add_argument("--gradmem-cap", type=float, default=4.0)
    parser.add_argument("--gdn-lr", type=float, default=32.0)
    parser.add_argument("--gdn-decay", type=float, default=1.0)
    parser.add_argument("--gdn-cap", type=float, default=2.0)
    parser.add_argument("--out", type=Path, default=Path("reports/gradmem_isolated/gradmem_isolated.json"))
    args = parser.parse_args()

    result = run(args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2))
    plot_path = args.out.with_suffix(".png")
    write_plot(result, plot_path)
    print(f"wrote {args.out}")
    print(f"wrote {plot_path}")
    print(json.dumps(result["history"][-1], indent=2))
    print("first_hit_loss_mean", json.dumps(result["first_hit_loss_mean"], indent=2))
    print("post_first_write_loss_mean", json.dumps(result["post_first_write_loss_mean"], indent=2))


if __name__ == "__main__":
    main()
