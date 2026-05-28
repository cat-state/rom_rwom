#!/usr/bin/env python3
import argparse
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt


VAL_RE = re.compile(r"step:(?P<step>\d+)/\d+\s+val_loss:(?P<loss>nan|[0-9.]+)")
CFG_RE = re.compile(r"rom_sparse_adam_lr_mul=(?P<lr>[0-9.]+)")
RUN_RE = re.compile(r"sparseadam(?P<lr>[0-9.]+)_")


def read_lr(path: Path) -> float:
    text = path.read_text(errors="replace")
    if match := CFG_RE.search(text):
        return float(match.group("lr"))
    if match := RUN_RE.search(path.name):
        return float(match.group("lr"))
    raise ValueError(f"Could not infer lr multiplier from {path}")


def read_points(path: Path) -> tuple[list[int], list[float]]:
    steps: list[int] = []
    losses: list[float] = []
    for match in VAL_RE.finditer(path.read_text(errors="replace")):
        loss = float(match.group("loss"))
        if math.isnan(loss):
            continue
        steps.append(int(match.group("step")))
        losses.append(loss)
    return steps, losses


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("logs", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, default=Path("sparse_adam_lr_sweep.png"))
    args = parser.parse_args()

    rows = []
    for path in sorted(args.logs):
        steps, losses = read_points(path)
        if not losses:
            continue
        rows.append((read_lr(path), path, steps, losses))
    rows.sort(key=lambda row: row[0])
    if not rows:
        raise SystemExit("No validation points found")

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.0), dpi=180)
    cmap = plt.get_cmap("viridis")
    denom = max(len(rows) - 1, 1)

    for idx, (lr, path, steps, losses) in enumerate(rows):
        color = cmap(idx / denom)
        label = f"LR x{lr:g}"
        axes[0].plot(steps, losses, linewidth=2.0, color=color, label=label)
        axes[0].annotate(
            f"{losses[-1]:.4f}",
            xy=(steps[-1], losses[-1]),
            xytext=(5, 0),
            textcoords="offset points",
            va="center",
            fontsize=8,
            color=color,
        )

    lrs = [row[0] for row in rows]
    final_losses = [row[3][-1] for row in rows]
    final_steps = [row[2][-1] for row in rows]
    axes[1].plot(lrs, final_losses, linewidth=2.0, color="#333333")
    axes[1].scatter(lrs, final_losses, s=42, color="#333333")
    for lr, loss, step in zip(lrs, final_losses, final_steps):
        axes[1].annotate(f"{loss:.4f}\n@{step}", xy=(lr, loss), xytext=(0, 8), textcoords="offset points", ha="center", fontsize=8)

    axes[0].set_title("Validation Trajectory")
    axes[0].set_xlabel("training steps")
    axes[0].set_ylabel("validation loss")
    axes[0].grid(True, color="#d9d9d9", linewidth=0.7, alpha=0.7)
    axes[0].legend(frameon=False, fontsize=8)

    axes[1].set_title("Last Validation Loss by SparseAdam LR Mult")
    axes[1].set_xlabel("ROM_SPARSE_ADAM_LR_MUL")
    axes[1].set_ylabel("validation loss")
    axes[1].set_xscale("log")
    axes[1].grid(True, which="both", color="#d9d9d9", linewidth=0.7, alpha=0.7)

    fig.suptitle("Layer-2 Hashed ROM SparseAdam LR Sweep")
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out)
    if args.out.suffix.lower() != ".svg":
        fig.savefig(args.out.with_suffix(".svg"))


if __name__ == "__main__":
    main()
