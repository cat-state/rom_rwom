#!/usr/bin/env python3
import argparse
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt


VAL_RE = re.compile(r"step:(?P<step>\d+)/\d+\s+val_loss:(?P<loss>nan|[0-9.]+)")
BETA2_RE = re.compile(r"rom_sparse_adam_beta2=(?P<beta2>[0-9.]+)")
SGD_MOM_RE = re.compile(r"rom_sparse_sgd_momentum=(?P<momentum>[0-9.]+)")


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


def label_for(path: Path) -> str:
    text = path.read_text(errors="replace")
    name = path.name
    if "sparsesgd" in name:
        momentum = SGD_MOM_RE.search(text)
        suffix = momentum.group("momentum") if momentum else "?"
        return f"SparseSGD mom={suffix}"
    beta2 = BETA2_RE.search(text)
    if beta2:
        return f"SparseAdam beta2={float(beta2.group('beta2')):g}"
    return path.stem


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("logs", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, default=Path("sparse_optimizer_ablation.png"))
    args = parser.parse_args()

    rows = []
    for path in args.logs:
        steps, losses = read_points(path)
        if not losses:
            continue
        rows.append((label_for(path), steps, losses))
    if not rows:
        raise SystemExit("No completed validation curves found")

    rows.sort(key=lambda row: (row[0].startswith("SparseSGD"), row[0]))

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.0), dpi=180)
    colors = ["#2f6fbb", "#c44536", "#2e8b57", "#6b4e9b", "#555555"]

    for idx, (label, steps, losses) in enumerate(rows):
        color = colors[idx % len(colors)]
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

    labels = [row[0] for row in rows]
    finals = [row[2][-1] for row in rows]
    bar_colors = [colors[idx % len(colors)] for idx in range(len(rows))]
    axes[1].bar(labels, finals, color=bar_colors, width=0.62)
    for idx, loss in enumerate(finals):
        axes[1].annotate(f"{loss:.4f}", xy=(idx, loss), xytext=(0, 4), textcoords="offset points", ha="center", fontsize=8)

    axes[0].set_title("Validation Trajectory")
    axes[0].set_xlabel("training steps")
    axes[0].set_ylabel("validation loss")
    axes[0].grid(True, color="#d9d9d9", linewidth=0.7, alpha=0.7)
    axes[0].legend(frameon=False, fontsize=8)

    axes[1].set_title("Last Validation Loss")
    axes[1].set_ylabel("validation loss")
    axes[1].tick_params(axis="x", rotation=18)
    axes[1].grid(True, axis="y", color="#d9d9d9", linewidth=0.7, alpha=0.7)

    fig.suptitle("Layer-2 Hashed ROM Sparse State Optimizer Ablation")
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out)
    if args.out.suffix.lower() != ".svg":
        fig.savefig(args.out.with_suffix(".svg"))


if __name__ == "__main__":
    main()
