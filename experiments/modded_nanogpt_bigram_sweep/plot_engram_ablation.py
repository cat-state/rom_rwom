#!/usr/bin/env python3
import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt


VAL_RE = re.compile(r"step:(?P<step>\d+)/\d+\s+val_loss:(?P<loss>nan|[0-9.]+)")
CFG_RE = re.compile(
    r"engram_dim=(?P<dim>\d+).*?engram_heads=(?P<heads>\d+).*?engram_max_ngram=(?P<ngram>\d+)",
    re.S,
)


def read_points(path: Path) -> tuple[list[int], list[float]]:
    steps: list[int] = []
    losses: list[float] = []
    for match in VAL_RE.finditer(path.read_text(errors="replace")):
        loss = float(match.group("loss"))
        if loss != loss:
            continue
        steps.append(int(match.group("step")))
        losses.append(loss)
    return steps, losses


def label_for(path: Path) -> str:
    text = path.read_text(errors="replace")
    match = CFG_RE.search(text)
    if match:
        return f"d{match.group('dim')} h{match.group('heads')} n2-{match.group('ngram')}"
    return path.stem


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("logs", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, default=Path("engram_ablation.png"))
    args = parser.parse_args()

    rows = []
    for path in args.logs:
        steps, losses = read_points(path)
        if losses:
            rows.append((label_for(path), path, steps, losses))
    if not rows:
        raise SystemExit("No validation points found")
    rows.sort(key=lambda row: row[3][-1])

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.0), dpi=180)
    colors = ["#2e8b57", "#3b6fb6", "#8b5a2b", "#7a4eb3", "#c44536", "#555555"]

    for idx, (label, _path, steps, losses) in enumerate(rows):
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
    finals = [row[3][-1] for row in rows]
    bar_colors = [colors[idx % len(colors)] for idx in range(len(rows))]
    axes[1].bar(labels, finals, color=bar_colors, width=0.62)
    for idx, loss in enumerate(finals):
        axes[1].annotate(f"{loss:.4f}", xy=(idx, loss), xytext=(0, 4), textcoords="offset points", ha="center", fontsize=8)

    axes[0].set_title("Validation Trajectory")
    axes[0].set_xlabel("training steps")
    axes[0].set_ylabel("validation loss")
    axes[0].grid(True, color="#d9d9d9", linewidth=0.7, alpha=0.7)
    axes[0].legend(frameon=False, fontsize=8)

    axes[1].set_title("Final 500-Step Validation Loss")
    axes[1].set_ylabel("validation loss")
    axes[1].tick_params(axis="x", rotation=18)
    axes[1].grid(True, axis="y", color="#d9d9d9", linewidth=0.7, alpha=0.7)

    fig.suptitle("Layer-2 Engram-Style Memory Ablations")
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out)
    if args.out.suffix.lower() != ".svg":
        fig.savefig(args.out.with_suffix(".svg"))


if __name__ == "__main__":
    main()
