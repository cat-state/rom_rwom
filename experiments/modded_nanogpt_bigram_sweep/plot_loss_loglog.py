#!/usr/bin/env python3
import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt


VAL_RE = re.compile(r"step:(?P<step>\d+)/\d+\s+val_loss:(?P<loss>nan|[0-9.]+)")


def read_points(path: Path, *, include_step_zero: bool = False, finite_only: bool = True) -> tuple[list[int], list[float]]:
    steps: list[int] = []
    losses: list[float] = []
    for match in VAL_RE.finditer(path.read_text(errors="replace")):
        step = int(match.group("step"))
        if step == 0 and not include_step_zero:
            continue
        loss = float(match.group("loss"))
        if finite_only and loss != loss:
            continue
        steps.append(step)
        losses.append(loss)
    return steps, losses


def first_nan_step(path: Path, *, min_step: int) -> int | None:
    for match in VAL_RE.finditer(path.read_text(errors="replace")):
        step = int(match.group("step"))
        if step >= min_step and match.group("loss") == "nan":
            return step
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--rom", type=Path, required=True)
    parser.add_argument("--live", type=Path)
    parser.add_argument("--live-label", default="live: per-token ROM GA2")
    parser.add_argument("--series", nargs=3, action="append", metavar=("LABEL", "PATH", "COLOR"), default=[])
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    series = [
        ("baseline: 5x hashed bigram table", args.baseline, "#3b6fb6"),
        ("ROM: 5x layer-2 hashed bigram state", args.rom, "#c44949"),
    ]
    if args.live is not None:
        series.append((args.live_label, args.live, "#2f8f5b"))
    for label, path, color in args.series:
        series.append((label, Path(path), color))

    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.2), dpi=180)
    omitted_notes: list[str] = []
    for ax, min_step, title in [
        (axes[0], 1, "Full Run"),
        (axes[1], 1000, "Post-1k Steps"),
    ]:
        for label, path, color in series:
            steps, losses = read_points(path)
            points = [(step, loss) for step, loss in zip(steps, losses) if step >= min_step]
            if not points:
                all_steps, all_losses = read_points(path, include_step_zero=True)
                if all_steps == [0] and min_step == 1:
                    omitted_notes.append(f"{label}: step 0 val {all_losses[0]:.4f} omitted from log-x")
                elif not all_steps and min_step == 1:
                    omitted_notes.append(f"{label}: no validation point emitted yet")
                continue
            nan_step = first_nan_step(path, min_step=min_step)
            if nan_step is not None and min_step == 1:
                omitted_notes.append(f"{label}: NaN from step {nan_step}")
            plot_steps, plot_losses = zip(*points)
            ax.plot(plot_steps, plot_losses, linewidth=2.0, color=color, label=label)
            ax.annotate(
                f"{plot_losses[-1]:.4f}",
                xy=(plot_steps[-1], plot_losses[-1]),
                xytext=(6, 0),
                textcoords="offset points",
                va="center",
                fontsize=9,
                color=color,
            )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("training steps (log scale)")
        ax.set_title(title)
        ax.grid(True, which="both", color="#d9d9d9", linewidth=0.7, alpha=0.7)

    axes[0].set_ylabel("validation loss (log scale)")
    if omitted_notes:
        axes[0].text(
            0.03,
            0.04,
            "\n".join(omitted_notes),
            transform=axes[0].transAxes,
            fontsize=8.5,
            color="#555555",
            va="bottom",
        )
    axes[1].legend(frameon=False, loc="upper right", fontsize=9)
    fig.suptitle("Validation Loss Scaling: Baseline vs ROM Variants")
    fig.tight_layout()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out)
    if args.out.suffix.lower() != ".svg":
        fig.savefig(args.out.with_suffix(".svg"))


if __name__ == "__main__":
    main()
